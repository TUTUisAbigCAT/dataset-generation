"""Numerical unit tests of the real sampling loop, using CPU test tensors only."""
from __future__ import annotations

import ast
import contextlib
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_original_sampler


def independent_corrected_pipeline():
    """Build the reference through an AST edit, independent of the runtime patch."""
    noise, _, compiled, _ = evaluate_original_sampler.extract_original_pipeline()
    tree = ast.parse(compiled)
    replacements = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "vari"
                and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult)):
            node.value.left = ast.Call(
                func=ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()),
                                   attr="sqrt", ctx=ast.Load()),
                args=[node.value.left], keywords=[])
            replacements += 1
    if replacements != 1:
        raise AssertionError("Expected exactly one posterior-noise coefficient")
    return noise, ast.unparse(ast.fix_missing_locations(tree))


class ZeroDenoiser(torch.nn.Module):
    def __init__(self, record_states=False):
        super().__init__()
        self.record_states = record_states
        self.states = []
        self.steps = []

    def forward(self, x, times):
        self.steps.append(int(times[0, 0]))
        if self.record_states:
            self.states.append(x[:, :, 0].detach().clone())
        # Preserve the input graph so the unmodified guidance expression runs.
        return (x[:, :, :1] * 0).permute(0, 2, 1)


def run_loop(betas, *, corrected, samples=8192, seed=6151, record_states=False):
    extractor = (evaluate_original_sampler.extract_corrected_pipeline if corrected
                 else evaluate_original_sampler.extract_original_pipeline)
    noise, _, compiled, _ = extractor()
    betas = torch.as_tensor(betas, dtype=torch.float32).reshape(-1, 1)
    windows = torch.zeros(samples // 2, 2, 2)
    windows[:, :, 1] = 1
    model = ZeroDenoiser(record_states).eval()
    scope = dict(torch=torch, sqrt=torch.sqrt,
        args=SimpleNamespace(window_size=2, stride=2), model=model,
        device=torch.device("cpu"),
        diffusion_config={"T": len(betas), "betas": betas,
                          "alphas": 1 - betas, "alpha_bars": torch.cumprod(1 - betas, dim=0)},
        test_dataset=SimpleNamespace(inputs=windows), test_dataloader=[windows],
        mask_dataloader=[torch.ones(windows.shape[:2], dtype=torch.bool)],
        hierarchical_column_indices=torch.tensor([1]), non_hier_cols=torch.tensor([0]),
        num_ops=0, trial=0)
    original_normal = torch.normal
    draws = []

    def record_normal(*args, **kwargs):
        result = original_normal(*args, **kwargs)
        draws.append((tuple(result.shape), result.dtype, result.device))
        return result

    # No production API is weakened: this directly executes the extracted
    # mathematical loop with a CPU test model and an entirely unknown signal.
    with torch.random.fork_rng(), mock.patch("torch.normal", side_effect=record_normal):
        torch.manual_seed(seed)
        exec(compile(noise, "reference_noise", "exec"), scope)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(compiled, "posterior_noise_test", "exec"), scope)
        rng_after = torch.get_rng_state().clone()
    result = scope["synth_tensor"][:, 0].detach().reshape(windows.shape[:2])
    if torch.count_nonzero(scope["grad"]):
        raise AssertionError("Independent all-unknown windows must have zero guidance")
    return result, model, draws, rng_after


class PosteriorNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_intermediate_innovations_have_posterior_variance(self):
        betas = [.03, .12, .24, .35]
        output, model, _, _ = run_loop(betas, corrected=True, samples=16384, record_states=True)
        after = model.states[1:] + [output]
        alpha_bar = [math.prod(1 - b for b in betas[:i + 1]) for i in range(len(betas))]
        self.assertEqual(model.steps, [3, 2, 1, 0])
        for t, before, following in zip(model.steps, model.states, after):
            # Recover the added innovation from consecutive actual model inputs.
            innovation = following.double() - before.double() / math.sqrt(1 - betas[t])
            with self.subTest(step=t):
                if t == 0:
                    self.assertLess(innovation.abs().max().item(), 2e-6)
                else:
                    expected_variance = betas[t] * (1 - alpha_bar[t - 1]) / (1 - alpha_bar[t])
                    self.assertAlmostEqual(innovation.var(unbiased=False).item(),
                                           expected_variance, delta=.045 * expected_variance)
                    self.assertLess(abs(innovation.mean().item()),
                                    5 * math.sqrt(expected_variance / innovation.numel()))

    def test_raw_and_corrected_consume_identical_random_draws(self):
        for steps in (1, 5, 200):
            with self.subTest(steps=steps):
                betas = torch.linspace(.0001, .02, steps)
                corrected, _, draws, state = run_loop(betas, corrected=True, samples=8)
                raw, _, raw_draws, raw_state = run_loop(betas, corrected=False, samples=8)
                self.assertEqual(draws, raw_draws)
                # Initial noise + T conditional draws + T temporary-state draws
                # + (T-1) posterior innovations: exactly 3T calls.
                self.assertEqual(len(draws), 3 * steps)
                self.assertTrue(torch.equal(state, raw_state))
                if steps == 1:
                    self.assertTrue(torch.equal(corrected, raw))
                else:
                    self.assertFalse(torch.equal(corrected, raw))

    def test_one_step_has_no_posterior_noise(self):
        output, model, draws, _ = run_loop([.2], corrected=True, samples=32, record_states=True)
        self.assertEqual(model.steps, [0])
        self.assertEqual(len(draws), 3)
        self.assertFalse(any(shape[-1] == 1 for shape, _, _ in draws))
        torch.testing.assert_close(output.double(), model.states[0].double() / math.sqrt(.8),
                                   rtol=2e-7, atol=2e-7)

    def test_200_step_output_matches_closed_form_variance_raw_does_not(self):
        betas = torch.linspace(.0001, .02, 200).double().tolist()
        # Independent closed-form propagation for epsilon=0: each innovation
        # is amplified by all subsequent reverse steps, not a copied sampler.
        cumulative = [math.prod(1 - b for b in betas[:i + 1]) for i in range(len(betas))]
        posterior = [betas[t] * (1 - cumulative[t - 1]) / (1 - cumulative[t])
                     for t in range(1, len(betas))]
        expected = 1 / cumulative[-1] + sum(q / cumulative[t - 1]
                                           for t, q in enumerate(posterior, 1))
        corrected, model, _, _ = run_loop(betas, corrected=True)
        raw, _, _, _ = run_loop(betas, corrected=False)
        actual = corrected.double().var(unbiased=False).item()
        raw_variance = raw.double().var(unbiased=False).item()
        self.assertEqual(model.steps, list(range(199, -1, -1)))
        self.assertAlmostEqual(actual, expected, delta=.07 * expected)
        self.assertLess(abs(corrected.double().mean().item()),
                        5 * math.sqrt(expected / corrected.numel()))
        self.assertGreater(abs(raw_variance - expected), .2 * expected)


if __name__ == "__main__":
    unittest.main()
