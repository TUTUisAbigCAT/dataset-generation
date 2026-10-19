"""Independent validation adapter; the CPU cases test mathematics only."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_original_sampler import extract_original_pipeline
from validation_sampling import _sample_windows_from_original, sample_validation_windows


class TinyDenoiser(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.05, device=device))
        self.seen = []

    def forward(self, x, times):
        self.seen.append((int(times[0, 0]), x.device, times.device))
        return (x[:, :, :1] * self.scale + x[:, :, 1:2] * .01).permute(0, 2, 1)


def fixture(device="cpu", length=6, steps=200):
    device = torch.device(device)
    args = SimpleNamespace(window_size=length, stride=length // 2, timesteps=steps)
    windows = torch.arange(2 * length * 10, device=device, dtype=torch.float32).reshape(2, length, 10) / 100
    betas = torch.linspace(.0001, .02, steps, device=device).reshape(steps, 1)
    config = {"T": steps, "betas": betas, "alphas": 1 - betas,
              "alpha_bars": torch.cumprod(1 - betas, dim=0)}
    return TinyDenoiser(device).eval(), windows, config, args


def direct_upstream(model, clean, config, args):
    """Independent setup of the raw excerpt, without invoking the adapter."""
    noise, _, generation, _ = extract_original_pipeline()
    windows = clean.clone()
    windows[:, :, 0] = 0
    device = clean.device
    scope = dict(torch=torch, sqrt=torch.sqrt,
        args=SimpleNamespace(window_size=args.window_size, stride=args.window_size),
        model=model, device=device, diffusion_config=config,
        test_dataset=SimpleNamespace(inputs=windows), test_dataloader=[windows],
        mask_dataloader=[torch.ones(windows.shape[:2], dtype=torch.bool, device=device)],
        hierarchical_column_indices=torch.arange(1, 10, device=device),
        non_hier_cols=torch.tensor([0], device=device), num_ops=0, trial=0)
    exec(compile(noise, "upstream_noise", "exec"), scope)
    with torch.device(device), contextlib.redirect_stdout(io.StringIO()):
        exec(compile(generation, "upstream_generation", "exec"), scope)
    return scope["synth_tensor"].reshape_as(windows)[:, :, 0].detach()


class ValidationSamplingTests(unittest.TestCase):
    def test_all_200_steps_match_exact_upstream_and_preserve_training_state(self):
        model, clean, config, args = fixture()
        original_input = clean.clone()
        original_args = vars(args).copy()
        original_schedule = {name: value.clone() for name, value in config.items() if torch.is_tensor(value)}
        model.scale.grad = torch.tensor(3.)
        original_parameter = model.scale.detach().clone()
        with torch.random.fork_rng():
            torch.manual_seed(1951)
            generated, metadata = _sample_windows_from_original(model, clean, config, args, "cpu")
            torch.manual_seed(1951)
            expected = direct_upstream(TinyDenoiser().eval(), clean, config, args)
        self.assertTrue(torch.equal(generated, expected))
        self.assertEqual(tuple(generated.shape), (2, 6))
        self.assertFalse(generated.requires_grad)
        self.assertTrue(torch.isfinite(generated).all())
        self.assertEqual([row[0] for row in model.seen], list(range(199, -1, -1)))
        self.assertEqual(metadata["denoiser_calls"], 200)
        self.assertEqual(metadata["diffusion_steps"], 200)
        self.assertEqual(metadata["training_stride"], 3)
        self.assertEqual(metadata["inference_stride"], 6)
        self.assertTrue(metadata["condition_columns_bitwise_preserved"])
        self.assertTrue(metadata["source_provenance"]["excerpts_byte_identical_to_upstream"])
        self.assertFalse(metadata["source_provenance"]["sampler_expressions_edited"])
        self.assertFalse(metadata["clipping_applied"])
        self.assertEqual(vars(args), original_args)
        self.assertTrue(torch.equal(clean, original_input))
        self.assertTrue(torch.equal(model.scale.detach(), original_parameter))
        self.assertEqual(model.scale.grad.item(), 3.)
        for name, expected_schedule in original_schedule.items():
            self.assertTrue(torch.equal(config[name], expected_schedule))

    def test_no_target_leak_no_cross_window_guidance_and_conditions_used(self):
        model, clean, config, args = fixture(steps=5)
        gradient_calls = []
        original_grad = torch.autograd.grad
        capturing = False

        def capture_grad(*positional, **keyword):
            nonlocal capturing
            # torch.device uses __torch_function__ and re-enters this patched
            # public function with tuple-normalized arguments. Record once.
            if capturing:
                return original_grad(*positional, **keyword)
            capturing = True
            try:
                result = original_grad(*positional, **keyword)
            finally:
                capturing = False
            gradient_calls.append((positional[0].detach().clone(), result[0].detach().clone()))
            return result

        def sample(value):
            torch.manual_seed(70)
            return _sample_windows_from_original(model, value, config, args, "cpu")[0]

        with torch.random.fork_rng(), mock.patch("torch.autograd.grad", side_effect=capture_grad):
            initial = sample(clean)
            replaced_target = clean.clone()
            replaced_target[:, :, 0] = -9876
            target_changed = sample(replaced_target)
            other_conditions = clean.clone()
            other_conditions[1, :, 1] += 10
            conditions_changed = sample(other_conditions)
        self.assertTrue(torch.equal(initial, target_changed))
        self.assertTrue(torch.equal(initial[0], conditions_changed[0]))
        self.assertFalse(torch.equal(initial[1], conditions_changed[1]))
        self.assertEqual(len(gradient_calls), 15)
        for loss, gradient in gradient_calls:
            self.assertEqual(torch.count_nonzero(loss).item(), 0)
            self.assertEqual(torch.count_nonzero(gradient).item(), 0)

    def test_production_api_rejects_cpu_without_fallback(self):
        model, clean, config, args = fixture()
        with self.assertRaisesRegex(RuntimeError, "CUDA is required.*CPU fallback is disabled"):
            sample_validation_windows(model, clean, config, args, "cpu")

    def test_incomplete_schedule_and_training_mode_are_rejected(self):
        model, clean, config, args = fixture(steps=5)
        with self.assertRaisesRegex(ValueError, "complete training diffusion schedule"):
            _sample_windows_from_original(model, clean, {**config, "T": 4}, args, "cpu")
        with self.assertRaisesRegex(ValueError, "eval mode"):
            _sample_windows_from_original(model.train(), clean, config, args, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable; not a GPU execution result")
    def test_public_api_executes_full_schedule_on_cuda(self):
        device = torch.device("cuda", torch.cuda.current_device())
        model, clean, config, args = fixture(device=device, length=3000)
        with torch.random.fork_rng(devices=[device]):
            generated, metadata = sample_validation_windows(model, clean, config, args, device)
        self.assertEqual(generated.device, device)
        self.assertEqual(generated.shape, (2, 3000))
        self.assertEqual(metadata["denoiser_calls"], 200)
        self.assertTrue(all(x == device and t == device for _, x, t in model.seen))


if __name__ == "__main__":
    unittest.main()
