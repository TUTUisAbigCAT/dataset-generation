"""Whole-night coverage and real sampling-loop checks with tiny test denoisers."""
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
import night_sampling
from diffusion_schedule import build_diffusion_config
from tests.test_posterior_noise import independent_corrected_pipeline


class TinyNightDenoiser(torch.nn.Module):
    def __init__(self, device="cpu", record=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.99, device=device))
        self.record = record
        self.seen = []

    def forward(self, x, times):
        if self.record:
            self.seen.append((int(times[0, 0]), x.detach().clone()))
        return (self.scale * x[:, :, :1] + .001 * x[:, :, 1:2]).permute(0, 2, 1)


def fixture(length=8, span_length=14, steps=200, schedule="linear_power", device="cpu"):
    device = torch.device(device)
    args = SimpleNamespace(window_size=length, s4_lmax=length, stride=2,
                           timesteps=steps, beta_0=.0001, beta_T=.02,
                           diffusion_schedule=schedule, terminal_alpha_bar=1e-8)
    config = {key: value.to(device) if torch.is_tensor(value) else value
              for key, value in build_diffusion_config(args).items()}
    clean = torch.arange(span_length * 10, dtype=torch.float32, device=device).reshape(span_length, 10) / 100
    return args, config, clean


def direct_all_unknown(model, clean, config, args, stride):
    """Actual corrected upstream loop, built by an independent AST correction."""
    noise, generation = independent_corrected_pipeline()
    clean = clean.clone()
    clean[:, 0] = 0
    windows = clean.unfold(0, args.window_size, stride).transpose(1, 2)
    device = clean.device
    scope = dict(torch=torch, sqrt=torch.sqrt,
                 args=SimpleNamespace(window_size=args.window_size, stride=stride),
                 model=model, device=device, diffusion_config=config,
                 test_dataset=SimpleNamespace(inputs=windows), test_dataloader=[windows],
                 mask_dataloader=[torch.ones(windows.shape[:2], dtype=torch.bool, device=device)],
                 hierarchical_column_indices=torch.arange(1, 10, device=device),
                 non_hier_cols=torch.tensor([0], device=device), num_ops=0, trial=0)
    exec(compile(noise, "<independent_noise>", "exec"), scope)
    with torch.device(device), contextlib.redirect_stdout(io.StringIO()):
        exec(compile(generation, "<independent_corrected_loop>", "exec"), scope)
    return scope["synth_tensor"].detach().clone()


class NightSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_plan_covers_tail_once_and_accounts_for_global_prefix(self):
        for length, total, stride, groups in ((8, 8, 3, 2), (8, 20, 3, 2),
                                               (8, 22, 3, 2), (8, 30, 8, 2),
                                               (8, 27, 2, 4)):
            with self.subTest(length=length, total=total, stride=stride, groups=groups):
                plan = night_sampling.plan_night(total, length, stride=stride, windows_per_group=groups)
                covered = 0
                all_starts = []
                for index, group in enumerate(plan):
                    start, end = group["start_sample"], group["end_sample"]
                    self.assertEqual(group["index"], index)
                    self.assertGreaterEqual(group["windows"], 1)
                    self.assertLessEqual(group["windows"], groups)
                    self.assertEqual(end-start, length+(group["windows"]-1)*stride)
                    self.assertLessEqual(start, covered)
                    self.assertGreater(end, covered)
                    self.assertEqual(group["known_prefix_samples"], covered-start)
                    self.assertEqual(group["append_samples"], end-covered)
                    all_starts.extend(start + stride*k for k in range(group["windows"]))
                    covered = end
                self.assertEqual(covered, total)
                self.assertEqual(sum(group["append_samples"] for group in plan), total)
                expected = list(range(0, total-length+1, stride))
                if expected[-1] != total-length:
                    expected.append(total-length)
                    self.assertEqual(plan[-1]["windows"], 1)
                self.assertEqual(all_starts, expected)

    def test_all_unknown_group_matches_original_corrected_sampler_and_rng(self):
        args, config, clean = fixture()
        model = TinyNightDenoiser().eval()
        original_clean = clean.clone()
        original_args = vars(args).copy()
        model.scale.grad = torch.tensor(3.)
        with torch.random.fork_rng():
            torch.manual_seed(321)
            got, _ = night_sampling._sample_group(model, clean, 0, config, args, "cpu", stride=2)
            rng_after = torch.get_rng_state().clone()
            torch.manual_seed(321)
            expected = direct_all_unknown(TinyNightDenoiser().eval(), clean, config, args, 2)
            expected_rng_after = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(got, expected))
        self.assertTrue(torch.equal(rng_after, expected_rng_after))
        self.assertTrue(torch.equal(clean, original_clean))
        self.assertEqual(vars(args), original_args)
        self.assertEqual(model.scale.grad.item(), 3.)
        self.assertFalse(got.requires_grad)

    def test_prefix_is_projected_at_every_step_in_every_overlapping_window(self):
        args, config, clean = fixture()
        prefix, stride = 6, 2
        model = TinyNightDenoiser(record=True).eval()
        draws, gradients = [], []
        original_normal, original_grad = torch.normal, torch.autograd.grad
        recording_grad = False

        def record_normal(*positional, **keyword):
            result = original_normal(*positional, **keyword)
            draws.append(result.detach().clone())
            return result

        def record_grad(*positional, **keyword):
            nonlocal recording_grad
            # torch.device's __torch_function__ can re-enter the patched API.
            if recording_grad:
                return original_grad(*positional, **keyword)
            recording_grad = True
            try:
                result = original_grad(*positional, **keyword)
            finally:
                recording_grad = False
            gradients.append(result[0].detach().clone())
            return result

        with torch.random.fork_rng(), mock.patch("torch.normal", side_effect=record_normal), \
             mock.patch("torch.autograd.grad", side_effect=record_grad):
            torch.manual_seed(321)
            got, _ = night_sampling._sample_group(model, clean, prefix, config, args, "cpu", stride=stride)
        self.assertEqual(len(model.seen), 200)
        self.assertEqual(len(draws), 600)
        self.assertEqual(len(gradients), 200)
        windows = clean.unfold(0, args.window_size, stride).transpose(1, 2)
        known = (torch.arange(len(clean)) < prefix).unfold(0, args.window_size, stride)
        self.assertEqual(known.sum(1).tolist(), [6, 4, 2, 0])
        for index, (t, observed) in enumerate(model.seen):
            self.assertEqual(t, 199-index)
            noise = draws[1+3*index].unfold(0, args.window_size, stride).transpose(1, 2)
            abar = config["alpha_bars"][t]
            expected_prefix = abar.sqrt()*windows[:, :, 0] + (1-abar).sqrt()*noise[:, :, 0]
            self.assertTrue(torch.equal(observed[:, :, 0][known], expected_prefix[known]))
            self.assertTrue(torch.equal(observed[:, :, 1:], windows[:, :, 1:]))
            self.assertTrue(bool(torch.isfinite(observed).all()))
        self.assertTrue(torch.equal(got[:prefix], clean[:prefix]))
        self.assertTrue(torch.equal(got[:, 1:], clean[:, 1:]))
        self.assertTrue(bool(torch.isfinite(got).all()))
        # An old x0/alpha_bar known-value penalty gives ~1e4 gradients for this
        # 0.99*x toy at terminal SNR 1e-8. Overlap-only guidance stays bounded.
        self.assertLess(max(float(g.abs().max()) for g in gradients), 100.)

    def test_unknown_target_values_are_not_used(self):
        args, config, clean = fixture(steps=5, schedule="linear")
        changed = clean.clone()
        changed[6:, 0] = -9876
        with torch.random.fork_rng():
            torch.manual_seed(19)
            first, _ = night_sampling._sample_group(TinyNightDenoiser().eval(), clean, 6, config, args, "cpu", stride=2)
            torch.manual_seed(19)
            second, _ = night_sampling._sample_group(TinyNightDenoiser().eval(), changed, 6, config, args, "cpu", stride=2)
        self.assertTrue(torch.equal(first, second))

    def test_generation_reuses_previous_prefix_across_groups_and_keeps_tail(self):
        args, config, clean = fixture(span_length=22, steps=5, schedule="linear")
        conditions = clean[:, 1:].clone()
        before = conditions.clone()
        calls = []
        actual_group = night_sampling._sample_group

        def capture_group(model, span, prefix, diffusion, training_args, device, *, stride):
            generated, metadata = actual_group(model, span, prefix, diffusion, training_args, device, stride=stride)
            calls.append((span.detach().clone(), prefix, generated.detach().clone()))
            return generated, metadata

        with torch.random.fork_rng(), mock.patch.object(night_sampling, "_sample_group", side_effect=capture_group):
            torch.manual_seed(51)
            signal, _ = night_sampling._generate_night(TinyNightDenoiser().eval(), conditions, config, args,
                                                       "cpu", stride=3, windows_per_group=2)
        plan = night_sampling.plan_night(len(clean), args.window_size, stride=3, windows_per_group=2)
        self.assertEqual(len(calls), len(plan))
        self.assertEqual(signal.shape, (22,))
        self.assertTrue(bool(torch.isfinite(signal).all()))
        self.assertTrue(torch.equal(conditions, before))
        for (span, prefix, generated), group in zip(calls, plan):
            start, end = group["start_sample"], group["end_sample"]
            self.assertEqual(prefix, group["known_prefix_samples"])
            self.assertTrue(torch.equal(span[:, 1:], conditions[start:end]))
            self.assertTrue(torch.equal(span[:prefix, 0], signal[start:start+prefix]))
            self.assertTrue(torch.equal(generated[:prefix], span[:prefix]))
            self.assertTrue(torch.equal(generated[prefix:, 0], signal[start+prefix:end]))

    def test_production_entry_rejects_cpu(self):
        args, config, clean = fixture(steps=3, schedule="linear")
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            night_sampling.generate_night(TinyNightDenoiser().eval(), clean[:, 1:], config, args, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable; actual GPU check skipped")
    def test_production_entry_runs_prefix_groups_on_cuda(self):
        device = torch.device("cuda", torch.cuda.current_device())
        args, config, clean = fixture(length=1500, span_length=3750, steps=3,
                                      schedule="linear", device=device)
        signal, _ = night_sampling.generate_night(TinyNightDenoiser(device).eval(), clean[:, 1:], config,
                                                 args, device, stride=750, windows_per_group=2)
        self.assertEqual(signal.device, device)
        self.assertEqual(signal.shape, (3750,))
        self.assertTrue(bool(torch.isfinite(signal).all()))


if __name__ == "__main__":
    unittest.main()
