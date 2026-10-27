"""Shared noise values must not imply shared writable sampling state."""
from __future__ import annotations

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
from scripts.evaluate_original_sampler import extract_corrected_pipeline, extract_original_pipeline


def noise_function(corrected):
    source = (extract_corrected_pipeline if corrected else extract_original_pipeline)()[0]
    namespace = {"torch": torch}
    exec(compile(source, "<noise_storage_test>", "exec"), namespace)
    return namespace["create_pipelined_noise"]


def arguments():
    return SimpleNamespace(window_size=8, s4_lmax=8, stride=2, timesteps=5,
                           beta_0=.0001, beta_T=.02, diffusion_schedule="linear",
                           terminal_alpha_bar=1e-8)


class PositionDependentDenoiser(torch.nn.Module):
    """Adjacent windows predict different values at the same global position."""
    def forward(self, x, times):
        local_position = torch.arange(x.shape[1], dtype=x.dtype, device=x.device)[None, :]
        return (.02*x[:, :, 0] + .04*local_position).unsqueeze(1)


class PipelinedNoiseStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def check_noise_storage(self, device):
        args = arguments()
        batch = torch.zeros(4, 8, 10, device=device)
        with torch.device(device):
            noise = noise_function(True)(batch, args)
        self.assertEqual(noise.device, device)
        self.assertTrue(torch.equal(noise[1:, :6], noise[:-1, 2:]))
        neighbor_before = noise[1, 0].clone()
        noise[0, 2] += 13
        self.assertTrue(torch.equal(noise[1, 0], neighbor_before))
        desired = torch.arange(4, dtype=noise.dtype, device=device)[:, None, None].expand(4, 8, 1)
        noise[:, :, torch.tensor([0], device=device)] = desired
        self.assertTrue(torch.equal(noise[:, :, 0], desired[:, :, 0]))

    def test_overlap_noise_values_equal_but_writes_are_independent(self):
        self.check_noise_storage(torch.device("cpu"))

    def test_materialization_preserves_every_noise_value_and_random_state(self):
        args = arguments()
        batch = torch.zeros(4, 8, 10)
        original = noise_function(False)
        corrected = noise_function(True)
        with torch.random.fork_rng():
            torch.manual_seed(1742)
            expected = original(batch, args).clone()
            expected_state = torch.get_rng_state().clone()
            torch.manual_seed(1742)
            actual = corrected(batch, args)
            actual_state = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_state, expected_state))

    def test_actual_overlap_loop_produces_nonzero_stitch_loss_and_gradient(self):
        args = arguments()
        config = build_diffusion_config(args)
        span = torch.zeros(14, 10)
        span[:, 1:] = torch.arange(14)[:, None]/10
        trace = []
        original_grad = torch.autograd.grad
        inside = False

        def record_gradient(outputs, inputs, *positional, **keyword):
            nonlocal inside
            # Local device contexts can re-enter patched torch functions.
            if inside:
                return original_grad(outputs, inputs, *positional, **keyword)
            inside = True
            try:
                result = original_grad(outputs, inputs, *positional, **keyword)
            finally:
                inside = False
            trace.append((outputs.detach().clone(), result[0].detach().clone()))
            return result

        with torch.random.fork_rng(), mock.patch("torch.autograd.grad", side_effect=record_gradient):
            torch.manual_seed(1742)
            generated, metadata = night_sampling._sample_group(
                PositionDependentDenoiser().eval(), span, 0, config, args, "cpu", stride=2)
        self.assertEqual(len(trace), args.timesteps)
        # All IBI is unknown, so the only possible nonzero loss is stitching.
        self.assertTrue(all(float(loss[1:].sum()) > 0 for loss, _ in trace))
        self.assertTrue(all(int(torch.count_nonzero(gradient)) > 0 for _, gradient in trace))
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for _, gradient in trace))
        self.assertTrue(all(float(loss[0]) == 0 for loss, _ in trace))
        self.assertTrue(torch.equal(generated[:, 1:], span[:, 1:]))
        self.assertTrue(metadata["all_finite"])

    def test_one_window_sampling_preserves_values_and_random_state(self):
        args = arguments()
        config = build_diffusion_config(args)
        span = torch.arange(80, dtype=torch.float32).reshape(8, 10)/100
        corrected_noise, generation, provenance = night_sampling._night_sampler()
        raw_noise = compile(extract_original_pipeline()[0], "<raw_noise_reference>", "exec")
        with torch.random.fork_rng():
            torch.manual_seed(1742)
            actual, _ = night_sampling._sample_group(
                PositionDependentDenoiser().eval(), span, 0, config, args, "cpu", stride=2)
            actual_state = torch.get_rng_state().clone()
            torch.manual_seed(1742)
            # Keep corrected DDPM and night projection; vary storage only.
            with mock.patch.object(night_sampling, "_night_sampler", return_value=(raw_noise, generation, provenance)):
                expected, _ = night_sampling._sample_group(
                    PositionDependentDenoiser().eval(), span, 0, config, args, "cpu", stride=2)
            expected_state = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_state, expected_state))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable; actual GPU storage check skipped")
    def test_overlap_storage_on_actual_cuda_device(self):
        self.check_noise_storage(torch.device("cuda", torch.cuda.current_device()))


if __name__ == "__main__":
    unittest.main()
