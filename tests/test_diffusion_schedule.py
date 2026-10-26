"""Schedule/prior contracts without constructing a model or invoking CUDA."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_schedule import (
    build_diffusion_config,
    diffusion_identity,
    diffusion_settings,
    validate_diffusion_config,
    verify_diffusion_identity,
)
from training_utils import fetchDiffusionConfig
from scripts.evaluate_denoising import DEFAULT_TRAINING_ARGS, load_training_run


def arguments(**changes):
    values = dict(beta_0=.0001, beta_T=.02, timesteps=200)
    values.update(changes)
    return SimpleNamespace(**values)


class DiffusionScheduleTests(unittest.TestCase):
    def test_missing_or_explicit_linear_is_bit_exact_legacy(self):
        for count in (1, 50, 200, 1000):
            for schedule in (None, "linear"):
                with self.subTest(steps=count, schedule=schedule):
                    args = arguments(timesteps=count)
                    if schedule is not None:
                        args.diffusion_schedule = schedule
                    actual = build_diffusion_config(args)
                    expected = fetchDiffusionConfig(args)
                    self.assertEqual(set(actual), set(expected))
                    self.assertEqual(actual["T"], count)
                    for key in ("betas", "alphas", "alpha_bars"):
                        self.assertEqual(actual[key].dtype, torch.float32)
                        self.assertEqual(actual[key].shape, (count, 1))
                        self.assertEqual(actual[key].device.type, "cpu")
                        self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def test_power_endpoint_and_smooth_step_sizes_match_independent_values(self):
        args = arguments(diffusion_schedule="linear_power", terminal_alpha_bar=1e-8)
        config = build_diffusion_config(args)
        # Independently calculated in float64 from the archived 200-step baseline.
        self.assertAlmostEqual(float(config["betas"][0]), .0009099374385618314, delta=1e-10)
        self.assertAlmostEqual(float(config["betas"][-1]), .16798639832500553, delta=1e-8)
        self.assertAlmostEqual(float(config["alpha_bars"][-1]), 1e-8, delta=1e-14)
        self.assertLess(float(config["betas"].max()), .2)
        self.assertLess(float(config["alphas"][-1].rsqrt()), 1.1)
        validate_diffusion_config(args, config)

    def test_power_schedule_is_finite_monotone_and_cumulatively_consistent(self):
        for count, target in ((50, 1e-8), (200, 1e-8), (200, 1e-6), (1000, 1e-8)):
            with self.subTest(steps=count, target=target):
                config = build_diffusion_config(arguments(
                    timesteps=count, diffusion_schedule="linear_power", terminal_alpha_bar=target))
                beta, alpha, abar = (config[key].reshape(-1) for key in ("betas", "alphas", "alpha_bars"))
                for value in (beta, alpha, abar):
                    self.assertTrue(bool(torch.isfinite(value).all()))
                    self.assertTrue(bool((value > 0).all()))
                    self.assertTrue(bool((value < 1).all()))
                self.assertTrue(bool((beta[1:] >= beta[:-1]).all()))
                self.assertTrue(bool((abar[1:] < abar[:-1]).all()))
                torch.testing.assert_close(alpha + beta, torch.ones_like(alpha), rtol=0, atol=1.2e-7)
                torch.testing.assert_close(torch.cumprod(alpha, 0), abar, rtol=1e-5, atol=0)
                self.assertAlmostEqual(float(abar[-1]) / target, 1., delta=1e-6)
                # The epsilon sampler and x0 diagnostic must have no zero denominators.
                self.assertTrue(bool(torch.isfinite(beta / torch.sqrt(1-abar)).all()))
                self.assertTrue(bool(torch.isfinite(torch.rsqrt(alpha)).all()))
                self.assertTrue(bool(torch.isfinite(torch.rsqrt(abar)).all()))
                previous = torch.cat((torch.ones(1), abar[:-1]))
                posterior_variance = beta * (1-previous) / (1-abar)
                self.assertEqual(float(posterior_variance[0]), 0.)
                self.assertTrue(bool((posterior_variance >= 0).all()))
                self.assertTrue(bool(torch.isfinite(posterior_variance.sqrt()).all()))

    def test_invalid_schedule_settings_and_nonrepresentable_endpoints_are_rejected(self):
        cases = [
            dict(timesteps=value) for value in (0, -1, True, 200., "200")
        ] + [
            {key: value} for key in ("beta_0", "beta_T")
            for value in (0., -1., 1., float("nan"), float("inf"), True)
        ] + [
            dict(beta_0=.03, beta_T=.02),
            dict(beta_0=1e-12),
            dict(diffusion_schedule="cosine"),
            dict(diffusion_schedule=True),
        ] + [
            dict(diffusion_schedule="linear_power", terminal_alpha_bar=value)
            for value in (0., -1., 1., float("nan"), float("inf"), True, 1e-320)
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                build_diffusion_config(arguments(**changes))

    def test_runtime_validation_detects_wrong_schedule_and_mutated_tensors(self):
        args = arguments(diffusion_schedule="linear_power", terminal_alpha_bar=1e-8)
        config = build_diffusion_config(args)
        with self.assertRaises(ValueError):
            validate_diffusion_config(args, fetchDiffusionConfig(args))
        for key in ("betas", "alphas", "alpha_bars"):
            broken = copy.deepcopy(config)
            broken[key][42] *= .99
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_diffusion_config(args, broken)
        broken = copy.deepcopy(config)
        broken["T"] = 199
        with self.assertRaises(ValueError):
            validate_diffusion_config(args, broken)

    def test_schedule_construction_preserves_rng_and_does_not_mutate_args(self):
        args = arguments(diffusion_schedule="linear_power", terminal_alpha_bar=1e-8)
        before_args = vars(args).copy()
        torch_rng = torch.get_rng_state().clone()
        numpy_rng = copy.deepcopy(np.random.get_state())
        build_diffusion_config(args)
        diffusion_identity(args)
        self.assertEqual(vars(args), before_args)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        after = np.random.get_state()
        self.assertEqual(numpy_rng[0], after[0])
        self.assertTrue(np.array_equal(numpy_rng[1], after[1]))
        self.assertEqual(numpy_rng[2:], after[2:])

    def test_frozen_identity_round_trip_and_schedule_mismatch(self):
        args = arguments(diffusion_schedule="linear_power", terminal_alpha_bar=1e-8)
        identity = diffusion_identity(args)
        self.assertEqual(json.loads(json.dumps(identity, allow_nan=False)), identity)
        self.assertEqual(identity, diffusion_identity(args, build_diffusion_config(args)))
        verify_diffusion_identity(args, identity)
        other = arguments(diffusion_schedule="linear_power", terminal_alpha_bar=1e-6)
        self.assertNotEqual(identity, diffusion_identity(other))
        with self.assertRaises(ValueError):
            verify_diffusion_identity(other, identity)
        with self.assertRaises(ValueError):
            verify_diffusion_identity(arguments(), identity)
        with self.assertRaises(ValueError):
            verify_diffusion_identity(args, None)
        verify_diffusion_identity(arguments(), None)

    def test_identity_hashes_cover_exact_runtime_tensor_bytes(self):
        args = arguments(diffusion_schedule="linear_power", terminal_alpha_bar=1e-8)
        config = build_diffusion_config(args)
        identity = diffusion_identity(args, config)
        self.assertEqual(identity["tensor_shape"], [200, 1])
        self.assertEqual(identity["tensor_dtype"], "float32")
        for key in ("betas", "alphas", "alpha_bars"):
            raw_bytes = config[key].numpy().astype("<f4", copy=False).tobytes()
            self.assertEqual(identity["tensor_sha256"][key], hashlib.sha256(raw_bytes).hexdigest())
        self.assertEqual(identity["betas_float32"], config["betas"].reshape(-1).tolist())
        for field in ("betas_float32", "tensor_sha256"):
            broken = copy.deepcopy(identity)
            if field == "betas_float32":
                broken[field][42] += .001
            else:
                broken[field]["alpha_bars"] = "0" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_diffusion_identity(args, broken)

    def test_new_run_default_requires_explicit_opt_in(self):
        self.assertEqual(diffusion_settings({})["diffusion_schedule"], "linear")
        self.assertEqual(diffusion_settings({}, default_schedule="linear_power")["diffusion_schedule"],
                         "linear_power")
        self.assertEqual(diffusion_settings({"diffusion_schedule": "linear"},
                         default_schedule="linear_power")["diffusion_schedule"], "linear")

    def test_current_global_configuration_cannot_change_legacy_manifest_schedule(self):
        # The only permitted read is the frozen manifest: no current JSON access.
        manifest = ROOT / "tests" / "old_run_test_only.json"
        config = ROOT / "tests" / "current_config_test_only.json"
        historical = dict(DEFAULT_TRAINING_ARGS)
        historical.pop("diffusion_schedule", None)
        historical.pop("terminal_alpha_bar", None)
        s4_hash = hashlib.sha256((ROOT / "TSImputers/S4Model.py").read_bytes()).hexdigest()
        text = json.dumps(dict(training_args=historical, data_identity={"window_epochs": 4},
                               source_sha256={"TSImputers/S4Model.py": s4_hash}))

        def read_frozen(path, *args, **kwargs):
            self.assertEqual(path, manifest, "Historical evaluation must not read current global config")
            return text

        with mock.patch.dict(os.environ, {"DREAMT_CONFIG_PATH": str(config)}), \
                mock.patch.object(Path, "read_text", autospec=True, side_effect=read_frozen), \
                mock.patch("scripts.evaluate_denoising.sha256", return_value=s4_hash):
            args, _, _ = load_training_run(manifest)
            actual = build_diffusion_config(args)
        self.assertEqual(getattr(args, "diffusion_schedule", "linear"), "linear")
        self.assertTrue(torch.equal(actual["alpha_bars"], fetchDiffusionConfig(args)["alpha_bars"]))
        self.assertAlmostEqual(float(actual["alpha_bars"][-1]), .13218275425061793, delta=1e-8)

    def test_power_manifest_requires_and_verifies_frozen_schedule_identity(self):
        training = dict(DEFAULT_TRAINING_ARGS, diffusion_schedule="linear_power", terminal_alpha_bar=1e-8)
        identity = diffusion_identity(SimpleNamespace(**training))
        s4_hash = hashlib.sha256((ROOT / "TSImputers/S4Model.py").read_bytes()).hexdigest()
        manifest = dict(training_args=training, data_identity={"window_epochs": 4},
                        source_sha256={"TSImputers/S4Model.py": s4_hash})
        path = ROOT / "tests" / "power_run_test_only.json"

        def load(value):
            with mock.patch.object(Path, "read_text", return_value=json.dumps(value)), \
                    mock.patch("scripts.evaluate_denoising.sha256", return_value=s4_hash):
                return load_training_run(path)

        with self.assertRaises(ValueError):
            load(manifest)
        manifest["diffusion_identity"] = identity
        restored, _, _ = load(manifest)
        self.assertEqual(diffusion_identity(restored), identity)
        manifest["diffusion_identity"] = copy.deepcopy(identity)
        manifest["diffusion_identity"]["tensor_sha256"]["alpha_bars"] = "0" * 64
        with self.assertRaises(ValueError):
            load(manifest)

    def test_corrected_200_step_sampler_is_finite_with_power_schedule(self):
        # Exercise the real reverse expressions with an analytic known-window
        # epsilon oracle. No S4 model, optimizer, or production CPU fallback.
        from validation_sampling import _sample_windows_from_original

        args = arguments(diffusion_schedule="linear_power", terminal_alpha_bar=1e-8,
                         window_size=6, s4_lmax=6, stride=3)
        config = build_diffusion_config(args)
        clean = torch.arange(120, dtype=torch.float32).reshape(2, 6, 10) / 100
        outer = self

        class KnownWindowOracle(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("reference", clean[:, :, 0].clone())
                self.calls = []

            def forward(self, value, timestep):
                step = int(timestep[0, 0])
                self.calls.append(step)
                outer.assertTrue(bool(torch.isfinite(value).all()))
                outer.assertTrue(torch.equal(value[:, :, 1:], clean[:, :, 1:]))
                abar = config["alpha_bars"][step]
                epsilon = (value[:, :, 0] - abar.sqrt() * self.reference) / (1-abar).sqrt()
                outer.assertTrue(bool(torch.isfinite(epsilon).all()))
                return epsilon[:, None, :]

        model = KnownWindowOracle().eval()
        original_grad = torch.autograd.grad
        guidance_checks, reentrant = [], False

        def verify_zero_guidance(*positional, **keyword):
            nonlocal reentrant
            if reentrant:
                return original_grad(*positional, **keyword)
            reentrant = True
            try:
                result = original_grad(*positional, **keyword)
            finally:
                reentrant = False
            self.assertEqual(torch.count_nonzero(positional[0]).item(), 0)
            self.assertEqual(torch.count_nonzero(result[0]).item(), 0)
            guidance_checks.append(True)
            return result

        with torch.random.fork_rng(), mock.patch("torch.autograd.grad", side_effect=verify_zero_guidance):
            torch.manual_seed(1951)
            generated, metadata = _sample_windows_from_original(model, clean, config, args, "cpu")
        self.assertEqual(model.calls, list(range(199, -1, -1)))
        self.assertEqual(len(guidance_checks), 200)
        self.assertTrue(bool(torch.isfinite(generated).all()))
        torch.testing.assert_close(generated, clean[:, :, 0], rtol=1e-5, atol=2e-5)
        self.assertTrue(metadata["condition_columns_bitwise_preserved"])
        self.assertTrue(metadata["last_step_guidance_exactly_zero"])
        self.assertTrue(metadata["all_finite"])
        self.assertEqual(metadata["denoiser_calls"], 200)
        self.assertEqual(metadata["diffusion_identity"], diffusion_identity(args))


if __name__ == "__main__":
    unittest.main()
