"""Direct-checkpoint inference contracts; tiny real S4, no optimization."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils import Preprocessor
from night_checkpoint import load_checkpoint_state, load_inference_config
from scripts.evaluate_denoising import DEFAULT_TRAINING_ARGS
from training_utils import fetchModel


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def small_s4(length=32):
    args = SimpleNamespace(**dict(DEFAULT_TRAINING_ARGS, dataset="DREAMT",
        window_size=length, s4_lmax=length, num_res_layers=2, res_channels=4,
        skip_channels=4, s4_dstate=8, diff_step_embed_in=8,
        diff_step_embed_mid=8, diff_step_embed_out=8))
    return fetchModel(10, 1, args).eval()


class ComplexFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.+2.j, 3.-4.j], dtype=torch.complex128))
        self.register_buffer("phase", torch.tensor([.3+.7j, -.9+.1j], dtype=torch.complex64))


class NightCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def check_full_s4_restore(self, device):
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            torch.manual_seed(42)
            original = small_s4().to(device)
            with torch.no_grad():
                # The baseline starts with a zero last layer. Make the test
                # sensitive to every S4 kernel instead of comparing two zeros.
                original.final_conv[2].conv.weight.normal_(mean=.15, std=.1)
                original.final_conv[2].conv.bias.fill_(.02)
            saved = snapshot(original)
            inputs = torch.randn(2, 32, 10, device=device)
            timesteps = torch.tensor([[17], [199]], device=device)
            with torch.no_grad():
                expected = original(inputs, timesteps)
            self.assertGreater(float(expected.std()), 1e-4)
            torch.manual_seed(901)
            restored = small_s4().to(device)
            fixed = next(name for name in saved if name.endswith("kernel.log_dt"))
            self.assertFalse(torch.equal(saved[fixed], restored.state_dict()[fixed]))
            with torch.no_grad():
                before = restored(inputs, timesteps)
            self.assertFalse(torch.equal(expected, before))
            metadata = load_checkpoint_state(restored, saved)
            with torch.no_grad():
                actual = restored(inputs, timesteps)
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(metadata["all_state_bitwise_equal"])
            self.assertTrue(metadata["parameters_and_buffers_restored"])
            self.assertFalse(metadata["training_source_identity_verified"])
            self.assertIn(fixed, metadata["buffer_storage_materialized"])
            self.assertEqual(set(metadata["s4_cached_lengths"].values()), {32})
            for name, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, saved[name]), name)
                self.assertEqual(value.device, device)

    def test_full_s4_fft_forward_restores_nonzero_output_and_different_fixed_buffers_cpu(self):
        self.check_full_s4_restore(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; actual GPU checkpoint check skipped")
    def test_full_s4_fft_forward_restores_nonzero_output_cuda(self):
        self.check_full_s4_restore(torch.device("cuda", torch.cuda.current_device()))

    def test_materialization_handles_actual_s4_stride_zero_buffers(self):
        with torch.random.fork_rng():
            original = small_s4()
            saved = snapshot(original)
            target = small_s4()
            broadcast = {name for name, value in target.named_buffers()
                         if any(size > 1 and stride == 0 for size, stride in zip(value.shape, value.stride()))}
            self.assertTrue(broadcast)
            with self.assertRaisesRegex(RuntimeError, "single memory location"):
                target.load_state_dict(saved, strict=True)
            metadata = load_checkpoint_state(target, saved)
            self.assertTrue(broadcast.issubset(metadata["buffer_storage_materialized"]))
            self.assertTrue(all(torch.equal(value, saved[name]) for name, value in target.state_dict().items()))

    def test_complex_parameters_and_buffers_keep_both_components_and_dtype(self):
        original = ComplexFixture()
        saved = snapshot(original)
        target = ComplexFixture()
        with torch.no_grad():
            target.weight.zero_()
            target.phase.zero_()
        metadata = load_checkpoint_state(target, saved)
        self.assertTrue(metadata["all_state_bitwise_equal"])
        self.assertEqual(target.weight.dtype, torch.complex128)
        self.assertEqual(target.phase.dtype, torch.complex64)
        self.assertTrue(torch.equal(target.weight, saved["weight"]))
        self.assertTrue(torch.equal(target.phase, saved["phase"]))
        self.assertTrue(bool((target.phase.imag != 0).all()))

    def test_incompatible_state_is_rejected_before_any_copy_or_buffer_materialization(self):
        source = ComplexFixture()
        valid = snapshot(source)
        invalid = [
            ({"state_dict": valid}, "raw state_dict"),
            ({"weight": valid["weight"]}, "raw state_dict"),
            ({**valid, "extra": torch.ones(1)}, "raw state_dict"),
            ({**valid, "weight": torch.ones(3, dtype=torch.complex128)}, "shape mismatch.*window_epochs"),
            ({**valid, "phase": valid["phase"].real.clone()}, "dtype mismatch"),
            ({**valid, "phase": torch.tensor([complex(1, float('nan')), 2j], dtype=torch.complex64)}, "nonfinite"),
            ({**valid, "weight": torch.tensor([complex(float('inf'), 0), 2j], dtype=torch.complex128)}, "nonfinite"),
            ({**valid, "phase": [1, 2]}, "only tensors"),
        ]
        for saved, message in invalid:
            with self.subTest(message=message):
                target = ComplexFixture()
                before = snapshot(target)
                buffer_identity = target.phase
                with self.assertRaisesRegex(ValueError, message):
                    load_checkpoint_state(target, saved)
                self.assertIs(target.phase, buffer_identity)
                self.assertTrue(all(torch.equal(value, before[name]) for name, value in target.state_dict().items()))

    def test_s4_cache_dimensions_require_matching_window_configuration(self):
        with torch.random.fork_rng():
            source = small_s4(32)
            with self.assertRaisesRegex(ValueError, "shape mismatch.*window_epochs"):
                load_checkpoint_state(small_s4(64), snapshot(source))
            target = small_s4(32)
            target.residual_layer.residual_blocks[0].S41.s4_layer.kernel.kernel.L = 64
            with self.assertRaisesRegex(ValueError, "cached length.*window_epochs"):
                load_checkpoint_state(target, snapshot(source))


class NightInferenceConfigTests(unittest.TestCase):
    def test_explicit_k4_k8_relative_paths_and_fixed_baseline_ignore_environment(self):
        with tempfile.TemporaryDirectory(prefix="night_checkpoint_", dir=ROOT) as directory:
            root = Path(directory)
            path = root / "config.json"
            for epochs in (4, 8):
                with self.subTest(window_epochs=epochs):
                    config = dict(data_dir="records", metadata_path="meta/participants.csv",
                        split_path="split.json", window_epochs=epochs, max_epochs=None,
                        validation={"batch_size": 7})
                    path.write_text(json.dumps(config), encoding="utf-8")
                    with mock.patch.dict(os.environ, {"DREAMT_CONFIG_PATH": "missing.json",
                            "DREAMT_DATA_DIR": "wrong-records", "DREAMT_SPLIT_PATH": "wrong-split.json"}):
                        args, data, provenance = load_inference_config(path)
                    self.assertEqual(args.dataset, "DREAMT")
                    self.assertEqual(args.window_size, 750 * epochs)
                    self.assertEqual(args.s4_lmax, args.window_size)
                    self.assertEqual(args.stride, 750)
                    self.assertEqual(args.timesteps, 200)
                    self.assertEqual(args.diffusion_schedule, "linear_power")
                    self.assertEqual(args.terminal_alpha_bar, 1e-8)
                    self.assertEqual(args.res_channels, DEFAULT_TRAINING_ARGS["res_channels"])
                    self.assertEqual(data["data_dir"], str(root / "records"))
                    self.assertEqual(data["metadata_path"], str(root / "meta" / "participants.csv"))
                    self.assertEqual(data["split_path"], str(root / "split.json"))
                    self.assertEqual(data["validation"], config["validation"])
                    self.assertFalse(provenance["training_manifest_read"])
                    self.assertFalse(provenance["training_identity_verified"])
                    self.assertFalse(provenance["training_schedule_identity_verified"])
                    self.assertFalse(provenance["training_scaler_identity_verified"])
                    self.assertEqual(json.loads(path.read_text(encoding="utf-8")), config)

    def test_actual_preprocessor_uses_explicit_data_config_despite_environment(self):
        with tempfile.TemporaryDirectory(prefix="night_checkpoint_", dir=ROOT) as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(dict(data_dir=str(ROOT / "example_data"),
                metadata_path=None, split_path=None, diffusion_schedule="linear")), encoding="utf-8")
            with mock.patch.dict(os.environ, {"DREAMT_CONFIG_PATH": "missing.json",
                    "DREAMT_DATA_DIR": "missing-folder", "DREAMT_METADATA_PATH": "missing.csv",
                    "DREAMT_SPLIT_PATH": "missing-splits.json", "DREAMT_MULTIPLIER": "1000"}):
                args, data, _ = load_inference_config(path)
                processor = Preprocessor(args.dataset, data_config=data)
            self.assertEqual(processor.data_dir, (ROOT / "example_data").resolve())
            self.assertEqual(processor.multiplier, 1.)
            self.assertEqual(args.diffusion_schedule, "linear")
            self.assertIsNone(data["metadata_path"])
            self.assertIsNone(data["split_path"])

    def test_invalid_config_is_rejected_without_reading_other_files(self):
        with tempfile.TemporaryDirectory(prefix="night_checkpoint_", dir=ROOT) as directory:
            path = Path(directory) / "config.json"
            for config in ([], None, {"window_epochs": True}, {"window_epochs": 0},
                           {"diffusion_schedule": "unknown"}, {"data_dir": ""}, {"split_path": 3}):
                with self.subTest(config=config):
                    path.write_text(json.dumps(config), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_inference_config(path)


if __name__ == "__main__":
    unittest.main()
