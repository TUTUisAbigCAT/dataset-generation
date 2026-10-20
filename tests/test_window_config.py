"""Configured sleep-epoch windows across data loading and launcher boundaries."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils import Preprocessor, WindowDataset
from epoch_validation import EpochValidator
from scripts import run_dreamt, run_simulation
from tests.test_early_stopping import TinyTestModel
from validation_metrics import summarize_distribution
from window_config import window_settings


class WindowConfigurationTests(unittest.TestCase):
    def processor(self, epochs=4, data_dir=None, **extra):
        config = dict(data_dir=str(data_dir or ROOT / "example_data"),
                      window_epochs=epochs, split_seed=42, **extra)
        return Preprocessor("DREAMT", config_override=config)

    def test_default_and_integer_epoch_settings(self):
        self.assertEqual(window_settings({}), dict(window_epochs=4, window_size=3000,
                         epoch_samples=750, sample_rate_hz=25, window_seconds=120))
        for epochs in (1, 2, 4, 8):
            with self.subTest(epochs=epochs):
                self.assertEqual(window_settings({"window_epochs": epochs}),
                                 dict(window_epochs=epochs, window_size=750 * epochs,
                                      epoch_samples=750, sample_rate_hz=25, window_seconds=30 * epochs))

    def test_invalid_epoch_config_is_rejected_in_helper_and_preprocessor(self):
        for value in (True, False, 0, -1, 1.0, 1.5, "4", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    window_settings({"window_epochs": value})
                with self.assertRaises(ValueError):
                    self.processor(value)

    def test_example_records_cover_lengths_counts_and_original_stage_order(self):
        mapped_stages = np.array([0, 1, 1, 3, 2, 1, 1, 0])
        for epochs in (1, 2, 4, 8):
            with self.subTest(epochs=epochs):
                processor = self.processor(epochs)
                size = epochs * 750
                self.assertEqual(processor.window_epochs, epochs)
                self.assertEqual(processor.window_size, size)
                training = processor.make_training_dataset(window_size=size, stride=750)
                self.assertEqual(len(training), len(processor.splits["train"]) * (9 - epochs))
                for index, (_, start) in enumerate(training.window_index):
                    self.assertEqual(start % 750, 0)
                    window = training[index]
                    self.assertEqual(tuple(window.shape), (size, 10))
                    first = start // 750
                    expected = np.repeat(mapped_stages[first:first + epochs], 750)
                    np.testing.assert_array_equal(window[:, 1:5].numpy(), np.eye(4)[expected])
                validation = processor.make_validation_dataset()
                expected_count = len(processor.splits["val"]) * ((8 - epochs) // 2 + 1)
                self.assertEqual(len(validation), expected_count)
                self.assertEqual(tuple(validation[0].shape), (size, 10))
                with self.assertRaises(ValueError):
                    WindowDataset(processor, "train", size + 750, 750)

    def test_legacy_validation_selector_uses_configured_length_and_short_window_stride(self):
        for epochs in (1, 2, 4, 8):
            with self.subTest(epochs=epochs):
                processor = self.processor(epochs)
                size, stride = epochs * 750, min(1500, epochs * 750)
                selected = processor.select_validation_windows(windows_per_subject=99)
                expected_starts = list(range(0, 6000 - size + 1, stride))
                for filename in processor.splits["val"]:
                    self.assertEqual(sorted(start for name, start in selected if name == filename), expected_starts)
                self.assertEqual(processor.validation_selection["window_size"], size)
                self.assertEqual(processor.validation_selection["stride"], stride)

    def test_long_span_assembly_remains_possible_with_short_configured_windows(self):
        processor = self.processor(2)
        filename = processor.splits["train"][0]
        whole_record = processor.window_array(filename, 0, 6000)
        pieces = [processor.window_array(filename, start, 1500) for start in range(0, 6000, 1500)]
        np.testing.assert_array_equal(whole_record, np.concatenate(pieces, axis=0))

    def test_invalid_epoch_and_time_gap_filter_each_configured_window_length(self):
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            for source in (ROOT / "example_data").iterdir():
                if source.is_file():
                    shutil.copy2(source, data_dir / source.name)
            filename = "EXAMPLE001.npz"
            with np.load(data_dir / filename, allow_pickle=False) as archive:
                payload = {key: archive[key].copy() for key in archive.files}
            payload["stages"][2 * 750:3 * 750] = 5
            payload["epoch_index"] = np.array([0, 1, 2, 3, 4, 6, 7, 8])
            np.savez(data_dir / filename, **payload)
            splits = dict(train=[filename, "EXAMPLE002.npz"],
                          val=["EXAMPLE003.npz"], test=["EXAMPLE004.npz"])
            (data_dir / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
            expected_epochs = {1: [0, 1, 3, 4, 5, 6, 7], 2: [0, 3, 5, 6], 4: [], 8: []}
            for epochs, expected in expected_epochs.items():
                with self.subTest(epochs=epochs):
                    processor = self.processor(epochs, data_dir=data_dir)
                    dataset = processor.make_training_dataset(epochs * 750, 750)
                    starts = [start for name, start in dataset.window_index if name == filename]
                    self.assertEqual(starts, [value * 750 for value in expected])
                    self.assertEqual(dataset.excluded_windows, 9 - epochs - len(expected))
                    self.assertEqual(processor.record(filename)["ibi"].shape, (6000,))

    def test_identity_freezes_window_epochs_and_legacy_defaults_to_four(self):
        original = self.processor(2)
        identity = original.data_identity()
        self.assertEqual(identity["window_epochs"], 2)
        with tempfile.TemporaryDirectory() as temp:
            unrelated = Path(temp) / "config.json"
            unrelated.write_text(json.dumps(dict(data_dir="missing", window_epochs=8)), encoding="utf-8")
            with mock.patch.dict(os.environ, {"DREAMT_CONFIG_PATH": str(unrelated)}):
                restored = Preprocessor.from_identity(identity)
                self.assertEqual(restored.window_size, 1500)
                self.assertEqual(restored.data_identity(), identity)
                legacy = self.processor(4).data_identity()
                del legacy["window_epochs"]
                restored_legacy = Preprocessor.from_identity(legacy)
                self.assertEqual(restored_legacy.window_size, 3000)
                self.assertEqual(restored_legacy.data_identity(), dict(legacy, window_epochs=4))


class WindowLauncherTests(unittest.TestCase):
    def test_check_only_reports_full_validation_counts_without_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            for epochs, expected_train, expected_validation in ((2, 14, 4), (8, 2, 1)):
                with self.subTest(epochs=epochs):
                    config_path, output = base / f"config_{epochs}.json", base / f"run_{epochs}"
                    config_path.write_text(json.dumps(dict(data_dir=str(ROOT / "example_data"),
                        window_epochs=epochs, validation={"max_windows": None})), encoding="utf-8")
                    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_dreamt.py"),
                        "--config", str(config_path), "--check-only", "--output-dir", str(output)],
                        cwd=base, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                        capture_output=True, text=True, timeout=45)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
                    self.assertEqual(preflight["data_identity"]["window_epochs"], epochs)
                    self.assertEqual(preflight["training_input_shape"], [epochs * 750, 10])
                    self.assertEqual(preflight["training_windows"], expected_train)
                    self.assertEqual(len(preflight["validation_windows"]), expected_validation)
                    self.assertEqual([row["start_sample"] for row in preflight["validation_windows"]],
                                     list(range(0, 6000, epochs * 750)))
                    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
                    self.assertEqual(status["status"], "checked")
                    self.assertFalse((output / "workspace").exists())

    def test_snapshot_and_frozen_training_config_preserve_epoch_count(self):
        # CUDA and external compute stages are explicitly mocked; source copying,
        # data preflight and config freezing execute their production paths.
        with tempfile.TemporaryDirectory() as temp:
            base, config = Path(temp), dict(data_dir=str(ROOT / "example_data"), window_epochs=2)
            config_path, output = base / "config.json", base / "run"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            argv = ["run_dreamt.py", "--config", str(config_path), "--output-dir", str(output),
                    "--epochs", "1", "--skip-generation"]
            with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(run_dreamt, "require_cuda", return_value=torch.device("cuda")), \
                 mock.patch.object(run_dreamt, "cuda_environment", return_value={"unit_test_mock": True}), \
                 mock.patch.object(run_dreamt.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
                 contextlib.redirect_stdout(io.StringIO()):
                run_dreamt.main()
            workspace = output / "workspace"
            self.assertEqual((workspace / "window_config.py").read_bytes(), (ROOT / "window_config.py").read_bytes())
            frozen = json.loads((workspace / "dataset_config.json").read_text(encoding="utf-8"))
            self.assertEqual(frozen["window_epochs"], 2)
            self.assertEqual(Preprocessor("DREAMT", config_override=frozen).window_size, 1500)

    def test_simulation_launches_configured_window_and_s4_lengths(self):
        # Reuse the tiny explicit CPU mock to exercise the actual training loop
        # and manifest plumbing without claiming a CUDA model run.
        for epochs, expected_windows, expected_updates in ((2, 14, 4), (8, 2, 1)):
            with self.subTest(epochs=epochs), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                workspace, output = base / "workspace", base / "training"
                run_dreamt.snapshot_source(workspace)
                config_path = workspace / "dataset_config.json"
                config_path.write_text(json.dumps(dict(data_dir=str(ROOT / "example_data"),
                    window_epochs=epochs)), encoding="utf-8")
                env = {key: value for key, value in os.environ.items() if not key.startswith("DREAMT_")}
                env["DREAMT_CONFIG_PATH"] = str(config_path)
                argv = ["run_simulation.py", "--dataset", "DREAMT", "--epochs", "1",
                        "--batch-size", "4", "--output-dir", str(output)]

                def generate(validator, model):
                    return summarize_distribution(validator.reference, validator.reference), {"unit_test_mock": True}, []

                with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(sys, "argv", argv), \
                     mock.patch.object(run_simulation, "ROOT", workspace), \
                     mock.patch.object(run_simulation, "require_cuda", return_value=torch.device("cpu")), \
                     mock.patch.object(run_simulation, "cuda_environment", return_value={"unit_test_mock": True}), \
                     mock.patch("cuda_runtime.require_cuda", return_value=torch.device("cpu")), \
                     mock.patch("cuda_runtime.assert_cuda_model"), \
                     mock.patch("epoch_validation._require_device", return_value=torch.device("cpu")), \
                     mock.patch("training_utils.fetchModel", side_effect=lambda *args: TinyTestModel()), \
                     mock.patch.object(EpochValidator, "_generate", generate), \
                     mock.patch("torch.cuda.synchronize"), contextlib.redirect_stdout(io.StringIO()):
                    run_simulation.main()
                manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["training_args"]["window_size"], epochs * 750)
                self.assertEqual(manifest["training_args"]["s4_lmax"], epochs * 750)
                self.assertEqual(manifest["training_args"]["stride"], 750)
                self.assertEqual(manifest["data"]["data_identity"]["window_epochs"], epochs)
                self.assertEqual(manifest["train_windows"], expected_windows)
                self.assertEqual(manifest["updates"], expected_updates)
                self.assertEqual(manifest["status"], "completed")


if __name__ == "__main__":
    unittest.main()
