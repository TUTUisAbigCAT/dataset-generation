"""Epoch coordination tests with explicit CPU-only CUDA-guard/generation mocks.

No production CPU inference fallback is enabled. These tests exercise selection,
physical-unit reporting, history and training-state preservation without a GPU model run.
"""
from __future__ import annotations

from collections import Counter
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils import Preprocessor
import epoch_validation as coordinator



def rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


class TwoWindowPreprocessor:
    """Two constant windows whose pooled variance comes entirely between windows."""
    def __init__(self):
        self.splits = {"train": ["train.npz"], "val": ["val_a.npz", "val_b.npz"], "test": ["test.npz"]}
        self.scaler = {"signal_std_seconds": .5, "signal_mean_seconds": 1., "fit_split": "train"}
        self.split_fingerprint = "fixed-test-split"
        names = ["IBI", "stage_W", "stage_Light", "stage_REM", "stage_Deep", "AHI", "BMI", "AGE", "gender_F", "gender_M"]
        self.columns = SimpleNamespace(tolist=lambda: names.copy())
        self.records = {
            name: {"ibi": np.full(3000, value, dtype=np.float64),
                   "stages": np.full(3000, index, dtype=np.int8), "file_sha256": str(index) * 64}
            for index, (name, value) in enumerate(zip(self.splits["val"], (.5, 1.5)))
        }

    def make_validation_dataset(self, *, window_size, stride):
        if window_size != 3000 or stride != 750:
            raise AssertionError("Expected complete windows searched every 30 seconds")
        return SimpleNamespace(window_index=[(name, 0) for name in self.splits["val"]])

    def record(self, name):
        if name not in self.splits["val"]:
            raise AssertionError("Validation opened a non-validation record")
        return self.records[name]

    def window_array(self, name, start, length):
        result = np.zeros((length, 10), dtype=np.float32)
        result[:, 0] = (self.record(name)["ibi"][start:start + length] - 1.) / .5
        result[:, 1] = 1.
        result[:, 5] = 10000.  # Clean conditions must never enter IBI metrics.
        return result

    def provenance(self):
        return {"data_synthetic": True}


class ValidationOptionsAndSelectionTests(unittest.TestCase):
    def test_options_defaults_and_strict_numeric_ranges(self):
        defaults = coordinator.validation_options({})
        self.assertEqual(defaults, coordinator.DEFAULT_VALIDATION)
        defaults["seed"] = 0
        self.assertNotEqual(defaults, coordinator.DEFAULT_VALIDATION)
        valid = {"validation": {"max_windows": None, "batch_size": 1, "seed": 2**32 - 1}}
        self.assertEqual(coordinator.validation_options(valid), valid["validation"])
        bad_values = [
            {"validation": None}, {"validation": []}, {"validation": {"unknown": 1}},
            *({"validation": {"max_windows": value}} for value in (0, -1, True, 1., "2")),
            *({"validation": {"batch_size": value}} for value in (0, -1, False, None, 1., "2")),
            *({"validation": {"seed": value}} for value in (-1, 2**32, True, None, 1., "2")),
        ]
        for config in bad_values:
            with self.subTest(config=config), self.assertRaises(ValueError):
                coordinator.validation_options(config)

    def test_fixed_selection_balances_subjects_and_preserves_gap_offset(self):
        index = [("a.npz", start) for start in range(750, 7501, 750)]
        index += [("b.npz", start) for start in range(0, 6001, 750)]
        index += [("c.npz", 1500)]
        processor = mock.Mock(splits={"val": ["a.npz", "b.npz", "c.npz"]})
        processor.make_validation_dataset.return_value = SimpleNamespace(window_index=index)
        options = {"max_windows": 5, "batch_size": 2, "seed": 42}
        before = rng_state()
        selected, info = coordinator.select_fixed_windows(processor, options)
        repeated, repeated_info = coordinator.select_fixed_windows(processor, options)
        self.assertEqual(selected, repeated)
        self.assertEqual(info, repeated_info)
        self.assertEqual(Counter(name for name, _ in selected), {"a.npz": 2, "b.npz": 2, "c.npz": 1})
        self.assertEqual(info["available_nonoverlapping_windows"], 7)
        self.assertEqual(info["selected_subjects"], 3)
        all_selected, _ = coordinator.select_fixed_windows(processor, {**options, "max_windows": None})
        self.assertEqual(len(all_selected), 7)
        self.assertIn(("a.npz", 750), all_selected)
        for name in processor.splits["val"]:
            starts = [start for subject, start in all_selected if subject == name]
            self.assertTrue(all(right - left >= 3000 for left, right in zip(starts, starts[1:])))
        after = rng_state()
        self.assertEqual(before[0], after[0])
        self.assertTrue(np.array_equal(before[1][1], after[1][1]))
        self.assertEqual(before[1][2:], after[1][2:])
        self.assertTrue(torch.equal(before[2], after[2]))
        processor.make_validation_dataset.assert_called_with(window_size=3000, stride=750)

    def test_selection_rejects_training_records_and_empty_candidates(self):
        processor = mock.Mock(splits={"val": ["val.npz"]})
        processor.make_validation_dataset.return_value = SimpleNamespace(window_index=[("train.npz", 0)])
        with self.assertRaisesRegex(RuntimeError, "escaped"):
            coordinator.select_fixed_windows(processor, coordinator.DEFAULT_VALIDATION)
        processor.make_validation_dataset.return_value = SimpleNamespace(window_index=[])
        with self.assertRaisesRegex(ValueError, "No complete"):
            coordinator.select_fixed_windows(processor, coordinator.DEFAULT_VALIDATION)


class EpochCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="epoch_validation_unit_")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.root = self.directory / "outputs"
        self.root.mkdir()
        self.output = self.root / "run" / "training" / "validation"
        self.config = {"output_root": str(self.root), "validation": {"max_windows": 64, "batch_size": 2, "seed": 7}}
        self.config_path = self.directory / "dataset_config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.environment = mock.patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for name in list(os.environ):
            if name.startswith(("DG_", "DREAMT_")) or name == "SLURM_JOB_ID":
                del os.environ[name]
        os.environ["DREAMT_CONFIG_PATH"] = str(self.config_path)
        self.args = SimpleNamespace(window_size=3000, s4_lmax=3000)
        self.model = torch.nn.Sequential(torch.nn.Linear(1, 1))
        self.model.register_buffer("fixed", torch.tensor([7.]))
        self.model.train()
        self.model[0].eval()  # A mixed mode tree must remain intact.
        for parameter in self.model.parameters():
            parameter.grad = torch.full_like(parameter, .25)
        self.processor = TwoWindowPreprocessor()

    def make_validator(self, processor=None, output=None, config=None):
        # Explicit unit-test mock: real production device checks still reject CPU.
        with mock.patch.object(coordinator, "_require_device", return_value=torch.device("cpu")):
            return coordinator.EpochValidator(
                processor or self.processor, self.args, self.model, {"T": 3},
                torch.device("cpu"), output or self.output, config or self.config)

    def assertRngEqual(self, before):
        after = rng_state()
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[1][0], after[1][0])
        self.assertTrue(np.array_equal(before[1][1], after[1][1]))
        self.assertEqual(before[1][2:], after[1][2:])
        self.assertTrue(torch.equal(before[2], after[2]))

    def model_state(self):
        return ({name: tensor.clone() for name, tensor in self.model.state_dict().items()},
                [module.training for module in self.model.modules()],
                [None if parameter.grad is None else parameter.grad.clone() for parameter in self.model.parameters()])

    def assertModelEqual(self, before):
        state, modes, gradients = before
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertTrue(all(torch.equal(value, self.model.state_dict()[name]) for name, value in state.items()))
        for old, parameter in zip(gradients, self.model.parameters()):
            self.assertTrue(old is None and parameter.grad is None or torch.equal(old, parameter.grad))

    def test_real_example_loader_frozen_windows_and_test_sealed(self):
        processor = Preprocessor("DREAMT", config_override={"data_dir": str(ROOT / "example_data"), "split_seed": 42})
        identity = processor.data_identity()
        before = rng_state()
        with mock.patch("numpy.load", wraps=np.load) as load:
            validator = self.make_validator(processor)
        self.assertRngEqual(before)
        self.assertEqual(identity, processor.data_identity())
        self.assertEqual(tuple(validator.windows.shape[1:]), (3000, 10))
        self.assertEqual(validator.reference.dtype, torch.float64)
        self.assertFalse(set(processor.accessed_files) & set(processor.splits["test"]))
        self.assertTrue(all(Path(call.args[0]).name in processor.splits["val"] for call in load.call_args_list))
        for index, metadata in enumerate(validator.windows_metadata):
            self.assertIn(metadata["filename"], processor.splits["val"])
            record = processor.record(metadata["filename"])
            start = metadata["start_sample"]
            expected = torch.from_numpy(record["ibi"][start:start + 3000].astype(np.float64))
            self.assertTrue(torch.equal(validator.reference[index], expected))
            self.assertEqual(metadata["file_sha256"], record["file_sha256"])
        protocol = json.loads((self.output / "protocol.json").read_text())
        self.assertFalse(protocol["test_npz_opened"])
        self.assertEqual(protocol["scaler"]["fit_split"], "train")
        self.assertEqual(validator.summary()["completed_epochs"], 0)

    def test_actual_loader_finds_window_after_invalid_first_epoch(self):
        records = self.directory / "gap_records"
        records.mkdir()
        for name in ("train", "val", "test"):
            epochs = 5
            ibi = (.8 + .01 * np.sin(np.arange(epochs * 750) / 31.)).astype(np.float32)
            labels = [-1, 1, 2, 3, 4] if name == "val" else [0, 1, 2, 3, 4]
            np.savez(records / (name + ".npz"), data=ibi, stages=np.repeat(labels, 750), fs=np.array(25), ahi=np.array(5.))
        (records / "participant_info.csv").write_text(
            "SID,AHI,BMI,AGE,GENDER\ntrain,5,25,40,F\nval,5,26,50,M\ntest,5,27,60,F\n", encoding="utf-8")
        (records / "splits.json").write_text(json.dumps({key: [key + ".npz"] for key in ("train", "val", "test")}), encoding="utf-8")
        processor = Preprocessor("DREAMT", config_override={"data_dir": str(records)})
        selected, metadata = coordinator.select_fixed_windows(processor, coordinator.DEFAULT_VALIDATION)
        self.assertEqual(selected, [("val.npz", 750)])
        self.assertEqual(metadata["selected_windows"], 1)
        self.assertNotIn("test.npz", processor.accessed_files)

    def test_pooled_metrics_history_and_unchanged_training(self):
        before_init = rng_state()
        validator = self.make_validator()
        self.assertRngEqual(before_init)
        frozen = validator.reference.clone()
        self.processor.records["val_a.npz"]["ibi"][:] = 99.
        self.assertTrue(torch.equal(validator.reference, frozen))
        generated = [frozen + .1, 1. + (frozen - 1.) * 2., 1. + (frozen - 1.) * 2.]
        reports = []
        for epoch, sample in enumerate(generated, 1):
            with torch.no_grad():
                self.model[0].weight.fill_(float(epoch))  # Simulate completed optimizer updates.
            model_before = self.model_state()
            random_before = rng_state()

            def fake_generate(model):
                self.assertIs(model, self.model)
                random.random()
                np.random.rand(3)
                torch.rand(3)
                return sample.clone(), {"unit_test_mock": True}, [{"unit_test_mock": True}]

            with mock.patch.object(validator, "_generate", side_effect=fake_generate), contextlib.redirect_stdout(io.StringIO()):
                reports.append(validator.run(epoch, self.model, .3 / epoch))
            self.assertRngEqual(random_before)
            self.assertModelEqual(model_before)
        first, second, third = reports
        self.assertAlmostEqual(first["aggregate"]["reference_mean_ms"], 1000.)
        self.assertAlmostEqual(first["aggregate"]["reference_variance_ms2"], 250000.)
        self.assertTrue(all(row["reference_variance_ms2"] == 0. for row in first["per_window"]))
        self.assertAlmostEqual(first["aggregate"]["mean_absolute_error_ms"], 100.)
        self.assertAlmostEqual(second["aggregate"]["mean_absolute_error_ms"], 0.)
        self.assertAlmostEqual(second["aggregate"]["variance_absolute_error_ms2"], 750000.)
        summary = validator.summary()
        self.assertEqual(summary["completed_epochs"], 3)
        with (self.output.parent / "epoch_metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["epoch"]) for row in rows], [1, 2, 3])
        for row, expected in zip(rows, (.3, .15, .1)):
            self.assertAlmostEqual(float(row["train_mean_loss"]), expected)
        for epoch in (1, 2, 3):
            report_path = self.output / f"epoch_{epoch:04d}" / "metrics.json"
            self.assertEqual(json.loads(report_path.read_text())["epoch"], epoch)
        self.assertEqual(json.loads((self.output / "summary.json").read_text())["completed_epochs"], 3)

    def test_generation_failure_restores_rng_and_keeps_training_model(self):
        validator = self.make_validator()
        random_before, model_before = rng_state(), self.model_state()

        def fail(model):
            random.random()
            np.random.rand()
            torch.rand(2)
            raise RuntimeError("mock sampler failure")

        with mock.patch.object(validator, "_generate", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "mock sampler failure"):
                validator.run(1, self.model, .1)
        self.assertRngEqual(random_before)
        self.assertModelEqual(model_before)
        self.assertEqual(validator.summary()["completed_epochs"], 0)

    def test_epoch_order_and_nonfinite_loss_rejected(self):
        validator = self.make_validator()
        for epoch, loss in ((0, .1), (2, .1), (1, float("nan")), (1, float("inf"))):
            with self.subTest(epoch=epoch, loss=loss), self.assertRaises(ValueError):
                validator.run(epoch, self.model, loss)
        self.assertEqual(list(self.output.glob("epoch_*")), [])

    def test_production_guard_rejects_cpu_before_model_execution(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA is required"):
            coordinator._require_device(torch.device("cpu"), self.model)

    def test_actual_generation_uses_fresh_model_fixed_noise_and_restores_rng(self):
        from validation_sampling import _sample_windows_from_original

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(()))
                self.register_buffer("fixed", torch.tensor([7.]))

            def forward(self, windows, time):
                return (windows[:, :, :1] * self.weight).permute(0, 2, 1)

        self.model = TinyModel().train()
        self.model.weight.grad = torch.tensor(.25)
        self.args = SimpleNamespace(window_size=3000, s4_lmax=3000, stride=1500, timesteps=3)
        validator = self.make_validator()
        betas = torch.linspace(.0001, .02, 3).reshape(3, 1)
        validator.diffusion_config = dict(T=3, betas=betas, alphas=1-betas,
                                         alpha_bars=torch.cumprod(1-betas, dim=0))
        original_rng, original_model = rng_state(), self.model_state()
        created = []

        def constructor(*args):
            model = TinyModel()
            created.append(model)
            return model

        # Only the device guard is mocked: execute the actual copy, full sampler,
        # inverse scaling, statistics and logging, using a tiny unit-test model.
        with mock.patch("training_utils.fetchModel", side_effect=constructor), \
             mock.patch.object(coordinator, "sample_validation_windows", side_effect=_sample_windows_from_original), \
             contextlib.redirect_stdout(io.StringIO()):
            first = validator.run(1, self.model, .1)
            second = validator.run(2, self.model, .1)
        self.assertEqual(len(created), 2)
        self.assertIsNot(created[0], created[1])
        self.assertTrue(all(model is not self.model and not model.training for model in created))
        self.assertTrue(all(torch.equal(model.weight, self.model.weight) for model in created))
        self.assertEqual(first["aggregate"], second["aggregate"])
        self.assertRngEqual(original_rng)
        self.assertModelEqual(original_model)

if __name__ == "__main__":
    unittest.main()
