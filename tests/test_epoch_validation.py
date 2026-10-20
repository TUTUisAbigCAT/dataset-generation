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
import math
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
from validation_metrics import summarize_distribution



def rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


class TwoWindowPreprocessor:
    """Two constant windows whose pooled variance comes entirely between windows."""
    def __init__(self, window_epochs=4):
        self.window_epochs = window_epochs
        self.window_size = window_epochs * 750
        self.splits = {"train": ["train.npz"], "val": ["val_a.npz", "val_b.npz"], "test": ["test.npz"]}
        self.scaler = {"signal_std_seconds": .5, "signal_mean_seconds": 1., "fit_split": "train"}
        self.split_fingerprint = "fixed-test-split"
        names = ["IBI", "stage_W", "stage_Light", "stage_REM", "stage_Deep", "AHI", "BMI", "AGE", "gender_F", "gender_M"]
        self.columns = SimpleNamespace(tolist=lambda: names.copy())
        self.records = {
            name: {"ibi": np.full(self.window_size, value, dtype=np.float64),
                   "stages": np.full(self.window_size, index, dtype=np.int8), "file_sha256": str(index) * 64}
            for index, (name, value) in enumerate(zip(self.splits["val"], (.5, 1.5)))
        }

    def make_validation_dataset(self, *, window_size, stride):
        if window_size != self.window_size or stride != 750:
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
        self.assertEqual(defaults["kl_variance_floor_ms2"], 1.0)
        defaults["seed"] = 0
        self.assertNotEqual(defaults, coordinator.DEFAULT_VALIDATION)
        valid = {"validation": {"max_windows": None, "batch_size": 1, "seed": 2**32 - 1, "kl_variance_floor_ms2": .25}}
        self.assertEqual(coordinator.validation_options(valid), valid["validation"])
        bad_values = [
            {"validation": None}, {"validation": []}, {"validation": {"unknown": 1}},
            *({"validation": {"max_windows": value}} for value in (0, -1, True, 1., "2")),
            *({"validation": {"batch_size": value}} for value in (0, -1, False, None, 1., "2")),
            *({"validation": {"seed": value}} for value in (-1, 2**32, True, None, 1., "2")),
            *({"validation": {"kl_variance_floor_ms2": value}}
              for value in (0, -1, True, False, None, "1", float("nan"), float("inf"))),
        ]
        for config in bad_values:
            with self.subTest(config=config), self.assertRaises(ValueError):
                coordinator.validation_options(config)

    def test_fixed_selection_balances_subjects_and_preserves_gap_offset(self):
        index = [("a.npz", start) for start in range(750, 7501, 750)]
        index += [("b.npz", start) for start in range(0, 6001, 750)]
        index += [("c.npz", 1500)]
        processor = mock.Mock(splits={"val": ["a.npz", "b.npz", "c.npz"]}, window_size=3000, window_epochs=4)
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
        processor = mock.Mock(splits={"val": ["val.npz"]}, window_size=3000, window_epochs=4)
        processor.make_validation_dataset.return_value = SimpleNamespace(window_index=[("train.npz", 0)])
        with self.assertRaisesRegex(RuntimeError, "escaped"):
            coordinator.select_fixed_windows(processor, coordinator.DEFAULT_VALIDATION)
        processor.make_validation_dataset.return_value = SimpleNamespace(window_index=[])
        with self.assertRaisesRegex(ValueError, "No complete"):
            coordinator.select_fixed_windows(processor, coordinator.DEFAULT_VALIDATION)

    def test_default_selection_includes_all_windows_beyond_previous_sixty_four_cap(self):
        index = [("val.npz", start) for start in range(0, 80 * 3000, 750)]
        processor = mock.Mock(splits={"val": ["val.npz"]}, window_size=3000, window_epochs=4)
        processor.make_validation_dataset.return_value = SimpleNamespace(window_index=index)
        options = coordinator.validation_options({})
        self.assertIsNone(options["max_windows"])
        selected, metadata = coordinator.select_fixed_windows(processor, options)
        self.assertEqual(selected, [("val.npz", start) for start in range(0, 80 * 3000, 3000)])
        self.assertEqual(metadata["available_nonoverlapping_windows"], 80)
        self.assertEqual(metadata["selected_windows"], 80)

    def test_selection_uses_configured_whole_epoch_window_lengths(self):
        for epochs in (1, 2, 3, 8):
            with self.subTest(window_epochs=epochs):
                length = epochs * 750
                processor = mock.Mock(splits={"val": ["val.npz"]},
                                      window_size=length, window_epochs=epochs)
                index = [("val.npz", start) for start in range(750, 750 + 3 * length, 750)]
                processor.make_validation_dataset.return_value = SimpleNamespace(window_index=index)
                selected, metadata = coordinator.select_fixed_windows(processor, coordinator.validation_options({}))
                self.assertEqual(selected, [("val.npz", 750 + offset * length) for offset in range(3)])
                self.assertEqual(metadata["window_size_samples"], length)
                self.assertEqual(metadata["candidate_stride_samples"], 750)
                processor.make_validation_dataset.assert_called_once_with(window_size=length, stride=750)


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

    def test_configured_window_length_controls_shapes_metrics_stages_and_protocol(self):
        for epochs in (1, 2, 3, 8):
            with self.subTest(window_epochs=epochs):
                length = epochs * 750
                self.args = SimpleNamespace(window_size=length, s4_lmax=length)
                processor = TwoWindowPreprocessor(window_epochs=epochs)
                validator = self.make_validator(processor, output=self.output / f"window_{epochs}",
                                                config={**self.config, "window_epochs": epochs})
                self.assertEqual(validator.windows.shape, (2, length, 10))
                self.assertEqual(validator.reference.shape, (2, length))
                self.assertEqual([len(row["sleep_stages"]) for row in validator.windows_metadata], [epochs, epochs])
                protocol = json.loads((validator.output / "protocol.json").read_text(encoding="utf-8"))
                self.assertEqual(protocol["window"]["window_epochs"], epochs)
                self.assertEqual(protocol["window"]["window_seconds"], epochs * 30)
                self.assertEqual(protocol["schema"], "ibi_epoch_diagonal_gaussian_kl_v3")
                self.assertEqual(protocol["selection"]["window_size_samples"], length)
                measured = summarize_distribution(validator.reference + .1, validator.reference)
                output = io.StringIO()
                with mock.patch.object(validator, "_generate", return_value=(measured, {}, [])), \
                     contextlib.redirect_stdout(output):
                    report = validator.run(1, self.model, .1)
                started = json.loads(output.getvalue().splitlines()[0])
                self.assertEqual(started["window_size"], length)
                self.assertEqual(started["window_epochs"], epochs)
                self.assertAlmostEqual(report["aggregate"]["gaussian_kl_real_to_generated"], length * .02)
                self.assertEqual(validator.summary()["window"]["window_size"], length)

    def test_validation_rejects_configuration_mismatch_and_insufficient_kernel(self):
        for args, processor, config in (
                (SimpleNamespace(window_size=3000, s4_lmax=6000), TwoWindowPreprocessor(2), self.config),
                (SimpleNamespace(window_size=1500, s4_lmax=6000), self.processor, self.config),
                (SimpleNamespace(window_size=3000, s4_lmax=6000), self.processor, {**self.config, "window_epochs": 2}),
                (SimpleNamespace(window_size=3000, s4_lmax=1500), self.processor, self.config)):
            with self.subTest(args=args, config=config), self.assertRaisesRegex(ValueError, "window_epochs|kernel length"):
                self.args = args
                self.make_validator(processor=processor, config=config)

    def test_sequence_distribution_kl_history_and_unchanged_training(self):
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
                return summarize_distribution(sample, frozen), {"unit_test_mock": True}, [{"unit_test_mock": True}]

            with mock.patch.object(validator, "_generate", side_effect=fake_generate), contextlib.redirect_stdout(io.StringIO()):
                reports.append(validator.run(epoch, self.model, .3 / epoch))
            self.assertRngEqual(random_before)
            self.assertModelEqual(model_before)
        first, second, third = reports
        self.assertEqual(set(first["aggregate"]), {"gaussian_kl_real_to_generated"})
        # Across-window variance is 250000 ms^2 at every coordinate. A 100 ms
        # shift contributes .5 * 100^2 / 250000 = .02 nats per coordinate.
        self.assertAlmostEqual(first["aggregate"]["gaussian_kl_real_to_generated"], 3000. * .02)
        self.assertAlmostEqual(second["aggregate"]["gaussian_kl_real_to_generated"],
                               .5 * 3000 * (math.log(4.) + .25 - 1.))
        self.assertEqual(second["aggregate"], third["aggregate"])
        self.assertNotIn("per_window", first)
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

    def test_kl_checkpoint_updates_exactly_when_patience_resets(self):
        validator = self.make_validator(config={**self.config, "early_stopping": {"smoothing_window": 1}})
        for epoch, kl in enumerate((10., 9., 9., 8.), 1):
            with torch.no_grad():
                self.model[0].weight.fill_(epoch)
            measured = dict(aggregate=dict(gaussian_kl_real_to_generated=kl))
            with mock.patch.object(validator, "_generate", return_value=(measured, {}, [])), \
                 contextlib.redirect_stdout(io.StringIO()):
                report = validator.run(epoch, self.model, .1)
            selected_epoch = [1, 2, 2, 4][epoch-1]
            selected = validator.summary()["best_checkpoints"]["kl"]
            self.assertEqual(selected["epoch"], selected_epoch)
            saved = torch.load(selected["checkpoint"], map_location="cpu", weights_only=True)
            self.assertEqual(saved["0.weight"].item(), selected_epoch)
            self.assertEqual(report["early_stopping"]["metric_improvement"], epoch in (1, 2, 4))
            self.assertNotIn("joint_improvement", report["early_stopping"])
            self.assertEqual(report["early_stopping"]["bad_epochs"], [0, 0, 1, 0][epoch-1])
            self.assertEqual(set(validator.summary()["best_checkpoints"]), {"kl"})

    def test_smoothed_checkpoint_warmup_and_current_weights(self):
        validator = self.make_validator(config={**self.config, "early_stopping": {"smoothing_window": 3}})
        # Epoch four improves SMA 5 -> 4 while raw KL rises 0 -> 9.
        # Its current model must replace epoch three despite the worse raw KL.
        for epoch, kl in enumerate((12., 3., 0., 9., 8.), 1):
            with torch.no_grad():
                self.model[0].weight.fill_(epoch)
            measured = dict(aggregate=dict(gaussian_kl_real_to_generated=kl))
            with mock.patch.object(validator, "_generate", return_value=(measured, {}, [])), \
                 contextlib.redirect_stdout(io.StringIO()):
                report = validator.run(epoch, self.model, .1)
            checkpoints = validator.summary()["best_checkpoints"]
            if epoch < 3:
                self.assertEqual(checkpoints, {})
                self.assertEqual(list(self.output.parent.glob("best_*_model.pth")), [])
                self.assertFalse(report["early_stopping"]["smoothing_ready"])
            else:
                selected = checkpoints["kl"]
                expected_epoch = 3 if epoch == 3 else 4
                self.assertEqual(selected["epoch"], expected_epoch)
                saved = torch.load(selected["checkpoint"], map_location="cpu", weights_only=True)
                self.assertEqual(saved["0.weight"].item(), expected_epoch)
        self.assertEqual(selected["metric"], "gaussian_kl_real_to_generated")
        self.assertEqual(selected["selection_rule"], "smoothed_kl")
        self.assertEqual(selected["value"], 4.)
        self.assertEqual(selected["raw_metrics"], {"gaussian_kl_real_to_generated": 9.})
        self.assertEqual(validator.summary()["early_stopping"]["best_metrics"]["kl"]["epoch"], 3)
        self.assertEqual(selected["window_start_epoch"], 2)
        self.assertEqual(selected["smoothing_window"], 3)
        with (self.output.parent / "epoch_metrics.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, [
                "epoch", "train_mean_loss", "gaussian_kl_real_to_generated",
                "smoothed_gaussian_kl_real_to_generated", "validation_seconds",
                "smoothing_ready", "metric_improvement", "early_stopping_bad_epochs", "stopped_early"])
        self.assertEqual(rows[0]["smoothed_gaussian_kl_real_to_generated"], "")
        self.assertEqual(rows[0]["smoothing_ready"], "False")
        self.assertEqual(float(rows[3]["smoothed_gaussian_kl_real_to_generated"]), 4.)
        self.assertEqual(rows[3]["metric_improvement"], "True")
        self.assertEqual(rows[4]["metric_improvement"], "False")
        self.assertEqual(rows[4]["early_stopping_bad_epochs"], "1")
        for obsolete in ("mean", "variance", "joint"):
            self.assertFalse((self.output.parent / f"best_{obsolete}_model.pth").exists())

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

    def test_actual_generation_accumulates_coordinate_moments_across_uneven_batches(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.05))

        self.model = TinyModel().train()
        processor = TwoWindowPreprocessor()
        processor.splits["val"] = [f"val_{index}.npz" for index in range(5)]
        coordinate = np.linspace(-1., 1., 3000, dtype=np.float64)
        processor.records = {
            name: dict(ibi=.6 + index * .2 + coordinate * (.01 + .03 * index),
                       stages=np.full(3000, index % 4, dtype=np.int8),
                       file_sha256=str(index) * 64)
            for index, name in enumerate(processor.splits["val"])
        }
        validator = self.make_validator(processor, config={**self.config, "validation": {
            **self.config["validation"], "kl_variance_floor_ms2": 100000.}})
        self.assertEqual(validator.options["batch_size"], 2)
        batch_sizes, generated_batches = [], []

        def sample(model, clean, diffusion_config, args, device):
            self.assertFalse(model.training)
            batch_sizes.append(len(clean))
            pattern = torch.linspace(-.2, .3, 3000, dtype=clean.dtype, device=clean.device)
            generated = clean[:, :, 0] * .7 + pattern
            generated_batches.append(generated.to(torch.float64) * validator.scale + validator.center)
            return generated, {"unit_test_mock": True, "windows": len(clean)}

        with mock.patch("training_utils.fetchModel", side_effect=lambda *args: TinyModel()), \
             mock.patch.object(coordinator, "sample_validation_windows", side_effect=sample), \
             contextlib.redirect_stdout(io.StringIO()):
            report = validator.run(1, self.model, .1)
        self.assertEqual(batch_sizes, [2, 2, 1])
        self.assertEqual([entry["windows"] for entry in report["sampling"]], [2, 2, 1])
        generated_ms = torch.cat(generated_batches) * 1000.
        reference_ms = validator.reference * 1000.
        ref_variance, ref_mean = torch.var_mean(reference_ms, dim=0, unbiased=False)
        gen_variance, gen_mean = torch.var_mean(generated_ms, dim=0, unbiased=False)
        ref_variance = ref_variance.clamp_min(validator.options["kl_variance_floor_ms2"])
        gen_variance = gen_variance.clamp_min(validator.options["kl_variance_floor_ms2"])
        expected_kl = .5 * (torch.log(gen_variance / ref_variance)
                             + (ref_variance + (ref_mean - gen_mean).square()) / gen_variance - 1).sum()
        self.assertEqual(set(report["aggregate"]), {"gaussian_kl_real_to_generated"})
        self.assertAlmostEqual(report["aggregate"]["gaussian_kl_real_to_generated"], float(expected_kl), places=7)
        self.assertNotIn("per_window", report)
        self.assertEqual(report["early_stopping"]["best_metrics"]["kl"]["metric"], "gaussian_kl_real_to_generated")
        self.assertEqual(validator.summary()["best_checkpoints"], {})  # No checkpoint during warmup.

if __name__ == "__main__":
    unittest.main()
