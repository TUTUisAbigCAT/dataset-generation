"""Raw-KL bookkeeping and mocked checkpoint I/O; no model inference or training."""
from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import epoch_validation as coordinator
from diffusion_schedule import build_diffusion_config


METRIC = "gaussian_kl_real_to_generated"


def current_config(**options):
    return {"window_epochs": 2,
            "validation": {"max_windows": None, "batch_size": 2, "seed": 20261020},
            "early_stopping": {"enabled": False, "patience": 10, "smoothing_window": 1,
                               "selection_mode": "current", "min_delta_kl": 0.0, **options}}


class FixedEvaluationPreprocessor:
    """Host-only fixture with S092 in training and no held-out split."""
    window_epochs = 2
    window_size = 1500

    def __init__(self, *, explicit_evaluation=True):
        self.splits = {"train": ["S001.npz", "S092.npz"], "val": [], "test": []}
        if explicit_evaluation:
            self.evaluation_filenames = ["S092.npz"]
            self.evaluation_protocol = dict(mode="all_subjects_s092", held_out=False,
                                            training_includes_evaluation_subject=True)
        else:
            self.splits = {"train": ["S001.npz"], "val": ["S092.npz"], "test": []}
        self.scaler = {"signal_std_seconds": .5, "signal_mean_seconds": 1., "fit_split": "train"}
        self.split_fingerprint = "mock-all-subjects-training"
        self.columns = SimpleNamespace(tolist=lambda: ["IBI", "stage_W"])
        self._record = dict(ibi=np.full(6000, .8, dtype=np.float64),
                            stages=np.zeros(6000, dtype=np.int8), file_sha256="a" * 64)

    def make_validation_dataset(self, *, window_size, stride):
        assert window_size == self.window_size and stride == 750
        return SimpleNamespace(window_index=[("S092.npz", start) for start in range(0, 4501, stride)])

    def record(self, name):
        assert name == "S092.npz"
        return self._record

    def window_array(self, name, start, length):
        values = np.ones((length, 2), dtype=np.float32)
        values[:, 0] = (self.record(name)["ibi"][start:start + length] - 1.) / .5
        return values

    def provenance(self):
        return {"data_synthetic": True}


class CheckpointState(torch.nn.Module):
    """Only a state_dict marker; its forward must never execute."""
    def __init__(self):
        super().__init__()
        self.register_buffer("epoch_marker", torch.tensor(0))

    def forward(self, *args, **kwargs):
        raise AssertionError("This test must not execute model inference")


class CurrentKLSelectionTests(unittest.TestCase):
    def test_current_options_are_explicit_and_legacy_options_stay_identical(self):
        self.assertEqual(coordinator.early_stopping_options({}), coordinator.DEFAULT_EARLY_STOPPING)
        self.assertEqual(coordinator.early_stopping_options({"early_stopping": {"selection_mode": "smoothed"}}),
                         coordinator.DEFAULT_EARLY_STOPPING)
        self.assertEqual(coordinator.early_stopping_options(current_config()), current_config()["early_stopping"])
        for value in (None, True, 1, "raw", [], {}):
            with self.subTest(selection_mode=value), self.assertRaisesRegex(ValueError, "selection_mode"):
                coordinator.early_stopping_options(current_config(selection_mode=value))
        for window in (2, 5):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "requires smoothing_window=1"):
                coordinator.early_stopping_options(current_config(smoothing_window=window))

    def test_raw_minimum_from_first_epoch_without_sma_or_early_termination(self):
        stopping = coordinator.EarlyStopping(current_config())
        self.assertFalse(stopping.summary()["selection_ready"])
        values = (2., 2., 2., 100., 1.) + (1.,) * 12
        # A trailing mean worsens from 2 at epoch 3 to 103/3 at epoch 5.
        # Current KL nevertheless improves to 1 and must select epoch 5.
        with mock.patch.object(coordinator.math, "fsum", side_effect=AssertionError("No averaging in current mode")):
            results = [stopping.update(epoch, {METRIC: kl}) for epoch, kl in enumerate(values, 1)]
        self.assertEqual([index + 1 for index, row in enumerate(results) if row["metric_improvement"]], [1, 5])
        self.assertEqual(len(stopping.history), 0)
        for result in results:
            self.assertTrue(result["selection_ready"])
            self.assertEqual(result["improvement_rule"], "current_kl")
            self.assertEqual(result["smoothing_method"], "none")
            self.assertNotIn("smoothed_metrics", result)
            self.assertNotIn("smoothing_ready", result)
            self.assertFalse(result["stopped_early"])
        self.assertEqual(results[-1]["bad_epochs"], 12)
        self.assertEqual(results[-1]["best_metrics"]["kl"]["epoch"], 5)
        self.assertEqual(results[-1]["threshold_anchors"], {"kl": 1.})
        self.assertEqual(results[-1]["current_metrics"], {METRIC: 1.})
        self.assertEqual(results[0]["best_metrics"]["kl"]["epoch"], 1)

    def test_current_min_delta_is_strict_and_compares_with_selected_anchor(self):
        stopping = coordinator.EarlyStopping(current_config(min_delta_kl=1.))
        results = [stopping.update(epoch, {METRIC: kl})
                   for epoch, kl in enumerate((10., 9., 8.9, 8., 7.8), 1)]
        self.assertEqual([row["metric_improvement"] for row in results], [True, False, True, False, True])
        self.assertEqual([row["bad_epochs"] for row in results], [0, 1, 0, 1, 0])
        self.assertEqual(results[-1]["threshold_anchors"], {"kl": 7.8})

    def test_current_mode_rejects_invalid_metrics_and_epoch_order(self):
        stopping = coordinator.EarlyStopping(current_config())
        for epoch, value in ((2, 1.), (1, -1.), (1, float("inf")), (1, float("nan"))):
            with self.subTest(epoch=epoch, value=value), self.assertRaises(ValueError):
                stopping.update(epoch, {METRIC: value})
        self.assertEqual(stopping.summary()["completed_epochs"], 0)

    def test_all_s092_nonoverlapping_windows_are_fixed_and_explicitly_bounded(self):
        processor = FixedEvaluationPreprocessor()
        options = coordinator.validation_options(current_config())
        selected, selection = coordinator.select_fixed_windows(processor, options)
        self.assertEqual(selected, [("S092.npz", start) for start in (0, 1500, 3000, 4500)])
        self.assertEqual(selection["available_nonoverlapping_windows"], 4)
        self.assertEqual(selection["selected_windows"], 4)
        self.assertEqual(selection["selected_subjects"], 1)
        self.assertEqual(coordinator.select_fixed_windows(processor, options), (selected, selection))
        self.assertEqual(coordinator.select_fixed_windows(processor, {**options, "seed": 17}), (selected, selection))
        with mock.patch.object(processor, "make_validation_dataset", return_value=SimpleNamespace(
                window_index=[("S001.npz", 0)])):
            with self.assertRaisesRegex(RuntimeError, "escaped"):
                coordinator.select_fixed_windows(processor, options)


class CurrentKLCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "training" / "validation"
        self.model = CheckpointState()
        self.processor = FixedEvaluationPreprocessor()
        self.args = SimpleNamespace(window_size=1500, s4_lmax=1500, timesteps=3,
                                    beta_0=.0001, beta_T=.02, diffusion_schedule="linear",
                                    terminal_alpha_bar=1e-8)

    def make_validator(self, *, legacy=False):
        config = current_config()
        processor = self.processor
        if legacy:
            config.pop("early_stopping")
            processor = FixedEvaluationPreprocessor(explicit_evaluation=False)
        # Explicitly mocked device/generation: no CPU model forward or training.
        with mock.patch.object(coordinator, "_require_device", return_value=torch.device("cpu")):
            return coordinator.EpochValidator(processor, self.args, self.model,
                build_diffusion_config(self.args), "cpu", self.output, config)

    def test_current_checkpoint_bytes_metadata_and_csv_track_raw_minimum(self):
        validator = self.make_validator()
        checkpoint = self.output.parent / "best_kl_model.pth"
        previous_bytes = None
        for epoch, (kl, selected_epoch) in enumerate(zip((2., 2., 2., 100., 1., 1., 3.),
                                                        (1, 1, 1, 1, 5, 5, 5)), 1):
            self.model.epoch_marker.fill_(epoch)
            with mock.patch.object(validator, "_generate", return_value=({"aggregate": {METRIC: kl}}, {}, [])), \
                    contextlib.redirect_stdout(io.StringIO()):
                report = validator.run(epoch, self.model, .1)
            self.assertTrue(checkpoint.is_file())  # Includes epoch one.
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["epoch_marker"].item(), selected_epoch)
            if epoch != selected_epoch:
                self.assertEqual(checkpoint.read_bytes(), previous_bytes)
            previous_bytes = checkpoint.read_bytes()
            best = validator.summary()["best_checkpoints"]["kl"]
            self.assertEqual(best["selection_rule"], "current_kl")
            self.assertEqual(best["epoch"], selected_epoch)
            self.assertEqual(best["value"], 2. if selected_epoch == 1 else 1.)
            self.assertEqual(best["raw_metrics"], {METRIC: best["value"]})
            self.assertNotIn("smoothing_window", best)
            self.assertNotIn("window_start_epoch", best)
            self.assertEqual(report["early_stopping"]["metric_improvement"], epoch == selected_epoch)
            self.assertFalse(validator.should_stop)

        protocol = json.loads((self.output / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["early_stopping_improvement_rule"], "current_kl")
        self.assertEqual(protocol["early_stopping_smoothing_method"], "none")
        self.assertEqual(protocol["evaluation_protocol"], self.processor.evaluation_protocol)
        self.assertEqual(protocol["evaluation_filenames"], ["S092.npz"])
        self.assertEqual({row["filename"] for row in protocol["windows"]}, {"S092.npz"})
        self.assertEqual(protocol["selection"]["selected_windows"], 4)
        self.assertTrue(protocol["validation_seed_reset_each_epoch"])
        self.assertEqual(protocol["options"]["seed"], current_config()["validation"]["seed"])
        with (self.output.parent / "epoch_metrics.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, list(coordinator.CURRENT_EPOCH_METRIC_FIELDS))
            self.assertNotIn("smoothed_gaussian_kl_real_to_generated", reader.fieldnames)
            self.assertNotIn("smoothing_ready", reader.fieldnames)
        self.assertEqual([float(row[METRIC]) for row in rows], [2., 2., 2., 100., 1., 1., 3.])
        self.assertEqual([row["metric_improvement"] for row in rows],
                         ["True", "False", "False", "False", "True", "False", "False"])
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["best_checkpoints"]["kl"]["epoch"], 5)
        self.assertEqual(summary["early_stopping"]["improvement_rule"], "current_kl")

    def test_legacy_protocol_has_no_new_evaluation_or_selection_options(self):
        validator = self.make_validator(legacy=True)
        self.assertNotIn("evaluation_protocol", validator.protocol)
        self.assertNotIn("evaluation_filenames", validator.protocol)
        self.assertEqual(validator.protocol["early_stopping"], coordinator.DEFAULT_EARLY_STOPPING)
        self.assertEqual(validator.protocol["early_stopping_improvement_rule"], "smoothed_kl")
        self.assertEqual(validator.protocol["early_stopping_smoothing_method"], "simple_moving_average")


if __name__ == "__main__":
    unittest.main()
