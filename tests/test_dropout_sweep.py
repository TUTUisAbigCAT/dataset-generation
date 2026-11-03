"""Dropout sweep contracts; no model training, inference, or CPU substitution.

NPZ fixtures exercise preprocessing only. Execution reports and subprocess
boundaries are explicitly synthetic and do not establish GPU execution.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from scripts import evaluate_denoising
from scripts import run_dreamt
from scripts import run_dropout_sweep as driver
from s4_dropout_config import s4_dropout_options, s4_dropout_report
from signal_normalization import normalization_spec
from tests import test_preprocessing_experiment as preprocessing_fixtures
from tests import test_minmax_experiment as minmax_fixtures
from tests.test_extended_ablation import write_completion, write_json
from training_input_dropout import input_dropout_options


ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = ROOT / "configs/dropout_sweep.json"
PROBABILITIES = (.01, .03, .05, .1)
MINMAX = dict(method="minmax", min_seconds=0.0, max_seconds=2.0)


def temporary_directory():
    folder = ROOT / "work/dropout_sweep_tests"
    folder.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=folder)


def preflight(config):
    result = preprocessing_fixtures.real_preflight(config)
    result["training_input_dropout"] = input_dropout_options(config)
    result["s4_dropout"] = s4_dropout_options(config)
    return result


class DropoutGridTests(unittest.TestCase):
    def base_reference(self):
        config = preprocessing_fixtures.PreprocessingConfigTests().base_config()
        config["signal_normalization"] = dict(MINMAX)
        return dict(config=config, config_path=ROOT / "work/dropout_reference/config.json")

    def test_all_sixteen_distinct_combinations_activate_exactly_one_location(self):
        grid = driver.load_grid(GRID_PATH)
        self.assertEqual(len(grid), 16)
        self.assertEqual({row["id"] for row in grid}, set(range(16)))
        self.assertEqual(len({row["name"] for row in grid}), 16)
        actual = set()
        for row in grid:
            input_p, s4_p = row["input_probability"], row["s4_probability"]
            self.assertEqual(sum(p > 0 for p in (input_p, s4_p)), 1)
            self.assertEqual(row["dropout_location"], "input" if input_p else "s4")
            actual.add((tuple(row["normalization_range"]), row["dropout_location"], max(input_p, s4_p)))
        expected = {(bounds, location, probability)
                    for bounds in ((-1, 1), (0, 1))
                    for location in ("input", "s4") for probability in PROBABILITIES}
        self.assertEqual(actual, expected)

    def test_selection_is_independent_and_unknown_ids_are_rejected(self):
        grid = driver.load_grid(GRID_PATH)
        original = copy.deepcopy(grid)
        for row in grid:
            selected = driver.select_configuration(grid, row["id"])
            self.assertEqual(selected, row)
            selected["normalization_range"][0] = 900
        self.assertEqual(grid, original)
        for invalid_id in (-1, 16, 1000, True, "0", .5):
            with self.subTest(invalid_id=invalid_id), self.assertRaises(ValueError):
                driver.select_configuration(grid, invalid_id)

    def test_corrupt_or_incomplete_grids_are_rejected(self):
        original = json.loads(GRID_PATH.read_text(encoding="utf-8"))
        mutations = {
            "schema": lambda doc: doc.update(schema="unrecognized"),
            "missing_row": lambda doc: doc["configurations"].pop(),
            "duplicate_id": lambda doc: doc["configurations"][1].update(id=0),
            "duplicate_combination": lambda doc: doc["configurations"][1].update(
                {key: value for key, value in doc["configurations"][0].items() if key not in ("id", "name")}),
            "invalid_range": lambda doc: doc["configurations"][0].update(normalization_range=[-2, 2]),
            "both_active": lambda doc: doc["configurations"][0].update(input_probability=.03, s4_probability=.03),
            "both_zero": lambda doc: doc["configurations"][0].update(input_probability=0, s4_probability=0),
            "off_grid_probability": lambda doc: doc["configurations"][0].update(input_probability=.02),
            "invalid_location": lambda doc: doc["configurations"][0].update(dropout_location="all"),
        }
        with temporary_directory() as folder:
            path = Path(folder) / "grid.json"
            for name, mutate in mutations.items():
                corrupt = copy.deepcopy(original)
                mutate(corrupt)
                write_json(path, corrupt)
                with self.subTest(corruption=name), self.assertRaises(ValueError):
                    driver.load_grid(path)

    def test_each_config_changes_only_requested_normalization_and_dropout(self):
        reference = self.base_reference()
        original = copy.deepcopy(reference)
        for row in driver.load_grid(GRID_PATH):
            with self.subTest(configuration_id=row["id"]):
                config = driver.sweep_config(reference, row)
                self.assertEqual(normalization_spec(config), dict(MINMAX,
                                 **({"output_range": [0.0, 1.0]} if row["normalization_range"] == [0, 1] else {})))
                self.assertEqual(input_dropout_options(config), dict(probability=row["input_probability"], seed=42))
                self.assertEqual(s4_dropout_options(config), row["s4_probability"])
                self.assertEqual(config["missing_value_policy"], "preserve")
                self.assertEqual(config["validation"], original["config"]["validation"])
                self.assertFalse(config["early_stopping"]["enabled"])
                self.assertEqual(config["diffusion_schedule"], original["config"]["diffusion_schedule"])
                self.assertEqual(config["terminal_alpha_bar"], original["config"]["terminal_alpha_bar"])
                self.assertEqual(config["window_epochs"], 4)
                self.assertEqual(Path(config["data_dir"]), (reference["config_path"].parent / "data").resolve())
        self.assertEqual(reference, original)

    def test_active_or_changed_baseline_is_rejected(self):
        row = driver.select_configuration(driver.load_grid(GRID_PATH), 0)
        for changes in (dict(s4_dropout=.01), dict(training_input_dropout={"probability": .01}),
                        dict(missing_value_policy="masked"), dict(signal_normalization={"method": "zscore"}),
                        dict(signal_normalization=dict(MINMAX, output_range=[0, 1]))):
            reference = self.base_reference()
            reference["config"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                driver.sweep_config(reference, row)


class DropoutPreflightTests(unittest.TestCase):
    def setUp(self):
        self.fixture = preprocessing_fixtures.PreprocessingIdentityTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        config = copy.deepcopy(self.fixture.config)
        config["signal_normalization"] = dict(MINMAX)
        self.reference = dict(config=config, config_path=self.fixture.config_path)
        self.original = preflight(config)

    def current(self, row):
        return preflight(driver.sweep_config(self.reference, row))

    def test_real_npz_preflights_keep_data_schedule_and_windows_identical(self):
        for row in driver.load_grid(GRID_PATH):
            with self.subTest(configuration_id=row["id"]):
                current = self.current(row)
                checks = driver.compare_sweep_preflight(self.original, current, row)
                self.assertTrue(checks)
                self.assertTrue(all(checks.values()), checks)
                self.assertEqual(current["training_windows"], self.original["training_windows"])
                self.assertEqual(current["validation_windows"], self.original["validation_windows"])
                self.assertEqual(current["diffusion_identity"], self.original["diffusion_identity"])
                scaler = current["data_identity"]["scaler"]
                self.assertEqual((scaler["signal_center_seconds"], scaler["signal_scale_seconds"]),
                                 (0.0, 2.0) if row["normalization_range"] == [0, 1] else (1.0, 1.0))

    def test_preflight_rejects_inactive_dropout_and_immutable_protocol_drift(self):
        mutations = {
            "input_dropout": lambda p: p["training_input_dropout"].update(probability=.2),
            "input_seed": lambda p: p["training_input_dropout"].update(seed=999),
            "s4_dropout": lambda p: p.update(s4_dropout=.2),
            "split": lambda p: p["data_identity"].update(split_fingerprint="changed"),
            "raw_train_hash": lambda p: p["data_identity"]["train_npz_sha256"].update({"extra.npz": "0" * 64}),
            "training_count": lambda p: p.update(training_windows=p["training_windows"] + 1),
            "validation_position": lambda p: p["validation_windows"][0].update(start_sample=750),
            "schedule": lambda p: p["diffusion_identity"].update(timesteps=123),
            "condition_scaler": lambda p: p["data_identity"]["scaler"]["continuous_mean"].__setitem__(0, 999),
            "early_stopping": lambda p: p["early_stopping"].update(enabled=True),
        }
        for row in driver.load_grid(GRID_PATH):
            if max(row["input_probability"], row["s4_probability"]) != .03:
                continue
            original = self.current(row)
            for name, mutate in mutations.items():
                current = copy.deepcopy(original)
                mutate(current)
                with self.subTest(configuration_id=row["id"], drift=name), self.assertRaises(ValueError):
                    driver.compare_sweep_preflight(self.original, current, row)


class S4DropoutConfigurationTests(unittest.TestCase):
    def test_zero_is_the_legacy_default_and_sweep_probabilities_are_valid(self):
        self.assertEqual(s4_dropout_options({}), 0.0)
        for probability in (0, *PROBABILITIES, .999):
            with self.subTest(probability=probability):
                self.assertEqual(s4_dropout_options({"s4_dropout": probability}), probability)

    def test_invalid_s4_probabilities_are_rejected(self):
        for value in (None, True, False, -.01, 1, 1.01, "0.1", [], {},
                      float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                s4_dropout_options({"s4_dropout": value})
        for config in (None, [], "{}"):
            with self.subTest(config=config), self.assertRaises(ValueError):
                s4_dropout_options(config)

    def test_frozen_nonzero_s4_argument_reloads_without_loading_a_model(self):
        # A deliberately minimal saved specification and real implementation
        # digest exercise the actual frozen-argument validation path only.
        with temporary_directory() as folder:
            path = Path(folder) / "manifest.json"
            for probability in (0.0, *PROBABILITIES):
                manifest = dict(
                    training_args=dict(evaluate_denoising.DEFAULT_TRAINING_ARGS,
                                       dataset="DREAMT", s4_dropout=probability),
                    data_identity={"window_epochs": 4},
                    source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                   for name in ("TSImputers/S4Model.py", "s4_dropout_config.py")},
                )
                write_json(path, manifest)
                with self.subTest(probability=probability), \
                     mock.patch("torch.load", side_effect=AssertionError("No checkpoint loading")):
                    args, restored, information = evaluate_denoising.load_training_run(path)
                self.assertEqual(args.s4_dropout, probability)
                self.assertEqual(restored["training_args"]["s4_dropout"], probability)
                self.assertTrue(information["s4_source_identity"]["matched"])

    def test_frozen_invalid_dropout_does_not_relax_other_architecture_checks(self):
        with temporary_directory() as folder:
            path = Path(folder) / "manifest.json"
            base = dict(
                training_args=dict(evaluate_denoising.DEFAULT_TRAINING_ARGS,
                                   dataset="DREAMT", s4_dropout=.03),
                data_identity={"window_epochs": 4},
                source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ("TSImputers/S4Model.py", "s4_dropout_config.py")},
            )
            for updates in (dict(s4_dropout=1.0), dict(s4_dropout=-.1),
                            dict(s4_dropout=True), dict(s4_dropout="0.03"),
                            dict(res_channels=128), dict(num_res_layers=2)):
                changed = copy.deepcopy(base)
                changed["training_args"].update(updates)
                write_json(path, changed)
                with self.subTest(updates=updates), self.assertRaises(ValueError):
                    evaluate_denoising.load_training_run(path)

    def test_active_frozen_dropout_requires_matching_helper_source(self):
        with temporary_directory() as folder:
            path = Path(folder) / "manifest.json"
            manifest = dict(
                training_args=dict(evaluate_denoising.DEFAULT_TRAINING_ARGS, dataset="DREAMT", s4_dropout=.05),
                data_identity={"window_epochs": 4},
                source_sha256={"TSImputers/S4Model.py": hashlib.sha256(
                    (ROOT / "TSImputers/S4Model.py").read_bytes()).hexdigest()},
            )
            for value in (None, "0" * 64):
                if value is not None:
                    manifest["source_sha256"]["s4_dropout_config.py"] = value
                write_json(path, manifest)
                with self.subTest(recorded_hash=value), self.assertRaisesRegex(ValueError, "[Ss]4.*[Ss]ource|[Ss]ource.*[Ss]4"):
                    evaluate_denoising.load_training_run(path)

    def test_active_s4_launch_refuses_missing_cuda_before_any_compute(self):
        with temporary_directory() as folder:
            base = Path(folder)
            config = base / "config.json"
            output = base / "run"
            write_json(config, dict(s4_dropout=.1, data_dir=str(ROOT / "example_data")))
            arguments = ["run_dreamt.py", "--config", str(config), "--output-dir", str(output)]
            with mock.patch.object(sys, "argv", arguments), \
                 mock.patch.dict(os.environ, {}, clear=False), \
                 mock.patch("torch.cuda.is_available", return_value=False), \
                 mock.patch.object(run_dreamt, "snapshot_source") as snapshot, \
                 mock.patch.object(run_dreamt.subprocess, "run") as compute, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "CUDA is required.*CPU fallback is disabled"):
                    run_dreamt.main()
            snapshot.assert_not_called()
            compute.assert_not_called()
            self.assertFalse((output / "workspace").exists())
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["stages"], [])

    def test_check_only_records_s4_configuration_without_compute(self):
        with temporary_directory() as folder:
            base = Path(folder)
            config_path, output = base / "config.json", base / "check"
            config = preprocessing_fixtures.PreprocessingConfigTests().base_config()
            config.update(data_dir=str(ROOT / "example_data"), metadata_path=None, split_path=None,
                          s4_dropout=.05, signal_normalization=dict(MINMAX, output_range=[0, 1]))
            write_json(config_path, config)
            arguments = ["run_dreamt.py", "--config", str(config_path), "--output-dir", str(output), "--check-only"]
            with mock.patch.object(sys, "argv", arguments), \
                 mock.patch.dict(os.environ, {}, clear=False), \
                 mock.patch.object(run_dreamt, "require_cuda", side_effect=AssertionError("Preprocessing needs no CUDA")), \
                 mock.patch.object(run_dreamt.subprocess, "run") as compute, \
                 contextlib.redirect_stdout(io.StringIO()):
                run_dreamt.main()
            compute.assert_not_called()
            result = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(result["s4_dropout"], .05)
            self.assertEqual(result["training_input_dropout"], dict(probability=0.0, seed=42))
            self.assertFalse(result["test_npz_opened"])
            self.assertFalse((output / "workspace").exists())


class S4DropoutReportTests(unittest.TestCase):
    @staticmethod
    def wrappers(probability, count=8):
        class S4Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout2d(probability) if probability else torch.nn.Identity()

            def forward(self, *args, **kwargs):
                raise AssertionError("Structural dropout audit must never execute a forward pass")

        model = torch.nn.Module()
        model.wrappers = torch.nn.ModuleList(S4Layer() for _ in range(count))
        return model

    def test_runtime_wrapper_audit_checks_all_layers_without_forward(self):
        for probability in (0.0, *PROBABILITIES):
            model = self.wrappers(probability)
            with self.subTest(probability=probability):
                report = s4_dropout_report(model, probability, expected_layers=8)
                self.assertEqual(report["probability"], probability)
                self.assertIs(report["active"], probability > 0)
                self.assertEqual(report["layers"], 8)
                self.assertEqual(len(report["modules"]), 8)
                self.assertTrue(report["training"])
                model.eval()
                report = s4_dropout_report(model, probability, expected_layers=8)
                self.assertFalse(report["training"])
                self.assertTrue(all(not wrapper.dropout.training for wrapper in model.wrappers))

    def test_missing_misconfigured_or_wrong_kind_of_wrapper_is_rejected(self):
        for count in (0, 7, 9):
            with self.subTest(count=count), self.assertRaises(ValueError):
                s4_dropout_report(self.wrappers(.03, count), .03, expected_layers=8)
        replacements = (torch.nn.Identity(), torch.nn.Dropout(.03), torch.nn.Dropout2d(.1))
        for replacement in replacements:
            model = self.wrappers(.03)
            model.wrappers[3].dropout = replacement
            with self.subTest(replacement=type(replacement).__name__), self.assertRaises(ValueError):
                s4_dropout_report(model, .03, expected_layers=8)


class DropoutSweepWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = minmax_fixtures.MinmaxExperimentWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.base = self.fixture.base
        self.reference = self.fixture.make_reference()

    def invoke(self, row, output, *, check_only=False, mutate_preflight=None, mutate_manifest=None):
        calls = []

        def synthetic_execution(command, **kwargs):
            calls.append(command)
            self.assertTrue(kwargs["check"])
            self.assertFalse(any(key.startswith("DREAMT_") for key in kwargs["env"]))
            self.assertEqual(Path(kwargs["cwd"]), ROOT)
            config = json.loads(Path(command[command.index("--config") + 1]).read_text(encoding="utf-8"))
            check = preflight(config)
            if mutate_preflight:
                mutate_preflight(check)
            destination = Path(command[command.index("--output-dir") + 1])
            if "--check-only" in command:
                write_json(destination / "preflight.json", check)
            else:
                epochs = int(command[command.index("--epochs") + 1])
                manifest = write_completion(destination, check, epochs)
                checkpoint = destination / "workspace/saved_models/DREAMT/model.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"Synthetic completion evidence; not a model checkpoint")
                runtime_report = s4_dropout_report(S4DropoutReportTests.wrappers(check["s4_dropout"]),
                                                   check["s4_dropout"], expected_layers=8)
                manifest.update(
                    checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    training_input_dropout=self.fixture.dropout_report(check, epochs, manifest["updates"]),
                    s4_dropout=check["s4_dropout"], s4_dropout_runtime=runtime_report,
                    cuda=True, device="cuda:0", cuda_environment={"cpu_fallback_enabled": False},
                )
                if mutate_manifest:
                    mutate_manifest(manifest)
                write_json(destination / "training/run_manifest.json", manifest)
                write_json(destination / "training/training_report.json", dict(
                    training_input_dropout=manifest["training_input_dropout"],
                    s4_dropout=manifest["s4_dropout"], s4_dropout_runtime=manifest["s4_dropout_runtime"]))
                self.fixture.write_final_metrics(destination, epochs)
            return subprocess.CompletedProcess(command, 0)

        arguments = ["--configuration-id", str(row["id"]), "--reference-run", str(self.reference),
                     "--output-dir", str(output), "--epochs", "2", "--batch-size", "8", "--threads", "4"]
        if check_only:
            arguments.append("--check-only")
        with mock.patch.object(driver.subprocess, "run", side_effect=synthetic_execution), \
             mock.patch.dict(os.environ, {"DREAMT_DATA_DIR": "must-not-override-frozen-reference"}), \
             contextlib.redirect_stdout(io.StringIO()):
            result = driver.main(arguments)
        return result, calls

    def test_all_sixteen_selected_runs_are_fresh_independent_and_equal_budget(self):
        original = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in self.reference.rglob("*") if path.is_file()}
        completed = []
        for row in driver.load_grid(GRID_PATH):
            with self.subTest(configuration_id=row["id"]):
                output = self.base / row["name"]
                result, calls = self.invoke(row, output)
                self.assertEqual(result["status"], "completed")
                self.assertTrue(result["training_completed"])
                self.assertEqual(len(calls), 2)
                self.assertIn("--check-only", calls[0])
                self.assertIn("--skip-generation", calls[1])
                self.assertNotIn("--check-only", calls[1])
                self.assertNotIn("--resume", calls[1])
                self.assertNotIn("--checkpoint", calls[1])
                self.assertEqual(result["completed_budget"]["completed_epochs"], 2)
                completed.append(result["completed_budget"]["updates"])
                saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(input_dropout_options(saved)["probability"], row["input_probability"])
                self.assertEqual(s4_dropout_options(saved), row["s4_probability"])
        self.assertEqual(len(set(completed)), 1)
        self.assertEqual(original, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in original})

    def test_check_only_never_creates_training_outputs(self):
        for row in driver.load_grid(GRID_PATH):
            if row["normalization_range"] != [0, 1] or max(row["input_probability"], row["s4_probability"]) != .03:
                continue
            with self.subTest(location=row["dropout_location"]):
                output = self.base / ("check_" + row["name"])
                result, calls = self.invoke(row, output, check_only=True)
                self.assertEqual(result["status"], "checked")
                self.assertFalse(result["training_completed"])
                self.assertEqual(len(calls), 1)
                self.assertFalse((output / "run").exists())
                self.assertFalse((output / "training.log").exists())

    def test_changed_preflight_blocks_training_and_records_failure(self):
        row = driver.select_configuration(driver.load_grid(GRID_PATH), 0)
        output = self.base / "bad_preflight"
        with self.assertRaises(ValueError):
            self.invoke(row, output, mutate_preflight=lambda p: p.update(s4_dropout=.2))
        self.assertFalse((output / "run").exists())
        self.assertFalse((output / "training.log").exists())
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["training_completed"])

    def test_short_budget_or_changed_s4_runtime_cannot_complete(self):
        row = next(row for row in driver.load_grid(GRID_PATH) if row["s4_probability"] == .03)
        corruptions = {
            "short_budget": lambda manifest: manifest.update(updates=0),
            "wrong_probability": lambda manifest: manifest.update(s4_dropout=.1),
            "wrong_layer_count": lambda manifest: manifest["s4_dropout_runtime"].update(layers=7),
            "checkpoint": lambda manifest: manifest.update(checkpoint_sha256="0" * 64),
        }
        for name, mutate in corruptions.items():
            output = self.base / name
            with self.subTest(corruption=name), self.assertRaises(ValueError):
                self.invoke(row, output, mutate_manifest=mutate)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertFalse(summary["training_completed"])


if __name__ == "__main__":
    unittest.main()
