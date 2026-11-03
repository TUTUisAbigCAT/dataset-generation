"""All-subject sweep contracts using synthetic NPZs and mocked subprocesses.

These tests do not train a model, run inference, or substitute CPU execution for
CUDA. Synthetic completion evidence only exercises orchestration safeguards.
"""
from __future__ import annotations

import contextlib
import copy
from collections import Counter
from argparse import Namespace
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

import numpy as np

from data_utils import Preprocessor
from diffusion_schedule import diffusion_identity, diffusion_settings
from epoch_validation import early_stopping_options, select_fixed_windows, validation_options
from scripts import run_all88_s092_sweep as driver
from scripts import run_dreamt
from scripts import evaluate_denoising
from s4_dropout_config import s4_dropout_options
from signal_normalization import normalization_spec
from tests.test_extended_ablation import write_completion, write_json
from tests import test_minmax_experiment as minmax_fixtures
from training_input_dropout import input_dropout_options


ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = ROOT / "configs/all88_s092_sweep.json"
PROBABILITIES = (0.0, .03, .05)
RANGES = ((-1.0, 1.0), (0.0, 1.0))
PROTOCOL = dict(mode="all_subjects_train_single_subject_eval", subject="S092", expected_subjects=88)
MINMAX = dict(method="minmax", min_seconds=0.0, max_seconds=2.0)


def temporary_directory():
    parent = ROOT / "work/all88_s092_sweep_tests"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


def preflight(config):
    """Load artificial NPZs and build windows; never instantiate a model."""
    processor = Preprocessor("DREAMT", data_config=config)
    training = processor.make_training_dataset(window_size=processor.window_size, stride=750)
    options = validation_options(config)
    windows, selection = select_fixed_windows(processor, options)
    schedule = diffusion_settings(config, default_schedule="linear_power")
    return dict(status="passed", data_identity=processor.data_identity(),
                training_windows=len(training), training_input_shape=list(training[0].shape),
                training_subject_window_counts=dict(Counter(name for name, _ in training.window_index)),
                missing_value_policy=processor.missing_value_policy,
                window_epochs=processor.window_epochs, window_size=processor.window_size,
                validation_windows=[dict(filename=name, start_sample=start) for name, start in windows],
                validation_selection=selection, validation_options=options,
                evaluation_protocol=copy.deepcopy(config["evaluation_protocol"]),
                early_stopping=early_stopping_options(config),
                training_input_dropout=input_dropout_options(config), s4_dropout=s4_dropout_options(config),
                diffusion_settings=schedule,
                diffusion_identity=diffusion_identity(dict(schedule, beta_0=.0001, beta_T=.02, timesteps=200)),
                provenance=processor.provenance(), test_npz_opened=False)


def base_config():
    # Deliberately stale settings exercise the new protocol's full overrides.
    return dict(data_dir="data", metadata_path="data/participant_info.csv",
                split_path="old_splits.json", generate_splits=True,
                multiplier=2.0, downsample_rate=2, max_epochs=4, window_epochs=4,
                missing_value_policy="masked", signal_normalization={"method": "zscore"},
                training_input_dropout=dict(probability=.1, seed=999), s4_dropout=.1,
                diffusion_schedule="linear_power", terminal_alpha_bar=1e-8,
                validation=dict(max_windows=64, batch_size=2, seed=99,
                                reference_mask_policy="training", save_artifacts=False),
                early_stopping=dict(enabled=True, smoothing_window=5, min_delta_kl=.1))


def synthetic_best_checkpoint(run_dir, manifest, scores):
    """Write clearly artificial bytes and selection metadata, never tensors."""
    first_minimum = scores.index(min(scores)) + 1
    checkpoint = run_dir / "training/best_kl_model.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"Synthetic best-checkpoint evidence; not a model checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    best = dict(epoch=first_minimum, metric="gaussian_kl_real_to_generated",
                value=min(scores), selection_rule="current_kl", checkpoint_sha256=digest)
    manifest["validation"] = dict(best_checkpoints={"kl": best})
    selected = dict(checkpoint_epoch=first_minimum, checkpoint_sha256=digest,
                    checkpoint_selection=copy.deepcopy(best),
                    data_identity=copy.deepcopy(manifest["data_identity"]),
                    training_args=copy.deepcopy(manifest["training_args"]))
    write_json(run_dir / "training/best_kl_manifest.json", selected)
    return selected


class All88SweepGridTests(unittest.TestCase):
    def test_all_twelve_requested_combinations_include_both_zero_labels(self):
        grid = driver.load_grid(GRID_PATH)
        self.assertEqual(len(grid), 12)
        self.assertEqual({row["id"] for row in grid}, set(range(12)))
        self.assertEqual(len({row["name"] for row in grid}), 12)
        actual = set()
        zero_rows = []
        for row in grid:
            location = row["dropout_location"]
            input_p, s4_p = row["input_probability"], row["s4_probability"]
            probability = input_p if location == "input" else s4_p
            self.assertEqual(s4_p if location == "input" else input_p, 0.0)
            self.assertEqual(sum(p > 0 for p in (input_p, s4_p)), int(probability > 0))
            actual.add((tuple(row["normalization_range"]), location, probability))
            if probability == 0:
                zero_rows.append(row)
        expected = {(bounds, location, probability)
                    for bounds in RANGES for location in ("input", "s4")
                    for probability in PROBABILITIES}
        self.assertEqual(actual, expected)
        self.assertEqual(len(zero_rows), 4)
        self.assertEqual({row["name"] for row in zero_rows}, {
            "minmax_m11_input_p0p0", "minmax_m11_s4_p0p0",
            "minmax_01_input_p0p0", "minmax_01_s4_p0p0",
        })

    def test_select_configuration_is_independent_and_validates_id(self):
        grid = driver.load_grid(GRID_PATH)
        original = copy.deepcopy(grid)
        for row in grid:
            chosen = driver.select_configuration(grid, row["id"])
            self.assertEqual(chosen, row)
            chosen["normalization_range"][0] = 999
        self.assertEqual(grid, original)
        for invalid_id in (-1, 12, 1000, True, False, "0", .5):
            with self.subTest(configuration_id=invalid_id), self.assertRaises(ValueError):
                driver.select_configuration(grid, invalid_id)

    def test_missing_duplicate_renamed_and_off_grid_rows_are_rejected(self):
        original = json.loads(GRID_PATH.read_text(encoding="utf-8"))
        changes = {
            "schema": lambda doc: doc.update(schema="dreamt_dropout_sweep_v1"),
            "missing_row": lambda doc: doc["configurations"].pop(),
            "duplicate_id": lambda doc: doc["configurations"][1].update(id=0),
            "duplicate_combination": lambda doc: doc["configurations"][1].update(
                {key: value for key, value in doc["configurations"][0].items() if key != "id"}),
            "invalid_range": lambda doc: doc["configurations"][0].update(normalization_range=[-2, 2]),
            "both_active": lambda doc: doc["configurations"][0].update(input_probability=.03, s4_probability=.03),
            "off_grid_probability": lambda doc: doc["configurations"][0].update(input_probability=.01),
            "invalid_location": lambda doc: doc["configurations"][0].update(dropout_location="both"),
            "mismatched_name": lambda doc: doc["configurations"][0].update(name="minmax_m11_input_p0p03"),
            "unknown_field": lambda doc: doc["configurations"][0].update(resume=True),
        }
        with temporary_directory() as folder:
            path = Path(folder) / "grid.json"
            for name, change in changes.items():
                value = copy.deepcopy(original)
                change(value)
                write_json(path, value)
                with self.subTest(corruption=name), self.assertRaises(ValueError):
                    driver.load_grid(path)


class All88SweepConfigurationTests(unittest.TestCase):
    def test_each_configuration_sets_the_complete_requested_protocol(self):
        base = base_config()
        original = copy.deepcopy(base)
        path = ROOT / "work/all88_reference/config.json"
        for row in driver.load_grid(GRID_PATH):
            with self.subTest(configuration_id=row["id"]):
                result = driver.sweep_config(base, path, row)
                self.assertEqual(result["evaluation_protocol"], PROTOCOL)
                self.assertEqual(result["missing_value_policy"], "preserve")
                self.assertEqual(result["window_epochs"], 4)
                self.assertEqual(result["multiplier"], 1.0)
                self.assertEqual(result["downsample_rate"], 1)
                self.assertIsNone(result["max_epochs"])
                self.assertIsNone(result["split_path"])
                self.assertEqual(Path(result["data_dir"]), (path.parent / "data").resolve())
                self.assertEqual(Path(result["metadata_path"]), (path.parent / "data/participant_info.csv").resolve())
                spec = dict(MINMAX)
                if row["normalization_range"] == [0, 1]:
                    spec["output_range"] = [0.0, 1.0]
                self.assertEqual(normalization_spec(result), spec)
                self.assertEqual(input_dropout_options(result), dict(probability=row["input_probability"], seed=42))
                self.assertEqual(s4_dropout_options(result), row["s4_probability"])
                self.assertIsNone(result["validation"]["max_windows"])
                self.assertEqual(result["validation"]["batch_size"], 8)
                self.assertEqual(result["validation"]["seed"], 20261020)
                self.assertEqual(result["validation"]["reference_mask_policy"], "observed_only")
                self.assertTrue(result["validation"]["save_artifacts"])
                self.assertFalse(result["early_stopping"]["enabled"])
                self.assertEqual(result["early_stopping"]["selection_mode"], "current")
                self.assertEqual(result["early_stopping"]["smoothing_window"], 1)
                self.assertEqual(result["early_stopping"]["min_delta_kl"], 0.0)
                self.assertEqual(result["diffusion_schedule"], "linear_power")
                self.assertEqual(result["terminal_alpha_bar"], 1e-8)
        self.assertEqual(base, original)

    def test_zero_probability_labels_have_identical_computational_configuration(self):
        grid = driver.load_grid(GRID_PATH)
        for bounds in RANGES:
            pairs = [row for row in grid if tuple(row["normalization_range"]) == bounds
                     and row["input_probability"] == row["s4_probability"] == 0]
            self.assertEqual(len(pairs), 2)
            configs = [driver.sweep_config(base_config(), ROOT / "config.json", row) for row in pairs]
            with self.subTest(normalization_range=bounds):
                self.assertEqual(configs[0], configs[1])
                self.assertEqual(configs[0]["training_input_dropout"]["seed"], 42)

    def test_submission_allocates_twelve_separate_gpu_tasks_with_full_budget(self):
        path = ROOT / "submit_all88_s092_sweep.sh"
        source = path.read_text(encoding="utf-8")
        self.assertNotIn(b"\r\n", path.read_bytes())
        self.assertIn("set -euo pipefail", source)
        for flag in ("--account=def-plago", "--cpus-per-task=4", "--gpus-per-node=1", "--time=10:00:00"):
            self.assertIn("#SBATCH " + flag, source)
        self.assertRegex(source, r"(?m)^#SBATCH --array=0-11(?:%\d+)?$")
        self.assertIn('cd "${SLURM_SUBMIT_DIR}"', source)
        self.assertIn("source ./.venv/bin/activate", source)
        self.assertIn("scripts/run_all88_s092_sweep.py", source)
        self.assertIn('--configuration-id "${SLURM_ARRAY_TASK_ID}"', source)
        self.assertIn('--epochs 30 --batch-size 8 --threads "${SLURM_CPUS_PER_TASK:-4}"', source)
        self.assertNotIn("--reference-run", source)
        self.assertNotIn("--resume", source)
        self.assertNotIn("--checkpoint", source)

    def test_training_entry_refuses_missing_cuda_before_data_or_compute(self):
        row = driver.load_grid(GRID_PATH)[0]
        with temporary_directory() as folder:
            base = Path(folder)
            config_path, output = base / "config.json", base / "run"
            config = driver.sweep_config(base_config(), config_path, row)
            write_json(config_path, config)
            arguments = ["run_dreamt.py", "--config", str(config_path), "--output-dir", str(output)]
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


class All88SweepPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        data = self.base / "data"
        data.mkdir()
        subjects = [f"S{number:03d}" for number in range(1, 88)] + ["S092"]
        records = ["SID,AHI,BMI,AGE,GENDER"]
        for index, subject in enumerate(subjects):
            # An incomplete tail in S092 proves that only complete windows enter
            # evaluation; isolated raw zeros remain stored missing-value markers.
            length = 6000 + (37 if subject == "S092" else 0)
            values = np.full((length, 1), .7 + (index % 10) * .01, dtype=np.float32)
            values[::71] = 0
            stages = np.zeros(length, dtype=np.int8)
            np.savez(data / (subject + ".npz"), data=values, stages=stages,
                     fs=25, ahi=5.0, synthetic=True)
            records.append(f"{subject},5.0,{20 + index % 10},{30 + index % 40},{'M' if index % 2 else 'F'}")
        (data / "participant_info.csv").write_text("\n".join(records) + "\n", encoding="utf-8")
        self.config_path = self.base / "config.json"
        self.config = base_config()
        write_json(self.config_path, self.config)

    def current(self, row):
        return preflight(driver.sweep_config(self.config, self.config_path, row))

    def test_real_synthetic_npz_preflights_train_all_88_and_evaluate_all_s092_windows(self):
        identities = []
        for row in driver.load_grid(GRID_PATH):
            with self.subTest(configuration_id=row["id"]):
                current = self.current(row)
                checks = driver.validate_preflight(current, row)
                self.assertTrue(checks)
                self.assertTrue(all(checks.values()), checks)
                identity = current["data_identity"]
                self.assertEqual(identity["evaluation_protocol"], PROTOCOL)
                self.assertEqual(identity["split_source"], "all_subjects")
                self.assertEqual(len(identity["splits"]["train"]), 88)
                self.assertIn("S092.npz", identity["splits"]["train"])
                self.assertEqual(identity["splits"]["val"], [])
                self.assertEqual(identity["splits"]["test"], [])
                self.assertEqual(len(identity["train_npz_sha256"]), 88)
                self.assertEqual(len(identity["scaler_subjects"]), 88)
                self.assertIn("S092", identity["scaler_subjects"])
                self.assertEqual(current["validation_windows"], [
                    dict(filename="S092.npz", start_sample=0),
                    dict(filename="S092.npz", start_sample=3000),
                ])
                self.assertEqual(current["validation_selection"]["selected_windows"],
                                 current["validation_selection"]["available_nonoverlapping_windows"])
                self.assertEqual(current["training_windows"], 88 * 5)
                self.assertEqual(current["diffusion_identity"]["timesteps"], 200)
                self.assertFalse(current["provenance"]["evaluation_is_held_out"])
                identities.append((identity["train_npz_sha256"], identity["splits"],
                                   current["validation_windows"], current["diffusion_identity"]))
        self.assertTrue(all(value == identities[0] for value in identities))

    def test_preflight_refuses_holdout_truncation_dropout_and_smoothed_selection(self):
        row = next(row for row in driver.load_grid(GRID_PATH) if row["input_probability"] == .03)
        original = self.current(row)
        changes = {
            "87_subjects": lambda p: p["data_identity"]["splits"]["train"].pop(0),
            "subject_without_training": lambda p: p["training_subject_window_counts"].update({"S092.npz": 0}),
            "holdout_subject": lambda p: p["data_identity"]["splits"].update(val=["S092.npz"]),
            "test_split": lambda p: p["data_identity"]["splits"].update(test=["S001.npz"]),
            "evaluation_subject": lambda p: p["validation_windows"][0].update(filename="S001.npz"),
            "truncated_validation": lambda p: p["validation_options"].update(max_windows=64),
            "partial_selection": lambda p: p["validation_selection"].update(available_nonoverlapping_windows=3),
            "short_diffusion": lambda p: p["diffusion_identity"].update(timesteps=100),
            "input_dropout": lambda p: p["training_input_dropout"].update(probability=.05),
            "input_seed": lambda p: p["training_input_dropout"].update(seed=999),
            "other_dropout": lambda p: p.update(s4_dropout=.03),
            "smoothed_selection": lambda p: p["early_stopping"].update(selection_mode="smoothed"),
            "averaging_window": lambda p: p["early_stopping"].update(smoothing_window=5),
            "early_termination": lambda p: p["early_stopping"].update(enabled=True),
        }
        for name, change in changes.items():
            changed = copy.deepcopy(original)
            change(changed)
            with self.subTest(corruption=name), self.assertRaises(ValueError):
                driver.validate_preflight(changed, row)

    def invoke_check_only(self, row, output, *, mutate_preflight=None):
        calls = []

        def preprocessing_only(command, **kwargs):
            calls.append(command)
            self.assertTrue(kwargs["check"])
            self.assertEqual(Path(kwargs["cwd"]), ROOT)
            self.assertFalse(any(key.startswith("DREAMT_") for key in kwargs["env"]))
            self.assertIn("--check-only", command)
            config = json.loads(Path(command[command.index("--config") + 1]).read_text(encoding="utf-8"))
            check = preflight(config)
            if mutate_preflight is not None:
                mutate_preflight(check)
            destination = Path(command[command.index("--output-dir") + 1])
            write_json(destination / "preflight.json", check)
            return subprocess.CompletedProcess(command, 0)

        args = ["--configuration-id", str(row["id"]), "--config", str(self.config_path),
                "--output-dir", str(output), "--check-only"]
        with mock.patch.object(driver.subprocess, "run", side_effect=preprocessing_only), \
             mock.patch.dict(os.environ, {"DREAMT_DATA_DIR": "must-not-override-selected-config"}), \
             contextlib.redirect_stdout(io.StringIO()):
            result = driver.main(args)
        return result, calls

    def test_check_only_uses_fresh_defaults_without_compute_for_every_configuration(self):
        before = self.config_path.read_bytes()
        for row in driver.load_grid(GRID_PATH):
            output = self.base / row["name"]
            with self.subTest(configuration_id=row["id"]):
                result, calls = self.invoke_check_only(row, output)
                self.assertEqual(result["status"], "checked")
                self.assertFalse(result["training_completed"])
                self.assertEqual(len(calls), 1)
                for flag, expected in (("--epochs", "30"), ("--batch-size", "8"), ("--threads", "4")):
                    self.assertEqual(calls[0][calls[0].index(flag) + 1], expected)
                self.assertNotIn("--resume", calls[0])
                self.assertNotIn("--checkpoint", calls[0])
                self.assertFalse((output / "run").exists())
                self.assertFalse((output / "training.log").exists())
        self.assertEqual(self.config_path.read_bytes(), before)

    def test_changed_preflight_blocks_training_and_records_failure(self):
        row = driver.load_grid(GRID_PATH)[0]
        output = self.base / "invalid_preflight"
        with self.assertRaises(ValueError):
            self.invoke_check_only(row, output, mutate_preflight=lambda p: p["early_stopping"].update(selection_mode="smoothed"))
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["training_completed"])
        self.assertFalse((output / "run").exists())
        self.assertFalse((output / "training.log").exists())

    def test_actual_snapshot_preserves_all_subject_protocol_and_current_kl(self):
        row = next(row for row in driver.load_grid(GRID_PATH) if row["s4_probability"] == .03)
        config = driver.sweep_config(self.config, self.config_path, row)
        write_json(self.config_path, config)
        output = self.base / "snapshot"
        calls = []

        def no_model_subprocess(command, **kwargs):
            calls.append(command)
            frozen_path = Path(kwargs["env"]["DREAMT_CONFIG_PATH"])
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            self.assertEqual(frozen["evaluation_protocol"], PROTOCOL)
            self.assertEqual(frozen["early_stopping"]["selection_mode"], "current")
            self.assertEqual(frozen["s4_dropout"], .03)
            return subprocess.CompletedProcess(command, 0)

        args = ["run_dreamt.py", "--config", str(self.config_path), "--output-dir", str(output),
                "--epochs", "3", "--threads", "4", "--skip-generation"]
        # CUDA attestation and external computation are both synthetic. The
        # real snapshot/configuration code and NPZ preprocessor run unchanged.
        with mock.patch.object(sys, "argv", args), \
             mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(run_dreamt, "require_cuda", return_value="cuda:0"), \
             mock.patch.object(run_dreamt, "cuda_environment", return_value={"synthetic": True}), \
             mock.patch.object(run_dreamt.subprocess, "run", side_effect=no_model_subprocess), \
             contextlib.redirect_stdout(io.StringIO()):
            run_dreamt.main()
        self.assertEqual(len(calls), 2)
        saved = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
        self.assertTrue(all(driver.validate_preflight(saved, row).values()))
        frozen_path = output / "workspace/dataset_config.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        self.assertEqual(preflight(frozen)["data_identity"], saved["data_identity"])
        for name in driver.SNAPSHOT_SOURCES:
            self.assertEqual((output / "workspace" / name).read_bytes(), (ROOT / name).read_bytes())

    def test_frozen_evaluation_binding_keeps_s092_in_training_without_model_loading(self):
        check = self.current(driver.load_grid(GRID_PATH)[0])
        manifest = dict(data_identity=check["data_identity"])
        args = Namespace(**dict(evaluate_denoising.DEFAULT_TRAINING_ARGS,
                                dataset="DREAMT", window_size=3000, s4_lmax=3000))
        with mock.patch.dict(os.environ, {"DREAMT_DATA_DIR": "must-not-override-frozen-identity"}), \
             mock.patch("torch.load", side_effect=AssertionError("Metadata binding must not load a model")):
            processor, evidence = evaluate_denoising.bind_training_data(
                args, manifest, self.base / "run/training/run_manifest.json", Preprocessor)
        self.assertTrue(evidence["all_recorded_identity_fields_match"])
        self.assertEqual(processor.data_identity(), check["data_identity"])
        self.assertEqual(len(processor.splits["train"]), 88)
        self.assertIn("S092.npz", processor.splits["train"])
        self.assertEqual(processor.evaluation_filenames, ["S092.npz"])

    def test_synthetic_completion_preserves_fresh_budget_sources_and_raw_best(self):
        # One input-dropout and one S4-dropout configuration exercise both report
        # validators. These are mocked execution reports, not performance claims.
        grid = driver.load_grid(GRID_PATH)
        rows = [next(row for row in grid if row["input_probability"] == .03),
                next(row for row in grid if row["s4_probability"] == .03)]
        for row in rows:
            calls = []

            def synthetic_execution(command, **kwargs):
                calls.append(command)
                config_path = Path(command[command.index("--config") + 1])
                check = preflight(json.loads(config_path.read_text(encoding="utf-8")))
                destination = Path(command[command.index("--output-dir") + 1])
                if "--check-only" in command:
                    write_json(destination / "preflight.json", check)
                    return subprocess.CompletedProcess(command, 0)
                manifest = write_completion(destination, check, 3)
                scores = [.8, .2, .5]
                (destination / "training/epoch_metrics.csv").write_text(
                    "epoch,gaussian_kl_real_to_generated\n" +
                    "".join(f"{epoch},{score}\n" for epoch, score in enumerate(scores, 1)), encoding="utf-8")
                checkpoint = destination / "workspace/saved_models/DREAMT/model.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"Synthetic final-checkpoint evidence; not model tensors")
                probability = row["s4_probability"]
                layer_count = manifest["training_args"]["num_res_layers"] * 2
                report = dict(probability=probability, active=probability > 0, training=True,
                              layers=layer_count, configuration_inspected=True, disabled_in_eval=True,
                              modules=[dict(name=f"synthetic_layer_{number}", probability=probability,
                                            module="Dropout2d" if probability else "Identity", training=True)
                                       for number in range(layer_count)])
                manifest.update(
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    source_sha256={name: driver.sha256(ROOT / name) for name in driver.SNAPSHOT_SOURCES},
                    training_input_dropout=minmax_fixtures.MinmaxExperimentWorkflowTests.dropout_report(check, 3, manifest["updates"]),
                    s4_dropout=probability, s4_dropout_runtime=report)
                synthetic_best_checkpoint(destination, manifest, scores)
                write_json(destination / "training/run_manifest.json", manifest)
                write_json(destination / "training/validation/epoch_0003/metrics.json", dict(
                    status="completed", epoch=3, aggregate=dict(gaussian_kl_real_to_generated=.5),
                    diagnostics=dict(generated_nonpositive_fraction=.01)))
                return subprocess.CompletedProcess(command, 0)

            output = self.base / ("completed_" + row["name"])
            args = ["--config", str(self.config_path), "--configuration-id", str(row["id"]),
                    "--output-dir", str(output), "--epochs", "3"]
            with self.subTest(dropout_location=row["dropout_location"]), \
                 mock.patch.object(driver.subprocess, "run", side_effect=synthetic_execution), \
                 mock.patch("torch.load", side_effect=AssertionError("Synthetic orchestration must not load tensors")), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = driver.main(args)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["training_completed"])
            self.assertEqual(len(calls), 2)
            self.assertIn("--skip-generation", calls[-1])
            self.assertNotIn("--resume", calls[-1])
            self.assertNotIn("--checkpoint", calls[-1])
            self.assertEqual(result["completed_budget"]["updates"], 55 * 3)
            self.assertEqual(result["best_checkpoint"]["epoch"], 2)
            self.assertEqual(result["best_checkpoint"]["value"], .2)
            self.assertEqual(result["final_metrics"]["gaussian_kl_real_to_generated"], .5)


class All88BestCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.scores = [.8, .2, .2, .5]
        self.rows = [dict(epoch=str(epoch), gaussian_kl_real_to_generated=str(score))
                     for epoch, score in enumerate(self.scores, 1)]
        self.manifest = dict(data_identity={"synthetic_identity": True}, training_args={"synthetic_args": True})
        self.selected = synthetic_best_checkpoint(self.base, self.manifest, self.scores)

    def test_first_raw_minimum_is_selected_even_when_last_epoch_is_worse(self):
        with mock.patch("torch.load", side_effect=AssertionError("Selection verification only hashes bytes")):
            best = driver.validate_best_checkpoint(self.base, self.manifest, self.rows)
        self.assertEqual(best["epoch"], 2)
        self.assertEqual(best["value"], .2)
        self.assertEqual(best["selection_rule"], "current_kl")

    def test_smoothed_metrics_wrong_epoch_and_changed_checkpoint_evidence_are_rejected(self):
        changes = {
            "sma_column": lambda manifest, rows: rows[0].update(smoothed_gaussian_kl_real_to_generated="0.1"),
            "sma_ready_column": lambda manifest, rows: rows[0].update(smoothing_ready="True"),
            "wrong_selection": lambda manifest, rows: manifest["validation"]["best_checkpoints"]["kl"].update(selection_rule="smoothed_kl"),
            "last_epoch": lambda manifest, rows: manifest["validation"]["best_checkpoints"]["kl"].update(epoch=4, value=.5),
            "later_tied_epoch": lambda manifest, rows: manifest["validation"]["best_checkpoints"]["kl"].update(epoch=3),
            "bad_hash": lambda manifest, rows: manifest["validation"]["best_checkpoints"]["kl"].update(checkpoint_sha256="0" * 64),
            "nonfinite_kl": lambda manifest, rows: rows[1].update(gaussian_kl_real_to_generated="nan"),
        }
        for name, change in changes.items():
            manifest, rows = copy.deepcopy(self.manifest), copy.deepcopy(self.rows)
            change(manifest, rows)
            with self.subTest(corruption=name), self.assertRaises(ValueError):
                driver.validate_best_checkpoint(self.base, manifest, rows)
        for field in ("data_identity", "training_args"):
            selected = copy.deepcopy(self.selected)
            selected[field] = {"changed": True}
            write_json(self.base / "training/best_kl_manifest.json", selected)
            with self.subTest(corruption="best_" + field), self.assertRaises(ValueError):
                driver.validate_best_checkpoint(self.base, self.manifest, self.rows)


if __name__ == "__main__":
    unittest.main()
