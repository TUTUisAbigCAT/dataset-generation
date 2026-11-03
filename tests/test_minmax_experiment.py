"""Real synthetic NPZ preflights and mocked orchestration, never model training."""
from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest import mock

from scripts import run_minmax_experiment as driver
from scripts import run_preprocessing_experiment as preprocessing_driver
from scripts import run_missing_value_ablation as budget_helpers
from scripts.verify_upstream_boundary import remove_training_input_dropout_edits
from tests import test_preprocessing_experiment as preprocessing_fixtures
from tests.test_extended_ablation import write_completion, write_json


ROOT = Path(__file__).resolve().parents[1]
MINMAX = dict(method="minmax", min_seconds=0.0, max_seconds=2.0)
DISABLED_DROPOUT = dict(probability=0.0, seed=42)


def preflight(config):
    result = preprocessing_fixtures.real_preflight(config)
    result["training_input_dropout"] = copy.deepcopy(config.get("training_input_dropout", DISABLED_DROPOUT))
    return result


class MinmaxExperimentConfigTests(unittest.TestCase):
    def base_config(self):
        base = preprocessing_fixtures.PreprocessingConfigTests().base_config()
        base["signal_normalization"] = dict(MINMAX)
        return base

    def test_each_experiment_changes_only_one_factor(self):
        base = self.base_config()
        original = copy.deepcopy(base)
        path = ROOT / "work/minmax_reference/config.json"
        ranged = driver.experiment_config(base, path, "minmax01")
        dropped = driver.experiment_config(base, path, "minmax_dropout")
        self.assertEqual(base, original)
        self.assertEqual(ranged["signal_normalization"], dict(MINMAX, output_range=[0.0, 1.0]))
        self.assertEqual(ranged.get("training_input_dropout", DISABLED_DROPOUT), DISABLED_DROPOUT)
        self.assertEqual(dropped["signal_normalization"], MINMAX)
        self.assertEqual(dropped["training_input_dropout"], dict(probability=.1, seed=42))
        for config in (ranged, dropped):
            self.assertEqual(config["missing_value_policy"], "preserve")
            self.assertEqual(config["validation"], base["validation"])
            self.assertFalse(config["early_stopping"]["enabled"])
            self.assertEqual(config["diffusion_schedule"], base["diffusion_schedule"])
            self.assertEqual(config["terminal_alpha_bar"], base["terminal_alpha_bar"])
            self.assertEqual(Path(config["data_dir"]), (path.parent / "data").resolve())

    def test_nonbaseline_configs_and_unknown_experiment_are_rejected(self):
        base = self.base_config()
        for updates in (
            dict(missing_value_policy="masked"), dict(missing_value_policy="zero_fill_one"),
            dict(signal_normalization={"method": "zscore"}),
            dict(signal_normalization=dict(MINMAX, output_range=[0.0, 1.0])),
            dict(training_input_dropout=dict(probability=.1, seed=42)),
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                driver.experiment_config(dict(base, **updates), ROOT / "old/config.json", "minmax01")
        with self.assertRaises(ValueError):
            driver.experiment_config(base, ROOT / "old/config.json", "unknown")

    def test_two_slurm_jobs_are_separate_and_reference_completed_minmax(self):
        reference = "/home/yxtao/links/scratch/dataset-generation/runs/minmax_20261103T003905Z_1223635"
        for experiment in ("minmax01", "minmax_dropout"):
            with self.subTest(experiment=experiment):
                path = ROOT / f"submit_{experiment}.sh"
                source = path.read_text(encoding="utf-8")
                self.assertNotIn(b"\r\n", path.read_bytes())
                self.assertIn("set -euo pipefail", source)
                for flag in ("--account=def-plago", "--cpus-per-task=4", "--gpus-per-node=1", "--time=10:00:00"):
                    self.assertIn("#SBATCH " + flag, source)
                self.assertIn(reference, source)
                self.assertIn('cd "${SLURM_SUBMIT_DIR}"', source)
                self.assertIn("source ./.venv/bin/activate", source)
                self.assertEqual(source.count("srun python -u scripts/run_minmax_experiment.py"), 1)
                self.assertIn(f"--experiment {experiment}", source)
                self.assertIn(f"/runs/{experiment}_$(date -u", source)
                self.assertIn('--epochs 30 --batch-size 8 --threads "${SLURM_CPUS_PER_TASK:-4}"', source)
                self.assertNotIn("--resume", source)
                self.assertNotIn("run_missing_value_ablation.py", source)
                self.assertNotIn("run_preprocessing_experiment.py", source)
                self.assertNotIn('mkdir "$OUTPUT_ROOT"', source)


class MinmaxExperimentWorkflowTests(unittest.TestCase):
    def setUp(self):
        # Reuse fixture setup only, not its test methods. The files contain
        # artificial NPZs with inserted raw zeros and no trained model weights.
        self.fixture = preprocessing_fixtures.PreprocessingIdentityTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.base = self.fixture.base
        self.config_path = self.fixture.config_path
        self.config = copy.deepcopy(self.fixture.config)
        self.config["signal_normalization"] = dict(MINMAX)
        self.reference_preflight = preflight(self.config)

    def current(self, experiment):
        return preflight(driver.experiment_config(self.config, self.config_path, experiment))

    def test_real_loader_changes_affine_only_or_dropout_only(self):
        original = self.reference_preflight
        for experiment in ("minmax01", "minmax_dropout"):
            with self.subTest(experiment=experiment):
                current = self.current(experiment)
                checks = driver.compare_minmax_preflight(original, current, experiment)
                self.assertTrue(checks)
                self.assertTrue(all(checks.values()), checks)
                self.assertEqual(current["validation_windows"], original["validation_windows"])
                self.assertEqual(current["training_windows"], original["training_windows"])
                a, b = original["data_identity"]["scaler"], current["data_identity"]["scaler"]
                for key in ("signal_mean_seconds", "signal_std_seconds", "signal_fit_samples",
                            "continuous_mean", "continuous_std", "continuous_columns"):
                    self.assertEqual(a[key], b[key])
                if experiment == "minmax01":
                    self.assertEqual((b["signal_center_seconds"], b["signal_scale_seconds"]), (0.0, 2.0))
                else:
                    self.assertEqual(current["data_identity"], original["data_identity"])
                    self.assertEqual((b["signal_center_seconds"], b["signal_scale_seconds"]), (1.0, 1.0))

    def test_preflight_rejects_data_window_metadata_schedule_and_dropout_drift(self):
        changes = {
            "split": lambda p: p["data_identity"].update(split_fingerprint="wrong"),
            "raw_hash": lambda p: p["data_identity"]["train_npz_sha256"].update({"EXTRA.npz": "f"*64}),
            "opened_hash": lambda p: p["provenance"]["opened_npz_sha256"].update({"EXTRA.npz": "f"*64}),
            "condition_scaler": lambda p: p["data_identity"]["scaler"]["continuous_mean"].__setitem__(0, 999.0),
            "signal_statistic": lambda p: p["data_identity"]["scaler"].update(signal_mean_seconds=1.9),
            "training_windows": lambda p: p.update(training_windows=p["training_windows"] + 1),
            "validation_position": lambda p: p["validation_windows"][0].update(start_sample=750),
            "schedule": lambda p: p["diffusion_identity"].update(timesteps=999),
            "validation_seed": lambda p: p["validation_options"].update(seed=999),
            "scoring_policy": lambda p: p["validation_options"].update(reference_mask_policy="training"),
            "dropout_probability": lambda p: p["training_input_dropout"].update(probability=.2),
            "dropout_seed": lambda p: p["training_input_dropout"].update(seed=999),
        }
        for experiment in ("minmax01", "minmax_dropout"):
            original = self.current(experiment)
            for label, change in changes.items():
                with self.subTest(experiment=experiment, change=label):
                    current = copy.deepcopy(original)
                    change(current)
                    with self.assertRaises(ValueError):
                        driver.compare_minmax_preflight(self.reference_preflight, current, experiment)

    def make_reference(self):
        """Complete synthetic historical metadata, not an actually trained model."""
        root = self.base / "reference_minmax"
        check = copy.deepcopy(self.reference_preflight)
        # Real historical minmax ran before the optional dropout metadata existed.
        check.pop("training_input_dropout")
        write_json(root / "config.json", self.config)
        write_json(root / "preflight/preflight.json", check)
        manifest = write_completion(root / "run", check, 2)
        workspace = root / "run/workspace"
        workspace.mkdir(parents=True)
        hashes = {}
        names = set(preprocessing_driver.INVARIANT_SOURCES) | {
            "data_utils.py", "epoch_validation.py", "signal_normalization.py"}
        for name in names:
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            value = (ROOT / name).read_bytes()
            if name == "training_wavestitch.py":
                value = remove_training_input_dropout_edits(value)
            target.write_bytes(value)
            hashes[name] = hashlib.sha256(value).hexdigest()
        write_json(workspace / "dataset_config.json", self.config)
        hashes["dataset_config.json"] = hashlib.sha256((workspace / "dataset_config.json").read_bytes()).hexdigest()
        checkpoint = workspace / "saved_models/DREAMT/model.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic historical checkpoint bytes; no model training")
        manifest.update(source_sha256=hashes, checkpoint=str(checkpoint),
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        write_json(root / "run/training/run_manifest.json", manifest)
        self.write_final_metrics(root / "run", 2)
        rows = budget_helpers._epoch_rows(root / "run/training/epoch_metrics.csv")
        budget = budget_helpers.validate_training_budget(manifest, rows, check, 2, 8)
        write_json(root / "summary.json", dict(status="completed", training_completed=True,
                   experiment="minmax", epochs=2, completed_budget=budget,
                   final_checkpoint_sha256=manifest["checkpoint_sha256"]))
        return root

    @staticmethod
    def write_final_metrics(output, epochs):
        write_json(output / f"training/validation/epoch_{epochs:04d}/metrics.json",
                   dict(status="completed", epoch=epochs,
                        aggregate=dict(gaussian_kl_real_to_generated=1.0 / epochs),
                        diagnostics=dict(generated_nonpositive_fraction=.02)))

    @staticmethod
    def dropout_report(check, epochs, updates):
        # Deliberately fabricated execution evidence for orchestration tests.
        # This does not assert that CUDA or a real random mask ran locally.
        options = copy.deepcopy(check["training_input_dropout"])
        active = options["probability"] > 0
        eligible = check["training_windows"] * check["window_size"] * epochs if active else 0
        dropped = round(options["probability"] * eligible)
        return dict(options=options, active=active, application_calls=updates if active else 0,
                    eligible_values=eligible, dropped_values=dropped,
                    observed_fraction=dropped / eligible if eligible else 0.0,
                    counter_dtype="int64", generator_device="cuda:0" if active else None,
                    protocol="Synthetic mocked report; no actual training")

    def test_load_legacy_reference_and_exact_training_source_exception(self):
        root = self.make_reference()
        reference = driver.load_minmax_reference(root)
        self.assertEqual(reference["budget"]["completed_epochs"], 2)
        self.assertNotIn("training_input_dropout", reference["preflight"])
        evidence = driver.verify_reference_sources(dict(root=root, arms={"run": reference}))
        self.assertIn("run", evidence)
        source = root / "run/workspace/training_wavestitch.py"
        original = source.read_bytes()
        source.write_bytes(original + b"\n# unauthorized extra edit\n")
        manifest_path = root / "run/training/run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_sha256"]["training_wavestitch.py"] = hashlib.sha256(source.read_bytes()).hexdigest()
        write_json(manifest_path, manifest)
        with self.assertRaises(ValueError):
            driver.verify_reference_sources(dict(root=root, arms={"run": driver.load_minmax_reference(root)}))

    def test_reference_must_have_complete_budget_and_matching_actual_dropout(self):
        root = self.make_reference()
        summary_path = root / "summary.json"
        original_summary = summary_path.read_bytes()
        for update in (dict(status="checked"), dict(training_completed=False), dict(experiment="zero_fill_one"),
                       dict(final_checkpoint_sha256="0"*64)):
            with self.subTest(update=update):
                changed = json.loads(original_summary)
                changed.update(update)
                write_json(summary_path, changed)
                with self.assertRaises(ValueError):
                    driver.load_minmax_reference(root)
        summary_path.write_bytes(original_summary)
        path = root / "run/preflight.json"
        actual = json.loads(path.read_text(encoding="utf-8"))
        actual["training_input_dropout"] = dict(probability=.1, seed=42)
        write_json(path, actual)
        with self.assertRaisesRegex(ValueError, "dropout"):
            driver.load_minmax_reference(root)

    def test_dropout_report_requires_full_budget_cuda_int64_and_plausible_counts(self):
        for experiment in ("minmax01", "minmax_dropout"):
            check = self.current(experiment)
            updates = (check["training_windows"] + 7) // 8 * 2
            budget = dict(updates=updates, training_windows=check["training_windows"], completed_epochs=2)
            report = self.dropout_report(check, 2, updates)
            with self.subTest(experiment=experiment):
                self.assertEqual(driver.validate_dropout_report({"training_input_dropout": report}, check, budget), report)
            mutations = {
                "options": lambda d: d["options"].update(probability=.2),
                "calls": lambda d: d.update(application_calls=-1),
                "eligible": lambda d: d.update(eligible_values=-1),
                "bad_fraction": lambda d: d.update(observed_fraction=float("nan")),
            }
            if experiment == "minmax_dropout":
                mutations.update(
                    cpu_generator=lambda d: d.update(generator_device="cpu"),
                    float_counter=lambda d: d.update(counter_dtype="float32"),
                    no_drops=lambda d: d.update(dropped_values=0, observed_fraction=0.0),
                    all_drops=lambda d: d.update(dropped_values=d["eligible_values"], observed_fraction=1.0))
            for label, change in mutations.items():
                with self.subTest(experiment=experiment, corruption=label):
                    changed = copy.deepcopy(report)
                    change(changed)
                    with self.assertRaises(ValueError):
                        driver.validate_dropout_report({"training_input_dropout": changed}, check, budget)

    def invoke_main(self, root, output, experiment, *, check_only=False, mutate_preflight=None,
                    mutate_manifest=None, fail_training=False):
        commands = []

        def fake_subprocess(command, **kwargs):
            commands.append(command)
            self.assertTrue(kwargs["check"])
            self.assertEqual(Path(kwargs["cwd"]), ROOT)
            self.assertFalse(any(k.startswith("DREAMT_") for k in kwargs["env"]))
            if fail_training and "--check-only" not in command:
                raise subprocess.CalledProcessError(7, command)
            config = json.loads(Path(command[command.index("--config") + 1]).read_text(encoding="utf-8"))
            check = preflight(config)
            if mutate_preflight:
                mutate_preflight(check)
            output = Path(command[command.index("--output-dir") + 1])
            if "--check-only" in command:
                write_json(output / "preflight.json", check)
            else:
                epochs = int(command[command.index("--epochs") + 1])
                manifest = write_completion(output, check, epochs)
                checkpoint = output / "workspace/saved_models/DREAMT/model.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"synthetic generated checkpoint; no actual model execution")
                manifest.update(checkpoint=str(checkpoint),
                                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                training_input_dropout=self.dropout_report(check, epochs, manifest["updates"]))
                if mutate_manifest:
                    mutate_manifest(manifest)
                write_json(output / "training/run_manifest.json", manifest)
                write_json(output / "training/training_report.json",
                           dict(training_input_dropout=manifest["training_input_dropout"]))
                self.write_final_metrics(output, epochs)
            return subprocess.CompletedProcess(command, 0)

        arguments = ["--experiment", experiment, "--reference-run", str(root), "--output-dir", str(output),
                     "--epochs", "2", "--batch-size", "8", "--threads", "4"]
        if check_only:
            arguments.append("--check-only")
        with mock.patch.object(driver.subprocess, "run", side_effect=fake_subprocess), \
             mock.patch.dict("os.environ", {"DREAMT_DATA_DIR": "must-not-override-the-reference"}), \
             contextlib.redirect_stdout(io.StringIO()):
            result = driver.main(arguments)
        return result, commands

    def test_main_each_new_experiment_only_and_final_equal_budget_comparison(self):
        root = self.make_reference()
        previous = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
        for experiment in ("minmax01", "minmax_dropout"):
            with self.subTest(experiment=experiment):
                output = self.base / experiment
                result, commands = self.invoke_main(root, output, experiment)
                self.assertEqual(result["status"], "completed")
                self.assertTrue(result["training_completed"])
                self.assertEqual(len(commands), 2)
                self.assertIn("--check-only", commands[0])
                self.assertIn("--skip-generation", commands[1])
                self.assertNotIn("--resume", commands[1])
                self.assertNotIn("--checkpoint", commands[1])
                self.assertEqual(result["completed_dropout"]["active"], experiment == "minmax_dropout")
                with (output / "comparison.csv").open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([row["experiment"] for row in rows], ["minmax", experiment])
                self.assertEqual({row["epoch"] for row in rows}, {"2"})
                self.assertEqual(len({row["updates"] for row in rows}), 1)
                self.assertFalse((output / "preserve").exists())
                self.assertFalse((output / "masked").exists())
        self.assertEqual(previous, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in previous})

    def test_check_only_has_no_training_or_weights(self):
        root = self.make_reference()
        for experiment in ("minmax01", "minmax_dropout"):
            with self.subTest(experiment=experiment):
                output = self.base / experiment
                result, commands = self.invoke_main(root, output, experiment, check_only=True)
                self.assertEqual(result["status"], "checked")
                self.assertFalse(result["training_completed"])
                self.assertEqual(len(commands), 1)
                self.assertFalse((output / "run").exists())
                self.assertFalse((output / "training.log").exists())

    def test_preflight_drift_stops_before_training_and_records_failure(self):
        root = self.make_reference()
        output = self.base / "bad_preflight"
        with self.assertRaisesRegex(ValueError, "preflight mismatch"):
            self.invoke_main(root, output, "minmax_dropout",
                             mutate_preflight=lambda p: p.update(training_windows=p["training_windows"]+1))
        self.assertFalse((output / "run").exists())
        self.assertFalse((output / "training.log").exists())
        self.assertEqual(json.loads((output / "summary.json").read_text(encoding="utf-8"))["status"], "failed")

    def test_incomplete_execution_or_corrupt_checkpoint_cannot_complete(self):
        root = self.make_reference()
        cases = {
            "failed_process": dict(fail_training=True),
            "short_budget": dict(mutate_manifest=lambda m: m.update(updates=0)),
            "checkpoint": dict(mutate_manifest=lambda m: m.update(checkpoint_sha256="0"*64)),
            "missing_dropout": dict(mutate_manifest=lambda m: m.update(training_input_dropout={})),
            "short_dropout": dict(mutate_manifest=lambda m: m["training_input_dropout"].update(application_calls=0)),
        }
        for name, options in cases.items():
            with self.subTest(failure=name):
                output = self.base / name
                with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                    self.invoke_main(root, output, "minmax_dropout", **options)
                report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["training_completed"])
                self.assertFalse((output / "comparison.csv").exists())

    def test_budget_threads_and_readonly_output_constraints(self):
        root = self.make_reference()
        for flag, value in (("--epochs", "3"), ("--batch-size", "4"), ("--threads", "2")):
            with self.subTest(flag=flag):
                with mock.patch.object(driver.subprocess, "run") as fake, self.assertRaises(ValueError):
                    driver.main(["--experiment", "minmax01", "--reference-run", str(root), "--epochs", "2",
                                 "--output-dir", str(self.base / flag.lstrip("-")), "--check-only", flag, value])
                fake.assert_not_called()
        for output in (root, root / "nested"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "outside"):
                self.invoke_main(root, output, "minmax01", check_only=True)
        output = self.base / "existing"
        output.mkdir()
        sentinel = output / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.invoke_main(root, output, "minmax_dropout", check_only=True)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
