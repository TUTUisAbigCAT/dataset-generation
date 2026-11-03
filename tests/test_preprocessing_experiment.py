"""Synthetic workflow tests; subprocess mocks never run CUDA/model training."""
from __future__ import annotations

import copy
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import run_preprocessing_experiment as driver
from data_utils import Preprocessor
from diffusion_schedule import diffusion_identity, diffusion_settings
from epoch_validation import select_fixed_windows, validation_options, early_stopping_options
from scripts import run_missing_value_ablation as ablation
from tests.test_extended_ablation import write_completion, write_json


ROOT = Path(__file__).resolve().parents[1]


def real_preflight(config):
    """Exercise the actual NPZ loader on synthetic fixtures, without a model."""
    processor = Preprocessor("DREAMT", data_config=config)
    training = processor.make_training_dataset(window_size=processor.window_size, stride=750)
    options = validation_options(config)
    windows, selection = select_fixed_windows(processor, options)
    schedule = diffusion_settings(config, default_schedule="linear_power")
    first = training[0]
    shape = list((first["data"] if isinstance(first, dict) else first).shape)
    return dict(status="passed", data_identity=processor.data_identity(),
                training_windows=len(training), training_input_shape=shape,
                missing_value_policy=processor.missing_value_policy,
                window_epochs=processor.window_epochs, window_size=processor.window_size,
                validation_windows=[dict(filename=name, start_sample=start) for name, start in windows],
                validation_selection=selection, validation_options=options,
                early_stopping=early_stopping_options(config), diffusion_settings=schedule,
                diffusion_identity=diffusion_identity(dict(schedule, beta_0=.0001, beta_T=.02, timesteps=200)),
                provenance=processor.provenance(), test_npz_opened=False)


class PreprocessingConfigTests(unittest.TestCase):
    def base_config(self):
        return dict(data_dir="data", metadata_path="data/participant_info.csv",
                    split_path="data/splits.json", window_epochs=4,
                    missing_value_policy="preserve",
                    diffusion_schedule="linear_power", terminal_alpha_bar=1e-8,
                    validation=dict(max_windows=64, batch_size=8, seed=42,
                                    reference_mask_policy="observed_only",
                                    kl_variance_floor_ms2=1.0, save_artifacts=True),
                    early_stopping=dict(enabled=False))

    def test_experiments_keep_zero_replacement_and_minmax_separate(self):
        base = self.base_config()
        original = copy.deepcopy(base)
        config_path = ROOT / "work/reference/config_preserve.json"
        filled = driver.experiment_config(base, config_path, "zero_fill_one")
        scaled = driver.experiment_config(base, config_path, "minmax")
        self.assertEqual(base, original)
        self.assertEqual(filled["missing_value_policy"], "zero_fill_one")
        self.assertEqual(filled.get("signal_normalization", {"method": "zscore"}),
                         {"method": "zscore"})
        self.assertEqual(scaled["missing_value_policy"], "preserve")
        self.assertEqual(scaled["signal_normalization"],
                         dict(method="minmax", min_seconds=0.0, max_seconds=2.0))
        for config in (filled, scaled):
            self.assertEqual(config["validation"]["max_windows"], 64)
            self.assertEqual(config["validation"]["reference_mask_policy"], "observed_only")
            self.assertTrue(config["validation"]["save_artifacts"])
            self.assertFalse(config["early_stopping"]["enabled"])
            self.assertEqual(config["diffusion_schedule"], base["diffusion_schedule"])
            self.assertEqual(config["terminal_alpha_bar"], base["terminal_alpha_bar"])
            self.assertEqual(config["window_epochs"], 4)
            self.assertEqual(Path(config["data_dir"]), (config_path.parent / "data").resolve())

    def test_submission_scripts_start_separate_fresh_runs(self):
        reference = ("/home/yxtao/links/scratch/dataset-generation/runs/"
                     "missing_value_ablation_extended_20261102T170345Z_1221493/training_30epochs")
        for experiment in ("zero_fill_one", "minmax"):
            with self.subTest(experiment=experiment):
                path = ROOT / f"submit_{experiment}.sh"
                source = path.read_text(encoding="utf-8")
                self.assertNotIn(b"\r\n", path.read_bytes())
                self.assertIn("set -euo pipefail", source)
                self.assertIn('#SBATCH --account=def-plago', source)
                self.assertIn('#SBATCH --gpus-per-node=1', source)
                self.assertIn('#SBATCH --cpus-per-task=4', source)
                self.assertIn('#SBATCH --time=10:00:00', source)
                self.assertIn('cd "${SLURM_SUBMIT_DIR}"', source)
                self.assertIn("source ./.venv/bin/activate", source)
                self.assertIn(reference, source)
                self.assertEqual(source.count("srun python -u scripts/run_preprocessing_experiment.py"), 1)
                self.assertIn(f"--experiment {experiment}", source)
                self.assertIn(f"/runs/{experiment}_$(date -u", source)
                self.assertIn('--epochs 30 --batch-size 8 --threads "${SLURM_CPUS_PER_TASK:-4}"', source)
                self.assertNotIn("--resume", source)
                self.assertNotIn("run_missing_value_ablation.py", source)
                self.assertNotIn('--config dataset_config.json', source)
                self.assertNotIn('mkdir "$OUTPUT_ROOT"', source)


class PreprocessingIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = ROOT / "work/preprocessing_experiment_tests"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        data = self.base / "data"
        shutil.copytree(ROOT / "example_data", data)
        for path in data.glob("*.npz"):
            with np.load(path, allow_pickle=False) as stored:
                arrays = {name: stored[name].copy() for name in stored.files}
            arrays["data"].reshape(-1)[::71] = 0
            np.savez(path, **arrays)
        self.config_path = self.base / "config.json"
        self.config = PreprocessingConfigTests().base_config()
        self.config["data_dir"] = str(data)
        self.config["metadata_path"] = None
        self.config["split_path"] = None
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.reference = real_preflight(self.config)

    def current(self, experiment):
        return real_preflight(driver.experiment_config(self.config, self.config_path, experiment))

    def test_actual_loader_allows_only_expected_signal_changes(self):
        for experiment in ("zero_fill_one", "minmax"):
            with self.subTest(experiment=experiment):
                current = self.current(experiment)
                checks = driver.compare_experiment_preflight(self.reference, current, experiment)
                self.assertTrue(checks)
                self.assertTrue(all(checks.values()), checks)
                self.assertEqual(current["validation_windows"], self.reference["validation_windows"])
                self.assertEqual(current["training_windows"], self.reference["training_windows"])
                old_scaler = self.reference["data_identity"]["scaler"]
                new_scaler = current["data_identity"]["scaler"]
                for name in ("continuous_columns", "continuous_mean", "continuous_std"):
                    self.assertEqual(old_scaler[name], new_scaler[name])
                if experiment == "zero_fill_one":
                    self.assertNotEqual(old_scaler["signal_mean_seconds"], new_scaler["signal_mean_seconds"])
                else:
                    self.assertEqual(new_scaler["signal_center_seconds"], 1.0)
                    self.assertEqual(new_scaler["signal_scale_seconds"], 1.0)

    def test_preflight_drift_is_rejected_before_training(self):
        mutations = {
            "split": lambda p: p["data_identity"].update(split_fingerprint="wrong"),
            "raw_train": lambda p: p["data_identity"]["train_npz_sha256"].update({"EXTRA.npz": "f"*64}),
            "opened_raw": lambda p: p["provenance"]["opened_npz_sha256"].update({"EXTRA.npz": "f"*64}),
            "condition_scaler": lambda p: p["data_identity"]["scaler"]["continuous_mean"].__setitem__(0, 999.0),
            "window_count": lambda p: p.update(training_windows=p["training_windows"] + 1),
            "window_position": lambda p: p["validation_windows"][0].update(start_sample=750),
            "window_length": lambda p: p.update(window_size=6000),
            "schedule": lambda p: p["diffusion_identity"].update(timesteps=123),
            "reference_mask": lambda p: p["validation_options"].update(reference_mask_policy="training"),
            "validation_seed": lambda p: p["validation_options"].update(seed=999),
            "early_stopping": lambda p: p["early_stopping"].update(enabled=True),
        }
        for experiment in ("zero_fill_one", "minmax"):
            original = self.current(experiment)
            for label, mutate in mutations.items():
                with self.subTest(experiment=experiment, drift=label):
                    changed = copy.deepcopy(original)
                    mutate(changed)
                    with self.assertRaises(ValueError):
                        driver.compare_experiment_preflight(self.reference, changed, experiment)

    def make_reference_run(self):
        """Artificial completion metadata and bytes; no historical model training."""
        root = self.base / "reference"
        summary = dict(status="completed", training_completed=True, epochs=2, arms={})
        for policy in ("preserve", "masked"):
            config = copy.deepcopy(self.config)
            config["missing_value_policy"] = policy
            check = real_preflight(config)
            write_json(root / f"config_{policy}.json", config)
            write_json(root / f"preflight_{policy}/preflight.json", check)
            manifest = write_completion(root / policy, check, 2)
            workspace = root / policy / "workspace"
            workspace.mkdir(parents=True)
            sources = set(driver.INVARIANT_SOURCES) | {"data_utils.py", "epoch_validation.py"}
            hashes = {}
            for name in sources:
                target = workspace / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
                hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
            write_json(workspace / "dataset_config.json", config)
            hashes["dataset_config.json"] = hashlib.sha256((workspace / "dataset_config.json").read_bytes()).hexdigest()
            checkpoint = workspace / "saved_models/DREAMT/model.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic checkpoint bytes, intentionally not loadable")
            manifest.update(source_sha256=hashes,
                            checkpoint=str(checkpoint),
                            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            write_json(root / policy / "training/run_manifest.json", manifest)
            write_json(root / policy / "training/validation/epoch_0002/metrics.json",
                       dict(status="completed", epoch=2,
                            aggregate=dict(gaussian_kl_real_to_generated=.5),
                            diagnostics=dict(generated_nonpositive_fraction=.01)))
            summary["arms"][policy] = dict(training_windows=check["training_windows"],
                                            completed_epochs=2, updates=manifest["updates"])
        write_json(root / "summary.json", summary)
        return root

    def test_frozen_sources_and_checkpoint_hashes_are_verified_without_loading_weights(self):
        root = self.make_reference_run()
        reference = ablation.load_reference_run(root)
        evidence = driver.verify_reference_sources(reference)
        self.assertEqual(set(evidence), {"preserve", "masked"})
        workspace = root / "preserve/workspace"
        manifest_path = root / "preserve/training/run_manifest.json"
        original_manifest = manifest_path.read_bytes()
        original_source = (workspace / "training_utils.py").read_bytes()
        (workspace / "training_utils.py").write_bytes(original_source + b"\n# tamper\n")
        with self.assertRaises(ValueError):
            driver.verify_reference_sources(ablation.load_reference_run(root))
        # Even a matching rewritten historical hash cannot authorize changed
        # immutable training/model/sampler code relative to current execution.
        manifest = json.loads(original_manifest)
        manifest["source_sha256"]["training_utils.py"] = hashlib.sha256(
            (workspace / "training_utils.py").read_bytes()).hexdigest()
        write_json(manifest_path, manifest)
        with self.assertRaises(ValueError):
            driver.verify_reference_sources(ablation.load_reference_run(root))
        (workspace / "training_utils.py").write_bytes(original_source)
        manifest_path.write_bytes(original_manifest)
        checkpoint = workspace / "saved_models/DREAMT/model.pth"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
        with self.assertRaises(ValueError):
            driver.verify_reference_sources(ablation.load_reference_run(root))

    def test_signal_scaler_is_independently_checked_against_raw_zero_counts(self):
        root = self.make_reference_run()
        reference = ablation.load_reference_run(root)
        for experiment in ("zero_fill_one", "minmax"):
            with self.subTest(experiment=experiment):
                current = self.current(experiment)
                result = driver.verify_signal_scaler(reference, current, experiment)
                self.assertTrue(result["verified"])
                self.assertGreater(result["raw_zero_fraction"], 0.0)
                changed = copy.deepcopy(current)
                changed["data_identity"]["scaler"]["signal_mean_seconds"] += .1
                with self.assertRaisesRegex(ValueError, "scaler differs"):
                    driver.verify_signal_scaler(reference, changed, experiment)

    def invoke_main(self, root, output, experiment, *, check_only=False, mutate_preflight=None,
                    mutate_completion=None, failure=None, extra=()):
        seen = []

        def fake_subprocess(command, **kwargs):
            # Only the subprocess boundary is mocked. Config, source identity,
            # real synthetic NPZ loading and all driver checks execute normally.
            seen.append(command)
            self.assertTrue(kwargs["check"])
            self.assertEqual(Path(kwargs["cwd"]), ROOT)
            self.assertFalse(any(key.startswith("DREAMT_") for key in kwargs["env"]))
            if failure == ("check" if "--check-only" in command else "training"):
                raise subprocess.CalledProcessError(7, command)
            config = json.loads(Path(command[command.index("--config") + 1]).read_text(encoding="utf-8"))
            check = real_preflight(config)
            if mutate_preflight:
                mutate_preflight(check)
            target = Path(command[command.index("--output-dir") + 1])
            if "--check-only" in command:
                write_json(target / "preflight.json", check)
            else:
                epochs = int(command[command.index("--epochs") + 1])
                manifest = write_completion(target, check, epochs)
                checkpoint = target / "workspace/saved_models/DREAMT/model.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"synthetic new checkpoint; no model training occurred")
                manifest.update(checkpoint=str(checkpoint),
                                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                metrics = dict(status="completed", epoch=epochs,
                               aggregate=dict(gaussian_kl_real_to_generated=1.0 / epochs),
                               diagnostics=dict(generated_nonpositive_fraction=.02))
                if mutate_completion:
                    mutate_completion(manifest, metrics)
                write_json(target / "training/run_manifest.json", manifest)
                write_json(target / f"training/validation/epoch_{epochs:04d}/metrics.json", metrics)
            return subprocess.CompletedProcess(command, 0)

        arguments = ["--experiment", experiment, "--reference-run", str(root), "--output-dir", str(output),
                     "--epochs", "2", "--batch-size", "8", "--threads", "4", *extra]
        if check_only:
            arguments.append("--check-only")
        with mock.patch.object(driver.subprocess, "run", side_effect=fake_subprocess), \
             mock.patch.dict("os.environ", {"DREAMT_DATA_DIR": "must-not-override-frozen-config"}), \
             contextlib.redirect_stdout(io.StringIO()):
            result = driver.main(arguments)
        return result, seen

    def test_main_trains_only_selected_fresh_experiment_and_compares_final_epochs(self):
        root = self.make_reference_run()
        old_files = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
        for experiment in ("zero_fill_one", "minmax"):
            with self.subTest(experiment=experiment):
                output = self.base / experiment
                result, calls = self.invoke_main(root, output, experiment)
                self.assertEqual(result["status"], "completed")
                self.assertTrue(result["training_completed"])
                self.assertTrue(result["completed_budget"]["full_budget_verified"])
                self.assertEqual(len(calls), 2)
                self.assertIn("--check-only", calls[0])
                self.assertNotIn("--check-only", calls[1])
                self.assertIn("--skip-generation", calls[1])
                self.assertNotIn("--resume", calls[1])
                self.assertNotIn("--checkpoint", calls[1])
                self.assertFalse((output / "preserve").exists())
                self.assertFalse((output / "masked").exists())
                with (output / "comparison.csv").open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([row["experiment"] for row in rows], ["preserve", "masked", experiment])
                self.assertEqual({row["epoch"] for row in rows}, {"2"})
                self.assertEqual(len({row["updates"] for row in rows}), 1)
                self.assertEqual(result["final_checkpoint_sha256"],
                                 hashlib.sha256(Path(result["final_checkpoint"]).read_bytes()).hexdigest())
        self.assertEqual(old_files, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in old_files})

    def test_main_check_only_never_creates_training_run(self):
        root = self.make_reference_run()
        result, calls = self.invoke_main(root, self.base / "checked", "zero_fill_one", check_only=True)
        self.assertEqual(result["status"], "checked")
        self.assertFalse(result["training_completed"])
        self.assertEqual(len(calls), 1)
        self.assertIn("--check-only", calls[0])
        self.assertFalse((self.base / "checked/run").exists())
        self.assertFalse((self.base / "checked/training.log").exists())

    def test_main_preflight_drift_blocks_training_and_records_failed_status(self):
        root = self.make_reference_run()
        output = self.base / "bad_preflight"
        with self.assertRaisesRegex(ValueError, "preflight mismatch"):
            self.invoke_main(root, output, "minmax",
                             mutate_preflight=lambda p: p.update(training_windows=p["training_windows"] + 1))
        report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["training_completed"])
        self.assertFalse((output / "run").exists())
        self.assertFalse((output / "training.log").exists())

    def test_main_failure_or_short_training_is_not_reported_as_complete(self):
        root = self.make_reference_run()
        cases = {
            "subprocess": dict(failure="training"),
            "short_updates": dict(mutate_completion=lambda manifest, metrics: manifest.update(updates=0)),
            "checkpoint_hash": dict(mutate_completion=lambda manifest, metrics: manifest.update(checkpoint_sha256="0"*64)),
            "metric_mismatch": dict(mutate_completion=lambda manifest, metrics: metrics["aggregate"].update(gaussian_kl_real_to_generated=99)),
            "missing_nonpositive": dict(mutate_completion=lambda manifest, metrics: metrics.update(diagnostics={})),
        }
        for label, options in cases.items():
            with self.subTest(failure=label):
                output = self.base / label
                with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                    self.invoke_main(root, output, "zero_fill_one", **options)
                report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["training_completed"])
                self.assertFalse((output / "comparison.csv").exists())

    def test_epoch_batch_and_threads_must_match_reference_before_subprocess(self):
        root = self.make_reference_run()
        for flag, value in (("--epochs", "3"), ("--batch-size", "4"), ("--threads", "2")):
            with self.subTest(flag=flag):
                output = self.base / flag.lstrip("-")
                with mock.patch.object(driver.subprocess, "run") as subprocess_mock, \
                     self.assertRaises(ValueError):
                    driver.main(["--reference-run", str(root), "--experiment", "minmax", "--epochs", "2",
                                 "--output-dir", str(output), "--check-only", flag, value])
                subprocess_mock.assert_not_called()
                self.assertEqual(json.loads((output / "summary.json").read_text(encoding="utf-8"))["status"], "failed")

    def test_output_cannot_overwrite_or_live_inside_reference(self):
        root = self.make_reference_run()
        for output in (root, root / "nested"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "outside"):
                self.invoke_main(root, output, "minmax", check_only=True)
        output = self.base / "existing"
        output.mkdir()
        sentinel = output / "sentinel"
        sentinel.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            self.invoke_main(root, output, "minmax", check_only=True)
        self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_missing_source_identity_cannot_be_silently_accepted(self):
        root = self.make_reference_run()
        path = root / "preserve/training/run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        del manifest["source_sha256"]["training_utils.py"]
        write_json(path, manifest)
        with self.assertRaises(ValueError):
            driver.verify_reference_sources(ablation.load_reference_run(root))


if __name__ == "__main__":
    unittest.main()
