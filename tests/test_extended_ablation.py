"""Driver identity/budget tests; mocked subprocesses never execute GPU training."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import run_missing_value_ablation as driver


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def preflight(policy, data_path):
    return dict(status="passed", training_windows=9, training_input_shape=[3000, 10],
                window_epochs=4, window_size=3000,
                data_identity=dict(data_directory=str(data_path), metadata_file_sha256="metadata-sha",
                    splits={"train": ["T.npz"], "val": ["V.npz"]}, split_fingerprint="split-sha",
                    train_npz_sha256={"T.npz": "train-sha"}, window_epochs=4,
                    multiplier=1.0, downsample_rate=1, max_epochs=None,
                    missing_value_policy=policy, zero_value_policy=policy,
                    scaler=dict(signal_mean_seconds=.8 if policy == "preserve" else .9,
                                signal_std_seconds=.2)),
                validation_windows=[dict(filename="V.npz", start_sample=0)],
                validation_selection=dict(selected_windows=1),
                validation_options=dict(max_windows=64, batch_size=8, seed=42,
                    reference_mask_policy="observed_only", kl_variance_floor_ms2=1.0),
                early_stopping=dict(enabled=False),
                diffusion_settings=dict(diffusion_schedule="linear_power", terminal_alpha_bar=1e-8),
                diffusion_identity={"tensor_hash": "schedule-sha"},
                provenance=dict(opened_npz_sha256={"T.npz": "train-sha", "V.npz": "val-sha"}))


def completed_manifest(check, epochs, batch_size=8):
    updates = ((check["training_windows"] + batch_size - 1) // batch_size) * epochs
    return dict(status="completed", training_args=driver.expected_training_args(check, epochs, batch_size),
                threads=4, data_identity=copy.deepcopy(check["data_identity"]),
                diffusion_identity=copy.deepcopy(check["diffusion_identity"]),
                train_windows=check["training_windows"], batches_per_epoch=updates // epochs,
                updates=updates, expected_updates=updates, completed_epochs=epochs,
                requested_epochs=epochs, checkpoint_epoch=epochs, stopped_early=False)


def write_completion(output, check, epochs, *, updates_delta=0):
    manifest = completed_manifest(check, epochs)
    manifest["updates"] += updates_delta
    write_json(output / "preflight.json", check)
    write_json(output / "training/run_manifest.json", manifest)
    (output / "training/epoch_metrics.csv").write_text(
        "epoch,gaussian_kl_real_to_generated\n" + "".join(f"{epoch},{1/epoch}\n" for epoch in range(1, epochs+1)),
        encoding="utf-8")
    write_json(output / f"training/validation/epoch_{epochs:04d}/metrics.json",
               dict(diagnostics=dict(generated_nonpositive_fraction=0.0)))
    return manifest


def reference_fixture(root):
    summary = dict(status="completed", training_completed=True, epochs=2, arms={})
    for policy in ("preserve", "masked"):
        check = preflight(policy, root / "data")
        config = dict(data_dir=str(root / "data"), window_epochs=4,
                      validation=check["validation_options"], missing_value_policy=policy,
                      early_stopping=dict(enabled=False), **check["diffusion_settings"],
                      reference_arm_marker=policy)
        write_json(root / f"config_{policy}.json", config)
        write_json(root / f"preflight_{policy}/preflight.json", check)
        manifest = write_completion(root / policy, check, 2)
        summary["arms"][policy] = dict(training_windows=9, completed_epochs=2, updates=manifest["updates"])
    write_json(root / "summary.json", summary)
    return root


class ExtendedAblationTests(unittest.TestCase):
    def test_reference_load_requires_both_arms_completed_and_coherent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = reference_fixture(Path(temp) / "old")
            loaded = driver.load_reference_run(root)
            self.assertEqual(loaded["arms"]["masked"]["budget"]["updates"], 4)
            manifest_path = root / "masked/training/run_manifest.json"
            changed = json.loads(manifest_path.read_text())
            changed["status"] = "training"
            write_json(manifest_path, changed)
            with self.assertRaisesRegex(ValueError, "not complete"):
                driver.load_reference_run(root)

    def test_reference_rejects_checked_only_and_partial_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = reference_fixture(Path(temp) / "old")
            summary_path = root / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["training_completed"] = False
            write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "completed both"):
                driver.load_reference_run(root)
            summary["training_completed"] = True
            write_json(summary_path, summary)
            manifest_path = root / "preserve/training/run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["updates"] -= 1
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "budget/completion"):
                driver.load_reference_run(root)

    def test_reference_preflight_only_ignores_artifact_option(self):
        old = preflight("masked", Path("data"))
        new = copy.deepcopy(old)
        new["validation_options"]["save_artifacts"] = True
        self.assertTrue(all(driver.compare_reference_preflights(old, new).values()))
        mutations = [
            ("scaler", lambda p: p["data_identity"]["scaler"].update(signal_mean_seconds=.95)),
            ("train hash", lambda p: p["data_identity"]["train_npz_sha256"].update({"T.npz": "changed"})),
            ("val hash", lambda p: p["provenance"]["opened_npz_sha256"].update({"V.npz": "changed"})),
            ("split", lambda p: p["data_identity"].update(split_fingerprint="changed")),
            ("window", lambda p: p.update(window_size=6000)),
            ("train windows", lambda p: p.update(training_windows=8)),
            ("schedule", lambda p: p["diffusion_identity"].update(tensor_hash="changed")),
            ("seed", lambda p: p["validation_options"].update(seed=43)),
            ("positions", lambda p: p["validation_windows"][0].update(start_sample=750)),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                changed = copy.deepcopy(new)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "preflight mismatch"):
                    driver.compare_reference_preflights(old, changed)

    def test_training_config_can_extend_epochs_but_not_batch_or_model(self):
        check = preflight("preserve", Path("data"))
        old = driver.expected_training_args(check, 10, 8)
        new = driver.expected_training_args(check, 30, 8)
        driver._check_training_args(old, new, ignore_epochs=True)
        for key, value in (("batch_size", 16), ("lr", .001), ("s4_dstate", 32), ("stride", 1500)):
            with self.subTest(key=key):
                changed = dict(new, **{key: value})
                with self.assertRaisesRegex(ValueError, key):
                    driver._check_training_args(old, changed, ignore_epochs=True)

    def test_equal_budget_requires_equal_windows_even_when_ceil_batches_match(self):
        arms = dict(preserve=dict(training_windows=9, expected_updates=60),
                    masked=dict(training_windows=10, expected_updates=60))
        with self.assertRaisesRegex(ValueError, "training_windows"):
            driver.require_equal_budgets(arms)
        arms["masked"]["training_windows"] = 9
        driver.require_equal_budgets(arms)
        arms["masked"]["expected_updates"] = 59
        with self.assertRaisesRegex(ValueError, "expected_updates"):
            driver.require_equal_budgets(arms)

    def test_completion_rejects_update_epoch_and_batch_drift(self):
        check = preflight("masked", Path("data"))
        manifest = completed_manifest(check, 30)
        rows = [dict(epoch=str(epoch)) for epoch in range(1, 31)]
        self.assertEqual(driver.validate_training_budget(manifest, rows, check, 30, 8)["updates"], 60)
        for field in ("updates", "completed_epochs", "train_windows", "batches_per_epoch", "checkpoint_epoch"):
            with self.subTest(field=field):
                changed = dict(manifest, **{field: manifest[field]-1})
                with self.assertRaisesRegex(ValueError, "budget/completion"):
                    driver.validate_training_budget(changed, rows, check, 30, 8)
        with self.assertRaisesRegex(ValueError, "sequence"):
            driver.validate_training_budget(manifest, rows[:-1], check, 30, 8)
        changed = copy.deepcopy(manifest)
        changed["training_args"]["batch_size"] = 4
        with self.assertRaisesRegex(ValueError, "batch_size"):
            driver.validate_training_budget(changed, rows, check, 30, 8)

    def run_mocked_main(self, root, output, *, check_only=False, override=None, unequal=False, short=False):
        seen = []
        def fake_subprocess(command, **kwargs):
            # Exercise the real driver, with artificial preflight/training output.
            # There is deliberately no CUDA or model execution in this test.
            seen.append(command)
            config = json.loads(Path(command[command.index("--config")+1]).read_text())
            policy = config["missing_value_policy"]
            target = Path(command[command.index("--output-dir")+1])
            check = preflight(policy, root / "data")
            check["validation_options"] = config["validation"]
            if unequal and policy == "masked":
                check["training_windows"] = 10
            if "--check-only" in command:
                write_json(target / "preflight.json", check)
            else:
                epochs = int(command[command.index("--epochs")+1])
                write_completion(target, check, epochs, updates_delta=-1 if short else 0)
            return subprocess.CompletedProcess(command, 0)
        arguments = ["--output-dir", str(output), "--epochs", "30", "--batch-size", "8",
                     "--require-equal-updates", "--save-validation-artifacts"]
        if not unequal:
            arguments += ["--reference-run", str(root)]
        if override:
            arguments += ["--config", str(override)]
        if check_only:
            arguments.append("--check-only")
        with mock.patch.object(driver.subprocess, "run", side_effect=fake_subprocess), \
             contextlib.redirect_stdout(io.StringIO()):
            driver.main(arguments)
        return seen

    def test_main_reuses_each_reference_config_and_enables_artifacts_without_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = reference_fixture(base / "old")
            output = base / "new"
            seen = self.run_mocked_main(root, output)
            result = json.loads((output / "summary.json").read_text())
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["equal_expected_updates"])
            self.assertEqual(result["arms"]["masked"]["updates"], 60)
            self.assertIn("fresh model", result["initialization"])
            self.assertEqual(len(seen), 4)
            for policy in ("preserve", "masked"):
                frozen = json.loads((output / f"config_{policy}.json").read_text())
                self.assertEqual(frozen["reference_arm_marker"], policy)
                self.assertTrue(frozen["validation"]["save_artifacts"])
                self.assertNotIn("--resume", result["arms"][policy]["command"])
                self.assertTrue(result["arms"][policy]["completed_budget"]["full_budget_verified"])

    def test_main_check_only_performs_no_training_and_explicit_override_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = reference_fixture(base / "old")
            override = base / "override.json"
            config = json.loads((root / "config_preserve.json").read_text())
            config["reference_arm_marker"] = "explicit"
            write_json(override, config)
            seen = self.run_mocked_main(root, base / "new", check_only=True, override=override)
            self.assertTrue(all("--check-only" in command for command in seen))
            result = json.loads((base / "new/summary.json").read_text())
            self.assertEqual(result["config_source"], "explicit shared --config override")
            self.assertEqual(result["status"], "checked")
            self.assertFalse(result["training_completed"])

    def test_main_stops_before_training_if_windows_differ_even_with_equal_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = reference_fixture(base / "old")
            with self.assertRaisesRegex(ValueError, "training_windows"):
                self.run_mocked_main(root, base / "new", unequal=True, override=root / "config_preserve.json")
            result = json.loads((base / "new/summary.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("updates", result["arms"]["preserve"])

    def test_main_fails_instead_of_claiming_success_when_training_is_short(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = reference_fixture(base / "old")
            with self.assertRaisesRegex(ValueError, "budget/completion"):
                self.run_mocked_main(root, base / "new", short=True)
            result = json.loads((base / "new/summary.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["training_completed"])

    def test_cannot_write_new_output_inside_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = reference_fixture(Path(temp) / "old")
            with self.assertRaisesRegex(ValueError, "outside"):
                driver.main(["--reference-run", str(root), "--output-dir", str(root / "new"), "--check-only"])

    def test_real_loader_reference_preflight_preserves_identity_with_artifact_export(self):
        # Genuine loader preflights on repository synthetic fixtures. Historical
        # completion metadata is explicitly constructed; no model was trained.
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            shutil.copytree(driver.ROOT / "example_data", base / "data")
            config_path = base / "config.json"
            write_json(config_path, dict(data_dir="data", window_epochs=1))
            old = base / "old"
            command = [sys.executable, "-B", str(driver.ROOT / "scripts/run_missing_value_ablation.py")]
            result = subprocess.run(command + ["--config", str(config_path), "--output-dir", str(old),
                "--epochs", "2", "--check-only"], capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((old / "summary.json").read_text())
            for policy in ("preserve", "masked"):
                check = json.loads((old / f"preflight_{policy}/preflight.json").read_text())
                manifest = write_completion(old / policy, check, 2)
                summary["arms"][policy].update(completed_epochs=2, updates=manifest["updates"])
            summary.update(status="completed", training_completed=True)
            write_json(old / "summary.json", summary)
            # A changed original/global config must not replace either frozen arm config.
            write_json(config_path, dict(data_dir="does-not-exist", window_epochs=8))
            output = base / "new"
            result = subprocess.run(command + ["--reference-run", str(old), "--output-dir", str(output),
                "--epochs", "30", "--require-equal-updates", "--save-validation-artifacts", "--check-only"],
                capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            actual = json.loads((output / "summary.json").read_text())
            self.assertEqual(actual["status"], "checked")
            self.assertFalse(actual["training_completed"])
            for policy in ("preserve", "masked"):
                checks = actual["arms"][policy]["reference_checks"]
                self.assertTrue(all(checks.values()), checks)
                self.assertEqual(actual["arms"][policy]["expected_updates"],
                                 summary["arms"][policy]["expected_updates"] * 15)
                self.assertFalse((output / policy).exists())

    def test_submission_exports_before_fresh_training_and_propagates_failure(self):
        source = (driver.ROOT / "submit_missing_value_ablation_extended.sh").read_text()
        self.assertIn("set -euo pipefail", source)
        self.assertLess(source.index("python scripts/export_ablation_validation.py"),
                        source.index("python scripts/run_missing_value_ablation.py"))
        self.assertIn('--output-dir "$OUTPUT_ROOT/epoch10_diagnostics"', source)
        self.assertIn('--output-dir "$OUTPUT_ROOT/training_30epochs"', source)
        self.assertIn("EXTENDED_EPOCHS=30", source)
        self.assertIn('--epochs "$EXTENDED_EPOCHS" --batch-size 8 --threads 4 --validation-max-windows 64', source)
        self.assertIn("--require-equal-updates --save-validation-artifacts", source)
        self.assertIn("#SBATCH --gpus-per-node=1", source)
        self.assertNotIn("--config dataset_config.json", source)
        self.assertEqual(source.count("python scripts/plot_validation_comparison.py"), 2)
        self.assertLess(source.index("python scripts/plot_validation_comparison.py"),
                        source.index("python scripts/run_missing_value_ablation.py"))
        self.assertGreater(source.rindex("python scripts/plot_validation_comparison.py"),
                           source.index("python scripts/run_missing_value_ablation.py"))


if __name__ == "__main__":
    unittest.main()
