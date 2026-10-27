"""Diagnostic CLI integration with artificial DREAMT records and mocked GPU sampling.

These checks do not train a model, execute CUDA kernels, or validate real S092
generation quality. Identity checks use real data/source digests; checkpoint
contents are explicitly non-model fixture bytes and are never deserialized.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils import Preprocessor
from diffusion_schedule import diffusion_identity
from night_checkpoint import load_inference_config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


class DiagnoseGenerationCommandTests(unittest.TestCase):
    def setUp(self):
        from scripts import diagnose_generation
        self.command = diagnose_generation
        self.temporary = tempfile.TemporaryDirectory(prefix="diagnose_cli_", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.records = self.directory / "records"
        self.records.mkdir()
        rows = []
        names = ["S001.npz", "S002.npz", "S092.npz"]
        self.target_epochs = 12
        for index, filename in enumerate(names):
            epochs = 8 if index < 2 else self.target_epochs
            positions = np.arange(epochs*750)
            stages = np.resize(np.array([0, 1, 2, 4, 3], dtype=np.int8), epochs)
            signal = (.75+.03*index+.025*np.sin(positions/37.)).astype(np.float32)
            np.savez_compressed(self.records / filename, data=signal[:, None],
                stages=np.repeat(stages, 750), fs=np.array(25), ahi=np.array(10.+index),
                synthetic=np.array(True))
            rows.append(dict(SID=filename[:-4], AHI=10.+index, BMI=24.+index,
                AGE=40+index, GENDER="M" if index == 1 else "F"))
        with (self.records / "participant_info.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(self.records / "splits.json", dict(train=names[:1], val=names[1:2], test=names[2:]))
        self.config_path = self.directory / "dataset_config.json"
        self.config = dict(data_dir="records", metadata_path="records/participant_info.csv",
            split_path="records/splits.json", multiplier=1., downsample_rate=1,
            max_epochs=4, split_seed=42, window_epochs=4,
            diffusion_schedule="linear_power", terminal_alpha_bar=1e-8)
        write_json(self.config_path, self.config)
        self.args, data_config, _ = load_inference_config(self.config_path)
        self.processor = Preprocessor("DREAMT", data_config=data_config)
        self.training = self.directory / "training"
        self.training.mkdir()
        self.checkpoint = self.training / "best_kl_model.pth"
        self.checkpoint.write_bytes(b"Artificial hash fixture: not actual model weights.")
        self.manifest_path = self.training / "run_manifest.json"
        source_names = ("training_utils.py", "TSImputers/SSSDS4Imputer.py", "TSImputers/S4Model.py",
            "utils/util.py", "data_utils.py", "watchsleep_loader.py", "window_config.py", "diffusion_schedule.py")
        self.manifest = dict(training_args=vars(self.args), data_identity=self.processor.data_identity(),
            diffusion_identity=diffusion_identity(self.args),
            source_sha256={name: sha256(ROOT / name) for name in source_names})
        write_json(self.manifest_path, self.manifest)
        self.validation_path = self.training / "validation" / "summary.json"
        self.validation_path.parent.mkdir()
        self.validation = dict(best_checkpoints={"kl": dict(checkpoint=str(self.checkpoint),
            checkpoint_sha256=sha256(self.checkpoint), epoch=5, selection_rule="smoothed_kl")})
        write_json(self.validation_path, self.validation)

    def arguments(self, output="output", *, check_only=False):
        result = ["--checkpoint", str(self.checkpoint), "--config", str(self.config_path),
                  "--output-dir", str(self.directory / output)]
        return result + (["--check-only"] if check_only else [])

    def invoke(self, argv):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.command.main(argv)

    def saved_report(self, output="output"):
        return json.loads((self.directory / output / "summary.json").read_text(encoding="utf-8"))

    def assert_blocked_without_sampling(self, argv):
        with mock.patch.object(self.command, "_sample") as sample, \
             mock.patch("cuda_runtime.require_cuda", side_effect=AssertionError("Blocked config must not access CUDA")), \
             mock.patch("torch.load", side_effect=AssertionError("Blocked config must not load weights")):
            with self.assertRaises(ValueError):
                self.invoke(argv)
        sample.assert_not_called()
        report = self.saved_report()
        self.assertEqual(report["status"], "blocked_config")
        self.assertFalse(report["config_audit"]["can_sample"])
        self.assertTrue((self.directory / "output" / "config_audit.json").is_file())
        self.assertFalse((self.directory / "output" / "diagnostics.npz").exists())

    def test_matched_check_only_uses_neighbor_metadata_without_cuda_or_loading_weights(self):
        with mock.patch.object(self.command, "_sample", side_effect=AssertionError("check-only must not sample")), \
             mock.patch("cuda_runtime.require_cuda", side_effect=AssertionError("check-only must not access CUDA")), \
             mock.patch("torch.load", side_effect=AssertionError("check-only must not deserialize weights")):
            report = self.invoke(self.arguments(check_only=True))
        self.assertEqual(report["status"], "checked_only")
        self.assertTrue(report["config_audit"]["can_sample"])
        self.assertEqual(self.saved_report()["status"], "checked_only")
        audit = json.loads((self.directory / "output" / "config_audit.json").read_text(encoding="utf-8"))
        self.assertTrue(audit["can_sample"])
        self.assertFalse((self.directory / "output" / "diagnostics.npz").exists())

    def test_explicit_metadata_paths_and_config_relative_data_paths(self):
        manifest = self.directory / "relocated_manifest.json"
        validation = self.directory / "relocated_validation.json"
        write_json(manifest, self.manifest)
        write_json(validation, self.validation)
        self.manifest_path.unlink()
        self.validation_path.unlink()
        report = self.invoke(self.arguments(check_only=True) + [
            "--training-manifest", str(manifest), "--validation-summary", str(validation),
            "--subject", "s092"])
        self.assertEqual(report["status"], "checked_only")
        self.assertTrue(report["config_audit"]["can_sample"])

    def test_missing_training_manifest_blocks_and_persists_audit(self):
        self.manifest_path.unlink()
        self.assert_blocked_without_sampling(self.arguments())

    def test_missing_best_checkpoint_binding_is_not_filled_from_current_file_hash(self):
        # A current-file digest cannot establish that training selected this best.
        self.validation_path.unlink()
        self.assert_blocked_without_sampling(self.arguments(check_only=True))

    def test_altered_checkpoint_bytes_block_before_sampling(self):
        self.checkpoint.write_bytes(b"Different artificial checkpoint fixture bytes")
        self.assert_blocked_without_sampling(self.arguments())

    def test_changed_scaler_in_historical_identity_blocks_before_sampling(self):
        bad = copy.deepcopy(self.manifest)
        bad["data_identity"]["scaler"]["signal_mean_seconds"] += .1
        write_json(self.manifest_path, bad)
        self.assert_blocked_without_sampling(self.arguments(check_only=True))

    def test_changed_s4_source_identity_blocks_before_sampling(self):
        bad = copy.deepcopy(self.manifest)
        bad["source_sha256"]["TSImputers/S4Model.py"] = "0"*64
        write_json(self.manifest_path, bad)
        self.assert_blocked_without_sampling(self.arguments())

    def previous_target_arguments(self, previous):
        # Isolate target identity from the already-covered model/train audit.
        path = self.directory / "previous_summary.json"
        write_json(path, previous)
        return self.arguments(check_only=True)+["--inference-summary", str(path)]

    def matched_prior_configuration_audit(self):
        return dict(status="matched", can_sample=True, blocking_checks=[],
                    expected_protocol_differences=[], historical_generation_comparison_requested=True)

    def test_previous_target_subject_and_file_digest_match_actual_selected_night(self):
        # These keys match the real generate_night.py summary schema.
        previous = dict(subject_id="S092", source_filename="S092.npz",
            data={"source_file_sha256": sha256(self.records / "S092.npz")})
        with mock.patch("diagnostic_identity.audit_training_identity",
                        return_value=self.matched_prior_configuration_audit()), \
             mock.patch.object(self.command, "_sample", side_effect=AssertionError("check-only")):
            report = self.invoke(self.previous_target_arguments(previous))
        self.assertEqual(report["status"], "checked_only")
        self.assertTrue(report["config_audit"]["can_sample"])
        rows = {row["name"]: row for row in report["config_audit"]["blocking_checks"]}
        self.assertEqual(rows["previous.target.subject_id"]["status"], "matched")
        self.assertEqual(rows["previous.target.source_file_sha256"]["status"], "matched")

    def test_previous_target_from_another_participant_blocks_even_if_training_identity_matches(self):
        previous = dict(subject_id="S002", source_filename="S002.npz",
            data={"source_file_sha256": sha256(self.records / "S002.npz")})
        with mock.patch("diagnostic_identity.audit_training_identity",
                        return_value=self.matched_prior_configuration_audit()):
            self.assert_blocked_without_sampling(self.previous_target_arguments(previous))
        checks = {row["name"]: row for row in self.saved_report()["config_audit"]["blocking_checks"]}
        self.assertEqual(checks["previous.target.subject_id"]["status"], "mismatch")

    def test_previous_target_changed_test_file_blocks_without_train_scaler_changes(self):
        previous = dict(subject_id="S092", source_filename="S092.npz",
            data={"source_file_sha256": "0"*64})
        with mock.patch("diagnostic_identity.audit_training_identity",
                        return_value=self.matched_prior_configuration_audit()):
            self.assert_blocked_without_sampling(self.previous_target_arguments(previous))
        checks = {row["name"]: row for row in self.saved_report()["config_audit"]["blocking_checks"]}
        self.assertEqual(checks["previous.target.subject_id"]["status"], "matched")
        self.assertEqual(checks["previous.target.source_file_sha256"]["status"], "mismatch")

    def test_previous_target_missing_identity_fields_cannot_be_treated_as_verified(self):
        previous = {"status": "completed"}
        with mock.patch("diagnostic_identity.audit_training_identity",
                        return_value=self.matched_prior_configuration_audit()):
            self.assert_blocked_without_sampling(self.previous_target_arguments(previous))
        checks = {row["name"]: row for row in self.saved_report()["config_audit"]["blocking_checks"]}
        self.assertEqual(checks["previous.target.subject_id"]["status"], "incomplete")
        self.assertEqual(checks["previous.target.source_file_sha256"]["status"], "incomplete")

    def test_matching_identity_still_requires_cuda_for_real_sampling(self):
        with mock.patch.object(self.command, "_sample") as sample, \
             mock.patch("cuda_runtime.require_cuda", side_effect=RuntimeError("CUDA required; test has no GPU")), \
             mock.patch("torch.load", side_effect=AssertionError("No CPU fallback")):
            with self.assertRaisesRegex(RuntimeError, "CUDA required"):
                self.invoke(self.arguments())
        sample.assert_not_called()
        report = self.saved_report()
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["config_audit"]["can_sample"])

    def test_mocked_gpu_sampling_receives_aligned_data_and_writes_exact_diagnostics(self):
        # Explicitly artificial numerical arrays; no model or CUDA executes.
        expected_reference = np.linspace(.6, .9, self.target_epochs*750, dtype=np.float64)
        expected_generated = np.stack([expected_reference+.01, expected_reference-.02])
        expected_generated[0, 7] = -.25  # Diagnostic arrays must not be clipped.
        expected_report = {"artificial_mock": True,
            "stages": {"comparison": {"rows": [{"seed": 123, "mean_bias_ms": 10.}]}}}

        def fake_sample(args, checkpoint, prepared, device, **options):
            self.assertEqual(checkpoint, self.checkpoint.resolve())
            self.assertEqual(device, torch.device("cuda:0"))
            self.assertEqual(args.window_size, 3000)
            self.assertEqual(prepared["subject_id"], "S092")
            self.assertEqual(prepared["conditions"].shape, (9000, 9))
            self.assertEqual(prepared["reference_seconds"].shape, (9000,))
            self.assertTrue(prepared["record"]["synthetic"])
            self.assertEqual(list(options["starts"]), [0, 3000])
            self.assertEqual(options["repeats"], 2)
            self.assertEqual(options["comparison_repeats"], 1)
            self.assertEqual(options["seed"], 123)
            self.assertEqual(options["threads"], 2)
            self.assertEqual(options["stride"], 750)
            self.assertEqual(options["windows_per_group"], 4)
            self.assertEqual(options["kl_variance_floor_ms2"], 1.)
            options["progress"]({"fixture": True, "completed": 1})
            return expected_report, {"reference_seconds": expected_reference,
                "independent_night_seconds": expected_generated}, {
                "artificial_mock": True, "loaded_checkpoint_sha256": sha256(checkpoint)}

        argv = self.arguments() + ["--window-start-epochs", "0", "4", "--repeats", "2",
            "--comparison-repeats", "1", "--seed", "123", "--threads", "2"]
        with mock.patch("cuda_runtime.require_cuda", return_value=torch.device("cuda:0")), \
             mock.patch.object(self.command, "_sample", side_effect=fake_sample) as sample, \
             mock.patch("torch.load", side_effect=AssertionError("Mocked execution must not load a model")):
            report = self.invoke(argv)
        self.assertEqual(sample.call_count, 1)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["config_audit"]["can_sample"])
        self.assertEqual(report["diagnostics"], expected_report)
        with np.load(self.directory / "output" / "diagnostics.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["reference_seconds"], expected_reference)
            np.testing.assert_array_equal(arrays["independent_night_seconds"], expected_generated)
        self.assertEqual(self.saved_report()["status"], "completed")

    def test_sampling_failure_is_recorded_without_claiming_completed_arrays(self):
        with mock.patch("cuda_runtime.require_cuda", return_value=torch.device("cuda:0")), \
             mock.patch.object(self.command, "_sample", side_effect=FloatingPointError("Artificial sampler failure")):
            with self.assertRaisesRegex(FloatingPointError, "Artificial sampler failure"):
                self.invoke(self.arguments())
        report = self.saved_report()
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["config_audit"]["can_sample"])
        self.assertFalse((self.directory / "output" / "diagnostics.npz").exists())

    def test_checkpoint_change_between_audit_and_loading_cannot_be_reported_as_verified(self):
        def replaced_checkpoint(*args, **kwargs):
            return {"artificial_mock": True}, {"fixture": np.zeros(3)}, {
                "loaded_checkpoint_sha256": "0"*64}
        with mock.patch("cuda_runtime.require_cuda", return_value=torch.device("cuda:0")), \
             mock.patch.object(self.command, "_sample", side_effect=replaced_checkpoint):
            with self.assertRaises((ValueError, RuntimeError)):
                self.invoke(self.arguments())
        self.assertNotEqual(self.saved_report()["status"], "completed")

    def test_existing_output_directory_and_contents_are_not_overwritten(self):
        output = self.directory / "output"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.invoke(self.arguments(check_only=True))
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_nonpositive_counts_and_invalid_seed_are_rejected_by_cli(self):
        for extra in (["--threads", "0"], ["--repeats", "0"], ["--comparison-repeats", "0"],
                      ["--stride-epochs", "0"], ["--windows-per-group", "0"],
                      ["--seed", "-1"], ["--seed", str(2**32)]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.invoke(self.arguments(check_only=True)+extra)
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
