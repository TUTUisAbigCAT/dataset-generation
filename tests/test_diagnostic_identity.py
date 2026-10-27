"""Identity evidence boundaries; no weights, CUDA, data files or model calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diagnostic_identity import audit_training_identity, REQUIRED_CORE_SOURCES, DATA_SOURCES
from diffusion_schedule import diffusion_identity


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture():
    args = dict(dataset="DREAMT", backbone="S4", beta_0=.0001, beta_T=.02, timesteps=200,
        diffusion_schedule="linear_power", terminal_alpha_bar=1e-8,
        window_size=3000, stride=750, num_res_layers=4, res_channels=64, skip_channels=64,
        diff_step_embed_in=32, diff_step_embed_mid=64, diff_step_embed_out=64,
        s4_lmax=3000, s4_dstate=64, s4_dropout=0., s4_bidirectional=True, s4_layernorm=True,
        propCycEnc=False, lr=.0001, epochs=50, batch_size=8, hdim=64, layers=4)
    scaler = dict(signal_mean_seconds=.8848376, signal_std_seconds=.3061459,
        continuous_columns=["AHI", "BMI", "AGE"], continuous_mean=[10., 25., 50.],
        continuous_std=[8., 3., 12.], continuous_raw_std=[8., 3., 12.],
        constant_columns=[], signal_fit_samples=3000, metadata_fit_subjects=1,
        fit_split="train", signal_fit_policy="stored zero markers included")
    data = dict(schema="dreamt_data_identity_v1", schema_version=1, dataset="DREAMT",
        data_directory="/training/records", metadata_path="/training/participant_info.csv",
        metadata_file_sha256=sha("metadata"), split_path=None,
        split_source="generated_subject_split", split_file_sha256=None,
        split_seed=42, split_fingerprint=sha("split"),
        splits=dict(train=["A.npz"], val=["B.npz"], test=["S092.npz"]),
        sample_rate_hz=25, epoch_samples=750, window_epochs=4,
        multiplier=1., downsample_rate=1, max_epochs=None,
        columns=["IBI", "stage_W", "stage_Light", "stage_REM", "stage_Deep", "AHI", "BMI", "AGE", "gender_F", "gender_M"],
        stage_names=["W", "Light", "REM", "Deep"],
        raw_stage_mapping={"0": 0, "1": 1, "2": 1, "3": 3, "4": 2, "other": -1},
        zero_value_policy="preserve stored zero markers", scaler=scaler,
        scaler_subjects=["A"], train_npz_sha256={"A.npz": sha("training record")})
    sources = {name: sha(name) for name in REQUIRED_CORE_SOURCES + DATA_SOURCES}
    checkpoint = sha("best checkpoint bytes")
    manifest = dict(training_args=deepcopy(args), data_identity=deepcopy(data),
        source_sha256=deepcopy(sources), diffusion_identity=diffusion_identity(args),
        checkpoint="/training/model.pth", checkpoint_sha256=sha("final bytes"),
        validation=dict(best_checkpoints=dict(kl=dict(checkpoint="/training/best_kl_model.pth",
            checkpoint_sha256=checkpoint, epoch=7))))
    return manifest, args, data, sources, checkpoint


def previous_summary(manifest, args, data, sources, checkpoint):
    return dict(sampling_performed=True, model_state_validated=True, status="failed",
        error="Generated IBI must be finite and nonnegative; values are not clipped",
        model_args=deepcopy(args), configured_data_identity=deepcopy(data),
        scaler=deepcopy(data["scaler"]), diffusion_identity=diffusion_identity(args),
        source_sha256=deepcopy(sources),
        source_provenance={"imported_core_sha256": {name: sources[name] for name in REQUIRED_CORE_SOURCES}},
        execution=dict(loaded_checkpoint_sha256=checkpoint, sampling_seed=42),
        checkpoint_identity=dict(sha256=checkpoint, hash_scope="loaded_checkpoint_bytes"),
        stride_samples=750, windows_per_group=4)


class DiagnosticIdentityTests(unittest.TestCase):
    def assert_status(self, report, expected):
        self.assertEqual(report["status"], expected,
            [check for check in report["blocking_checks"] if check["status"] != "matched"])
        self.assertEqual(report["can_sample"], expected == "matched")
        json.dumps(report, allow_nan=False)

    def test_exact_identity_matches_namespace_without_requiring_training_batch_size(self):
        manifest, args, data, sources, checkpoint = fixture()
        args.update(batch_size=1024, epochs=1000, stride=1500, validation_output=None)
        report = audit_training_identity(manifest, SimpleNamespace(**args), data, sources, checkpoint)
        self.assert_status(report, "matched")
        self.assertTrue(any(item["name"] == "training.batch_size" for item in report["expected_protocol_differences"]))

    def test_model_dimensions_and_each_affine_transform_mismatch_block(self):
        for change in ("model", "signal", "metadata", "units", "column_order"):
            with self.subTest(change=change):
                manifest, args, data, sources, checkpoint = fixture()
                if change == "model":
                    args["s4_dstate"] = 32
                elif change == "signal":
                    data["scaler"]["signal_std_seconds"] *= 2
                elif change == "metadata":
                    data["scaler"]["continuous_mean"][0] += 1
                elif change == "units":
                    data["multiplier"] = .001
                else:
                    data["columns"][5], data["columns"][6] = data["columns"][6], data["columns"][5]
                self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "mismatch")

    def test_missing_legacy_schedule_means_linear_and_can_be_reconstructed(self):
        manifest, args, data, sources, checkpoint = fixture()
        manifest["training_args"].pop("diffusion_schedule")
        manifest["training_args"].pop("terminal_alpha_bar")
        manifest.pop("diffusion_identity")
        manifest["data_identity"].pop("window_epochs")
        args["diffusion_schedule"] = "linear"
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "matched")
        args["diffusion_schedule"] = "linear_power"
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "mismatch")

    def test_nonlinear_schedule_requires_identity_and_rejects_corrupt_hash(self):
        manifest, args, data, sources, checkpoint = fixture()
        recorded = manifest.pop("diffusion_identity")
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "incomplete")
        manifest["diffusion_identity"] = recorded
        recorded["tensor_sha256"]["alpha_bars"] = sha("wrong tensor")
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "mismatch")

    def test_actual_terminal_target_change_is_detected(self):
        manifest, args, data, sources, checkpoint = fixture()
        args["terminal_alpha_bar"] = 1e-6
        report = audit_training_identity(manifest, args, data, sources, checkpoint)
        self.assert_status(report, "mismatch")
        self.assertTrue(any(check["name"].endswith("tensor_sha256") and check["status"] == "mismatch"
                            for check in report["blocking_checks"]))

    def test_final_hash_is_never_substituted_for_missing_best(self):
        manifest, args, data, sources, checkpoint = fixture()
        manifest.pop("validation")
        manifest["checkpoint"] = "/training/best_kl_model.pth"
        manifest["checkpoint_sha256"] = checkpoint
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "incomplete")

    def test_external_summary_allows_interrupted_training_but_conflicts_fail(self):
        manifest, args, data, sources, checkpoint = fixture()
        summary = manifest.pop("validation")
        manifest["status"] = "training"
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint,
                                                  validation_summary=summary), "matched")
        manifest["validation"] = deepcopy(summary)
        manifest["validation"]["best_checkpoints"]["kl"]["checkpoint_sha256"] = sha("different best")
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint,
                                                  validation_summary=summary), "mismatch")

    def test_selected_best_manifest_is_sufficient_evidence(self):
        manifest, args, data, sources, checkpoint = fixture()
        manifest.pop("validation")
        manifest.update(checkpoint=r"C:\run\best_kl_model.pth", checkpoint_sha256=checkpoint,
            checkpoint_selection={"metric": "gaussian_kl_real_to_generated", "selection_rule": "smoothed_kl"})
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "matched")

    def test_missing_critical_s4_hash_is_incomplete_not_matching(self):
        manifest, args, data, sources, checkpoint = fixture()
        manifest["source_sha256"].pop("TSImputers/S4Model.py")
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "incomplete")

    def test_cross_platform_paths_and_locations_preserve_hash_identity(self):
        manifest, args, data, sources, checkpoint = fixture()
        manifest["source_sha256"] = {"C:\\old\\checkout\\" + name.replace("/", "\\"): value
                                      for name, value in sources.items()}
        sources = {"/project/new/checkout/" + name: value for name, value in sources.items()}
        data.update(data_directory="D:/records", metadata_path="D:/participants.csv", split_path="D:/split.json",
                    split_source="split_file", split_fingerprint=sha("different provenance only"))
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "matched")

    def test_changed_or_conflicting_source_aliases_are_not_ignored(self):
        manifest, args, data, sources, checkpoint = fixture()
        sources["C:\\new\\TSImputers\\S4Model.py"] = sha("other contents")
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "mismatch")
        manifest, args, data, sources, checkpoint = fixture()
        manifest["source_sha256"]["TSImputers/Embedding.py"] = sha("embedding")
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "incomplete")
        sources["TSImputers/Embedding.py"] = sha("embedding")
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "matched")

    def test_changed_training_contents_and_uncovered_records_block(self):
        manifest, args, data, sources, checkpoint = fixture()
        data["train_npz_sha256"]["A.npz"] = sha("different stored values")
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "mismatch")
        manifest, args, data, sources, checkpoint = fixture()
        for identity in (manifest["data_identity"], data):
            identity["train_npz_sha256"] = {}
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "incomplete")

    def test_legacy_data_nesting_and_train_hash_alias_remain_supported(self):
        manifest, args, data, sources, checkpoint = fixture()
        identity = manifest.pop("data_identity")
        identity["training_npz_sha256"] = identity.pop("train_npz_sha256")
        manifest["data"] = {"data_identity": identity}
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "matched")

    def test_conflicting_training_hash_aliases_are_not_silently_preferred(self):
        manifest, args, data, sources, checkpoint = fixture()
        manifest["data_identity"]["training_npz_sha256"] = {"A.npz": sha("contradictory identity")}
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "mismatch")

    def test_previous_export_failure_does_not_invalidate_completed_sampling(self):
        manifest, args, data, sources, checkpoint = fixture()
        summary = previous_summary(manifest, args, data, sources, checkpoint)
        summary["source_sha256"]["scripts/evaluate_original_sampler.py"] = sha("old sampler")
        sources["scripts/evaluate_original_sampler.py"] = sha("noise clone correction")
        summary["execution"]["sampling_seed"] = 999
        report = audit_training_identity(manifest, args, data, sources, checkpoint, inference_summary=summary)
        self.assert_status(report, "matched")
        self.assertTrue(any(item["name"].startswith("previous.sampler.") for item in report["expected_protocol_differences"]))

    def test_previous_actual_checkpoint_scaler_or_core_difference_blocks_comparison(self):
        for change in ("checkpoint", "scaler", "core", "not_sampled"):
            with self.subTest(change=change):
                manifest, args, data, sources, checkpoint = fixture()
                summary = previous_summary(manifest, args, data, sources, checkpoint)
                if change == "checkpoint":
                    summary["execution"]["loaded_checkpoint_sha256"] = sha("other checkpoint")
                elif change == "scaler":
                    summary["scaler"]["signal_mean_seconds"] += .01
                elif change == "core":
                    summary["source_provenance"]["imported_core_sha256"]["TSImputers/S4Model.py"] = sha("old math")
                else:
                    summary["sampling_performed"] = False
                self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint,
                                                          inference_summary=summary), "mismatch")

    def test_previous_preflight_digest_is_not_actual_load_evidence(self):
        manifest, args, data, sources, checkpoint = fixture()
        summary = previous_summary(manifest, args, data, sources, checkpoint)
        summary["execution"].pop("loaded_checkpoint_sha256")
        summary["checkpoint_identity"]["hash_scope"] = "preflight_file"
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint,
                                                  inference_summary=summary), "incomplete")

    def test_malformed_or_missing_evidence_is_json_serializable_and_never_passes(self):
        report = audit_training_identity({}, {}, {}, {}, None)
        self.assertFalse(report["can_sample"])
        json.dumps(report, allow_nan=False)
        manifest, args, data, sources, checkpoint = fixture()
        for identity in (manifest["data_identity"], data):
            identity["scaler"]["signal_std_seconds"] = float("nan")
        report = audit_training_identity(manifest, args, data, sources, checkpoint)
        self.assertFalse(report["can_sample"])
        json.dumps(report, allow_nan=False)
        manifest, args, data, sources, checkpoint = fixture()
        args["s4_layernorm"] = manifest["training_args"]["s4_layernorm"] = "True"
        self.assert_status(audit_training_identity(manifest, args, data, sources, checkpoint), "incomplete")

    def test_inputs_are_not_mutated(self):
        values = fixture()
        original = deepcopy(values)
        audit_training_identity(*values)
        self.assertEqual(values, original)


if __name__ == "__main__":
    unittest.main()
