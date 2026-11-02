"""Artificial data/identity integration and explicitly mocked sampling tests.

No real DREAMT subject, production S4 forward pass, or CUDA run is performed.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
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
from diffusion_schedule import build_diffusion_config, diffusion_identity
from epoch_validation import EpochValidator
from scripts.evaluate_denoising import DEFAULT_TRAINING_ARGS
from scripts import export_ablation_validation as export
from scripts import evaluate_stage_generation as stage


class TinyModel(torch.nn.Module):
    """Checkpoint plumbing only; not an S4 replacement quality experiment."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2))
        self.register_buffer("fixed", torch.randn(2))


class StageGenerationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = (ROOT / "work/stage_generation_workflow_tests").resolve()
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.base = Path(self.temp.name)
        self.data_dir = self.base / "data"
        self.data_dir.mkdir()
        self.subjects = ("S001", "S002", "S003", "S092")
        metadata = ["SID,AHI,BMI,AGE,GENDER"]
        for index, subject in enumerate(self.subjects):
            signal = .7 + .1 * index + .03 * np.sin(np.arange(6000) / 31.0)
            signal[::47] = 0
            # All labels are original five-class values. N1/N2 merge to Light.
            labels = np.repeat([0, 1, 2, 3, 4, 0, 3, 4], 750)
            ahi = 5.0 + 4 * index
            np.savez(self.data_dir / f"{subject}.npz", data=signal.astype(np.float32)[:, None],
                     stages=labels, fs=np.array(25), ahi=np.array(ahi), synthetic=np.array(True))
            metadata.append(f"{subject},{ahi},{22+index*1.5},{30+index*8},{'F' if index%2==0 else 'M'}")
        (self.data_dir / "participant_info.csv").write_text("\n".join(metadata)+"\n", encoding="utf-8")
        export.save_json(self.data_dir / "splits.json", dict(train=["S001.npz", "S002.npz"],
            val=["S003.npz"], test=["S092.npz"]))
        self.run = self.base / "reference"
        for arm in export.ARMS:
            self.create_arm(arm)

    def tearDown(self):
        target = Path(self.temp.name).resolve()
        if not target.is_relative_to(self.temp_root) or target == self.temp_root:
            raise RuntimeError("Refuse cleanup outside the test temporary root")
        self.temp.cleanup()

    def create_arm(self, arm):
        root = self.run / arm
        workspace = root / "workspace"
        workspace.mkdir(parents=True)
        config = dict(data_dir=str(self.data_dir), split_path=str(self.data_dir / "splits.json"),
            generate_splits=False, window_epochs=4, missing_value_policy=arm,
            diffusion_schedule="linear_power", terminal_alpha_bar=1e-8,
            validation=dict(max_windows=64, batch_size=8, seed=20261020,
                kl_variance_floor_ms2=1.0, reference_mask_policy="observed_only"),
            early_stopping=dict(enabled=False))
        export.save_json(workspace / "dataset_config.json", config)
        processor = Preprocessor("DREAMT", data_config=config)
        args = SimpleNamespace(**dict(DEFAULT_TRAINING_ARGS, dataset="DREAMT", batch_size=8,
            epochs=30, stride=750, diffusion_schedule="linear_power", terminal_alpha_bar=1e-8))
        schedule = build_diffusion_config(args)
        # Construct the real frozen protocol on CPU without a model or sampler.
        with mock.patch("epoch_validation._require_device", return_value=torch.device("cpu")):
            EpochValidator(processor, args, None, schedule, torch.device("cpu"),
                           root / "training/validation", config)
        torch.manual_seed(42)
        model = TinyModel()
        model.weight.data.fill_(3.0 if arm == "preserve" else 4.0)
        checkpoint = workspace / "saved_models/DREAMT/model.pth"
        checkpoint.parent.mkdir(parents=True)
        torch.save(model.state_dict(), checkpoint)
        sources = {name: export.sha256(ROOT / name) for name in export.RUNTIME_SOURCES}
        sources["dataset_config.json"] = export.sha256(workspace / "dataset_config.json")
        # Explicit fixture metadata, not a claimed 30-epoch model training run.
        manifest = dict(status="completed", checkpoint=str(checkpoint),
            checkpoint_sha256=export.sha256(checkpoint), checkpoint_epoch=30, completed_epochs=30,
            requested_epochs=30, batches_per_epoch=1, updates=30, training_args=vars(args),
            source_sha256=sources, data_identity=processor.data_identity(),
            diffusion_identity=diffusion_identity(args, schedule), missing_value_policy=arm, threads=4)
        export.save_json(root / "training/run_manifest.json", manifest)
        final_dir = root / "training/validation/epoch_0030"
        final_dir.mkdir()
        export.save_json(final_dir / "metrics.json", dict(status="completed", epoch=30,
            aggregate=dict(gaussian_kl_real_to_generated=1.0 if arm == "preserve" else 2.0)))
        (root / "training/best_kl_manifest.json").write_text("not JSON", encoding="utf-8")
        (root / "training/best_kl_model.pth").write_bytes(b"deliberately irrelevant best checkpoint")

    def load_pair(self):
        return stage.load_reference_pair(self.run)[0]

    def arguments(self, output, *extra):
        return ["--ablation-run", str(self.run), "--subject", "S092", "--expected-epochs", "30",
                "--repeats", "2", "--seed", "123", "--selection-seed", "42", "--windows-per-stage", "1",
                "--batch-size", "1", "--threads", "1", "--output-dir", str(output), *extra]

    def test_real_identity_restore_opens_explicit_test_subject_without_refitting(self):
        contexts = self.load_pair()
        before = {path: export.sha256(path) for path in self.run.rglob("*") if path.is_file()}
        processors = []
        actual_bind = stage.bind_training_data
        def bind(*args):
            result = actual_bind(*args)
            processors.append(result[0])
            return result
        with mock.patch.object(stage, "bind_training_data", side_effect=bind):
            data, checks = stage.prepare_pair(contexts, "S092", 1, 42)
        self.assertEqual(data["subject_id"], "S092")
        self.assertEqual(data["conditions"].shape, (2, 3000, 9))
        self.assertEqual(data["stage_epochs_four"].tolist(), [[0, 1, 1, 3], [2, 0, 3, 2]])
        np.testing.assert_array_equal(data["scoring_mask"], data["reference_seconds"] > 0)
        self.assertTrue(all(checks["shared_inputs"].values()))
        for arm, processor in zip(export.ARMS, processors):
            self.assertTrue(checks[arm]["training_data_identity_checks"]["all_recorded_identity_fields_match"])
            self.assertTrue(checks[arm]["training_identity_unchanged"])
            provenance = checks[arm]["subject_provenance"]
            self.assertEqual(provenance["subject_role"], "test")
            self.assertFalse(provenance["subject_in_original_training_scaler"])
            self.assertFalse(provenance["scaler_refitted_for_generation"])
            self.assertNotIn("S092.npz", processor.accessed_files)
            self.assertEqual(processor.data_identity(), contexts[arm]["identity"])
            with self.assertRaisesRegex(ValueError, "test is sealed"):
                processor.record("S092.npz")
            self.assertEqual(contexts[arm]["checkpoint"].name, "model.pth")
        self.assertNotEqual(processors[0].scaler["signal_mean_seconds"], processors[1].scaler["signal_mean_seconds"])
        self.assertEqual(before, {path: export.sha256(path) for path in before})

    def test_check_only_uses_real_identity_pipeline_without_cuda_or_model(self):
        output = self.base / "check"
        with mock.patch.object(stage, "require_cuda") as cuda, \
             mock.patch.object(stage, "fetchModel") as model, \
             mock.patch.object(stage, "sample_arm") as sample, redirect_stdout(io.StringIO()):
            report = stage.main(self.arguments(output, "--check-only"))
        cuda.assert_not_called()
        model.assert_not_called()
        sample.assert_not_called()
        self.assertEqual(report["status"], "checked")
        self.assertFalse(report["sampling_performed"])
        self.assertFalse(report["training_executed"])
        self.assertFalse(report["best_checkpoint_selected"])
        self.assertTrue(all(report["shared_reference_checks"].values()))
        protocol = export.read_json(output / "protocol.json")
        self.assertEqual(protocol["seeds"], [123, 124])
        self.assertFalse(protocol["generation"]["reference_ibi_used_as_model_input"])
        self.assertEqual(protocol["generation"]["real_ibi_prefix_samples"], 0)
        self.assertEqual(protocol["generation"]["overlap_samples"], 0)
        for name, digest in report["file_hashes"].items():
            self.assertEqual(export.sha256(output / name), digest)

    def test_wrong_expected_epoch_and_unequal_updates_fail_before_sampling(self):
        with mock.patch.object(stage, "sample_arm") as sample, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "expected-epochs"):
                stage.main(self.arguments(self.base / "wrong_epoch", "--expected-epochs", "29"))
        sample.assert_not_called()
        path = self.run / "masked/training/run_manifest.json"
        manifest = export.read_json(path)
        manifest.update(updates=60, batches_per_epoch=2)
        export.save_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "updates"):
            self.load_pair()

    def test_wrong_final_hash_fails_without_best_fallback(self):
        path = self.run / "masked/workspace/saved_models/DREAMT/model.pth"
        path.write_bytes(b"changed final weights")
        with self.assertRaisesRegex(ValueError, "digest"):
            self.load_pair()

    def test_changed_training_data_or_scaler_is_rejected_by_real_binding(self):
        contexts = self.load_pair()
        changed = deepcopy(contexts)
        changed["masked"]["manifest"]["data_identity"]["scaler"]["signal_mean_seconds"] += .1
        with self.assertRaisesRegex(ValueError, "identity|scaler"):
            stage.prepare_pair(changed, "S092", 1, 42)
        path = self.data_dir / "S001.npz"
        with np.load(path, allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
        arrays["data"][1, 0] += .1
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, "identity|scaler"):
            stage.prepare_pair(contexts, "S092", 1, 42)

    def test_training_and_validation_subjects_are_not_silently_used_as_test(self):
        contexts = self.load_pair()
        for subject in ("S001", "S003"):
            with self.subTest(subject=subject), self.assertRaisesRegex(ValueError, "held-out test"):
                stage.prepare_pair(contexts, subject, 1, 42)

    def test_pair_rejects_condition_drift_even_without_changing_reference(self):
        contexts = self.load_pair()
        actual = stage.prepare_night
        count = 0
        def prepare(*args, **kwargs):
            nonlocal count
            result = actual(*args, **kwargs)
            count += 1
            if count == 2:
                result["conditions"] = result["conditions"].copy()
                result["conditions"][:, 4] += 1.0
            return result
        with mock.patch.object(stage, "prepare_night", side_effect=prepare):
            with self.assertRaisesRegex(ValueError, "test inputs differ.*conditions"):
                stage.prepare_pair(contexts, "S092", 1, 42)

    def test_no_cpu_fallback_in_main_or_sample_arm(self):
        output = self.base / "cpu_rejected"
        with mock.patch.object(stage, "require_cuda", return_value=torch.device("cpu")), \
             mock.patch.object(stage, "fetchModel") as model, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                stage.main(self.arguments(output))
        model.assert_not_called()
        report = export.read_json(output / "summary.json")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["sampling_performed"])
        # The direct model helper must independently enforce the same boundary.
        with mock.patch.object(stage, "fetchModel") as model:
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                stage.sample_arm({}, {}, "cpu", seeds=[123], batch_size=1, output_dir=self.base / "direct_cpu")
        model.assert_not_called()
        self.assertFalse((self.base / "direct_cpu").exists())

    def test_missing_cuda_reports_failure_without_constructing_model(self):
        output = self.base / "no_gpu"
        with mock.patch.object(stage, "require_cuda", side_effect=RuntimeError("CUDA unavailable")), \
             mock.patch.object(stage, "fetchModel") as model, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                stage.main(self.arguments(output))
        model.assert_not_called()
        self.assertEqual(export.read_json(output / "summary.json")["status"], "failed")

    def test_sample_arm_passes_only_zero_signal_with_original_conditions_and_paired_noise(self):
        contexts = self.load_pair()
        data, _ = stage.prepare_pair(contexts, "S092", 1, 42)
        initial_reference = data["reference_seconds"].copy()
        seen = {arm: [] for arm in export.ARMS}
        inputs_seen = []
        models = []
        active = [None]
        def model_factory(*args):
            model = TinyModel()
            models.append(model)
            return model
        def fake_sampler(model, inputs, schedule, args, device):
            self.assertEqual(device.type, "cpu")  # Explicit test-only patch below.
            self.assertFalse(model.training)
            self.assertTrue(torch.equal(model.weight, torch.full((2,), 3.0 if active[0] == "preserve" else 4.0)))
            self.assertEqual(tuple(inputs.shape[1:]), (3000, 10))
            self.assertTrue(torch.equal(inputs[:, :, 0], torch.zeros_like(inputs[:, :, 0])))
            batch = len(inputs_seen) % 2
            np.testing.assert_array_equal(inputs[:, :, 1:].numpy(), data["conditions"][batch:batch+1])
            inputs_seen.append(inputs.detach().clone())
            signal = torch.randn(inputs.shape[:2]) * .1
            signal[:, 0] = -100.0  # Negative even with this fixture's small masked std.
            seen[active[0]].append(signal.detach().clone())
            return signal, {"fixture_sampler": True, "no_model_forward_executed": True}
        random.seed(91)
        np.random.seed(92)
        torch.manual_seed(93)
        states = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
        reports = {}
        with mock.patch.object(stage, "_require_sampling_device", return_value=torch.device("cpu")), \
             mock.patch.object(stage, "assert_cuda_model"), \
             mock.patch.object(stage, "fetchModel", side_effect=model_factory), \
             mock.patch.object(stage, "sample_validation_windows", side_effect=fake_sampler):
            for arm in export.ARMS:
                active[0] = arm
                reports[arm] = stage.sample_arm(contexts[arm], data, "cpu", seeds=[123, 124],
                    batch_size=1, output_dir=self.base / f"sample_{arm}")
        self.assertEqual(states[0], random.getstate())
        self.assertEqual(states[1][0], np.random.get_state()[0])
        np.testing.assert_array_equal(states[1][1], np.random.get_state()[1])
        self.assertEqual(states[1][2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(states[2], torch.random.get_rng_state()))
        np.testing.assert_array_equal(data["reference_seconds"], initial_reference)
        self.assertEqual(len(inputs_seen), 8)
        for a, b in zip(seen["preserve"], seen["masked"]):
            self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(seen["preserve"][0], seen["preserve"][2]))
        for arm in export.ARMS:
            report = reports[arm]
            self.assertFalse(report["training_executed"])
            self.assertFalse(report["best_checkpoint_selected"])
            self.assertEqual(report["checkpoint_kind"], "final")
            self.assertEqual(report["epoch"], 30)
            self.assertEqual(report["stages"]["seeds"], [123, 124])
            scaler = contexts[arm]["identity"]["scaler"]
            with np.load(self.base / f"sample_{arm}/draw_000/waveforms.npz", allow_pickle=False) as stored:
                expected = torch.cat(seen[arm][:2]).double().numpy() * scaler["signal_std_seconds"] + scaler["signal_mean_seconds"]
                np.testing.assert_array_equal(stored["generated_ibi_seconds"], expected)
                self.assertTrue(np.all(stored["generated_ibi_seconds"][:, 0] < 0))
            with (self.base / f"sample_{arm}/draw_000/per_epoch_metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({int(row["source_epoch_index"]) for row in rows}), 8)

    def test_checkpoint_changed_after_preflight_is_rejected_before_torch_load(self):
        contexts = self.load_pair()
        data, _ = stage.prepare_pair(contexts, "S092", 1, 42)
        contexts["preserve"]["checkpoint"].write_bytes(b"changed after verified identity")
        with mock.patch.object(stage, "_require_sampling_device", return_value=torch.device("cpu")), \
             mock.patch.object(stage, "fetchModel", side_effect=lambda *args: TinyModel()), \
             mock.patch.object(stage, "assert_cuda_model"), mock.patch.object(stage.torch, "load") as load:
            with self.assertRaisesRegex(ValueError, "changed after identity"):
                stage.sample_arm(contexts["preserve"], data, "cpu", seeds=[123], batch_size=1,
                                 output_dir=self.base / "changed")
        load.assert_not_called()

    def test_duplicate_draw_seed_is_rejected_before_model_or_output(self):
        with mock.patch.object(stage, "_require_sampling_device", return_value=torch.device("cpu")), \
             mock.patch.object(stage, "fetchModel") as model:
            with self.assertRaisesRegex(ValueError, "distinct sampling seeds"):
                stage.sample_arm({}, {}, "cpu", seeds=[123, 123], batch_size=1,
                                 output_dir=self.base / "duplicate")
        model.assert_not_called()
        self.assertFalse((self.base / "duplicate").exists())

    def test_complete_main_writes_hashed_artifacts_and_summary_without_optimizer(self):
        output = self.base / "complete_fixture"
        calls = []
        def fake_sampler(model, inputs, schedule, args, device):
            self.assertTrue(torch.equal(inputs[:, :, 0], torch.zeros_like(inputs[:, :, 0])))
            calls.append(inputs.shape)
            return torch.randn(inputs.shape[:2]) * .1, {"fixture_sampler": True}
        previous_threads = torch.get_num_threads()
        try:
            with mock.patch.object(stage, "require_cuda", return_value=torch.device("cpu")), \
                 mock.patch.object(stage, "_require_sampling_device", return_value=torch.device("cpu")), \
                 mock.patch.object(stage, "cuda_environment", return_value={"fixture_only": True}), \
                 mock.patch.object(stage, "assert_cuda_model"), \
                 mock.patch.object(stage, "fetchModel", side_effect=lambda *args: TinyModel()), \
                 mock.patch.object(stage, "sample_validation_windows", side_effect=fake_sampler), \
                 mock.patch.object(torch.optim, "Adam") as optimizer, redirect_stdout(io.StringIO()):
                report = stage.main(self.arguments(output))
        finally:
            torch.set_num_threads(previous_threads)
        optimizer.assert_not_called()
        self.assertEqual(len(calls), 8)
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["training_executed"])
        self.assertFalse(report["best_checkpoint_selected"])
        for name, digest in report["file_hashes"].items():
            self.assertFalse(Path(name).is_absolute())
            self.assertEqual(export.sha256(output / name), digest)
        for arm in export.ARMS:
            self.assertEqual(report["arms"][arm]["stages"]["seeds"], [123, 124])
            self.assertTrue((output / arm / "draw_001/waveforms.npz").is_file())
            self.assertTrue((output / arm / "draw_001/per_stage_metrics.csv").is_file())
        with (output / "stage_summary.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["arm"] for row in rows}, {"preserve", "masked"})
        self.assertTrue(all(row["draws"] == "2" for row in rows))
        self.assertIn("gaussian_kl_real_to_generated_mean", rows[0])
        self.assertIn("gaussian_kl_real_to_generated_std", rows[0])

    def test_output_cannot_overwrite_input_or_existing_folder(self):
        with mock.patch.object(stage, "load_reference_pair") as load:
            for path in (self.run, self.run / "new/deeper"):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "outside"):
                    stage.main(self.arguments(path, "--check-only"))
            existing = self.base / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                stage.main(self.arguments(existing, "--check-only"))
        load.assert_not_called()

    def test_submission_runs_evaluation_then_plot_with_fixed_final_budget(self):
        source = (ROOT / "submit_stage_evaluation.sh").read_text()
        self.assertIn("set -euo pipefail", source)
        for text in ("#SBATCH --cpus-per-task=4", "#SBATCH --gpus-per-node=1", "#SBATCH --time=10:00:00",
                     "missing_value_ablation_extended_20261102T170345Z_1221493/training_30epochs",
                     "--subject S092 --expected-epochs 30", "--repeats 8 --seed 20261102 --selection-seed 42",
                     "--windows-per-stage 8", "--batch-size 8 --threads 4", "${SLURM_JOB_ID:-local}"):
            self.assertIn(text, source)
        self.assertLess(source.index("python scripts/evaluate_stage_generation.py"),
                        source.index("python scripts/plot_stage_evaluation.py"))
        self.assertIn('--input-dir "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/plots"', source)
        self.assertNotIn('mkdir "$OUTPUT_ROOT"', source)
        self.assertNotIn("run_dreamt.py", source)


if __name__ == "__main__":
    unittest.main()
