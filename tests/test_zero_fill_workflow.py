"""Zero-fill workflow checks with synthetic data and explicit CPU test boundaries.

The tiny S4 update executes the actual production loop on CPU as a unit test;
it is not a production CUDA training or signal-quality experiment.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
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
from diffusion_schedule import build_diffusion_config
import epoch_validation as coordinator
from night_checkpoint import load_inference_config
from scripts import evaluate_denoising as audit
from scripts import run_dreamt as launcher
from signal_normalization import signal_affine_parameters
from training_utils import fetchModel


def execute_node(node, namespace):
    tree = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]))
    exec(compile(tree, "production_fragment::explicit_cpu_unit_test", "exec"), namespace)


class ZeroFillWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp_root = ROOT / "work/zero_fill_workflow_tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="zero_fill_workflow_", dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.data = self.folder / "data"
        self.data.mkdir()
        for index, sid in enumerate(("A", "B", "V", "T")):
            values = np.tile(np.array([0., .6, 1.2, 1.8], dtype=np.float32), 1500)
            values[values > 0] += index * .05
            np.savez(self.data / f"{sid}.npz", data=values[:, None], fs=25,
                     stages=np.array([0, 1, 2, 3, 4, 1, 0, 3]), ahi=5.+index*10, synthetic=True)
        (self.data / "participant_info.csv").write_text(
            "SID,AHI,BMI,AGE,GENDER\nA,5,20,20,F\nB,15,30,40,M\nV,25,40,60,F\nT,35,50,80,M\n",
            encoding="utf-8")
        (self.data / "splits.json").write_text(json.dumps(
            dict(train=["A.npz", "B.npz"], val=["V.npz"], test=["T.npz"])), encoding="utf-8")
        self.config = self.folder / "config.json"
        self.write_config()

    def write_config(self, **options):
        config = dict(data_dir="data", window_epochs=1, missing_value_policy="zero_fill_one")
        config.update(options)
        self.config.write_text(json.dumps(config), encoding="utf-8")
        return config

    def processor(self, policy="zero_fill_one", **options):
        return Preprocessor("DREAMT", data_config=dict(data_dir=str(self.data), window_epochs=1,
                                                       missing_value_policy=policy, **options))

    def make_validator(self, processor, **options):
        args = SimpleNamespace(window_size=750, s4_lmax=750, timesteps=3,
            beta_0=.0001, beta_T=.02, diffusion_schedule="linear", terminal_alpha_bar=1e-8)
        config = dict(window_epochs=1, validation=dict(batch_size=2, **options))
        with mock.patch.object(coordinator, "_require_device", return_value=torch.device("cpu")):
            return coordinator.EpochValidator(processor, args, torch.nn.Linear(1, 1),
                build_diffusion_config(args), "cpu", self.folder / ("validation_" + str(len(list(self.folder.glob("validation_*"))))),
                config)

    def test_cli_preflight_accepts_policy_override_and_ignores_stale_environment(self):
        self.write_config(missing_value_policy="preserve")
        output = self.folder / "preflight"
        command = ["run_dreamt.py", "--config", str(self.config), "--output-dir", str(output),
                   "--missing-value-policy", "zero_fill_one", "--check-only"]
        with mock.patch.object(sys, "argv", command), mock.patch.dict(os.environ,
                {"DREAMT_DATA_DIR": "nonexistent-stale-data", "DREAMT_MISSING_VALUE_POLICY": "masked"}), \
             mock.patch.object(launcher, "require_cuda", side_effect=AssertionError("check-only must not load CUDA")), \
             contextlib.redirect_stdout(io.StringIO()):
            launcher.main()
        preflight = json.loads((output / "preflight.json").read_text())
        self.assertEqual(preflight["missing_value_policy"], "zero_fill_one")
        self.assertEqual(preflight["data_identity"]["missing_value_policy"], "zero_fill_one")
        self.assertEqual(preflight["training_input_shape"], [750, 10])
        self.assertFalse(preflight["test_npz_opened"])
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "checked")

    def test_freeze_and_inference_restore_same_policy_scaler_and_training_target(self):
        output = self.folder / "frozen"
        commands = []
        def fake_subprocess(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0)
        with mock.patch.object(sys, "argv", ["run_dreamt.py", "--config", str(self.config),
                "--output-dir", str(output), "--epochs", "1", "--skip-generation"]), \
             mock.patch.dict(os.environ, {}), \
             mock.patch.object(launcher, "require_cuda", return_value=torch.device("cuda:0")), \
             mock.patch.object(launcher, "cuda_environment", return_value={"mocked_test_boundary": True}), \
             mock.patch.object(launcher.subprocess, "run", side_effect=fake_subprocess), \
             contextlib.redirect_stdout(io.StringIO()):
            launcher.main()
        # Only orchestration is mocked: snapshot files and both preprocessors are real.
        self.assertEqual(len(commands), 2)
        frozen_path = output / "workspace/dataset_config.json"
        frozen = json.loads(frozen_path.read_text())
        self.assertEqual(frozen["missing_value_policy"], "zero_fill_one")
        for name in ("data_utils.py", "epoch_validation.py", "scripts/evaluate_denoising.py"):
            self.assertEqual((output / "workspace" / name).read_bytes(), (ROOT / name).read_bytes())
        args, data_config, provenance = load_inference_config(frozen_path)
        with mock.patch.dict(os.environ, {"DREAMT_CONFIG_PATH": "invalid.json", "DREAMT_DATA_DIR": "invalid"}):
            restored = Preprocessor("DREAMT", data_config=data_config)
        self.assertFalse(provenance["environment_overrides_used"])
        self.assertEqual(args.window_size, 750)
        expected = json.loads((output / "preflight.json").read_text())["data_identity"]
        self.assertEqual(restored.data_identity(), expected)
        original = self.processor()
        np.testing.assert_array_equal(restored.window_array("A.npz", 0, 750), original.window_array("A.npz", 0, 750))

    def test_frozen_manifest_binding_preserves_policy_in_schema_and_legacy_branches(self):
        processor = self.processor()
        expected = processor.data_identity()
        args = SimpleNamespace(dataset="DREAMT", window_size=750, s4_lmax=750, propCycEnc=False)
        for legacy in (False, True):
            identity = copy.deepcopy(expected)
            if legacy:
                identity.pop("schema")
            restored, checks = audit.bind_training_data(args, {"data_identity": identity},
                self.folder / "run_manifest.json", Preprocessor)
            self.assertTrue(checks["all_recorded_identity_fields_match"])
            self.assertTrue(checks["checks"]["missing_value_policy"])
            self.assertTrue(checks["checks"]["zero_value_policy"])
            self.assertEqual(restored.missing_value_policy, "zero_fill_one")
            self.assertEqual(restored.scaler, processor.scaler)

    def test_unknown_policy_rejected_by_json_cli_and_denoising(self):
        self.write_config(missing_value_policy="zero_fill_typo")
        with mock.patch.object(sys, "argv", ["run_dreamt.py", "--config", str(self.config),
                "--output-dir", str(self.folder / "bad"), "--check-only"]), \
             mock.patch.dict(os.environ, {}), self.assertRaisesRegex(ValueError, "missing_value_policy"):
            launcher.main()
        with mock.patch.object(sys, "argv", ["run_dreamt.py", "--missing-value-policy", "unknown"]), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            launcher.main()
        self.assertEqual(error.exception.code, 2)
        with self.assertRaisesRegex(ValueError, "Unknown frozen missing_value_policy"):
            audit._validation_observed_masks(SimpleNamespace(missing_value_policy="unknown"), [], 750)

    def test_validation_default_scores_raw_positive_positions_not_filled_training_target(self):
        processor = self.processor()
        validator = self.make_validator(processor)
        self.assertIsNotNone(validator.reference_mask)
        self.assertEqual(validator.protocol["missing_value_policy"], "zero_fill_one")
        self.assertEqual(validator.protocol["schema"], "ibi_epoch_masked_window_gaussian_kl_v5")
        for index, row in enumerate(validator.windows_metadata):
            raw = processor.record(row["filename"])["ibi"][row["start_sample"]:row["start_sample"]+750]
            np.testing.assert_array_equal(validator.reference[index].numpy(), raw)
            np.testing.assert_array_equal(validator.reference_mask[index].numpy(), raw > 0)
            filled = processor.inverse_signal(validator.windows[index, :, 0].numpy())
            np.testing.assert_allclose(filled[raw == 0], 1., atol=1e-7)
            self.assertTrue(torch.all(validator.reference[index][~validator.reference_mask[index]] == 0))

    def test_paired_validation_has_identical_raw_reference_mask_and_window_selection(self):
        validators = [self.make_validator(self.processor(policy), reference_mask_policy="observed_only")
                      for policy in ("preserve", "masked", "zero_fill_one")]
        for current in validators[1:]:
            self.assertTrue(torch.equal(current.reference, validators[0].reference))
            self.assertTrue(torch.equal(current.reference_mask, validators[0].reference_mask))
            self.assertEqual(current.windows_metadata, validators[0].windows_metadata)

    def test_validation_rejects_filled_only_windows_without_two_raw_observations(self):
        path = self.data / "V.npz"
        with np.load(path, allow_pickle=False) as stored:
            values = {key: stored[key].copy() for key in stored.files}
        values["data"][:750] = 0
        np.savez(path, **values)
        processor = self.processor()
        self.assertIn(("V.npz", 0), processor.make_validation_dataset(stride=750).window_index)
        selected, metadata = coordinator.select_fixed_windows(processor, coordinator.DEFAULT_VALIDATION)
        self.assertNotIn(("V.npz", 0), selected)
        self.assertEqual(metadata["excluded_insufficient_observation_candidates"], 1)
        from scripts.evaluate_original_sampler import select_sampling_window
        self.assertEqual(select_sampling_window(processor, "V"), ("V.npz", 750))
        with self.assertRaisesRegex(ValueError, "at least two raw positive"):
            select_sampling_window(processor, "V", start_sample=0)

    def test_teacher_forced_metrics_use_all_points_and_explicit_filled_target_protocol(self):
        processor = self.processor()
        with mock.patch.object(processor, "window_observed_mask", side_effect=AssertionError("all-point training objective")):
            self.assertIsNone(audit._validation_observed_masks(processor, [("V.npz", 0)], 750))
        predicted = torch.tensor([[1., 2., 3.]])
        target = torch.zeros_like(predicted)
        result = audit._teacher_forced_window_metrics(predicted, target, torch.tensor([1., .5, 1.5]),
            torch.tensor([2., .5, 1.5]), None, torch.nn.MSELoss())
        self.assertAlmostEqual(result["epsilon_mse"].item(), 14/3, places=6)
        self.assertEqual(result["reference_mean_ms"].item(), 1000.)
        main = next(node for node in ast.parse(Path(audit.__file__).read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        branch = next(node for node in main.body if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and any(isinstance(value, ast.Constant) and value.value == "zero_fill_one" for value in node.test.comparators))
        context = dict(preprocessor=processor, report={"protocol": {}})
        execute_node(branch, context)
        protocol = context["report"]["protocol"]
        self.assertIn("all-point", protocol["loss"])
        self.assertTrue(protocol["filled_training_target_scored"])
        self.assertFalse(protocol["raw_observed_mask_exported"])
        self.assertIn("not the original raw", protocol["reference_target"])

    def test_actual_s4_production_loop_supervises_filled_positions_with_ordinary_mse(self):
        processor = self.processor()
        dataset = processor.make_training_dataset(750, 750)
        batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [0, 1]), batch_size=2)))
        self.assertTrue(torch.is_tensor(batch))
        raw_mask = torch.from_numpy(np.stack([processor.window_observed_mask(name, start, 750)
                                             for name, start in dataset.window_index[:2]]))
        self.assertTrue((~raw_mask).any())
        args = SimpleNamespace(**dict(audit.DEFAULT_TRAINING_ARGS, window_size=750, s4_lmax=750,
            epochs=1, res_channels=8, skip_channels=8, num_res_layers=1, s4_dstate=8,
            diff_step_embed_in=8, diff_step_embed_mid=8, diff_step_embed_out=8,
            timesteps=4, diffusion_schedule="linear_power", terminal_alpha_bar=1e-8))
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(), contextlib.redirect_stdout(io.StringIO()):
                torch.manual_seed(42)
                model = fetchModel(10, 1, args)
                before = {name: value.detach().clone() for name, value in model.named_parameters()}
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                captures = []
                def criterion(predicted, target):
                    predicted.retain_grad()
                    loss = torch.nn.functional.mse_loss(predicted, target)
                    captures.append((predicted, target.detach().clone(), float(loss.detach())))
                    return loss
                forbidden = mock.Mock(side_effect=AssertionError("zero_fill_one must not call masked loss"))
                reports = []
                loop = next(node for node in ast.walk(ast.parse((ROOT / "training_wavestitch.py").read_text()))
                            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "epoch")
                namespace = dict(torch=torch, np=np, model=model, optimizer=optimizer, criterion=criterion,
                    device=torch.device("cpu"), args=args, dataloader=[batch], diffusion_config=build_diffusion_config(args),
                    non_hier_cols=torch.tensor([0]), randint=torch.randint, normal=torch.normal, sqrt=torch.sqrt,
                    masked_noise_mse=forbidden, training_input_dropout=lambda batch: batch, validator=SimpleNamespace(should_stop=False,
                    run=lambda epoch, live_model, loss: reports.append(loss)))
                execute_node(loop, namespace)
            forbidden.assert_not_called()
            self.assertEqual(len(captures), 1)
            predicted, target, loss = captures[0]
            self.assertEqual(reports, [loss])
            expected_gradient = 2*(predicted.detach()-target)/target.numel()
            torch.testing.assert_close(predicted.grad, expected_gradient)
            self.assertTrue(torch.all(predicted.grad[:, 0][~raw_mask] != 0))
            self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.named_parameters()))
            self.assertTrue(all(torch.isfinite(value).all() for value in model.parameters()))
        finally:
            torch.set_num_threads(old_threads)

    def test_minmax_snapshot_and_inference_binding_freeze_range_instead_of_population_statistics(self):
        spec = dict(method="minmax", min_seconds=0, max_seconds=2)
        self.write_config(missing_value_policy="preserve", signal_normalization=spec)
        output = self.folder / "minmax_frozen"
        with mock.patch.object(sys, "argv", ["run_dreamt.py", "--config", str(self.config),
                "--output-dir", str(output), "--epochs", "1", "--skip-generation"]), \
             mock.patch.dict(os.environ, {}), \
             mock.patch.object(launcher, "require_cuda", return_value=torch.device("cuda:0")), \
             mock.patch.object(launcher, "cuda_environment", return_value={"mocked_test_boundary": True}), \
             mock.patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
             contextlib.redirect_stdout(io.StringIO()):
            launcher.main()
        frozen_path = output / "workspace/dataset_config.json"
        frozen = json.loads(frozen_path.read_text())
        self.assertEqual(frozen["signal_normalization"], spec)
        self.assertEqual((output / "workspace/signal_normalization.py").read_bytes(),
                         (ROOT / "signal_normalization.py").read_bytes())
        args, config, _ = load_inference_config(frozen_path)
        restored = Preprocessor("DREAMT", data_config=config)
        self.assertEqual(signal_affine_parameters(restored.scaler), (1., 1.))
        self.assertNotEqual(restored.scaler["signal_std_seconds"], 1.)
        np.testing.assert_array_equal(restored.inverse_signal(np.array([-1., 0., 1.])), [0., 1., 2.])
        expected = restored.data_identity()
        for legacy in (False, True):
            identity = copy.deepcopy(expected)
            if legacy:
                identity.pop("schema")
            bound, checks = audit.bind_training_data(args, {"data_identity": identity},
                self.folder / "manifest.json", Preprocessor)
            self.assertTrue(checks["checks"]["signal_normalization"])
            self.assertEqual(bound.scaler, restored.scaler)
            np.testing.assert_array_equal(bound.window_array("V.npz", 0, 750), restored.window_array("V.npz", 0, 750))

    def test_minmax_validation_uses_affine_inverse_with_raw_reference_and_fixed_observed_mask(self):
        processor = self.processor("preserve", signal_normalization=dict(method="minmax", min_seconds=0, max_seconds=2))
        validator = self.make_validator(processor, reference_mask_policy="observed_only", save_artifacts=True)
        self.assertEqual((validator.center, validator.scale), (1., 1.))
        self.assertNotEqual(validator.scale, processor.scaler["signal_std_seconds"])
        training = torch.nn.Linear(1, 1)
        def generated_zero(model, clean, schedule, args, device):
            return torch.zeros(clean.shape[:2], device=device), {"unit_test_sampler": True}
        with mock.patch("training_utils.fetchModel", side_effect=lambda *args: torch.nn.Linear(1, 1)), \
             mock.patch.object(coordinator, "sample_validation_windows", side_effect=generated_zero):
            result, _, _ = validator._generate(training)
        np.testing.assert_array_equal(result["_artifact_payload"]["generated_seconds"],
                                      np.ones(validator.reference.shape))
        self.assertTrue(torch.all(validator.reference[~validator.reference_mask] == 0))
        self.assertTrue(torch.all(validator.windows[:, :, 0][~validator.reference_mask] == -1))
        self.assertFalse(result["diagnostics"]["generated_nonpositive_samples"])

    def test_minmax_teacher_forced_actual_row_inverse_uses_range_not_training_mean_std(self):
        main = next(node for node in ast.parse(Path(audit.__file__).read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        evaluate = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
        row_loop = next(node for node in ast.walk(evaluate) if isinstance(node, ast.For)
                       and isinstance(node.target, ast.Name) and node.target.id == "i")
        processor = self.processor("preserve", signal_normalization=dict(method="minmax", min_seconds=0, max_seconds=2))
        center, scale = signal_affine_parameters(processor.scaler)
        context = dict(torch=torch, preprocessor=processor, signal_center=center, signal_scale=scale,
            batch=torch.tensor([[[-1.], [0.], [1.]]]), estimate=torch.tensor([[[0., 0., 0.]]]),
            predicted=torch.zeros(1, 1, 3), target=torch.ones(1, 1, 3), observed_batch=None,
            criterion=torch.nn.MSELoss(), rows=[], identities=[{"subject_id":"synthetic"}], offset=0,
            timesteps=torch.tensor([0]), alpha_bars=torch.tensor([.9]), state_name="trained", scenario="test",
            _teacher_forced_window_metrics=audit._teacher_forced_window_metrics)
        execute_node(row_loop, context)
        row = context["rows"][0]
        self.assertEqual(row["reference_mean_ms"].item(), 1000.)
        self.assertEqual(row["reconstructed_mean_ms"].item(), 1000.)
        self.assertEqual(row["epsilon_mse"].item(), 1.)

    def test_minmax_run_cannot_omit_or_change_normalization_source_identity(self):
        processor = self.processor("preserve", signal_normalization=dict(method="minmax", min_seconds=0, max_seconds=2))
        manifest = dict(training_args=dict(audit.DEFAULT_TRAINING_ARGS, dataset="DREAMT",
            window_size=750, s4_lmax=750), data_identity=processor.data_identity(),
            source_sha256={"TSImputers/S4Model.py": audit.sha256(ROOT / "TSImputers/S4Model.py")})
        path = self.folder / "run_manifest.json"
        for digest in (None, "0"*64):
            if digest is not None:
                manifest["source_sha256"]["signal_normalization.py"] = digest
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "normalization source identity is missing or differs"):
                audit.load_training_run(path)
        manifest["source_sha256"]["signal_normalization.py"] = audit.sha256(ROOT / "signal_normalization.py")
        path.write_text(json.dumps(manifest), encoding="utf-8")
        _, _, info = audit.load_training_run(path)
        self.assertTrue(info["signal_normalization_source_identity"]["matched"])
        # A true legacy z-score manifest remains supported without this new file.
        manifest["data_identity"] = self.processor("preserve").data_identity()
        manifest["source_sha256"].pop("signal_normalization.py")
        path.write_text(json.dumps(manifest), encoding="utf-8")
        _, _, legacy = audit.load_training_run(path)
        self.assertFalse(legacy["signal_normalization_source_identity"]["required"])

    def test_minmax_controlled_diagnostics_block_missing_helper_or_invalid_affine_metadata(self):
        from diagnostic_identity import audit_training_identity, DATA_SOURCES, REQUIRED_CORE_SOURCES
        processor = self.processor("preserve", signal_normalization=dict(method="minmax", min_seconds=0, max_seconds=2))
        identity = processor.data_identity()
        arguments = dict(audit.DEFAULT_TRAINING_ARGS, dataset="DREAMT", window_size=750, s4_lmax=750,
                         diffusion_schedule="linear", terminal_alpha_bar=1e-8)
        sources = {name: audit.sha256(ROOT / name) for name in DATA_SOURCES + REQUIRED_CORE_SOURCES}
        checkpoint_hash = "a"*64
        manifest = dict(training_args=arguments, data_identity=identity, source_sha256=dict(sources),
            validation=dict(best_checkpoints=dict(kl=dict(checkpoint_sha256=checkpoint_hash, epoch=1))))
        report = audit_training_identity(manifest, arguments, identity, sources, checkpoint_hash)
        self.assertEqual(report["status"], "matched")
        manifest["source_sha256"].pop("signal_normalization.py")
        missing = audit_training_identity(manifest, arguments, identity, sources, checkpoint_hash)
        self.assertFalse(missing["can_sample"])
        self.assertTrue(any(row["name"].endswith("signal_normalization.py") and row["status"] == "incomplete"
                            for row in missing["blocking_checks"]))
        manifest["source_sha256"] = sources
        identity["scaler"]["signal_center_seconds"] = 0.
        invalid = audit_training_identity(manifest, arguments, identity, sources, checkpoint_hash)
        self.assertFalse(invalid["can_sample"])
        self.assertTrue(any(row["name"].endswith("scaler_values") and row["status"] == "incomplete"
                            for row in invalid["blocking_checks"]))


if __name__ == "__main__":
    unittest.main()
