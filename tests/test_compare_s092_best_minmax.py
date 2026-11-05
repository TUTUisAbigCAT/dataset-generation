"""Artificial identity fixtures and read-only frozen-worker integration.

The checkpoint restoration regression loads a small S4 state without sampling.
Other tests use artificial metadata and fake checkpoint bytes.
"""
from __future__ import annotations

import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from data_utils import Preprocessor
from diffusion_schedule import diffusion_identity
from scripts.evaluate_denoising import DEFAULT_TRAINING_ARGS
from scripts import compare_s092_best_minmax as driver


ROOT = Path(__file__).resolve().parents[1]
EXTRA_SOURCES = ("missing_data.py", "training_wavestitch.py", "training_input_dropout.py",
                 "scripts/evaluate_denoising.py", "scripts/verify_upstream_boundary.py",
                 "TSImputers/TimeGAN.py", "third_party/WatchSleepNet/data_setup.py")


class CompareBestMinmaxTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = (ROOT / "work/compare_best_minmax_tests").resolve()
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.base = Path(self.temp.name)
        self.data = self.base / "data"
        self.data.mkdir()
        metadata = ["SID,AHI,BMI,AGE,GENDER"]
        for i, name in enumerate(("S001", "S002", "S092")):
            signal = .7 + .12 * i + .02 * np.sin(np.arange(3000) / 31)
            signal[[0, 749, 1500, 2999]] = 0
            np.savez(self.data / (name + ".npz"), data=signal.astype(np.float32)[:, None],
                     stages=np.repeat([0, 1, 4, 3], 750), fs=np.array(25), ahi=np.array(10.+i))
            metadata.append(f"{name},{10+i},{24+i},{40+i},{'F' if i%2==0 else 'M'}")
        self.metadata = self.data / "participant_info.csv"
        self.metadata.write_text("\n".join(metadata)+"\n", encoding="utf-8")
        driver.save_json(self.data / "splits.json", dict(train=["S001.npz"], val=["S002.npz"], test=["S092.npz"]))
        self.args = dict(DEFAULT_TRAINING_ARGS, dataset="DREAMT", batch_size=8, epochs=30,
                         stride=750, diffusion_schedule="linear_power", terminal_alpha_bar=1e-8)
        self.roots, self.manifests = {}, {}
        for arm, epoch in (("minmax", 27), ("minmax01", 29)):
            root = self.base / arm
            workspace, training = root / "run/workspace", root / "run/training"
            workspace.mkdir(parents=True)
            training.mkdir()
            norm = dict(method="minmax", min_seconds=0., max_seconds=2.)
            if arm == "minmax01":
                norm["output_range"] = [0., 1.]
            config = dict(data_dir=str(self.data), metadata_path=str(self.metadata),
                          split_path=str(self.data / "splits.json"), generate_splits=False,
                          window_epochs=4, signal_normalization=norm, missing_value_policy="preserve")
            preprocessor = Preprocessor("DREAMT", data_config=config)
            sources = {}
            for name in sorted(set(driver.REQUIRED_SOURCES) | set(EXTRA_SOURCES)):
                target = workspace / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
                sources[name] = driver.sha256(target)
            driver.save_json(workspace / "dataset_config.json", config)
            sources["dataset_config.json"] = driver.sha256(workspace / "dataset_config.json")
            checkpoint = training / "best_kl_model.pth"
            checkpoint.write_bytes(("artificial identity bytes, not model weights: " + arm).encode())
            selection = dict(epoch=epoch, metric="gaussian_kl_real_to_generated",
                             selection_rule="smoothed_kl", value=.5, smoothing_window=5,
                             window_start_epoch=epoch-4)
            manifest = dict(status="completed", training_args=self.args,
                checkpoint="/recorded/cluster/best_kl_model.pth", checkpoint_sha256=driver.sha256(checkpoint),
                checkpoint_epoch=epoch, completed_epochs=30, updates=192270, batches_per_epoch=6409,
                checkpoint_selection=selection,
                validation=dict(best_checkpoints=dict(kl=dict(selection, checkpoint_sha256=driver.sha256(checkpoint)))),
                source_sha256=sources, data_identity=preprocessor.data_identity(),
                diffusion_identity=diffusion_identity(SimpleNamespace(**self.args)))
            if arm == "minmax01":
                manifest["training_input_dropout"] = dict(options=dict(probability=0., seed=42), active=False,
                    application_calls=0, eligible_values=0, dropped_values=0, observed_fraction=0.)
            driver.save_json(training / "best_kl_manifest.json", manifest)
            # A final model must never be selected or even parsed by this task.
            (training / "run_manifest.json").write_text("deliberately invalid final manifest", encoding="utf-8")
            self.roots[arm], self.manifests[arm] = root, manifest
        # Inference must succeed without reopening any train or val NPZ.
        (self.data / "S001.npz").unlink()
        (self.data / "S002.npz").unlink()

    def tearDown(self):
        target = Path(self.temp.name).resolve()
        if not target.is_relative_to(self.temp_root) or target == self.temp_root:
            raise RuntimeError("Unsafe test cleanup path")
        self.temp.cleanup()

    def context(self, arm="minmax"):
        return driver.load_arm(self.roots[arm], arm)

    def write_manifest(self, arm, manifest):
        driver.save_json(self.roots[arm] / "run/training/best_kl_manifest.json", manifest)

    def test_best_not_final_and_different_best_epochs_are_allowed(self):
        contexts = {arm: self.context(arm) for arm in driver.ARMS}
        self.assertEqual([contexts[a]["epoch"] for a in driver.ARMS], [27, 29])
        self.assertTrue(all(driver.compare_arms(contexts).values()))
        for context in contexts.values():
            self.assertEqual(Path(context["checkpoint"]).name, "best_kl_model.pth")
            self.assertEqual(context["checkpoint_sha256"], context["manifest"]["checkpoint_sha256"])
        self.assertNotEqual(contexts["minmax"]["identity"]["scaler"]["signal_center_seconds"],
                            contexts["minmax01"]["identity"]["scaler"]["signal_center_seconds"])

    def test_missing_or_changed_best_never_falls_back_to_final(self):
        path = self.roots["minmax"] / "run/training/best_kl_model.pth"
        original = path.read_bytes()
        path.write_bytes(b"different weights")
        with self.assertRaisesRegex(ValueError, "checkpoint SHA256"):
            self.context()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.context()
        path.write_bytes(original)

    def test_inconsistent_best_selection_rejected(self):
        for change in (dict(checkpoint_epoch=30), dict(status="running"),
                       dict(checkpoint_selection=dict(self.manifests["minmax"]["checkpoint_selection"], epoch=28))):
            with self.subTest(change=change):
                self.write_manifest("minmax", dict(self.manifests["minmax"], **change))
                with self.assertRaisesRegex(ValueError, "selected best KL"):
                    self.context()

    def test_source_mutation_escape_missing_and_conflicting_alias_rejected(self):
        original = self.manifests["minmax"]
        variants = []
        for name, digest in (("training_utils.py", "0" * 64), ("../outside.py", "0" * 64),
                             ("TSImputers\\S4Model.py", "0" * 64)):
            bad = deepcopy(original)
            bad["source_sha256"][name] = digest
            variants.append(bad)
        missing = deepcopy(original)
        del missing["source_sha256"]["diffusion_schedule.py"]
        variants.append(missing)
        for bad in variants:
            with self.subTest(keys=bad["source_sha256"]), self.assertRaises(ValueError):
                self.write_manifest("minmax", bad)
                self.context()

    def test_active_dropout_or_nonzero_counters_rejected(self):
        base = self.manifests["minmax01"]
        for changes in (dict(options=dict(probability=.1, seed=42)), dict(active=True),
                        dict(application_calls=1), dict(eligible_values=10), dict(dropped_values=1)):
            bad = deepcopy(base)
            bad["training_input_dropout"].update(changes)
            self.write_manifest("minmax01", bad)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "dropout"):
                self.context("minmax01")

    def test_subject_leakage_wrong_norm_and_zero_replacement_rejected(self):
        original = self.manifests["minmax"]
        variants = []
        bad = deepcopy(original)
        bad["data_identity"]["splits"]["train"].append("S092.npz")
        variants.append(bad)
        bad = deepcopy(original)
        bad["data_identity"]["scaler_subjects"].append("S092")
        variants.append(bad)
        bad = deepcopy(original)
        bad["data_identity"]["missing_value_policy"] = "zero_fill_one"
        variants.append(bad)
        bad = deepcopy(original)
        bad["data_identity"]["scaler"]["signal_normalization"]["output_range"] = [0., 1.]
        variants.append(bad)
        for bad in variants:
            with self.subTest(identity=bad["data_identity"]), self.assertRaises(ValueError):
                self.write_manifest("minmax", bad)
                self.context()

    def test_comparison_rejects_changed_data_conditions_schedule_and_core(self):
        contexts = {arm: self.context(arm) for arm in driver.ARMS}
        mutations = (
            lambda c: c["identity"].update(split_fingerprint="changed"),
            lambda c: c["identity"]["scaler"]["continuous_mean"].__setitem__(0, 999.),
            lambda c: c["manifest"]["diffusion_identity"].update(terminal_alpha_bar=1e-6),
            lambda c: c["identity"]["columns"].reverse(),
            lambda c: c["source_checks"].update({"TSImputers/S4Model.py": "0" * 64}),
            lambda c: c["manifest"]["training_input_dropout"]["options"].update(probability=.1),
        )
        for mutate in mutations:
            bad = deepcopy(contexts)
            mutate(bad["minmax01"])
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                driver.compare_arms(bad)

    def run_preflight(self, arm, *, start=0, expected_sha=None, output=None,
                      windows=1, require_all_stages=False):
        output = output or self.base / ("preflight_" + arm)
        output.mkdir()
        # The production driver deliberately pins the audited real S092 hash.
        # Only this fresh test process substitutes our artificial fixture hash.
        code = """import pathlib,sys
from scripts import compare_s092_best_minmax as d
d.S092_SHA256=sys.argv[1]
c=d.load_arm(pathlib.Path(sys.argv[2]),sys.argv[3])
d.worker_preflight(c,pathlib.Path(sys.argv[4]),pathlib.Path(sys.argv[5]),int(sys.argv[6]),pathlib.Path(sys.argv[7]),int(sys.argv[8]),bool(int(sys.argv[9])))
"""
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        result = subprocess.run([sys.executable, "-B", "-c", code,
            expected_sha or driver.sha256(self.data / "S092.npz"), str(self.roots[arm]), arm,
            str(self.data / "S092.npz"), str(self.metadata), str(start), str(output),
            str(windows), str(int(require_all_stages))],
            cwd=ROOT, env=environment, capture_output=True, text=True)
        return result, output

    def test_real_frozen_worker_preflight_needs_only_test_npz_and_csv(self):
        before = {p: driver.sha256(p) for p in self.base.rglob("*") if p.is_file()}
        outputs = self.base / "paired"
        outputs.mkdir()
        for arm in driver.ARMS:
            result, output = self.run_preflight(arm, output=outputs / arm)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = driver.read_json(output / "preflight.json")
            self.assertFalse(report["training_npz_reopened"])
            self.assertFalse(report["scaler_refitted"])
            self.assertFalse(report["sampling_performed"])
            self.assertEqual(report["raw_zero_samples"], 4)
            self.assertEqual(report["stages"], [0, 1, 2, 3])
            self.assertEqual((report["signal_center_seconds"], report["signal_scale_seconds"]),
                             (1., 1.) if arm == "minmax" else (0., 2.))
            imports = report["imported_sources"]
            self.assertIn("night_checkpoint", imports)
            for name, record in imports.items():
                if name in ("night_data", "night_checkpoint"):
                    self.assertEqual(Path(record["path"]), ROOT / (name + ".py"))
                    self.assertEqual(record["sha256"], driver.sha256(record["path"]))
                else:
                    self.assertTrue(Path(record["path"]).is_relative_to(self.roots[arm] / "run/workspace"))
            with np.load(output / "inputs.npz", allow_pickle=False) as values:
                self.assertTrue(np.array_equal(values["scoring_mask"], values["reference_ibi_seconds"] > 0))
                self.assertEqual(values["conditions"].shape, (3000, 9))
        self.assertTrue(all(driver.matching_inputs(outputs).values()))
        self.assertTrue(all(driver.sha256(path) == digest for path, digest in before.items()))

    def test_worker_restores_saved_s4_kernel_buffers_before_sampling(self):
        import torch
        import night_checkpoint
        from tests.test_night_checkpoint import small_s4, snapshot
        from scripts import evaluate_original_sampler

        self.addCleanup(np.random.set_state, np.random.get_state())
        with torch.random.fork_rng(devices=[]):
            target = small_s4()
            saved = snapshot(target)
            kernel_buffers = {name for name, _ in target.named_buffers()
                              if name.rsplit('.', 1)[-1] in {'B', 'P', 'w', 'Q'}}
            self.assertEqual({name.rsplit('.', 1)[-1] for name in kernel_buffers}, {'B', 'P', 'w', 'Q'})
            for name in kernel_buffers:
                saved[name].add_(1)
            with self.assertRaisesRegex(RuntimeError, 'Seed-42 reconstructed buffers differ'):
                evaluate_original_sampler.load_original_parameters(target, saved, torch)

            checkpoint = self.base / 'kernel_state.pth'
            torch.save(saved, checkpoint)
            context = dict(checkpoint=str(checkpoint), checkpoint_sha256=driver.sha256(checkpoint),
                           best_manifest_sha256='a' * 64)
            inputs = self.base / 'inputs.npz'
            np.savez(inputs, fixture=np.array(1))
            driver.save_json(self.base / 'preflight.json', dict(context, inputs_sha256=driver.sha256(inputs)))
            modules = {'training_utils': SimpleNamespace(fetchModel=lambda *args: target),
                       'cuda_runtime': SimpleNamespace(assert_cuda_model=lambda model: None),
                       'scripts.evaluate_original_sampler': evaluate_original_sampler,
                       'night_checkpoint': night_checkpoint}
            # Stop at eval(), immediately after state loading, before any model forward.
            with (mock.patch.object(driver, 'frozen_imports', return_value=(modules, None, {})),
                  mock.patch.object(driver, 'require_sampling_device', return_value=torch.device('cpu')),
                  mock.patch.object(target, 'eval', side_effect=StopIteration('restored')),
                  self.assertRaisesRegex(StopIteration, 'restored')):
                driver.worker_sample(context, self.base, 20261103, torch.get_num_threads())
            for name, value in target.state_dict().items():
                self.assertTrue(torch.equal(value, saved[name]), name)

    def test_three_windows_keep_model_length_and_require_all_stages(self):
        path = self.data / "S092.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {k: source[k].copy() for k in source.files}
        arrays['data'] = np.tile(arrays['data'], (3, 1))
        arrays['stages'] = np.tile(arrays['stages'], 3)
        np.savez(path, **arrays)
        result, output = self.run_preflight("minmax", windows=3, require_all_stages=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = driver.read_json(output / 'preflight.json')
        self.assertEqual(report['window_size'], 3000)
        self.assertEqual(report['model_args']['window_size'], 3000)
        self.assertEqual(report['total_samples'], 9000)
        self.assertEqual(report['duration_seconds'], 360)
        self.assertEqual(report['windows'], 3)
        self.assertTrue(report['all_four_stages_present'])
        with np.load(output / 'inputs.npz', allow_pickle=False) as inputs:
            ref, cond, mask = driver.sampling_arrays(inputs, 3000)
            np.testing.assert_array_equal(ref.reshape(-1), inputs['reference_ibi_seconds'])
            np.testing.assert_array_equal(cond.reshape(-1, 9), inputs['conditions'])
            np.testing.assert_array_equal(mask.reshape(-1), inputs['scoring_mask'])
            self.assertEqual(ref.shape, (3, 3000))
        arrays['stages'][arrays['stages'] == 3] = 1
        np.savez(path, **arrays)
        failed, _ = self.run_preflight("minmax", windows=3, require_all_stages=True,
                                      output=self.base / 'missing_deep')
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn('must contain W, Light, REM and Deep', failed.stderr)

    def test_partial_or_mismatched_sampling_arrays_rejected(self):
        valid = dict(reference_ibi_seconds=np.zeros(9000), conditions=np.zeros((9000, 9)),
                     scoring_mask=np.ones(9000, dtype=bool))
        for key, wrong in [('reference_ibi_seconds', np.zeros(8999)),
                           ('conditions', np.zeros((9000, 10))),
                           ('scoring_mask', np.ones(9000))]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'complete matching model windows'):
                driver.sampling_arrays(dict(valid, **{key: wrong}), 3000)

    def test_worker_rejects_changed_npz_metadata_and_invalid_window(self):
        result, _ = self.run_preflight("minmax", expected_sha="0" * 64, output=self.base / "bad_npz")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NPZ differs", result.stderr)
        result, _ = self.run_preflight("minmax", start=30, output=self.base / "too_long")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exceeds the record", result.stderr)
        result, _ = self.run_preflight("minmax", start=1, output=self.base / "unaligned")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("multiple of 30", result.stderr)
        self.metadata.write_text(self.metadata.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        result, _ = self.run_preflight("minmax", output=self.base / "bad_metadata")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("metadata SHA256 differs", result.stderr)

    def test_same_process_import_contamination_is_rejected(self):
        with mock.patch.dict(sys.modules, {"training_utils": SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError, "fresh worker"):
                driver.frozen_imports(self.context())

    def test_cuda_requirement_cannot_return_cpu(self):
        runtime = SimpleNamespace(require_cuda=lambda: SimpleNamespace(type="cpu"))
        with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
            driver.require_sampling_device(runtime)
        cuda = SimpleNamespace(type="cuda", index=0)
        self.assertIs(driver.require_sampling_device(SimpleNamespace(require_cuda=lambda: cuda)), cuda)

    def test_pairing_rejects_condition_or_mask_differences(self):
        output = self.base / "pairing"
        for arm in driver.ARMS:
            (output / arm).mkdir(parents=True)
        arrays = dict(reference_ibi_seconds=np.array([0., 1.]), scoring_mask=np.array([False, True]),
                      conditions=np.zeros((2, 9)), sleep_stages=np.array([0]))
        np.savez(output / "minmax/inputs.npz", **arrays)
        for key in ("conditions", "scoring_mask"):
            changed = deepcopy(arrays)
            changed[key].flat[0] = 1
            np.savez(output / "minmax01/inputs.npz", **changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Paired S092"):
                driver.matching_inputs(output)
        self.assertFalse((output / "reference.npz").exists())

    def test_check_only_orchestration_never_launches_sampling(self):
        output = self.base / "check_only"
        phases = []
        def fake_launch(cli, context, phase, destination):
            phases.append((context["arm"], phase))
            self.assertEqual(phase, "preflight")
            np.savez(destination / "inputs.npz", reference_ibi_seconds=np.array([0., 1.]),
                     scoring_mask=np.array([False, True]), conditions=np.zeros((2, 9)))
        with mock.patch.object(driver, "launch_worker", side_effect=fake_launch), contextlib.redirect_stdout(io.StringIO()):
            driver.main(["--minmax-run", str(self.roots["minmax"]),
                "--minmax01-run", str(self.roots["minmax01"]), "--output-dir", str(output), "--check-only"])
        self.assertEqual(phases, [(a, "preflight") for a in driver.ARMS])
        report = driver.read_json(output / "summary.json")
        self.assertEqual(report["status"], "checked")
        self.assertFalse(report["sampling_performed"])
        self.assertFalse(report["training_executed"])

    def test_output_cannot_be_inside_training_run(self):
        with self.assertRaisesRegex(ValueError, "outside the read-only"):
            driver.main(["--minmax-run", str(self.roots["minmax"]),
                "--minmax01-run", str(self.roots["minmax01"]),
                "--output-dir", str(self.roots["minmax"] / "new_output"), "--check-only"])


if __name__ == "__main__":
    unittest.main()
