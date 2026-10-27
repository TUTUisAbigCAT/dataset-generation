"""Whole-night CLI contracts using artificial records and mocked CUDA execution.

No test here is a real S092 inference or evidence of generated signal quality.
The private sampling-boundary test uses a tiny CPU fixture model with explicitly
mocked CUDA guards; production entry points retain their CUDA-only checks.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
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
from scripts import generate_night
from watchsleep_loader import SSDataset


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NightCommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="night_cli_", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.records = self.directory / "records"
        self.records.mkdir()
        names, rows = [], []
        self.target_epochs = 11
        for index, epochs in enumerate((8, 9, self.target_epochs)):
            sid = f"S{90 + index:03d}"
            names.append(sid + ".npz")
            samples = np.arange(epochs * 750)
            raw_stages = np.resize(np.array([0, 1, 2, 4, 3], dtype=np.int8), epochs)
            signal = (.78 + index * .03 + .04 * np.sin(samples / 43.)).astype(np.float32)
            np.savez(self.records / names[-1], data=signal[:, None],
                     stages=np.repeat(raw_stages, 750), fs=np.array(25),
                     ahi=np.array(10. + index), synthetic=np.array(True))
            rows.append(dict(SID=sid, AHI=10. + index, BMI=24. + index, AGE=40 + index,
                             GENDER="F" if index % 2 == 0 else "M", NOTE="artificial fixture"))
        with (self.records / "participant_info.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (self.records / "splits.json").write_text(json.dumps(
            dict(train=names[:1], val=names[1:2], test=names[2:])), encoding="utf-8")
        self.checkpoint = self.directory / "best_model.pth"
        self.checkpoint.write_bytes(b"Artificial checkpoint digest fixture, not model weights.")
        self.config_path = self.directory / "dataset_config.json"
        self.write_config()

    def write_config(self, window_epochs=4):
        self.processor = Preprocessor("DREAMT", data_config=dict(data_dir=str(self.records),
            window_epochs=window_epochs, max_epochs=4))
        config = dict(data_dir="records", metadata_path="records/participant_info.csv",
                      split_path="records/splits.json", window_epochs=window_epochs, max_epochs=4,
                      diffusion_schedule="linear_power", terminal_alpha_bar=1e-8)
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def arguments(self, output="output", *, check_only=False):
        argv = ["--checkpoint", str(self.checkpoint), "--config", str(self.config_path),
                "--output-dir", str(self.directory / output)]
        return argv + (["--check-only"] if check_only else [])

    def invoke(self, argv):
        with contextlib.redirect_stdout(io.StringIO()):
            return generate_night.main(argv)

    def fake_sample(self, args, checkpoint, conditions, scaler, device, **options):
        self.assertEqual(checkpoint, self.checkpoint.resolve())
        self.assertEqual(conditions.shape, (self.target_epochs * 750, 9))
        self.assertEqual(conditions.dtype, np.float32)
        self.assertTrue(np.isfinite(conditions).all())
        self.assertEqual(device, torch.device("cuda:0"))
        self.assertEqual(options["seed"], 123)
        self.assertEqual(options["threads"], 4)
        self.assertEqual(options["stride"], 750)
        self.assertEqual(options["windows_per_group"], 4)
        options["progress"](dict(group=0, groups=3, fixture=True))
        generated = np.linspace(.6, .9, len(conditions), dtype=np.float64)
        standardized = (generated - scaler["signal_mean_seconds"]) / scaler["signal_std_seconds"]
        return standardized, generated, {"test_fixture": True}, {
            "test_fixture": True, "loaded_checkpoint_sha256": sha256(checkpoint),
            "checkpoint_load": {"all_state_bitwise_equal": True,
                                "parameters_and_buffers_restored": True,
                                "training_source_identity_verified": False}}

    def test_check_only_full_stored_night_uses_configured_dimensions_without_cuda_or_model(self):
        for epochs in (1, 2, 4, 8):
            with self.subTest(window_epochs=epochs):
                self.write_config(window_epochs=epochs)
                argv = self.arguments(f"checked_{epochs}", check_only=True)
                argv += ["--subject", "s092.NPZ"]
                with mock.patch("cuda_runtime.require_cuda", side_effect=AssertionError("No GPU in check-only")), \
                     mock.patch.object(generate_night, "_sample", side_effect=AssertionError("No model in check-only")), \
                     mock.patch("torch.load", side_effect=AssertionError("No weights loaded in check-only")), \
                     mock.patch.dict(os.environ, {"DREAMT_DATA_DIR": "missing-current-folder"}):
                    report = self.invoke(argv)
                self.assertEqual(report["status"], "checked_only")
                self.assertEqual(report["schema"], "dreamt_whole_night_generation_v2")
                self.assertEqual(report["input_mode"], "direct_checkpoint")
                self.assertFalse(report["training_manifest_required"])
                self.assertEqual(report["subject_id"], "S092")
                self.assertEqual(report["target_samples"], self.target_epochs * 750)
                self.assertEqual(report["window_samples"], epochs * 750)
                self.assertEqual(report["data"]["training_max_epochs"], 4)
                self.assertIsNone(report["data"]["generation_max_epochs"])
                self.assertEqual(report["data"]["subject_role"], "test")
                self.assertFalse(report["model_constructed"])
                self.assertFalse(report["model_state_validated"])
                self.assertFalse(report["historical_training_identity_verified"])
                self.assertFalse(report["sampling_performed"])
                self.assertFalse(report["real_DREAMT_quality_verified"])
                self.assertEqual(report["source_provenance"]["night_sampler"], "wavestitch_grouped_hard_prefix")
                self.assertEqual(report["source_provenance"]["night_adaptations"],
                                 ["project_known_prefix_each_step", "replace_soft_self_loss_with_hard_projection"])
                self.assertEqual(len(report["source_provenance"]["night_sampler_sha256"]), 64)
                self.assertEqual(sum(row["append_samples"] for row in report["generation_plan"]), self.target_epochs * 750)
                self.assertFalse((self.directory / f"checked_{epochs}" / "S092.npz").exists())

    def test_config_relative_paths_resolve_from_its_directory_not_process_cwd(self):
        other = self.directory / "unrelated_cwd"
        other.mkdir()
        previous = Path.cwd()
        try:
            os.chdir(other)
            report = self.invoke(self.arguments(check_only=True))
        finally:
            os.chdir(previous)
        self.assertEqual(report["status"], "checked_only")
        self.assertEqual(report["subject_id"], "S092")
        self.assertEqual(report["target_samples"], 8250)

    def test_corrupt_neighboring_manifests_are_completely_ignored(self):
        for name in ("run_manifest.json", "best_kl_manifest.json"):
            (self.directory / name).write_bytes(b"{deliberately corrupt manifest")
        original_open = Path.open
        opened = []

        def forbid_manifest_open(path, *args, **kwargs):
            opened.append(path)
            if path.name in ("run_manifest.json", "best_kl_manifest.json"):
                raise AssertionError("Direct checkpoint generation must not read a manifest")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", autospec=True, side_effect=forbid_manifest_open):
            report = self.invoke(self.arguments(check_only=True))
        self.assertEqual(report["status"], "checked_only")
        self.assertTrue(opened)
        self.assertFalse(report["historical_training_identity_verified"])

    def test_generation_writes_loader_compatible_night_and_exact_participant_row(self):
        argv = self.arguments() + ["--subject", "s092", "--seed", "123"]
        with mock.patch("cuda_runtime.require_cuda", return_value=torch.device("cuda:0")), \
             mock.patch.object(generate_night, "_sample", side_effect=self.fake_sample) as sample:
            report = self.invoke(argv)
        self.assertEqual(sample.call_count, 1)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["data"]["coverage"], "all_complete_stored_epochs")
        self.assertFalse(report["reference_ibi_passed_to_sampler"])
        self.assertTrue(report["source_record_synthetic"])
        self.assertFalse(report["real_DREAMT_quality_verified"])
        output = self.directory / "output"
        loaded = SSDataset(output, "dreamt_ibi", file_list=["S092.npz"], max_epochs=None).read_record(0)
        self.assertEqual(loaded["stored_complete_epochs"], self.target_epochs)
        self.assertTrue(loaded["valid_epochs"].all())
        self.assertTrue(loaded["synthetic"])
        np.testing.assert_allclose(loaded["ibi_epochs"].reshape(-1),
                                   np.linspace(.6, .9, self.target_epochs * 750), rtol=1e-7)
        with np.load(self.records / "S092.npz") as source, np.load(output / "S092.npz") as generated:
            np.testing.assert_array_equal(generated["stages"], source["stages"])
            self.assertNotIn("reference_seconds", generated.files)
            self.assertFalse(np.array_equal(generated["data"], source["data"]))
        with (output / "participant_info.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["SID"], "S092")
        self.assertEqual(rows[0]["NOTE"], "artificial fixture")
        self.assertEqual(report["output_npz_sha256"], sha256(output / "S092.npz"))
        self.assertFalse((output / "diagnostic_generated.npz").exists())

    def test_report_identifies_loaded_snapshot_if_checkpoint_changes_during_sampling(self):
        loaded_hash = sha256(self.checkpoint)

        def sample_while_file_changes(*args, **kwargs):
            result = self.fake_sample(*args, **kwargs)
            self.checkpoint.write_bytes(b"New live checkpoint after the model was loaded")
            return result

        with mock.patch("cuda_runtime.require_cuda", return_value=torch.device("cuda:0")), \
             mock.patch.object(generate_night, "_sample", side_effect=sample_while_file_changes):
            report = self.invoke(self.arguments() + ["--seed", "123"])
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["checkpoint_identity"]["sha256"], loaded_hash)
        self.assertNotEqual(report["checkpoint_identity"]["sha256"], sha256(self.checkpoint))

    def test_invalid_generated_values_are_preserved_as_failed_diagnostics(self):
        for name, value in (("negative", -.5), ("nonfinite", np.nan)):
            with self.subTest(value=name):
                def bad_sample(*args, **kwargs):
                    standardized, generated, sampling, execution = self.fake_sample(*args, **kwargs)
                    generated[7] = value
                    return standardized, generated, sampling, execution

                argv = self.arguments(name) + ["--seed", "123"]
                with mock.patch("cuda_runtime.require_cuda", return_value=torch.device("cuda:0")), \
                     mock.patch.object(generate_night, "_sample", side_effect=bad_sample):
                    with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                        self.invoke(argv)
                output = self.directory / name
                report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["invalid_values_clipped_or_repaired"])
                self.assertFalse((output / "S092.npz").exists())
                with np.load(output / "diagnostic_generated.npz") as saved:
                    if np.isnan(value):
                        self.assertTrue(np.isnan(saved["generated_seconds"][7]))
                    else:
                        self.assertEqual(saved["generated_seconds"][7], value)

    def test_no_cuda_fails_before_loading_config_and_never_falls_back(self):
        argv = self.arguments("no_cuda")
        with mock.patch("cuda_runtime.require_cuda", side_effect=RuntimeError("CUDA required")), \
             mock.patch.object(generate_night, "load_inference_config") as load, \
             mock.patch.object(generate_night, "_sample") as sample:
            with self.assertRaisesRegex(RuntimeError, "CUDA required"):
                self.invoke(argv)
        load.assert_not_called()
        sample.assert_not_called()
        report = json.loads((self.directory / "no_cuda" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")

    def test_check_only_requires_checkpoint_file_but_does_not_invent_a_historical_digest(self):
        # A current file hash is provenance, not a match against a nonexistent
        # training manifest. Check-only deliberately does not deserialize it.
        self.checkpoint.write_bytes(b"Changed fixture bytes")
        report = self.invoke(self.arguments("current_hash", check_only=True))
        self.assertEqual(report["checkpoint_identity"]["sha256"], sha256(self.checkpoint))
        self.assertFalse(report["model_state_validated"])
        self.checkpoint.unlink()
        with mock.patch.object(generate_night, "_sample") as sample:
            with self.assertRaises((FileNotFoundError, ValueError)):
                self.invoke(self.arguments(check_only=True))
        sample.assert_not_called()

    def test_check_only_rejects_an_invalid_grouped_sampler_adapter(self):
        with mock.patch("night_sampling._night_sampler", side_effect=RuntimeError("Night adapter source mismatch")), \
             mock.patch.object(generate_night, "_sample") as sample:
            with self.assertRaisesRegex(RuntimeError, "adapter source mismatch"):
                self.invoke(self.arguments(check_only=True))
        sample.assert_not_called()
        report = json.loads((self.directory / "output" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")

    def test_failure_inside_sampler_records_summary_without_claiming_raw_diagnostics(self):
        with mock.patch("cuda_runtime.require_cuda", return_value=torch.device("cuda:0")), \
             mock.patch.object(generate_night, "_sample", side_effect=FloatingPointError("Nonfinite intermediate")):
            with self.assertRaisesRegex(FloatingPointError, "Nonfinite intermediate"):
                self.invoke(self.arguments())
        output = self.directory / "output"
        report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertNotIn("diagnostic_output", report)
        self.assertFalse((output / "diagnostic_generated.npz").exists())
        self.assertFalse((output / "S092.npz").exists())

    def test_existing_output_directory_is_never_overwritten(self):
        output = self.directory / "output"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.invoke(self.arguments(check_only=True))
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_manifest_cli_aliases_are_removed_and_threads_must_be_positive(self):
        for extra in (["--run-manifest", "unused.json"], ["--training-args-json", "unused.json"],
                      ["--threads", "0"], ["--threads", "-1"], ["--threads", "2.0"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.invoke(self.arguments(check_only=True) + extra)
                self.assertEqual(error.exception.code, 2)


class SamplingInitializationTests(unittest.TestCase):
    def test_full_checkpoint_state_is_restored_before_separately_seeded_sampling(self):
        from training_utils import fetchModel  # Import before patching its factory.
        from night_checkpoint import DEFAULT_TRAINING_ARGS

        class TinyFixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
                self.register_buffer("fixed", torch.rand(7))

        previous_threads = torch.get_num_threads()
        rng = torch.random.get_rng_state()
        numpy_rng = np.random.get_state()
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.addCleanup(torch.random.set_rng_state, rng)
        self.addCleanup(np.random.set_state, numpy_rng)
        torch.set_num_threads(2)
        # Deliberately unlike the seed42 constructor: strict restoration must
        # replace both parameters and buffers, not reconstruct old buffers.
        torch.manual_seed(7)
        saved = TinyFixture().state_dict()
        saved["weight"].fill_(3.5)
        saved["fixed"] = torch.linspace(0.2, 0.8, 7)
        constructed = []
        temporary = tempfile.TemporaryDirectory(prefix="night_sample_", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        checkpoint = Path(temporary.name) / "tiny_fixture.pth"
        checkpoint.write_bytes(b"Mocked state_dict payload snapshot")
        loaded_hash = sha256(checkpoint)

        def load_snapshot(stream, **options):
            self.assertEqual(stream.getvalue(), b"Mocked state_dict payload snapshot")
            checkpoint.write_bytes(b"Concurrent replacement of the on-disk checkpoint")
            return saved

        def construct(*_):
            constructed.append((torch.get_num_threads(), torch.initial_seed()))
            return TinyFixture()

        def sample(model, conditions, schedule, args, device, **kwargs):
            self.assertEqual(conditions.shape, (750, 9))
            self.assertEqual(torch.get_num_threads(), 1)
            self.assertFalse(model.training)
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
            self.assertTrue(torch.equal(model.weight, saved["weight"]))
            self.assertTrue(torch.equal(model.fixed, saved["fixed"]))
            return torch.randn(750), {"test_fixture": True}

        outputs = []
        args = SimpleNamespace(**DEFAULT_TRAINING_ARGS)
        with mock.patch("training_utils.fetchModel", side_effect=construct), \
             mock.patch("torch.load", side_effect=load_snapshot) as load, \
             mock.patch("cuda_runtime.assert_cuda_model"), \
             mock.patch("cuda_runtime.cuda_environment", return_value={"test_fixture_cpu": True}), \
             mock.patch("torch.cuda.synchronize"), \
             mock.patch("night_sampling.generate_night", side_effect=sample):
            for seed in (19, 19, 21):
                checkpoint.write_bytes(b"Mocked state_dict payload snapshot")
                values, seconds, metadata, execution = generate_night._sample(args,
                    checkpoint, np.ones((750, 9), dtype=np.float32),
                    {"signal_mean_seconds": .8, "signal_std_seconds": .05}, torch.device("cpu"),
                    seed=seed, threads=1, stride=750, windows_per_group=4, progress=lambda _: None)
                outputs.append(values)
                np.testing.assert_allclose(seconds, values.astype(np.float64) * .05 + .8)
                self.assertEqual(execution["initialization_seed"], 42)
                self.assertEqual(execution["initialization_threads"], 1)
                self.assertEqual(execution["sampling_seed"], seed)
                self.assertTrue(execution["checkpoint_load"]["all_state_bitwise_equal"])
                self.assertTrue(execution["checkpoint_load"]["parameters_and_buffers_restored"])
                self.assertFalse(execution["checkpoint_load"]["training_source_identity_verified"])
                self.assertEqual(execution["loaded_checkpoint_sha256"], loaded_hash)
                self.assertNotEqual(execution["loaded_checkpoint_sha256"], sha256(checkpoint))
        self.assertEqual(constructed, [(1, 42)] * 3)
        self.assertEqual(load.call_count, 3)
        for call in load.call_args_list:
            self.assertIsInstance(call.args[0], io.BytesIO)
            self.assertEqual(call.args[0].getvalue(), b"Mocked state_dict payload snapshot")
            self.assertEqual(call.kwargs, {"weights_only": True, "map_location": torch.device("cpu")})
        np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertFalse(np.array_equal(outputs[0], outputs[2]))


if __name__ == "__main__":
    unittest.main()
