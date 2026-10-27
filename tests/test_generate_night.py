"""Whole-night CLI contracts using artificial records and mocked CUDA execution.

No test here is a real S092 inference or evidence of generated signal quality.
The private sampling-boundary test uses a tiny CPU fixture model with explicitly
mocked CUDA guards; production entry points retain their CUDA-only checks.
"""
from __future__ import annotations

import contextlib
import csv
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
from scripts.evaluate_denoising import DEFAULT_TRAINING_ARGS, sha256
from watchsleep_loader import SSDataset


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
        self.checkpoint = self.directory / "digest_only_fixture.bin"
        self.checkpoint.write_bytes(b"Artificial checkpoint digest fixture, not model weights.")
        self.manifest_path = self.directory / "run_manifest.json"
        self.write_manifest()

    def write_manifest(self, window_epochs=4, threads=2):
        self.processor = Preprocessor("DREAMT", data_config=dict(data_dir=str(self.records),
            window_epochs=window_epochs, max_epochs=4))
        manifest = dict(training_args=dict(DEFAULT_TRAINING_ARGS, dataset="DREAMT",
            window_size=750 * window_epochs, s4_lmax=max(3000, 750 * window_epochs), stride=750),
            data_identity=self.processor.data_identity(), threads=threads,
            source_sha256={"TSImputers/S4Model.py": sha256(ROOT / "TSImputers/S4Model.py")},
            checkpoint=str(self.checkpoint), checkpoint_sha256=sha256(self.checkpoint),
            test_fixture_not_a_training_result=True)
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def arguments(self, output="output", *, check_only=False):
        argv = ["--run-manifest", str(self.manifest_path), "--output-dir", str(self.directory / output)]
        return argv + (["--check-only"] if check_only else [])

    def invoke(self, argv):
        with contextlib.redirect_stdout(io.StringIO()):
            return generate_night.main(argv)

    def fake_sample(self, args, manifest, checkpoint, conditions, scaler, device, **options):
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
        return standardized, generated, {"test_fixture": True}, {"test_fixture": True}

    def test_check_only_full_stored_night_uses_frozen_dimensions_without_cuda_or_model(self):
        for epochs in (1, 2, 4, 8):
            with self.subTest(window_epochs=epochs):
                self.write_manifest(window_epochs=epochs)
                argv = self.arguments(f"checked_{epochs}", check_only=True)
                argv[0] = "--training-args-json"
                argv += ["--subject", "s092.NPZ"]
                with mock.patch("cuda_runtime.require_cuda", side_effect=AssertionError("No GPU in check-only")), \
                     mock.patch.object(generate_night, "_sample", side_effect=AssertionError("No model in check-only")), \
                     mock.patch.dict(os.environ, {"DREAMT_DATA_DIR": "missing-current-folder"}):
                    report = self.invoke(argv)
                self.assertEqual(report["status"], "checked_only")
                self.assertEqual(report["subject_id"], "S092")
                self.assertEqual(report["target_samples"], self.target_epochs * 750)
                self.assertEqual(report["window_samples"], epochs * 750)
                self.assertEqual(report["data"]["training_max_epochs"], 4)
                self.assertIsNone(report["data"]["generation_max_epochs"])
                self.assertEqual(report["data"]["subject_role"], "test")
                self.assertFalse(report["model_constructed"])
                self.assertFalse(report["sampling_performed"])
                self.assertFalse(report["real_DREAMT_quality_verified"])
                self.assertEqual(report["source_provenance"]["night_sampler"], "wavestitch_grouped_hard_prefix")
                self.assertEqual(report["source_provenance"]["night_adaptations"],
                                 ["project_known_prefix_each_step", "replace_soft_self_loss_with_hard_projection"])
                self.assertEqual(len(report["source_provenance"]["night_sampler_sha256"]), 64)
                self.assertEqual(sum(row["append_samples"] for row in report["generation_plan"]), self.target_epochs * 750)
                self.assertFalse((self.directory / f"checked_{epochs}" / "S092.npz").exists())

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

    def test_no_cuda_fails_before_reading_manifest_and_never_falls_back(self):
        argv = self.arguments("no_cuda")
        with mock.patch("cuda_runtime.require_cuda", side_effect=RuntimeError("CUDA required")), \
             mock.patch.object(generate_night, "load_training_run") as load, \
             mock.patch.object(generate_night, "_sample") as sample:
            with self.assertRaisesRegex(RuntimeError, "CUDA required"):
                self.invoke(argv)
        load.assert_not_called()
        sample.assert_not_called()
        report = json.loads((self.directory / "no_cuda" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")

    def test_check_only_still_verifies_checkpoint_digest(self):
        self.checkpoint.write_bytes(b"Changed fixture bytes")
        with mock.patch.object(generate_night, "_sample") as sample:
            with self.assertRaisesRegex(ValueError, "checkpoint digest"):
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

    def test_recorded_constructor_threads_are_validated_and_legacy_default_is_explicit(self):
        self.assertEqual(generate_night.initialization_threads({}), 2)
        self.assertEqual(generate_night.initialization_threads({"threads": 8}), 8)
        for value in (0, -1, 2.0, True, None, "2"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Manifest threads"):
                generate_night.initialization_threads({"threads": value})


class SamplingInitializationTests(unittest.TestCase):
    def test_sampling_seed_is_separate_from_constructor_seed_and_fixed_buffers(self):
        from training_utils import fetchModel  # Import before patching its factory.

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
        torch.manual_seed(42)
        saved = TinyFixture().state_dict()
        constructed = []

        def construct(*_):
            constructed.append((torch.get_num_threads(), torch.initial_seed()))
            return TinyFixture()

        def sample(model, conditions, schedule, args, device, **kwargs):
            self.assertEqual(conditions.shape, (750, 9))
            self.assertEqual(torch.get_num_threads(), 1)
            self.assertFalse(model.training)
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
            self.assertTrue(torch.equal(model.fixed, saved["fixed"]))
            return torch.randn(750), {"test_fixture": True}

        outputs = []
        args = SimpleNamespace(**DEFAULT_TRAINING_ARGS)
        with mock.patch("training_utils.fetchModel", side_effect=construct), \
             mock.patch("torch.load", return_value=saved), \
             mock.patch("cuda_runtime.assert_cuda_model"), \
             mock.patch("cuda_runtime.cuda_environment", return_value={"test_fixture_cpu": True}), \
             mock.patch("torch.cuda.synchronize"), \
             mock.patch("night_sampling.generate_night", side_effect=sample):
            for seed in (19, 19, 21):
                values, seconds, metadata, execution = generate_night._sample(args, {"threads": 2},
                    Path("tiny_fixture.pt"), np.ones((750, 9), dtype=np.float32),
                    {"signal_mean_seconds": .8, "signal_std_seconds": .05}, torch.device("cpu"),
                    seed=seed, threads=1, stride=750, windows_per_group=4, progress=lambda _: None)
                outputs.append(values)
                np.testing.assert_allclose(seconds, values.astype(np.float64) * .05 + .8)
                self.assertEqual(execution["initialization_seed"], 42)
                self.assertEqual(execution["initialization_threads"], 2)
                self.assertEqual(execution["sampling_seed"], seed)
                self.assertTrue(execution["checkpoint_load"]["all_buffers_bitwise_equal"])
        self.assertEqual(constructed, [(2, 42)] * 3)
        np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertFalse(np.array_equal(outputs[0], outputs[2]))


if __name__ == "__main__":
    unittest.main()
