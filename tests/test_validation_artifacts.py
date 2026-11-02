"""Synthetic export tests; model/CUDA/sampling boundaries are mocked explicitly."""
import ast
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import epoch_validation as coordinator
from validation_artifacts import write_validation_artifacts
from validation_metrics import window_statistics
from tests import test_masked_validation as fixtures
from scripts import run_dreamt


def rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


class ValidationArtifactTests(unittest.TestCase):
    def temporary(self):
        temporary = tempfile.TemporaryDirectory(prefix="validation_artifacts_test_", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def writer_payload(self):
        reference = np.array([[1., 3., 0., np.nan], [2., 4., 6., 0.]], dtype=np.float64)
        generated = np.array([[2., 4., -99., 0.], [3., 5., 7., -5.]], dtype=np.float64)
        mask = np.isfinite(reference) & (reference > 0)
        statistics = window_statistics(torch.from_numpy(generated), torch.from_numpy(reference),
                                       observed_mask=torch.from_numpy(mask))
        metadata = [dict(validation_index=i, subject_id=f"synthetic_{i}", filename=f"synthetic_{i}.npz",
                         start_sample=750*i, start_second=30.*i, sleep_stages=[i]) for i in range(2)]
        return dict(generated_seconds=generated, reference_seconds=reference, scoring_mask=mask,
                    windows_metadata=metadata, conditions=np.ones((2, 4, 2), dtype=np.float32),
                    condition_columns=["stage_test", "attribute_test"],
                    statistics={name: value.numpy() for name, value in statistics.items()},
                    protocol_path=ROOT/"synthetic_protocol.json", epoch=2)

    def assert_rng_equal(self, expected):
        actual = rng_state()
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1][0], actual[1][0])
        np.testing.assert_array_equal(expected[1][1], actual[1][1])
        self.assertEqual(expected[1][2:], actual[1][2:])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_lossless_pickle_free_npz_retains_missing_zero_nan_and_negative_generation(self):
        payload = self.writer_payload()
        report = write_validation_artifacts(self.temporary(), **payload)
        with np.load(report["waveforms"], allow_pickle=False) as saved:
            for name in saved.files:
                self.assertNotEqual(saved[name].dtype.kind, "O", name)
            np.testing.assert_array_equal(saved["generated_ibi_seconds"], payload["generated_seconds"])
            np.testing.assert_array_equal(saved["reference_ibi_seconds"], payload["reference_seconds"])
            np.testing.assert_array_equal(saved["scoring_mask"], payload["scoring_mask"])
            np.testing.assert_array_equal(saved["reference_observed_mask"], payload["scoring_mask"])
            self.assertEqual(saved["generated_ibi_seconds"].shape, (2, 4))
            self.assertEqual(saved["reference_ibi_seconds"][0, 2], 0.)
            self.assertTrue(np.isnan(saved["reference_ibi_seconds"][0, 3]))
            self.assertEqual(saved["generated_ibi_seconds"][0, 2], -99.)
            self.assertEqual(saved["validation_index"].tolist(), [0, 1])
            self.assertEqual(saved["start_sample"].tolist(), [0, 750])
        for name, field in (("waveforms", "waveform_sha256"), ("per_window_metrics", "metrics_sha256")):
            self.assertEqual(report[field], hashlib.sha256(Path(report[name]).read_bytes()).hexdigest())
        index = json.loads(Path(report["index"]).read_text())
        self.assertFalse(index["waveform_clipping"])
        self.assertEqual(index["waveform_unit"], "seconds")

    def test_csv_moments_match_saved_arrays_at_scoring_mask_and_kl_terms_sum(self):
        report = write_validation_artifacts(self.temporary(), **self.writer_payload())
        with np.load(report["waveforms"], allow_pickle=False) as saved, \
             Path(report["per_window_metrics"]).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
            for i, row in enumerate(rows):
                mask = saved["scoring_mask"][i]
                generated = saved["generated_ibi_seconds"][i, mask] * 1000
                reference = saved["reference_ibi_seconds"][i, mask] * 1000
                self.assertEqual(int(row["validation_index"]), i)
                self.assertEqual(int(row["observed_counts"]), int(mask.sum()))
                self.assertAlmostEqual(float(row["generated_mean_ms"]), generated.mean(), places=10)
                self.assertAlmostEqual(float(row["reference_mean_ms"]), reference.mean(), places=10)
                self.assertAlmostEqual(float(row["generated_variance_ms2"]), generated.var(ddof=0), places=8)
                self.assertAlmostEqual(float(row["reference_variance_ms2"]), reference.var(ddof=0), places=8)
                self.assertAlmostEqual(float(row["mean_error_ms"]), generated.mean()-reference.mean(), places=10)
                self.assertAlmostEqual(float(row["variance_error_ms2"]), generated.var()-reference.var(), places=8)
                self.assertEqual(float(row["gaussian_kl_real_to_generated"]),
                                 float(row["kl_mean_term"])+float(row["kl_variance_term"]))

    def test_export_refuses_overwrite_without_changing_existing_bytes(self):
        output = self.temporary()
        report = write_validation_artifacts(output, **self.writer_payload())
        paths = [Path(report[key]) for key in ("waveforms", "per_window_metrics", "index")]
        before = {path: path.read_bytes() for path in paths}
        with self.assertRaises(FileExistsError):
            write_validation_artifacts(output, **self.writer_payload())
        self.assertEqual(before, {path: path.read_bytes() for path in paths})
        for name in ("waveforms.npz", "per_window_metrics.csv", "artifacts.json"):
            with self.subTest(name=name):
                fresh = self.temporary()
                (fresh/name).write_text("existing", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    write_validation_artifacts(fresh, **self.writer_payload())
                self.assertEqual((fresh/name).read_text(), "existing")

    def test_writer_rejects_misaligned_metadata_shapes_and_nonboolean_masks(self):
        mutations = [dict(scoring_mask=np.ones((2, 4))),
                     dict(scoring_mask=np.ones((1, 4), dtype=bool)),
                     dict(conditions=np.ones((2, 3, 2))),
                     dict(statistics={"generated_mean_ms": np.ones(1)}),
                     dict(windows_metadata=list(reversed(self.writer_payload()["windows_metadata"])))]
        for mutation in mutations:
            with self.subTest(fields=list(mutation)), self.assertRaises(ValueError):
                write_validation_artifacts(self.temporary(), **{**self.writer_payload(), **mutation})

    def make_validator(self, enabled, policy="masked"):
        return fixtures.MaskedEpochValidationTests.make_validator(
            self, fixtures.MaskedPreprocessor(policy, windows=5), {"save_artifacts": enabled})

    def sample_epoch(self, validator, model, epoch=1):
        generated = validator.reference + torch.arange(1, 6, dtype=torch.float64)[:, None]/10
        if validator.reference_mask is not None:
            generated[~validator.reference_mask] = -.25
        batch_sizes, conditions = [], []
        def sample(copied_model, clean, diffusion_config, args, device):
            self.assertIsNot(copied_model, model)
            start = sum(batch_sizes)
            batch_sizes.append(len(clean))
            conditions.append(clean[:, :, 1:].clone())
            random.random(); np.random.rand(); torch.rand(1)
            return (generated[start:start+len(clean)]-validator.center)/validator.scale, {"mock_sampling": True}
        before = rng_state()
        with mock.patch("training_utils.fetchModel", side_effect=lambda *_: torch.nn.Linear(1, 1)), \
             mock.patch.object(coordinator, "sample_validation_windows", side_effect=sample), \
             contextlib.redirect_stdout(io.StringIO()):
            report = validator.run(epoch, model, .25)
        self.assert_rng_equal(before)
        self.assertEqual(batch_sizes, [2, 2, 1])
        return report, generated, torch.cat(conditions).numpy(), batch_sizes

    def test_multibatch_epoch_export_keeps_fixed_order_conditions_and_reference_across_epochs(self):
        validator, model = self.make_validator(True)
        reports = []
        for epoch in (1, 2):
            report, generated, conditions, _ = self.sample_epoch(validator, model, epoch)
            reports.append(report)
            with np.load(report["artifacts"]["waveforms"], allow_pickle=False) as saved:
                self.assertEqual(saved["generated_ibi_seconds"].shape, (5, 750))
                self.assertEqual(saved["clean_conditions"].shape, (5, 750, 9))
                np.testing.assert_array_equal(saved["generated_ibi_seconds"], generated.numpy())
                np.testing.assert_array_equal(saved["reference_ibi_seconds"], validator.reference.numpy())
                np.testing.assert_array_equal(saved["clean_conditions"], conditions)
                np.testing.assert_array_equal(saved["scoring_mask"], validator.reference_mask.numpy())
                self.assertEqual(saved["validation_index"].tolist(), list(range(5)))
                self.assertEqual(saved["filename"].tolist(), [f"v{i}.npz" for i in range(5)])
            with Path(report["artifacts"]["per_window_metrics"]).open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["observed_counts"]) for row in rows], [2, 3, 4, 5, 6])
            self.assertEqual([int(row["generated_nonpositive_samples_all"]) for row in rows], [748, 747, 746, 745, 744])
            self.assertAlmostEqual(np.mean([float(row["gaussian_kl_real_to_generated"]) for row in rows]),
                                   report["aggregate"]["gaussian_kl_real_to_generated"], places=14)
        self.assertEqual(reports[0]["aggregate"], reports[1]["aggregate"])
        self.assertNotEqual(reports[0]["artifacts"]["waveforms"], reports[1]["artifacts"]["waveforms"])

    def test_export_toggle_does_not_change_generation_count_aggregate_diagnostics_or_rng(self):
        results = []
        for enabled in (False, True):
            validator, model = self.make_validator(enabled)
            report, generated, _, batches = self.sample_epoch(validator, model)
            results.append((report, generated, batches))
            self.assertEqual((validator.output/"epoch_0001"/"waveforms.npz").exists(), enabled)
            self.assertEqual((validator.output/"epoch_0001"/"per_window_metrics.csv").exists(), enabled)
            self.assertNotIn("_artifact_payload", report)
        self.assertEqual(results[0][0]["aggregate"], results[1][0]["aggregate"])
        self.assertEqual(results[0][0]["diagnostics"], results[1][0]["diagnostics"])
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        self.assertEqual(results[0][2], results[1][2])

    def test_preserve_training_export_uses_all_positions_for_scoring_without_changing_observed_mask(self):
        validator, model = self.make_validator(True, "preserve")
        report, _, _, _ = self.sample_epoch(validator, model)
        with np.load(report["artifacts"]["waveforms"], allow_pickle=False) as saved:
            self.assertTrue(saved["scoring_mask"].all())
            np.testing.assert_array_equal(saved["reference_observed_mask"], validator.reference.numpy() > 0)
            self.assertFalse(saved["reference_observed_mask"].all())
        with Path(report["artifacts"]["per_window_metrics"]).open(newline="", encoding="utf-8") as stream:
            self.assertEqual([int(row["observed_counts"]) for row in csv.DictReader(stream)], [750]*5)

    def test_save_artifacts_flag_is_strict_boolean_and_default_remains_omitted(self):
        self.assertNotIn("save_artifacts", coordinator.validation_options({}))
        for valid in (False, True):
            self.assertIs(coordinator.validation_options({"validation": {"save_artifacts": valid}})["save_artifacts"], valid)
        for invalid in (None, 0, 1, "false", "true", [], {}, np.bool_(True)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                coordinator.validation_options({"validation": {"save_artifacts": invalid}})

    def test_run_snapshot_copies_artifact_module_and_simulation_tracks_its_hash(self):
        destination = self.temporary()/"snapshot"
        copied = []
        def copy2(source, target):
            copied.append((Path(source), Path(target)))
        with mock.patch.object(run_dreamt.shutil, "copy2", side_effect=copy2), \
             mock.patch.object(run_dreamt.shutil, "copytree"):
            run_dreamt.snapshot_source(destination)
        self.assertIn((ROOT/"validation_artifacts.py", destination/"validation_artifacts.py"), copied)
        tree = ast.parse((ROOT/"scripts"/"run_simulation.py").read_text(encoding="utf-8"))
        paths = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "source_paths" for target in node.targets))
        self.assertTrue(any(isinstance(node, ast.Constant) and node.value == "validation_artifacts.py"
                            for node in ast.walk(paths)))


if __name__ == "__main__":
    unittest.main()
