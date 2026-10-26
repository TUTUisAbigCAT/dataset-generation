"""Folder-switch regressions using temporary ordinary DREAMT-format records.

No real data, models, or diffusion are executed. These fixtures deliberately
lack simulator markers, timeline extensions, saved splits, 60-minute records,
eight validation subjects, and 30-minute validation starts.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work" / "folder_switch_tests"
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from data_utils import Preprocessor


def evaluation_helpers():
    """Import only helper definitions; the evaluator's model code stays in main."""
    path = ROOT / "scripts/evaluate_denoising.py"
    spec = importlib.util.spec_from_file_location("folder_switch_evaluation_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def snapshot_files(directory):
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def write_metadata(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("SID", "AHI", "BMI", "AGE", "GENDER"))
        writer.writeheader()
        writer.writerows(rows)


def create_case(base, first_sid, epoch_counts, *, metadata_location="same", constant=False):
    directory = base / "records"
    directory.mkdir(parents=True)
    rows = []
    for index, epochs in enumerate(epoch_counts):
        sid = f"S{first_sid + index:03d}"
        # Float32 AHI in NPZ vs decimal CSV intentionally has harmless rounding.
        ahi = 12.3456789 if constant else 12.3456789 + index*.3
        row = dict(SID=sid, AHI=ahi, BMI=25. if constant else 23. + index,
                   AGE=50. if constant else 30. + index*3, GENDER="F" if index % 2 == 0 else "M")
        rows.append(row)
        samples = np.arange(epochs*750, dtype=np.float64)
        signal = (.72 + index*.025 + .025*np.sin(samples/37.)).astype(np.float32)
        stages = np.repeat(np.resize(np.array([0, 1, 2, 3, 4], dtype=np.int8), epochs), 750)
        np.savez(directory / (sid + ".npz"), data=signal[:, None], stages=stages,
                 fs=np.asarray(25, dtype=np.int32), ahi=np.asarray(ahi, dtype=np.float32))
    metadata_path = None
    if metadata_location is not None:
        metadata_path = (directory if metadata_location == "same" else base) / "participant_info.csv"
        write_metadata(metadata_path, rows)
    return dict(base=base, directory=directory, metadata=metadata_path, rows=rows,
                filenames=[row["SID"] + ".npz" for row in rows])


class FolderSwitchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="switch_", dir=WORK)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config_dir = self.directory / "configuration"
        self.config_dir.mkdir()
        self.config_path = self.config_dir / "dataset_config.json"
        self.cwd_dir = self.directory / "unrelated_cwd"
        self.cwd_dir.mkdir()
        self.a = create_case(self.directory / "folder_a", 101, [4, 6, 8, 10, 12])
        self.b = create_case(self.directory / "folder_b", 901, [4]*6, metadata_location="parent")
        self.environment = mock.patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for key in list(os.environ):
            if key.startswith("DREAMT_"):
                del os.environ[key]
        os.environ["DREAMT_CONFIG_PATH"] = str(self.config_path)
        self.write_config(self.a["directory"])

    def write_config(self, data_directory, **extra):
        config = dict(data_dir=os.path.relpath(data_directory, self.config_dir),
                      metadata_path=None, split_path=None, multiplier=1., downsample_rate=1)
        config.update(extra)
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        return config

    def processor(self, **kwargs):
        return Preprocessor("DREAMT", **kwargs)

    def frozen_manifest_fixture(self, processor):
        helpers = evaluation_helpers()
        run = self.directory / "data_only_manifest_fixture"
        run.mkdir()
        checkpoint = run / "digest_fixture.bin"
        checkpoint.write_bytes(b"Data-only unit-test digest fixture; not model weights.")
        manifest = dict(training_args=dict(helpers.DEFAULT_TRAINING_ARGS, dataset="DREAMT", batch_size=2, epochs=1),
                        source_sha256={"TSImputers/S4Model.py": hashlib.sha256((ROOT / "TSImputers/S4Model.py").read_bytes()).hexdigest()},
                        data_identity=processor.data_identity(), checkpoint=str(checkpoint),
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        test_fixture_not_a_training_result=True)
        path = run / "run_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        args, restored_manifest, _ = helpers.load_training_run(path)
        return helpers, path, args, restored_manifest

    def assert_generated_split(self, processor, case):
        splits = processor.splits
        self.assertEqual(set(splits), {"train", "val", "test"})
        self.assertTrue(all(len(splits[name]) >= 1 for name in splits))
        flattened = [filename for values in splits.values() for filename in values]
        self.assertEqual(len(set(flattened)), len(flattened))
        self.assertEqual(set(flattened), set(case["filenames"]))
        self.assertNotEqual(len(splits["val"]), 8)
        provenance = processor.provenance()
        self.assertEqual(provenance["split_source"], "generated_subject_split")
        self.assertIsNone(provenance["split_file_sha256"])
        self.assertEqual(provenance["split_seed"], 42)
        self.assertEqual(provenance["splits"], splits)
        self.assertRegex(provenance["split_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertFalse(provenance["test_npz_opened"])
        self.assertEqual(Path(provenance["data_directory"]), case["directory"].resolve())

    def assert_batch_and_validation(self, processor):
        dataset = processor.make_training_dataset(3000, 1500)
        local_generator = torch.Generator().manual_seed(91)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, generator=local_generator)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch.shape[1:]), (3000, 10))
        self.assertTrue(torch.isfinite(batch).all())
        self.assertTrue(torch.equal(batch[:, :, 1:5].sum(dim=-1), torch.ones(batch.shape[:2])))
        selected = processor.select_validation_windows(windows_per_subject=2)
        self.assertGreater(len(selected), 0)
        self.assertLessEqual(len(selected), 2*len(processor.splits["val"]))
        self.assertEqual(len(selected), len(set(selected)))
        for filename, start in selected:
            self.assertIn(filename, processor.splits["val"])
            self.assertNotIn(filename, processor.splits["train"] + processor.splits["test"])
            self.assertLess(start, 45000, "Short records cannot require a fixed 30-minute start")
            self.assertEqual(start % 1500, 0)
            array = processor.window_array(filename, start, 3000)
            self.assertEqual(array.shape, (3000, 10))
            self.assertTrue(np.isfinite(array).all())
        return selected

    def test_switch_only_config_data_dir_fits_batches_and_dynamic_validation(self):
        before_a, before_b = snapshot_files(self.a["base"]), snapshot_files(self.b["base"])
        for case in (self.a, self.b):
            with np.load(case["directory"] / case["filenames"][0], allow_pickle=False) as raw:
                self.assertEqual(set(raw.files), {"data", "stages", "fs", "ahi"})
            self.assertFalse((case["directory"] / "splits.json").exists())
        config_a = json.loads(self.config_path.read_text(encoding="utf-8"))
        processor_a = self.processor()
        self.assert_generated_split(processor_a, self.a)
        self.assert_batch_and_validation(processor_a)
        config_b = dict(config_a)
        config_b["data_dir"] = os.path.relpath(self.b["directory"], self.config_dir)
        self.config_path.write_text(json.dumps(config_b), encoding="utf-8")
        self.assertEqual([key for key in config_a if config_a[key] != config_b[key]], ["data_dir"])
        processor_b = self.processor()
        self.assert_generated_split(processor_b, self.b)
        selected_b = self.assert_batch_and_validation(processor_b)
        self.assertEqual(len(selected_b), len(processor_b.splits["val"]))
        self.assertTrue(all(start == 0 for _, start in selected_b), "Four-epoch records have exactly one window")
        self.assertTrue(set(processor_a.splits["train"]).isdisjoint(processor_b.splits["train"]))
        self.assertNotEqual(processor_a.provenance()["split_fingerprint"], processor_b.provenance()["split_fingerprint"])
        self.assertEqual(snapshot_files(self.a["base"]), before_a)
        self.assertEqual(snapshot_files(self.b["base"]), before_b)

    def test_auto_split_is_deterministic_local_rng_and_read_only(self):
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()
        before = snapshot_files(self.a["base"])
        first = self.processor()
        windows = first.select_validation_windows(windows_per_subject=2)
        second = self.processor()
        self.assertEqual(first.splits, second.splits)
        self.assertEqual(windows, second.select_validation_windows(windows_per_subject=2))
        self.assertEqual(first.provenance()["split_fingerprint"], second.provenance()["split_fingerprint"])
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))
        self.assertEqual(snapshot_files(self.a["base"]), before)

    def test_config_relative_path_uses_config_parent_even_with_other_cwd(self):
        with working_directory(self.cwd_dir):
            processor = self.processor()
        self.assertEqual(processor.data_dir, self.a["directory"].resolve())
        self.assert_generated_split(processor, self.a)

    def test_env_and_explicit_relative_data_paths_use_cwd_and_precedence(self):
        with working_directory(self.cwd_dir):
            os.environ["DREAMT_DATA_DIR"] = os.path.relpath(self.b["directory"], self.cwd_dir)
            via_env = self.processor()
            self.assertEqual(via_env.data_dir, self.b["directory"].resolve())
            via_explicit = self.processor(data_dir=os.path.relpath(self.a["directory"], self.cwd_dir))
            self.assertEqual(via_explicit.data_dir, self.a["directory"].resolve())

    def test_parent_metadata_discovery_and_same_directory_priority(self):
        self.write_config(self.b["directory"])
        self.assertEqual(self.processor().metadata_path, self.b["metadata"].resolve())
        same_folder = self.b["directory"] / "participant_info.csv"
        same_folder.write_bytes(self.b["metadata"].read_bytes())
        self.b["metadata"].write_text("deliberately_invalid_parent_csv\n", encoding="utf-8")
        processor = self.processor()
        self.assertEqual(processor.metadata_path, same_folder.resolve())
        self.assert_batch_and_validation(processor)

    def test_explicit_metadata_and_split_paths_override_config_and_environment(self):
        override = self.directory / "explicit"
        override.mkdir()
        metadata = override / "participants.csv"
        write_metadata(metadata, self.a["rows"])
        split_path = override / "partitions.json"
        names = self.a["filenames"]
        expected = dict(train=names[:3], val=[names[3]], test=[names[4]])
        split_path.write_text(json.dumps(expected), encoding="utf-8")
        self.write_config(self.b["directory"], metadata_path="missing_config.csv", split_path="missing_config.json")
        os.environ["DREAMT_METADATA_PATH"] = "missing_environment.csv"
        os.environ["DREAMT_SPLIT_PATH"] = "missing_environment.json"
        with working_directory(self.cwd_dir):
            processor = self.processor(data_dir=os.path.relpath(self.a["directory"], self.cwd_dir),
                                       metadata_path=os.path.relpath(metadata, self.cwd_dir),
                                       split_path=os.path.relpath(split_path, self.cwd_dir))
        self.assertEqual(processor.metadata_path, metadata.resolve())
        self.assertEqual(processor.splits, expected)
        provenance = processor.provenance()
        self.assertEqual(provenance["split_source"], "split_file")
        self.assertEqual(provenance["split_file_sha256"], hashlib.sha256(split_path.read_bytes()).hexdigest())

    def test_constant_continuous_metadata_uses_unit_scale_and_is_recorded(self):
        case = create_case(self.directory / "constant", 301, [4, 5, 6, 7, 8], constant=True)
        self.write_config(case["directory"])
        processor = self.processor()
        self.assertEqual(processor.scaler["constant_columns"], ["AHI", "BMI", "AGE"])
        np.testing.assert_array_equal(processor.scaler["continuous_raw_std"], [0., 0., 0.])
        np.testing.assert_array_equal(processor.scaler["continuous_std"], [1., 1., 1.])
        self.assert_batch_and_validation(processor)
        for filename in processor.splits["train"] + processor.splits["val"]:
            np.testing.assert_array_equal(processor.window_array(filename, 0, 3000)[:, 5:8], np.zeros((3000, 3)))

    def test_float32_npz_ahi_rounding_is_accepted_against_decimal_csv(self):
        processor = self.processor()
        for filename in processor.splits["train"]:
            record = processor.record(filename)
            self.assertNotEqual(record["ahi"], record["continuous"][0])
            self.assertTrue(np.isclose(record["ahi"], record["continuous"][0], rtol=1e-7, atol=1e-7))
            self.assertIn("NPZ", record["ahi_source"])

    def test_missing_csv_raises_without_loading_any_fallback_dataset(self):
        case = create_case(self.directory / "missing_metadata", 401, [4]*4, metadata_location=None)
        self.write_config(case["directory"])
        with mock.patch.object(np, "load", side_effect=AssertionError("Must not load a fallback dataset")):
            with self.assertRaises((FileNotFoundError, ValueError)) as failure:
                self.processor()
        self.assertRegex(str(failure.exception).lower(), r"metadata|participant|csv")
        self.assertFalse((case["directory"] / "participant_info.csv").exists())

    def test_missing_selected_config_does_not_fall_back_to_original_dataset(self):
        os.environ["DREAMT_CONFIG_PATH"] = str(self.directory / "missing_dataset_config.json")
        with mock.patch.object(np, "load", side_effect=AssertionError("Must not load a fallback dataset")):
            with self.assertRaises((FileNotFoundError, ValueError)):
                self.processor()

    def test_nonexistent_configured_data_directory_does_not_fall_back(self):
        self.write_config(self.directory / "missing_records")
        with mock.patch.object(np, "load", side_effect=AssertionError("Must not load a fallback dataset")):
            with self.assertRaises((FileNotFoundError, ValueError)):
                self.processor()

    def test_incorrect_existing_split_fails_instead_of_generating_a_new_one(self):
        names = self.a["filenames"]
        path = self.a["directory"] / "splits.json"
        invalid_splits = [dict(train=names[:3], val=[names[0], names[3]], test=[names[4]]),
                          dict(train=names[:2], val=[names[3]], test=[names[4]]),
                          dict(train=names[:3], val=["unknown.npz"], test=[names[4]])]
        for invalid in invalid_splits:
            with self.subTest(split=invalid):
                path.write_text(json.dumps(invalid), encoding="utf-8")
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    self.processor()
                self.assertEqual(path.read_bytes(), before)

    def test_valid_existing_split_is_used_without_reassignment(self):
        names = self.a["filenames"]
        expected = dict(train=names[1:4], val=[names[0]], test=[names[4]])
        path = self.a["directory"] / "splits.json"
        path.write_text(json.dumps(expected), encoding="utf-8")
        processor = self.processor()
        self.assertEqual(processor.splits, expected)
        self.assertEqual(processor.provenance()["split_source"], "split_file")
        self.assertIsNotNone(processor.provenance()["split_file_sha256"])

    def test_fewer_than_three_subjects_cannot_silently_skip_a_partition(self):
        case = create_case(self.directory / "two_subjects", 501, [4, 6], constant=True)
        self.write_config(case["directory"])
        with self.assertRaises(ValueError):
            self.processor()

    def test_validation_selector_skips_short_subject_and_accepts_single_window(self):
        case = create_case(self.directory / "short_validation", 601, [4, 5, 3, 4, 6])
        names = case["filenames"]
        (case["directory"] / "splits.json").write_text(
            json.dumps(dict(train=names[:2], val=names[2:4], test=[names[4]])), encoding="utf-8")
        self.write_config(case["directory"])
        processor = self.processor()
        self.assertEqual(processor.select_validation_windows(windows_per_subject=2), [(names[3], 0)])
        self.assertTrue(processor.validation_selection["excluded"])

    def test_validation_selector_raises_when_no_legal_window_exists(self):
        case = create_case(self.directory / "no_validation_windows", 701, [4, 3, 4], constant=True)
        names = case["filenames"]
        (case["directory"] / "splits.json").write_text(
            json.dumps(dict(train=[names[0]], val=[names[1]], test=[names[2]])), encoding="utf-8")
        self.write_config(case["directory"])
        processor = self.processor()
        with self.assertRaises(ValueError):
            processor.select_validation_windows(windows_per_subject=2)

    def test_frozen_manifest_keeps_folder_a_despite_global_folder_b(self):
        original = self.processor()
        helpers, manifest_path, args, manifest = self.frozen_manifest_fixture(original)
        identity = original.data_identity()
        original.select_validation_windows(windows_per_subject=2)
        self.assertEqual(original.data_identity(), identity, "Opening val must not change frozen training identity")
        first_train = original.splits["train"][0]
        expected_window = original.window_array(first_train, 0, 3000)
        self.write_config(self.b["directory"])
        os.environ["DREAMT_DATA_DIR"] = str(self.b["directory"])
        os.environ["DREAMT_METADATA_PATH"] = str(self.b["metadata"])
        os.environ["DREAMT_SPLIT_PATH"] = str(self.directory / "missing_global_split.json")
        os.environ["DREAMT_MULTIPLIER"] = "2"
        # A newly-created split cannot silently replace the recorded auto split.
        (self.a["directory"] / "splits.json").write_text('{"not_the_recorded_split": []}', encoding="utf-8")
        bound, checks = helpers.bind_training_data(args, manifest, manifest_path, Preprocessor)
        self.assertTrue(checks["global_config_ignored"])
        self.assertTrue(checks["all_recorded_identity_fields_match"])
        self.assertEqual(bound.data_dir, self.a["directory"].resolve())
        self.assertEqual(bound.data_identity(), identity)
        np.testing.assert_array_equal(bound.window_array(first_train, 0, 3000), expected_window)
        os.environ["DREAMT_CONFIG_PATH"] = str(self.directory / "missing_global_config.json")
        restored = Preprocessor.from_identity(identity, dataset="DREAMT")
        self.assertEqual(restored.data_identity(), identity)

    def test_frozen_manifest_rejects_metadata_changes(self):
        original = self.processor()
        helpers, manifest_path, args, manifest = self.frozen_manifest_fixture(original)
        rows = [dict(row) for row in self.a["rows"]]
        rows[0]["BMI"] += 1.
        write_metadata(self.a["metadata"], rows)
        with self.assertRaises(ValueError):
            helpers.bind_training_data(args, manifest, manifest_path, Preprocessor)

    def test_frozen_manifest_rejects_training_record_changes(self):
        original = self.processor()
        helpers, manifest_path, args, manifest = self.frozen_manifest_fixture(original)
        path = self.a["directory"] / original.splits["train"][0]
        with np.load(path, allow_pickle=False) as saved:
            payload = {key: saved[key].copy() for key in saved.files}
        payload["data"] += np.float32(.005)
        np.savez(path, **payload)
        with self.assertRaises(ValueError):
            helpers.bind_training_data(args, manifest, manifest_path, Preprocessor)

    def test_checkpoint_digest_accepts_identical_relocation_rejects_wrong_file(self):
        helpers, manifest_path, _, manifest = self.frozen_manifest_fixture(self.processor())
        recorded, info = helpers.resolve_checkpoint(manifest_path, manifest)
        self.assertTrue(info["recorded_digest_verified"])
        relocated = self.directory / "relocated_digest_fixture.bin"
        relocated.write_bytes(recorded.read_bytes())
        actual, info = helpers.resolve_checkpoint(manifest_path, manifest, override=relocated)
        self.assertEqual(actual, relocated.resolve())
        self.assertTrue(info["explicit_override"])
        relocated.write_bytes(b"A different unit-test checkpoint digest, never model weights.")
        with self.assertRaises(ValueError):
            helpers.resolve_checkpoint(manifest_path, manifest, override=relocated)
        recorded.write_bytes(b"Recorded unit-test checkpoint was modified.")
        with self.assertRaises(ValueError):
            helpers.resolve_checkpoint(manifest_path, manifest)

    def test_evaluation_requires_explicit_complete_manifest(self):
        helpers, manifest_path, _, manifest = self.frozen_manifest_fixture(self.processor())
        missing_data = dict(manifest)
        del missing_data["data_identity"]
        manifest_path.write_text(json.dumps(missing_data), encoding="utf-8")
        with self.assertRaises(ValueError):
            helpers.load_training_run(manifest_path)
        missing_args = dict(manifest)
        missing_args["training_args"] = {"dataset": "DREAMT"}
        manifest_path.write_text(json.dumps(missing_args), encoding="utf-8")
        with self.assertRaises(ValueError):
            helpers.load_training_run(manifest_path)

    def test_invalid_holdout_metadata_is_rejected_before_any_npz_load(self):
        split = self.processor().splits
        invalid_sid = Path(split["val"][0]).stem
        rows = [dict(row) for row in self.a["rows"]]
        for row in rows:
            if row["SID"] == invalid_sid:
                row["AGE"] = "not_numeric"
        write_metadata(self.a["metadata"], rows)
        with mock.patch.object(np, "load", side_effect=AssertionError("Metadata must be validated before NPZ loading")):
            with self.assertRaises(ValueError):
                self.processor()

    def test_extraction_zero_patterns_fit_batch_and_validate_without_imputation(self):
        expected = {}
        for filename in self.a["filenames"]:
            path = self.a["directory"] / filename
            with np.load(path, allow_pickle=False) as saved:
                payload = {key: saved[key].copy() for key in saved.files}
            epochs = payload["data"].reshape(-1, 750)
            epochs[:, :24] = 0.
            epochs[:, -31:] = 0.
            epochs[:, 200:207] = 0.
            epochs[2] = 0.
            expected[filename] = epochs.reshape(-1).copy()
            np.savez(path, **payload)
        processor = self.processor()
        self.assert_batch_and_validation(processor)
        train_signal = np.concatenate([expected[name] for name in processor.splits["train"]]).astype(np.float64)
        self.assertEqual(processor.scaler["signal_fit_samples"], len(train_signal))
        self.assertAlmostEqual(processor.scaler["signal_mean_seconds"], float(train_signal.mean()), places=9)
        self.assertAlmostEqual(processor.scaler["signal_std_seconds"], float(train_signal.std()), places=9)
        for filename in processor.splits["train"] + processor.splits["val"]:
            record = processor.record(filename)
            self.assertTrue(record["valid_epochs"].all())
            np.testing.assert_array_equal(record["ibi"], expected[filename])
            self.assertEqual(record["zero_samples"], int((expected[filename] == 0).sum()))
            self.assertEqual(record["zero_epochs"], 1)
            array = processor.window_array(filename, 0, 3000)
            np.testing.assert_allclose(processor.inverse_signal(array[:, 0]), expected[filename][:3000], atol=1e-7, rtol=0)
        identity, provenance = processor.data_identity(), processor.provenance()
        self.assertEqual(identity["zero_value_policy"], provenance["zero_value_policy"])
        for filename, alignment in provenance["record_alignment"].items():
            self.assertEqual(alignment["zero_samples"], int((expected[filename] == 0).sum()))
            self.assertEqual(alignment["zero_epochs"], 1)

    def test_entire_training_signal_zero_fails_with_explicit_variance_error(self):
        for filename in self.a["filenames"]:
            path = self.a["directory"] / filename
            with np.load(path, allow_pickle=False) as saved:
                payload = {key: saved[key].copy() for key in saved.files}
            payload["data"][:] = 0.
            np.savez(path, **payload)
        with self.assertRaisesRegex(ValueError, "variance"):
            self.processor()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
