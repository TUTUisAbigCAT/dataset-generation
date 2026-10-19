"""DREAMT adapter contracts against the frozen WatchSleepNet SSDataset.

Only small temporary NPZ/CSV fixtures are used. The original class is extracted
with AST to avoid importing its unrelated sklearn dependency. No model or
diffusion code is imported or executed by these data-only tests.
"""
from __future__ import annotations

import ast
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work" / "dreamt_loader_tests"
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from watchsleep_loader import SSDataset
from data_utils import Preprocessor, WindowDataset


def original_dataset_class():
    path = ROOT / "third_party/WatchSleepNet/data_setup.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SSDataset"]
    if len(nodes) != 1:
        raise AssertionError("Frozen source must define exactly one SSDataset")
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    scope = dict(np=np, torch=torch, Dataset=Dataset, pad_sequence=pad_sequence, json=json, os=os, Path=Path)
    exec(compile(module, str(path), "exec"), scope)
    return scope["SSDataset"]


OriginalSSDataset = original_dataset_class()
MISSING = object()
EPOCH = 750
EXPECTED_COLUMNS = ["IBI", "stage_W", "stage_Light", "stage_REM", "stage_Deep",
                    "AHI", "BMI", "AGE", "gender_F", "gender_M"]


def write_record(directory, filename, labels, *, fs=25, tail=0, column=True, ahi=3.5, offset=0., **extra):
    """Write the original four-field format unless a test requests extensions."""
    samples_per_epoch = int(fs * 30)
    count = len(labels) * samples_per_epoch + tail
    index = np.arange(count, dtype=np.float64)
    signal = (.76 + offset + .02*np.sin(index/31.) + .003*((index//samples_per_epoch) % 7)).astype(np.float32)
    stages = np.repeat(np.asarray(labels, dtype=np.int16), samples_per_epoch)
    if tail:
        stages = np.concatenate((stages, np.full(tail, labels[-1], dtype=np.int16)))
    payload = dict(data=signal[:, None] if column else signal, stages=stages, fs=np.asarray(fs))
    if ahi is not MISSING:
        payload["ahi"] = np.asarray(ahi)
    payload.update(extra)
    np.savez(directory / filename, **payload)
    return payload


def rewrite_record(path, transform):
    with np.load(path, allow_pickle=False) as saved:
        payload = {key: saved[key].copy() for key in saved.files}
    transform(payload)
    np.savez(path, **payload)
    return payload


class TemporaryDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fixture_", dir=WORK)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def assert_original_tuple(self, filename, **kwargs):
        oracle = OriginalSSDataset(dir=self.directory, dataset="dreamt", file_list=[filename], **kwargs)[0]
        actual = SSDataset(dir=self.directory, dataset="dreamt", file_list=[filename], **kwargs)[0]
        self.assertEqual(len(actual), len(oracle))
        self.assertTrue(torch.equal(actual[0], oracle[0]), "IBI differs from frozen SSDataset")
        self.assertTrue(torch.equal(actual[1], oracle[1]), "Labels differ from frozen SSDataset")
        self.assertEqual(actual[2:], oracle[2:])
        return actual


class SSDatasetCompatibilityTests(TemporaryDataTest):
    def test_original_four_fields_flat_and_column_match_tuple(self):
        for column in (False, True):
            with self.subTest(column=column):
                write_record(self.directory, "S001.npz", [0, 1, 2, 3, 4], column=column)
                with np.load(self.directory / "S001.npz", allow_pickle=False) as raw:
                    self.assertEqual(set(raw.files), {"data", "stages", "fs", "ahi"})
                actual = self.assert_original_tuple("S001.npz", return_file_name=True)
                self.assertEqual(actual[0].shape, (5, 750))
                self.assertEqual(actual[0].dtype, torch.float32)
                self.assertEqual(actual[1].tolist(), [0, 1, 1, 3, 2])
                self.assertEqual(actual[-1], "S001")

    def test_integer_downsampling_multiplier_and_tail_match_original(self):
        write_record(self.directory, "S002.npz", [4, 3, 2, 1, 0], fs=100, tail=173, column=False)
        actual = self.assert_original_tuple("S002.npz", multiplier=1.25, downsample_rate=4)
        self.assertEqual(actual[0].shape, (5, 750))
        record = SSDataset(self.directory, "dreamt", ["S002.npz"], multiplier=1.25, downsample_rate=4).read_record(0)
        self.assertEqual(record["discarded_tail_samples"], 44)
        np.testing.assert_array_equal(record["ibi_epochs"], actual[0].numpy())

    def test_four_class_mapping_invalid_labels_matches_original(self):
        write_record(self.directory, "S003.npz", [0, 1, 2, 3, 4, 5, -1, 99])
        actual = self.assert_original_tuple("S003.npz")
        self.assertEqual(actual[1].tolist(), [0, 1, 1, 3, 2, -1, -1, -1])
        record = SSDataset(self.directory, "dreamt", ["S003.npz"]).read_record(0)
        np.testing.assert_array_equal(record["valid_epochs"], [True]*5 + [False]*3)
        self.assertEqual(record["ibi_epochs"].shape, (8, 750))

    def test_sleep_wake_original_api_is_preserved(self):
        write_record(self.directory, "S004.npz", [0, 1, 2, 3, 4, 5])
        actual = self.assert_original_tuple("S004.npz", task="sleep_wake")
        self.assertEqual(actual[1].tolist(), [0, 1, 1, 1, 1, -1])

    def test_no_timeline_requires_only_stored_order_claim(self):
        write_record(self.directory, "S005.npz", [0, 1, 2, 3, 4])
        record = SSDataset(self.directory, "dreamt", ["S005.npz"]).read_record(0)
        self.assertEqual(record["continuity_source"], "stored_order_only")
        self.assertIsNone(record["epoch_index"])
        self.assertIsNone(record["epoch_start_seconds"])
        np.testing.assert_array_equal(record["adjacent"], np.ones(4, dtype=bool))

    def test_epoch_index_and_seconds_report_gaps_without_deletion(self):
        for extra, source in (({"epoch_index": np.array([10, 11, 13, 14, 15])}, "epoch_index"),
                              ({"epoch_start_seconds": np.array([0., 30., 90., 120., 150.])}, "epoch_start_seconds")):
            with self.subTest(source=source):
                write_record(self.directory, "S006.npz", [0, 1, 2, 3, 4], **extra)
                record = SSDataset(self.directory, "dreamt", ["S006.npz"]).read_record(0)
                self.assertEqual(record["continuity_source"], source)
                np.testing.assert_array_equal(record["adjacent"], [True, False, True, True])
                self.assertEqual(len(record["stage_epochs"]), 5)

    def test_two_timeline_sources_both_constrain_adjacency(self):
        write_record(self.directory, "S007.npz", [0, 1, 2, 3, 4],
                     epoch_index=np.array([0, 1, 3, 4, 5]),
                     epoch_start_seconds=np.array([0., 30., 60., 120., 150.]))
        record = SSDataset(self.directory, "dreamt", ["S007.npz"]).read_record(0)
        self.assertEqual(record["continuity_source"], "epoch_index_and_epoch_start_seconds")
        np.testing.assert_array_equal(record["adjacent"], [True, False, False, True])

    def test_nonfinite_or_mixed_label_epochs_remain_in_place(self):
        payload = write_record(self.directory, "S008.npz", [0, 1, 2, 3, 4])
        payload["data"][EPOCH + 17, 0] = np.nan
        payload["stages"][2*EPOCH + 17] = 4
        np.savez(self.directory / "S008.npz", **payload)
        record = SSDataset(self.directory, "dreamt", ["S008.npz"]).read_record(0)
        np.testing.assert_array_equal(record["valid_epochs"], [True, False, False, True, True])
        np.testing.assert_array_equal(record["stage_epochs"], [0, 1, 1, 3, 2])
        self.assertEqual(record["ibi_epochs"].shape, (5, 750))

    def test_extraction_zero_markers_and_all_zero_epoch_match_original_tuple(self):
        payload = write_record(self.directory, "ZEROS.npz", [0, 1, 2, 3, 4])
        epochs = payload["data"].reshape(5, EPOCH)
        epochs[:, :24] = 0.
        epochs[:, -31:] = 0.
        epochs[:, 200:207] = 0.
        epochs[2] = 0.
        np.savez(self.directory / "ZEROS.npz", **payload)
        actual = self.assert_original_tuple("ZEROS.npz")
        np.testing.assert_array_equal(actual[0].numpy(), epochs)
        record = SSDataset(self.directory, "dreamt", ["ZEROS.npz"]).read_record(0)
        np.testing.assert_array_equal(record["valid_epochs"], np.ones(5, dtype=bool))
        self.assertEqual(record["zero_samples"], int((epochs == 0).sum()))
        self.assertEqual(record["zero_epochs"], 1)
        self.assertEqual(record["epochs_with_zero"], 5)

    def test_negative_and_nonfinite_remain_invalid_while_zero_is_preserved(self):
        payload = write_record(self.directory, "ZERO_INVALID.npz", [0, 1, 2, 3, 4])
        epochs = payload["data"].reshape(5, EPOCH)
        epochs[0] = 0.
        epochs[1, 17] = -.1
        epochs[2, 17] = np.nan
        epochs[3, 17] = np.inf
        epochs[4, :30] = 0.
        np.savez(self.directory / "ZERO_INVALID.npz", **payload)
        record = SSDataset(self.directory, "dreamt", ["ZERO_INVALID.npz"]).read_record(0)
        np.testing.assert_array_equal(record["valid_epochs"], [True, False, False, False, True])
        self.assertEqual(record["ibi_epochs"].shape, (5, EPOCH))
        np.testing.assert_array_equal(record["ibi_epochs"][0], np.zeros(EPOCH, dtype=np.float32))

    def test_default_retains_more_than_1100_epochs_explicit_limit_matches_original(self):
        write_record(self.directory, "LONG.npz", [0, 1, 2, 3, 4]*221)
        dataset = SSDataset(self.directory, "dreamt", ["LONG.npz"])
        self.assertEqual(dataset[0][2], 1105)
        self.assertEqual(dataset.read_record(0)["ibi_epochs"].shape, (1105, 750))
        limited = SSDataset(self.directory, "dreamt", ["LONG.npz"], max_epochs=1100)[0]
        original = OriginalSSDataset(self.directory, "dreamt", ["LONG.npz"])[0]
        self.assertTrue(torch.equal(limited[0], original[0]))
        self.assertTrue(torch.equal(limited[1], original[1]))
        self.assertEqual(limited[2:], original[2:])

    def test_missing_ahi_is_explicit_in_read_record(self):
        write_record(self.directory, "S009.npz", [0, 1, 2, 3], ahi=MISSING)
        self.assertIsNone(SSDataset(self.directory, "dreamt", ["S009.npz"]).read_record(0)["ahi"])

    def test_original_collate_api_pads_without_changing_values(self):
        write_record(self.directory, "A.npz", [0, 1, 2, 3])
        write_record(self.directory, "B.npz", [4, 3, 2, 1, 0])
        new = SSDataset(self.directory, "dreamt", ["A.npz", "B.npz"])
        old = OriginalSSDataset(self.directory, "dreamt", ["A.npz", "B.npz"])
        a, b = SSDataset.collate_fn([new[0], new[1]]), OriginalSSDataset.collate_fn([old[0], old[1]])
        for left, right in zip(a[:3], b[:3]):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(a[3], b[3])


class PreprocessorTests(TemporaryDataTest):
    def setUp(self):
        super().setUp()
        self.data = self.directory / "npz"
        self.data.mkdir()
        self.metadata_path = self.directory / "participants.csv"
        self.split_path = self.directory / "subject_splits.json"
        self.splits = {"train": ["A.npz", "B.npz"], "val": ["V.npz"], "test": ["T.npz"]}
        self.metadata_rows = [
            dict(SID="A", AHI=2., BMI=22., AGE=30., GENDER="F"),
            dict(SID="B", AHI=10., BMI=28., AGE=55., GENDER="M"),
            dict(SID="V", AHI=40., BMI=40., AGE=75., GENDER="F"),
            dict(SID="T", AHI=60., BMI=45., AGE=80., GENDER="M"),
        ]
        self.payloads = {
            "A": write_record(self.data, "A.npz", [0, 1, 2, 3, 4, 0, 1, 2], ahi=2.),
            "B": write_record(self.data, "B.npz", [4, 3, 2, 1, 0, 4, 3, 2], ahi=10., offset=.15),
            "V": write_record(self.data, "V.npz", [0, 1, 2, 3, 4, 0, 1, 2], ahi=40., offset=10.),
            "T": write_record(self.data, "T.npz", [0, 1, 2, 3, 4, 0, 1, 2], ahi=60., offset=20.),
        }
        self.write_metadata()
        self.split_path.write_text(json.dumps(self.splits), encoding="utf-8")

    def write_metadata(self, fields=("SID", "AHI", "BMI", "AGE", "GENDER")):
        with self.metadata_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.metadata_rows)

    def preprocessor(self, **kwargs):
        return Preprocessor("DREAMT", data_dir=self.data, metadata_path=self.metadata_path,
                            split_path=self.split_path, **kwargs)

    def test_original_style_npz_requires_no_simulator_fields_or_timeline(self):
        processor = self.preprocessor()
        record = processor.record("A.npz")
        self.assertEqual(record["ibi"].shape, (8*EPOCH,))
        self.assertEqual(record["ibi"].dtype, np.float32)
        self.assertEqual(record["continuity_source"], "stored_order_only")
        np.testing.assert_array_equal(record["stage_epochs"], [0, 1, 1, 3, 2, 0, 1, 1])
        np.testing.assert_array_equal(record["stages"], np.repeat(record["stage_epochs"], EPOCH))
        self.assertEqual(len(processor.make_training_dataset(3000, 1500)), 6)

    def test_ten_channels_four_class_onehot_and_static_metadata(self):
        processor = self.preprocessor()
        self.assertEqual(list(processor.columns), EXPECTED_COLUMNS)
        self.assertEqual(list(processor.schema_frame.columns), EXPECTED_COLUMNS)
        self.assertEqual(list(processor.hierarchical_features_cyclic), EXPECTED_COLUMNS[1:])
        array = processor.window_array("A.npz", 0, 3000)
        self.assertEqual(array.shape, (3000, 10))
        self.assertEqual(array.dtype, np.float32)
        expected_stages = np.repeat([0, 1, 1, 3], EPOCH)
        np.testing.assert_array_equal(array[:, 1:5], np.eye(4, dtype=np.float32)[expected_stages])
        np.testing.assert_allclose(array[:, 5:8], np.broadcast_to([-1., -1., -1.], (3000, 3)), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(array[:, 8:], np.broadcast_to([1., 0.], (3000, 2)))
        np.testing.assert_allclose(processor.inverse_signal(array[:, 0]), self.payloads["A"]["data"][:3000, 0], atol=1e-7, rtol=0)
        tensor = processor.make_training_dataset(3000, 1500)[0]
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tuple(tensor.shape), (3000, 10))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_only_valid_training_epochs_fit_signal_scaler(self):
        def invalidate(payload):
            payload["stages"][2*EPOCH:3*EPOCH] = 5
            payload["data"][2*EPOCH:3*EPOCH] = 10000.
            payload["data"][5*EPOCH + 17] = np.nan
        changed = rewrite_record(self.data / "A.npz", invalidate)
        processor = self.preprocessor()
        valid_a = np.ones(8, dtype=bool)
        valid_a[[2, 5]] = False
        expected = np.concatenate((changed["data"].reshape(8, EPOCH)[valid_a].reshape(-1), self.payloads["B"]["data"].reshape(-1))).astype(np.float64)
        self.assertEqual(processor.scaler["signal_fit_samples"], 14*EPOCH)
        self.assertAlmostEqual(processor.scaler["signal_mean_seconds"], float(expected.mean()), places=9)
        self.assertAlmostEqual(processor.scaler["signal_std_seconds"], float(expected.std()), places=9)
        np.testing.assert_array_equal(processor.record("A.npz")["valid_epochs"], valid_a)

    def test_train_only_scaler_and_sealed_test_survive_extreme_holdout(self):
        loaded = []
        original_load = np.load
        def observed_load(path, *args, **kwargs):
            loaded.append(Path(path).name)
            return original_load(path, *args, **kwargs)
        with mock.patch.object(np, "load", side_effect=observed_load):
            processor = self.preprocessor()
            before = json.dumps(processor.scaler, sort_keys=True)
            self.assertNotIn("V.npz", loaded)
            self.assertNotIn("T.npz", loaded)
            processor.window_array("V.npz", 0, 3000)
            self.assertEqual(json.dumps(processor.scaler, sort_keys=True), before)
            with self.assertRaises((ValueError, PermissionError)):
                processor.record("T.npz")
            with self.assertRaises((ValueError, PermissionError)):
                WindowDataset(processor, "test", 3000, 1500)
            self.assertNotIn("T.npz", loaded)
        expected = np.concatenate([self.payloads[name]["data"].reshape(-1) for name in ("A", "B")]).astype(np.float64)
        self.assertAlmostEqual(processor.scaler["signal_mean_seconds"], float(expected.mean()), places=9)
        self.assertAlmostEqual(processor.scaler["signal_std_seconds"], float(expected.std()), places=9)
        np.testing.assert_allclose(processor.scaler["continuous_mean"], [6., 25., 42.5], atol=1e-12)
        self.assertEqual(processor.scaler["metadata_fit_subjects"], 2)

    def test_gap_excludes_cross_gap_windows_preserving_original_positions(self):
        rewrite_record(self.data / "A.npz", lambda p: p.update(epoch_index=np.array([0, 1, 2, 4, 5, 6, 7, 8])))
        processor = self.preprocessor()
        dataset = processor.make_training_dataset(3000, 1500)
        self.assertEqual([start for name, start in dataset.window_index if name == "A.npz"], [4*EPOCH])
        self.assertEqual(processor.record("A.npz")["ibi"].shape, (8*EPOCH,))
        for start in (0, 2*EPOCH):
            with self.subTest(start=start), self.assertRaises(ValueError):
                processor.window_array("A.npz", start, 3000)
        np.testing.assert_allclose(processor.inverse_signal(processor.window_array("A.npz", 4*EPOCH, 3000)[:, 0]),
                                   self.payloads["A"]["data"][4*EPOCH:, 0], atol=1e-7, rtol=0)

    def test_invalid_epoch_is_not_deleted_to_create_artificial_windows(self):
        def invalidate(payload):
            payload["stages"][2*EPOCH:3*EPOCH] = 5
        rewrite_record(self.data / "A.npz", invalidate)
        processor = self.preprocessor()
        dataset = processor.make_training_dataset(3000, 1500)
        self.assertEqual([start for name, start in dataset.window_index if name == "A.npz"], [4*EPOCH])
        record = processor.record("A.npz")
        self.assertEqual(record["ibi"].shape, (8*EPOCH,))
        self.assertEqual(len(record["valid_epochs"]), 8)
        self.assertFalse(record["valid_epochs"][2])
        for start in (0, 2*EPOCH):
            with self.subTest(start=start), self.assertRaises(ValueError):
                processor.window_array("A.npz", start, 3000)

    def test_missing_npz_ahi_falls_back_to_csv(self):
        rewrite_record(self.data / "A.npz", lambda p: p.pop("ahi"))
        processor = self.preprocessor()
        self.assertEqual(processor.record("A.npz")["continuous"][0], 2.)
        np.testing.assert_allclose(processor.window_array("A.npz", 0, 3000)[:, 5], -1., atol=1e-6)

    def test_present_npz_ahi_must_match_csv(self):
        rewrite_record(self.data / "A.npz", lambda p: p.update(ahi=np.asarray(25.)))
        with self.assertRaises(ValueError):
            self.preprocessor()

    def test_required_metadata_columns_and_invalid_rows_are_rejected(self):
        for removed in ("SID", "AHI", "BMI", "AGE", "GENDER"):
            with self.subTest(missing_column=removed):
                self.write_metadata([field for field in ("SID", "AHI", "BMI", "AGE", "GENDER") if field != removed])
                with self.assertRaises(ValueError):
                    self.preprocessor()
        self.write_metadata()
        for key, value in (("AGE", "not_numeric"), ("BMI", np.nan), ("AHI", np.inf), ("GENDER", "unknown")):
            with self.subTest(key=key, value=value):
                previous = self.metadata_rows[0][key]
                self.metadata_rows[0][key] = value
                self.write_metadata()
                with self.assertRaises(ValueError):
                    self.preprocessor()
                self.metadata_rows[0][key] = previous
        self.write_metadata()

    def test_duplicate_or_missing_subject_metadata_is_rejected(self):
        self.metadata_rows.append(self.metadata_rows[0].copy())
        self.write_metadata()
        with self.assertRaises(ValueError):
            self.preprocessor()
        self.metadata_rows = [row for row in self.metadata_rows if row["SID"] != "A"]
        self.write_metadata()
        with self.assertRaises(ValueError):
            self.preprocessor()

    def test_subject_split_overlap_is_rejected(self):
        self.splits["val"] = ["A.npz", "V.npz"]
        self.split_path.write_text(json.dumps(self.splits), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.preprocessor()

    def test_explicit_multiplier_downsampling_and_epoch_cap_reach_adapter(self):
        labels = {"A": [0, 1, 2, 3, 4, 0, 1, 2], "B": [4, 3, 2, 1, 0, 4, 3, 2],
                  "V": [0, 1, 2, 3, 4, 0, 1, 2], "T": [0, 1, 2, 3, 4, 0, 1, 2]}
        for row in self.metadata_rows:
            write_record(self.data, row["SID"] + ".npz", labels[row["SID"]], fs=50, tail=31,
                         ahi=row["AHI"], offset=.01*row["AHI"])
        processor = self.preprocessor(multiplier=1.2, downsample_rate=2, max_epochs=5)
        record = processor.record("A.npz")
        self.assertEqual(record["ibi"].shape, (5*EPOCH,))
        oracle = OriginalSSDataset(self.data, "dreamt", ["A.npz"], multiplier=1.2, downsample_rate=2)[0][0]
        np.testing.assert_array_equal(record["ibi"], oracle[:5].numpy().reshape(-1))
        self.assertEqual(len(processor.make_training_dataset(3000, 1500)), 2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
