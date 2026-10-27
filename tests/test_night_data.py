"""Whole-night inference data boundaries and original-loader output compatibility."""
from __future__ import annotations

import copy
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils import Preprocessor
from night_data import prepare_night, save_night_npz
from scripts.audit_watchsleep_chain import original_ssdataset
from watchsleep_loader import SSDataset


class NightDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.records = self.directory / "records"
        self.records.mkdir()
        self.raw_stages = np.array([0, 1, 2, 3, 4, 0, 4, 3], dtype=np.int64)
        self.four_stages = np.array([0, 1, 1, 3, 2, 0, 2, 3], dtype=np.int64)
        names = ["S001", "S002", "S092"]
        for index, sid in enumerate(names):
            signal = (0.75 + 0.03 * index + 0.025 * np.sin(np.arange(6000) / 29)).astype(np.float32)
            np.savez_compressed(self.records / (sid + ".npz"), data=signal[:, None],
                                stages=np.repeat(self.raw_stages, 750), fs=np.array(25),
                                ahi=np.array([10, 20, 40][index], dtype=np.float64))
        with (self.records / "participant_info.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["SID", "AHI", "BMI", "AGE", "GENDER"])
            writer.writeheader()
            for sid, ahi, bmi, age, gender in (("S001", 10, 22, 34, "F"),
                                               ("S002", 20, 27, 44, "F"),
                                               ("S092", 40, 32, 54, "M")):
                writer.writerow(dict(SID=sid, AHI=ahi, BMI=bmi, AGE=age, GENDER=gender))
        (self.records / "splits.json").write_text(
            '{"train":["S001.npz"],"val":["S002.npz"],"test":["S092.npz"]}', encoding="utf-8")
        self.processor = self.make_processor()

    def make_processor(self, **options):
        return Preprocessor("DREAMT", data_config=dict(data_dir=str(self.records), max_epochs=2,
                                                        window_epochs=4, **options))

    def replace_target(self, **updates):
        path = self.records / "S092.npz"
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays.update(updates)
        np.savez_compressed(path, **arrays)

    def test_test_subject_full_night_uses_frozen_scaler_without_changing_identity(self):
        before = copy.deepcopy(self.processor.data_identity())
        accessed = list(self.processor.accessed_files)
        prepared = prepare_night(self.processor, "s092.NPZ")
        self.assertEqual(prepared["subject_id"], "S092")
        self.assertEqual(prepared["filename"], "S092.npz")
        self.assertEqual(prepared["conditions"].shape, (6000, 9))
        self.assertEqual(prepared["conditions"].dtype, np.float32)
        self.assertEqual(prepared["reference_seconds"].dtype, np.float64)
        self.assertEqual(prepared["provenance"]["subject_role"], "test")
        self.assertFalse(prepared["provenance"]["scaler_refitted_for_generation"])
        self.assertFalse(prepared["provenance"]["subject_in_original_training_scaler"])
        self.assertEqual(prepared["provenance"]["training_max_epochs"], 2)
        self.assertIsNone(prepared["provenance"]["generation_max_epochs"])
        self.assertEqual(self.processor.data_identity(), before)
        self.assertEqual(self.processor.accessed_files, accessed)
        self.assertNotIn("S092.npz", self.processor._records)
        with self.assertRaisesRegex(ValueError, "test is sealed"):
            self.processor.record("S092.npz")

    def test_stage_channels_metadata_zscore_and_gender_are_time_aligned(self):
        prepared = prepare_night(self.processor)
        expected = np.eye(4, dtype=np.float32)[np.repeat(self.four_stages, 750)]
        np.testing.assert_array_equal(prepared["conditions"][:, :4], expected)
        # The sole training participant gives constant metadata columns: means
        # [10,22,34], std=[1,1,1]. S092 must not participate in a new fit.
        np.testing.assert_array_equal(prepared["conditions"][:, 4:],
                                      np.tile([30, 10, 20, 0, 1], (6000, 1)))
        np.testing.assert_array_equal(prepared["stage_epochs_four"], self.four_stages)
        np.testing.assert_array_equal(prepared["raw_stage_epochs"], self.raw_stages)
        self.assertEqual(prepared["ahi"], 40.0)

    def test_bad_reference_ibi_changes_no_condition_and_is_reported(self):
        original = prepare_night(self.processor)
        signal = np.full(6000, 10.0, dtype=np.float32)
        signal[[0, 2, 4]] = [np.nan, np.inf, -1]
        self.replace_target(data=signal[:, None])
        prepared = prepare_night(self.processor)
        np.testing.assert_array_equal(prepared["conditions"], original["conditions"])
        self.assertEqual(prepared["provenance"]["reference_nonfinite_samples"], 2)
        self.assertEqual(prepared["provenance"]["reference_negative_samples"], 1)
        self.assertTrue(np.isnan(prepared["reference_seconds"][0]))
        self.assertEqual(prepared["reference_seconds"][4], -1)

    def test_train_and_validation_subjects_are_allowed_without_reclassifying_them(self):
        for sid, role in (("s001", "train"), ("S002", "val")):
            with self.subTest(role=role):
                prepared = prepare_night(self.processor, sid)
                self.assertEqual(prepared["provenance"]["subject_role"], role)
                self.assertEqual(len(prepared["conditions"]), 6000)
                self.assertEqual(prepared["provenance"]["subject_in_original_training_scaler"], role == "train")

    def test_resolution_is_exact_case_insensitive_and_rejects_ambiguity_or_paths(self):
        for subject in ("S09", "unknown", "", None, "../S092", "C:\\S092", "S092/other"):
            with self.subTest(subject=subject), self.assertRaises(ValueError):
                prepare_night(self.processor, subject)
        ambiguous = copy.copy(self.processor)
        ambiguous.splits = dict(train=["S092.npz"], val=[], test=["s092.npz"])
        with self.assertRaisesRegex(ValueError, "found 2"):
            prepare_night(ambiguous, "S092")

    def test_unknown_and_mixed_raw_labels_cannot_be_silently_removed_or_merged(self):
        for label, sample in ((-1, 750), (5, 750), (2, 751)):
            labels = np.repeat(self.raw_stages, 750)
            labels[sample] = label
            self.replace_target(stages=labels)
            with self.subTest(label=label, sample=sample):
                # Mixing raw N1 and N2 within one epoch remains inconsistent,
                # even though both remap to Light.
                with self.assertRaisesRegex(ValueError, "Unknown or inconsistent.*epoch"):
                    prepare_night(self.processor)

    def test_known_source_gaps_fail_but_missing_time_axis_is_explicit(self):
        prepared = prepare_night(self.processor)
        self.assertEqual(prepared["provenance"]["continuity_source"], "stored_order_only")
        self.assertFalse(prepared["provenance"]["source_time_continuity_verified"])
        indices = np.arange(8)
        indices[4:] += 1
        self.replace_target(epoch_index=indices)
        with self.assertRaisesRegex(ValueError, "Known source-time gaps.*\\[3\\]"):
            prepare_night(self.processor)

    def test_epoch_label_input_and_incomplete_tail_are_explicitly_accounted_for(self):
        self.replace_target(data=np.ones((6101, 1), dtype=np.float32),
                            stages=np.append(self.raw_stages, 4))
        prepared = prepare_night(self.processor)
        self.assertEqual(prepared["conditions"].shape, (6000, 9))
        self.assertEqual(prepared["provenance"]["discarded_tail_samples"], 101)
        self.assertEqual(prepared["provenance"]["discarded_tail_seconds"], 101 / 25)
        self.assertEqual(prepared["provenance"]["coverage"], "all_complete_stored_epochs")
        self.assertEqual(prepared["record"]["label_granularity"], "epoch_level")

    def test_source_rate_stride_and_unit_multiplier_match_training_transform(self):
        expected = prepare_night(self.processor)
        for path in self.records.glob("*.npz"):
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in saved.files}
            arrays["data"] = np.repeat(arrays["data"].astype(np.float64) * 1000, 4, axis=0)
            arrays["stages"] = np.repeat(arrays["stages"], 4)
            arrays["fs"] = np.array(100)
            np.savez_compressed(path, **arrays)
        processor = self.make_processor(multiplier=0.001, downsample_rate=4)
        prepared = prepare_night(processor)
        np.testing.assert_array_equal(prepared["reference_seconds"], expected["reference_seconds"])
        np.testing.assert_array_equal(prepared["conditions"], expected["conditions"])
        self.assertEqual(prepared["provenance"]["source_fs"], 100)
        self.assertEqual(prepared["provenance"]["output_fs"], 25)
        self.assertEqual(prepared["provenance"]["output_multiplier"], 1)
        self.assertEqual(prepared["provenance"]["output_downsample_rate"], 1)

    def test_inconsistent_ahi_and_empty_night_are_rejected(self):
        self.replace_target(ahi=np.array(41.0))
        with self.assertRaisesRegex(ValueError, "AHI disagree"):
            prepare_night(self.processor)
        self.replace_target(data=np.empty((0, 1)), stages=np.empty(0, dtype=int), ahi=np.array(40.0))
        with self.assertRaisesRegex(ValueError, "no complete"):
            prepare_night(self.processor)

    def test_export_roundtrips_through_current_and_original_watchsleep_loaders(self):
        self.replace_target(epoch_index=np.arange(100, 108), epoch_start_seconds=np.arange(8) * 30.0)
        prepared = prepare_night(self.processor)
        generated = np.linspace(0.6, 1.0, 6000, dtype=np.float64)
        output = self.directory / "generated" / "S092.npz"
        result = save_night_npz(output, prepared, generated)
        self.assertEqual(result["samples"], 6000)
        with np.load(output, allow_pickle=False) as saved:
            self.assertEqual(saved["data"].shape, (6000, 1))
            self.assertEqual(saved["stages"].shape, (6000,))
            self.assertTrue(saved["synthetic"].item())
            np.testing.assert_array_equal(saved["stages"][::750], self.raw_stages)
            np.testing.assert_array_equal(saved["epoch_index"], np.arange(100, 108))
        record = SSDataset(output.parent, "dreamt_ibi", file_list=[output.name]).read_record(0)
        np.testing.assert_array_equal(record["ibi_epochs"].ravel(), generated.astype(np.float32))
        np.testing.assert_array_equal(record["stage_epochs"], self.four_stages)
        self.assertTrue(record["valid_epochs"].all())
        self.assertTrue(record["adjacent"].all())
        self.assertEqual(record["ahi"], 40.0)
        original = original_ssdataset()
        loaded = original(output.parent, "dreamt_ibi", file_list=[output.name])[0]
        np.testing.assert_array_equal(loaded[0].numpy().ravel(), generated.astype(np.float32))
        np.testing.assert_array_equal(loaded[1].numpy(), self.four_stages)
        self.assertEqual(loaded[2], 8)
        self.assertEqual(loaded[3], 40.0)

    def test_export_rejects_invalid_values_without_clipping_or_overwriting(self):
        prepared = prepare_night(self.processor)
        output = self.directory / "output.npz"
        for value in (np.nan, np.inf, -1, 1e40):
            generated = np.ones(6000, dtype=np.float64)
            generated[123] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                save_night_npz(output, prepared, generated)
            self.assertFalse(output.exists())
        for values in (np.ones(6001), np.ones((6000, 1)), np.ones(6000, dtype=int)):
            with self.subTest(shape=values.shape, dtype=values.dtype), self.assertRaises(ValueError):
                save_night_npz(output, prepared, values)
        generated = np.ones(6000, dtype=np.float32)
        save_night_npz(output, prepared, generated)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            save_night_npz(output, prepared, generated * 2)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
