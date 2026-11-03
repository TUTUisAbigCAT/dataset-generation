"""Data-only regressions for all-subject training with in-sample S092 checks.

Fixtures are generated locally; no model, training step, or sampling runs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils import ALL_SUBJECTS_EVALUATION_MODE, Preprocessor, evaluation_protocol_options


class AllSubjectProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = (ROOT / "work" / "all_subject_protocol_tests").resolve()
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.directory = Path(self.temp.name)
        self.subjects = ["S001", "S002", "S003", "S092"]
        self.rows = {}
        self.signals = {}
        for index, sid in enumerate(self.subjects):
            self.add_subject(sid, index)
        self.write_metadata()
        self.legacy_splits = dict(train=["S001.npz", "S002.npz"],
                                  val=["S003.npz"], test=["S092.npz"])
        (self.directory / "splits.json").write_text(json.dumps(self.legacy_splits), encoding="utf-8")
        self.protocol = dict(mode=ALL_SUBJECTS_EVALUATION_MODE, subject="S092", expected_subjects=4)

    def tearDown(self):
        resolved = self.directory.resolve()
        self.assertNotEqual(resolved, self.temp_root)
        self.assertTrue(resolved.is_relative_to(self.temp_root))
        self.temp.cleanup()

    def add_subject(self, sid, index, *, valid=True):
        self.rows[sid] = [5. + index, 20. + index, 30. + index * 2]
        signal = np.tile(np.array([0, .5, 1, 1.5], dtype=np.float32), 1500)
        signal[signal > 0] += index * .005
        self.signals[sid] = signal
        np.savez(self.directory / (sid + ".npz"), data=signal[:, None], fs=25,
                 stages=np.array([0, 1, 2, 3, 4, 1, 0, 3] if valid else [-1] * 8),
                 ahi=self.rows[sid][0], synthetic=True)

    def write_metadata(self):
        rows = ["SID,AHI,BMI,AGE,GENDER"]
        rows.extend(",".join([sid, *(str(value) for value in row), "F"])
                    for sid, row in self.rows.items())
        (self.directory / "participant_info.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    def processor(self, **extra):
        config = dict(data_dir=str(self.directory), window_epochs=4,
                      missing_value_policy="zero_fill_one", evaluation_protocol=dict(self.protocol))
        config.update(extra)
        return Preprocessor("DREAMT", config_override=config)

    def test_all_files_train_and_only_s092_is_evaluated_without_held_out_splits(self):
        before = {path.name: path.read_bytes() for path in self.directory.iterdir()}
        processor = self.processor()
        filenames = [sid + ".npz" for sid in self.subjects]
        self.assertEqual(processor.splits, dict(train=filenames, val=[], test=[]))
        self.assertEqual(processor.split_source, "all_subjects")
        self.assertIsNone(processor.split_path)
        self.assertIsNone(processor.split_file_sha256)
        self.assertEqual(processor.evaluation_filenames, ["S092.npz"])
        self.assertEqual(processor.accessed_files, filenames)
        train = processor.make_training_dataset(3000, 750)
        evaluation = processor.make_validation_dataset(stride=750)
        self.assertEqual({name for name, _ in train.window_index}, set(filenames))
        self.assertEqual({name for name, _ in evaluation.window_index}, {"S092.npz"})
        self.assertEqual(len(evaluation), 5)
        self.assertTrue(all(name == "S092.npz" for name, _ in processor.select_validation_windows(99)))
        provenance = processor.provenance()
        self.assertFalse(provenance["held_out_subjects"])
        self.assertFalse(provenance["evaluation_is_held_out"])
        self.assertTrue(provenance["evaluation_included_in_training"])
        self.assertEqual(provenance["evaluation_protocol"], self.protocol)
        self.assertEqual(provenance["evaluation_filenames"], ["S092.npz"])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.directory.iterdir()})

    def test_scalers_include_s092_and_all_other_subjects_once(self):
        processor = self.processor()
        raw = np.concatenate(list(self.signals.values())).astype(np.float64)
        expected = np.where(raw == 0, 1., raw)
        self.assertEqual(processor.scaler["metadata_fit_subjects"], 4)
        self.assertEqual(processor.scaler["signal_fit_samples"], 24000)
        self.assertEqual(processor.scaler_subjects, self.subjects)
        self.assertAlmostEqual(processor.scaler["signal_mean_seconds"], expected.mean())
        self.assertAlmostEqual(processor.scaler["signal_std_seconds"], expected.std())
        np.testing.assert_allclose(processor.scaler["continuous_mean"], np.mean(list(self.rows.values()), axis=0))
        np.testing.assert_allclose(processor.scaler["continuous_std"], np.std(list(self.rows.values()), axis=0))
        window = processor.window_array("S092.npz", 0, 3000)
        np.testing.assert_allclose(processor.inverse_signal(window[:, 0]),
                                   np.where(self.signals["S092"][:3000] == 0, 1., self.signals["S092"][:3000]),
                                   atol=1e-7)
        np.testing.assert_array_equal(processor.window_observed_mask("S092.npz", 0, 3000),
                                      self.signals["S092"][:3000] > 0)

    def test_all_subject_metadata_fit_includes_subject_without_valid_signal(self):
        self.add_subject("S001", 0, valid=False)
        processor = self.processor()
        self.assertEqual(processor.scaler["metadata_fit_subjects"], 4)
        self.assertEqual(processor.scaler_subjects, self.subjects)
        self.assertEqual(processor.scaler["signal_fit_samples"], 18000)
        np.testing.assert_allclose(processor.scaler["continuous_mean"], np.mean(list(self.rows.values()), axis=0))

    def test_production_count_88_includes_every_subject_and_refits_metadata(self):
        for index in range(84):
            self.add_subject(f"S{100 + index:03d}", index + 4)
        self.write_metadata()
        protocol = dict(self.protocol, expected_subjects=88)
        processor = self.processor(evaluation_protocol=protocol)
        self.assertEqual(len(processor.splits["train"]), 88)
        self.assertIn("S092.npz", processor.splits["train"])
        self.assertEqual(processor.scaler["metadata_fit_subjects"], 88)
        self.assertEqual(len(processor.scaler_subjects), 88)
        self.assertEqual(len(processor.data_identity()["train_npz_sha256"]), 88)
        np.testing.assert_allclose(processor.scaler["continuous_mean"], np.mean(list(self.rows.values()), axis=0))
        self.assertEqual({name for name, _ in processor.make_validation_dataset().window_index}, {"S092.npz"})

    def test_new_mode_ignores_local_or_explicit_split_file_and_roundtrips(self):
        (self.directory / "splits.json").write_text("invalid old split file", encoding="utf-8")
        processor = self.processor(split_path="does-not-exist.json", generate_splits=False)
        identity = processor.data_identity()
        self.assertEqual(identity["evaluation_protocol"], self.protocol)
        self.assertEqual(identity["evaluation_filenames"], ["S092.npz"])
        self.assertEqual(set(identity["train_npz_sha256"]), set(processor.splits["train"]))
        with mock.patch.dict(os.environ, {"DREAMT_CONFIG_PATH": "missing.json", "DREAMT_DATA_DIR": "missing",
                                          "DREAMT_SPLIT_PATH": "missing.json"}):
            restored = Preprocessor.from_training_identity(identity)
        self.assertEqual(restored.data_identity(), identity)
        self.assertEqual(restored.evaluation_filenames, ["S092.npz"])
        np.testing.assert_array_equal(restored.window_array("S092.npz", 0, 3000),
                                      processor.window_array("S092.npz", 0, 3000))

    def test_fingerprint_and_identity_bind_evaluation_membership_and_all_training_files(self):
        processor = self.processor()
        identity = processor.data_identity()
        other = self.processor(evaluation_protocol=dict(self.protocol, subject="S003"))
        self.assertNotEqual(processor.split_fingerprint, other.split_fingerprint)
        variants = []
        for field in ("evaluation_protocol", "evaluation_filenames"):
            altered = copy.deepcopy(identity)
            del altered[field]
            variants.append(altered)
        altered = copy.deepcopy(identity)
        altered["evaluation_filenames"] = ["S003.npz"]
        variants.append(altered)
        altered = copy.deepcopy(identity)
        altered["evaluation_protocol"]["subject"] = "S003"
        variants.append(altered)
        altered = copy.deepcopy(identity)
        altered["splits"]["train"].remove("S092.npz")
        altered["splits"]["test"].append("S092.npz")
        variants.append(altered)
        altered = copy.deepcopy(identity)
        altered["split_source"] = "generated_subject_split"
        variants.append(altered)
        for altered in variants:
            with self.subTest(identity=altered), self.assertRaises(ValueError):
                Preprocessor.from_identity(altered)
        # The monitored subject is part of the frozen training hash inventory.
        # Reverse samples to keep fitted moments and participant metadata fixed.
        np.savez(self.directory / "S092.npz", data=self.signals["S092"][::-1, None], fs=25,
                 stages=np.array([0, 1, 2, 3, 4, 1, 0, 3]), ahi=self.rows["S092"][0], synthetic=True)
        with self.assertRaisesRegex(ValueError, "train_npz_sha256"):
            Preprocessor.from_identity(identity)

    def test_constructing_and_selecting_evaluation_windows_preserves_global_numpy_rng(self):
        before = np.random.get_state()
        processor = self.processor()
        processor.select_validation_windows()
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_count_and_exact_subject_are_checked_before_reading_npz(self):
        invalid_protocols = [dict(self.protocol, expected_subjects=88),
                             dict(self.protocol, expected_subjects=3),
                             dict(self.protocol, subject="S999"),
                             dict(self.protocol, subject="s092")]
        with mock.patch("data_utils.SSDataset.read_record", side_effect=AssertionError("must not open NPZ")):
            for protocol in invalid_protocols:
                with self.subTest(protocol=protocol), self.assertRaises(ValueError):
                    self.processor(evaluation_protocol=protocol)

    def test_invalid_protocol_configuration_is_rejected(self):
        invalid = [None, False, [], {}, dict(self.protocol, unknown=1),
                   dict(self.protocol, mode="held_out"), dict(self.protocol, expected_subjects=True),
                   dict(self.protocol, expected_subjects=0), dict(self.protocol, expected_subjects=-1),
                   dict(self.protocol, expected_subjects=4.), dict(self.protocol, expected_subjects="4")]
        invalid.extend(dict(self.protocol, subject=subject)
                       for subject in (None, "", " S092", "S092 ", "../S092", "x/S092", "x\\S092", "S092.npz"))
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluation_protocol_options({"evaluation_protocol": value})

    def test_legacy_split_identity_and_sealed_test_behavior_are_unchanged(self):
        processor = Preprocessor("DREAMT", config_override=dict(data_dir=str(self.directory), window_epochs=4))
        self.assertIsNone(processor.evaluation_protocol)
        self.assertEqual(processor.evaluation_filenames, ["S003.npz"])
        self.assertEqual(processor.splits, self.legacy_splits)
        self.assertEqual(processor.split_source, "split_file")
        self.assertEqual(processor.accessed_files, ["S001.npz", "S002.npz"])
        with self.assertRaisesRegex(ValueError, "test is sealed"):
            processor.record("S092.npz")
        identity = processor.data_identity()
        self.assertNotIn("evaluation_protocol", identity)
        self.assertNotIn("evaluation_filenames", identity)
        self.assertNotIn("held_out_subjects", processor.provenance())
        historical_fingerprint_payload = dict(filenames=[sid + ".npz" for sid in self.subjects],
            seed=42, source="split_file", policy="explicit_file", splits=self.legacy_splits)
        expected = hashlib.sha256(json.dumps(historical_fingerprint_payload, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        self.assertEqual(processor.split_fingerprint, expected)
        self.assertEqual(Preprocessor.from_identity(identity).data_identity(), identity)


if __name__ == "__main__":
    unittest.main()
