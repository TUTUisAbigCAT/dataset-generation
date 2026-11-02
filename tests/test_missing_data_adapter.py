"""Missing-zero policy regression with small artificial WatchSleep-compatible files."""
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
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils import MASKED_ZERO_VALUE_POLICY, ZERO_VALUE_POLICY, Preprocessor


class MissingDataAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = (ROOT / "work" / "missing_data_adapter_tests").resolve()
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.root = Path(self.temp.name)
        self.signals = {
            "A": np.tile(np.array([0, .5, 1, 1.5], dtype=np.float32), 1500),
            "B": np.tile(np.array([0, .5, 1, 1.5], dtype=np.float32), 1500),
            "V": np.tile(np.array([0, 100, 200, 300], dtype=np.float32), 1500),
            "T": np.full(6000, 999, dtype=np.float32),
        }
        self.ahi = {"A": 5., "B": 15., "V": 25., "T": 35.}
        for sid in self.signals:
            self.save_record(sid)
        (self.root / "participant_info.csv").write_text(
            "SID,AHI,BMI,AGE,GENDER\nA,5,20,20,F\nB,15,30,40,M\nV,25,40,60,F\nT,35,50,80,M\n",
            encoding="utf-8")
        (self.root / "splits.json").write_text(json.dumps({
            "train": ["A.npz", "B.npz"], "val": ["V.npz"], "test": ["T.npz"]}), encoding="utf-8")

    def tearDown(self):
        resolved = self.root.resolve()
        self.assertNotEqual(resolved, self.temp_root)
        self.assertTrue(resolved.is_relative_to(self.temp_root))
        self.temp.cleanup()

    def save_record(self, sid, *, stages=None, epoch_index=None):
        payload = dict(data=self.signals[sid][:, None], fs=25,
                       stages=np.array([0, 1, 2, 3, 4, 1, 0, 3] if stages is None else stages),
                       ahi=self.ahi[sid], synthetic=True)
        if epoch_index is not None:
            payload["epoch_index"] = np.asarray(epoch_index)
        np.savez(self.root / f"{sid}.npz", **payload)

    def processor(self, policy="masked", window_epochs=4, **kwargs):
        config = dict(data_dir=str(self.root), window_epochs=window_epochs,
                      missing_value_policy=policy)
        return Preprocessor("DREAMT", config_override=config, **kwargs)

    def test_signal_scaler_excludes_raw_zero_and_uses_train_only(self):
        processor = self.processor()
        self.assertEqual(processor.scaler["signal_fit_samples"], 9000)
        self.assertEqual(processor.scaler["signal_mean_seconds"], 1.)
        self.assertAlmostEqual(processor.scaler["signal_std_seconds"], np.sqrt(1 / 6))
        self.assertEqual(processor.scaler["fit_split"], "train")
        np.testing.assert_array_equal(processor.scaler["continuous_mean"], [10, 25, 30])
        np.testing.assert_array_equal(processor.scaler["continuous_std"], [5, 5, 10])
        self.assertEqual(processor.accessed_files, ["A.npz", "B.npz"])
        before_scaler = copy.deepcopy(processor.scaler)
        before_identity = processor.data_identity()
        processor.window_array("V.npz", 0, 3000)
        processor.window_observed_mask("V.npz", 0, 3000)
        self.assertEqual(processor.scaler, before_scaler)
        self.assertEqual(processor.data_identity(), before_identity)
        self.assertNotIn("T.npz", processor.accessed_files)
        with self.assertRaisesRegex(ValueError, "test is sealed"):
            processor.record("T.npz")

    def test_raw_mask_distinguishes_valid_normalized_zero_from_missing_placeholder(self):
        processor = self.processor()
        original_record = processor.record("A.npz")["ibi"].copy()
        original_bytes = (self.root / "A.npz").read_bytes()
        window = processor.window_array("A.npz", 0, 3000)
        mask = processor.window_observed_mask("A.npz", 0, 3000)
        self.assertEqual(window.shape, (3000, 10))
        self.assertEqual(window.dtype, np.float32)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(window[0, 0], 0.)  # raw 0, unobserved
        self.assertFalse(mask[0])
        self.assertEqual(window[2, 0], 0.)  # raw 1 equals the fitted mean
        self.assertTrue(mask[2])
        np.testing.assert_array_equal(mask, self.signals["A"][:3000] > 0)
        np.testing.assert_array_equal(window[~mask, 0], 0.)
        np.testing.assert_allclose(processor.inverse_signal(window[mask, 0]),
                                   self.signals["A"][:3000][mask], atol=1e-7)
        mask[:] = False
        np.testing.assert_array_equal(processor.record("A.npz")["ibi"], original_record)
        self.assertEqual((self.root / "A.npz").read_bytes(), original_bytes)

    def test_masked_dataset_returns_mask_sidecar_and_preserve_retains_tensor(self):
        masked, preserved = self.processor(), self.processor("preserve")
        a = masked.make_training_dataset(3000, 750)[0]
        b = preserved.make_training_dataset(3000, 750)[0]
        self.assertEqual(set(a), {"data", "observed_mask"})
        self.assertEqual(a["data"].dtype, torch.float32)
        self.assertEqual(a["observed_mask"].dtype, torch.bool)
        self.assertEqual(tuple(a["observed_mask"].shape), (3000,))
        self.assertIsInstance(b, torch.Tensor)
        self.assertEqual(tuple(b.shape), (3000, 10))
        torch.testing.assert_close(a["data"][:, 1:], b[:, 1:], rtol=0, atol=0)
        self.assertNotEqual(float(b[0, 0]), 0.)
        batch = next(iter(DataLoader(masked.make_training_dataset(3000, 750), batch_size=2)))
        self.assertEqual(tuple(batch["data"].shape), (2, 3000, 10))
        self.assertEqual(tuple(batch["observed_mask"].shape), (2, 3000))
        self.assertEqual(batch["observed_mask"].dtype, torch.bool)
        np.testing.assert_array_equal(preserved.window_observed_mask("A.npz", 0, 3000),
                                      self.signals["A"][:3000] > 0)

    def test_all_window_paths_require_two_observations_only_in_masked_mode(self):
        for sid in ["A", "V"]:
            self.signals[sid][:1500] = 0
            self.signals[sid][750 + 10] = .5
            self.signals[sid][1500:2250] = 0
            self.signals[sid][1500:1502] = [.5, 1.5]
            self.save_record(sid)
        masked = self.processor(window_epochs=1)
        training = masked.make_training_dataset(750, 750)
        self.assertEqual([start for name, start in training.window_index if name == "A.npz"],
                         [1500, 2250, 3000, 3750, 4500, 5250])
        self.assertEqual(training.excluded_windows, 2)
        validation = masked.make_validation_dataset(stride=750)
        self.assertEqual([start for _, start in validation.window_index],
                         [1500, 2250, 3000, 3750, 4500, 5250])
        self.assertEqual(sorted(masked.select_validation_windows(99)), validation.window_index)
        for name in ["A.npz", "V.npz"]:
            for start in [0, 750]:
                with self.subTest(name=name, start=start):
                    with self.assertRaisesRegex(ValueError, "at least two"):
                        masked.window_array(name, start, 750)
                    with self.assertRaisesRegex(ValueError, "at least two"):
                        masked.window_observed_mask(name, start, 750)
        self.assertEqual(masked.window_observed_mask("A.npz", 1500, 750).sum(), 2)
        preserved = self.processor("preserve", window_epochs=1)
        self.assertEqual(len(preserved.make_training_dataset(750, 750)), 16)
        self.assertFalse(preserved.window_observed_mask("A.npz", 0, 750).any())
        self.assertTrue(torch.is_tensor(preserved.make_training_dataset(750, 750)[0]))

    def test_observation_requirement_applies_to_the_whole_configured_window(self):
        self.signals["V"][:] = 0
        self.signals["V"][5250:5252] = [.5, 1.5]
        self.save_record("V")
        processor = self.processor()
        validation = processor.make_validation_dataset(stride=750)
        self.assertEqual(validation.window_index, [("V.npz", 3000)])
        self.assertEqual(processor.select_validation_windows(99), [("V.npz", 3000)])
        self.assertEqual(processor.window_observed_mask("V.npz", 3000, 3000).sum(), 2)
        self.assertEqual(processor.window_array("V.npz", 3000, 3000).shape, (3000, 10))

    def test_no_observed_training_values_and_no_usable_validation_windows_reject(self):
        self.signals["V"][:] = 0
        self.save_record("V")
        processor = self.processor()
        with self.assertRaisesRegex(ValueError, "No complete valid"):
            processor.make_validation_dataset()
        with self.assertRaisesRegex(ValueError, "No complete valid"):
            processor.select_validation_windows()
        for sid in ["A", "B"]:
            self.signals[sid][:] = 0
            self.save_record(sid)
        with self.assertRaisesRegex(ValueError, "no observed positive"):
            self.processor()

    def test_existing_invalid_epochs_and_known_gaps_remain_excluded(self):
        self.signals["A"][750] = np.nan
        self.signals["A"][2250] = -.5
        self.save_record("A", stages=[0, 1, 5, 3, 4, 1, 0, 3],
                         epoch_index=[0, 1, 2, 3, 4, 5, 7, 8])
        processor = self.processor(window_epochs=2)
        record = processor.record("A.npz")
        np.testing.assert_array_equal(record["valid_epochs"],
                                      [True, False, False, False, True, True, True, True])
        actual = [start for name, start in processor.make_training_dataset(1500, 750).window_index
                  if name == "A.npz"]
        self.assertEqual(actual, [3000, 4500])
        for start in [0, 750, 1500, 2250, 3750]:
            with self.subTest(start=start):
                with self.assertRaisesRegex(ValueError, "invalid epoch or known time gap"):
                    processor.window_observed_mask("A.npz", start, 1500)
        train_values = []
        for name in processor.splits["train"]:
            row = processor.record(name)
            valid = row["ibi"].reshape(-1, 750)[row["valid_epochs"]]
            train_values.append(valid[valid > 0].astype(np.float64))
        expected = np.concatenate(train_values)
        self.assertEqual(processor.scaler["signal_fit_samples"], len(expected))
        self.assertAlmostEqual(processor.scaler["signal_mean_seconds"], expected.mean())
        self.assertAlmostEqual(processor.scaler["signal_std_seconds"], expected.std())

    def test_metadata_scaler_does_not_change_when_a_structurally_valid_subject_has_no_observations(self):
        self.signals["A"][:] = 0
        self.save_record("A")
        masked, preserved = self.processor(), self.processor("preserve")
        self.assertEqual(masked.scaler["signal_fit_samples"], 4500)
        self.assertEqual(masked.scaler["metadata_fit_subjects"], 2)
        self.assertEqual(masked.scaler["continuous_mean"], preserved.scaler["continuous_mean"])
        self.assertEqual(masked.scaler["continuous_std"], preserved.scaler["continuous_std"])
        self.assertTrue(all(name == "B.npz" for name, _ in
                            masked.make_training_dataset(3000, 750).window_index))

    def test_masked_identity_round_trip_ignores_current_configuration_and_rejects_policy_tampering(self):
        original = self.processor()
        identity = original.data_identity()
        self.assertEqual(identity["missing_value_policy"], "masked")
        self.assertEqual(identity["zero_value_policy"], MASKED_ZERO_VALUE_POLICY)
        bad_config = self.root / "unrelated.json"
        bad_config.write_text(json.dumps({"data_dir": "missing", "missing_value_policy": "preserve"}), encoding="utf-8")
        with mock.patch.dict(os.environ, {"DREAMT_CONFIG_PATH": str(bad_config), "DREAMT_DATA_DIR": "missing"}):
            restored = Preprocessor.from_identity(identity)
        self.assertEqual(restored.missing_value_policy, "masked")
        self.assertEqual(restored.data_identity(), identity)
        np.testing.assert_array_equal(restored.window_array("A.npz", 0, 3000),
                                      original.window_array("A.npz", 0, 3000))
        tampered = copy.deepcopy(identity)
        del tampered["missing_value_policy"]
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            Preprocessor.from_identity(tampered)
        provenance = restored.provenance()
        self.assertEqual(provenance["missing_value_policy"], "masked")
        self.assertEqual(provenance["zero_value_policy"], MASKED_ZERO_VALUE_POLICY)
        self.assertEqual(len(provenance["columns"]), 10)

    def test_legacy_preserve_identity_and_original_scaler_remain_unchanged(self):
        processor = self.processor("preserve")
        identity = processor.data_identity()
        self.assertNotIn("missing_value_policy", identity)
        self.assertEqual(identity["zero_value_policy"], ZERO_VALUE_POLICY)
        self.assertEqual(processor.scaler["signal_fit_samples"], 12000)
        self.assertEqual(processor.scaler["signal_mean_seconds"], .75)
        self.assertAlmostEqual(processor.scaler["signal_std_seconds"], np.sqrt(.3125))
        config = self.root / "masked.json"
        config.write_text(json.dumps({"data_dir": str(self.root), "missing_value_policy": "masked"}), encoding="utf-8")
        with mock.patch.dict(os.environ, {"DREAMT_CONFIG_PATH": str(config)}):
            restored = Preprocessor.from_identity(identity)
        self.assertEqual(restored.missing_value_policy, "preserve")
        self.assertEqual(restored.data_identity(), identity)
        self.assertIsInstance(restored.make_training_dataset(3000, 750)[0], torch.Tensor)
        serialized = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(serialized(restored.data_identity()), serialized(identity))

    def test_default_preserve_explicit_override_and_invalid_config_values(self):
        default = Preprocessor("DREAMT", config_override={"data_dir": str(self.root)})
        self.assertEqual(default.missing_value_policy, "preserve")
        explicit = self.processor("masked", missing_value_policy="preserve")
        self.assertEqual(explicit.missing_value_policy, "preserve")
        self.assertEqual(explicit.data_identity(), default.data_identity())
        for invalid in [None, True, False, 0, 1, "MASKED", "", "impute", []]:
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "missing_value_policy"):
                self.processor(invalid)

    def test_long_span_mask_and_array_stay_aligned_without_changing_epoch_labels(self):
        processor = self.processor(window_epochs=2)
        whole = processor.window_array("A.npz", 0, 6000)
        mask = processor.window_observed_mask("A.npz", 0, 6000)
        chunks = [processor.window_array("A.npz", i, 1500) for i in range(0, 6000, 1500)]
        masks = [processor.window_observed_mask("A.npz", i, 1500) for i in range(0, 6000, 1500)]
        np.testing.assert_array_equal(whole, np.concatenate(chunks))
        np.testing.assert_array_equal(mask, np.concatenate(masks))
        np.testing.assert_array_equal(whole[:, 1:5].argmax(axis=1),
                                      np.repeat([0, 1, 1, 3, 2, 1, 0, 3], 750))


if __name__ == "__main__":
    unittest.main()
