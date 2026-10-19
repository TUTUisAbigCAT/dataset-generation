"""Integration checks for independently composed original-loader windows.

Existing tests cover individual mappings, metadata validation and split rules.
These tests target composition across epoch loading, scaling, full conditional
windows and inverse scaling, plus the classification padding/length boundary.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/loader_chain_tests"
WORK.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("independent_watchsleep_chain_audit", ROOT / "scripts/audit_watchsleep_chain.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class NumericalLoaderChainTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="case_", dir=WORK)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_25hz_original_epochs_to_every_conditioned_window_and_back(self):
        result = audit.audit_numeric_case(self.directory / "fs25", fs=25, downsample_rate=1, multiplier=1.)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["train_window_count"], 3)
        self.assertEqual(result["val_window_count"], 2)
        self.assertTrue(result["original_epoch_signal_bitwise_equal"])
        self.assertTrue(result["condition_channels_exact"])
        self.assertEqual(result["scaler_fit_sample_count"], (7+5)*750)
        self.assertEqual([row["discarded_25hz_tail_samples"] for row in result["records"]], [29, 29, 29])
        self.assertGreater(sum(row["zero_markers_in_oracle_window"] for row in result["windows"]), 0)
        self.assertFalse(result["test_npz_opened"])

    def test_100hz_stride_unit_conversion_and_csv_ahi_to_windows(self):
        result = audit.audit_numeric_case(self.directory / "fs100", fs=100, downsample_rate=4,
                                          multiplier=.001, missing_ahi="S102.npz")
        self.assertEqual(result["train_window_count"], 3)
        self.assertEqual(result["val_window_count"], 2)
        self.assertEqual([row["discarded_25hz_tail_samples"] for row in result["records"]], [44, 44, 44])
        fallback = next(row for row in result["records"] if row["filename"] == "S102.npz")
        self.assertEqual(fallback["legacy_tuple_ahi"], 0.)
        self.assertGreater(fallback["condition_ahi"], 0.)
        self.assertEqual(fallback["ahi_source"], "participant_info.csv")
        self.assertLessEqual(result["maximum_inverse_ibi_error_seconds"], audit.INVERSE_ATOL_SECONDS)

    def test_classification_padding_and_length_cap_never_create_generator_samples(self):
        result = audit.audit_boundary_case(self.directory / "boundary")
        self.assertTrue(result["explicit_cap_1100_tuple_exact"])
        self.assertEqual(result["original_actual_lengths"], [1100, 3])
        self.assertEqual(result["generation_windows_from_short_record"], 0)
        self.assertEqual(result["generated_training_windows"], 551)
        self.assertEqual(result["windows_if_original_1100_cap_were_used"], 549)
        self.assertEqual(result["final_window_start_epoch"], 1100)
        self.assertFalse(result["padding_in_signal_scaler"])
        self.assertFalse(result["padding_in_generation_windows"])

    def test_oracle_refuses_a_changed_source_snapshot(self):
        changed = self.directory / "modified_data_setup.py"
        changed.write_bytes(audit.SOURCE.read_bytes() + b"\n# changed test-only copy\n")
        with self.assertRaisesRegex(AssertionError, "source hash mismatch"):
            audit.original_ssdataset(changed)


if __name__ == "__main__":
    audit.torch.set_num_threads(1)
    unittest.main(verbosity=2)
