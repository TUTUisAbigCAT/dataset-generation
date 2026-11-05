"""S092 all-window inference input checks; no model construction or sampling."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import night_data
import signal_normalization
from scripts import compare_s092_best_minmax as driver


class FullWindowInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.npz = self.root / "S092.npz"
        self.meta = self.root / "participant_info.csv"
        self.meta.write_text("SID,AHI,BMI,AGE,GENDER\nS092,21,41,53,F\n", encoding="utf-8")
        # One fully unobserved epoch is allowed in an otherwise observable
        # window. A negative raw reference later excludes a different window.
        self.signal = np.repeat(np.arange(16, dtype=np.float32) / 20 + .8, 750)
        self.signal[:750] = 0
        self.signal[3000] = -1
        self.signal = np.concatenate([self.signal, np.ones(75, dtype=np.float32)])
        labels = np.resize([0, 1, 4, 3], 17)
        self.stages = np.resize([0, 1, 2, 3], 16)
        np.savez(self.npz, data=self.signal, stages=labels, fs=25, ahi=21)
        self.digest = driver.sha256(self.npz)
        self.starts = [0, 3750, 6750]
        identity = dict(metadata_file_sha256=driver.sha256(self.meta), splits=dict(train=["S092.npz"]),
                        scaler_subjects=["S092"], multiplier=1., downsample_rate=1, max_epochs=None,
                        sample_rate_hz=25, evaluation_protocol=dict(subject="S092"),
                        train_npz_sha256={"S092.npz":self.digest},
                        scaler=dict(continuous_columns=["AHI", "BMI", "AGE"], continuous_mean=[10., 30., 40.],
                                    continuous_std=[3., 5., 15.], signal_mean_seconds=1., signal_std_seconds=1.))
        self.context = dict(workspace=str(self.root/"run/workspace"), arm="minmax",
                            epoch=25, source_checks={}, checkpoint_sha256="a"*64, best_manifest_sha256="b"*64,
                            manifest=dict(diffusion_identity={}), identity=identity)
        self.protocol_path = self.root/"run/training/validation/protocol.json"
        self.protocol_path.parent.mkdir(parents=True)
        self.protocol = dict(evaluation_filenames=["S092.npz"],evaluation_protocol=identity["evaluation_protocol"],
            window=dict(epoch_samples=750, sample_rate_hz=25, window_epochs=4, window_size=3000),
            options=dict(max_windows=None, reference_mask_policy="observed_only"),
            selection=dict(candidate_stride_samples=750, window_size_samples=3000,
                           selected_subjects=1, available_subjects=1,
                           available_nonoverlapping_windows=3, selected_windows=3),
            reference_mask_policy="observed_only", windows=[])
        for i, start in enumerate(self.starts):
            mask = self.signal[start:start+3000] > 0
            self.protocol["windows"].append(dict(validation_index=i, filename="S092.npz",
                subject_id="S092", start_sample=start, start_second=start/25, file_sha256=self.digest,
                sleep_stages=self.stages[start//750:start//750+4].tolist(),
                reference_mask=dict(observed_samples=int(mask.sum()), missing_samples=int((~mask).sum()),
                                    sha256=hashlib.sha256(mask.tobytes()).hexdigest())))

    def run_preflight(self, protocol, **options):
        driver.save_json(self.protocol_path, protocol)
        modules = dict(night_data=night_data, signal_normalization=signal_normalization)
        with (mock.patch.object(driver, "S092_SHA256", self.digest),
              mock.patch.object(driver, "frozen_imports", return_value=(modules, SimpleNamespace(window_size=3000), {}))):
            return driver.worker_preflight(self.context, self.npz, self.meta, 0, self.root, all_windows=True,
                                           **options)

    def test_full_length_stride_reproduces_frozen_nonoverlap_even_after_invalid_epoch(self):
        report = self.run_preflight(self.protocol, window_stride_seconds=120)
        self.assertEqual(report["source_samples"], len(self.signal))
        self.assertEqual(report["trailing_samples"], 2325)
        self.assertEqual(report["uncovered_samples"], 3075)
        self.assertEqual((report["total_samples"], report["windows"], report["duration_seconds"]), (9000, 3, 360))
        with np.load(self.root/"inputs.npz") as arrays:
            expected = np.concatenate([self.signal[start:start+3000] for start in self.starts])
            np.testing.assert_array_equal(arrays["start_samples"], self.starts)
            np.testing.assert_array_equal(arrays["reference_ibi_seconds"], expected)
            np.testing.assert_array_equal(arrays["scoring_mask"], expected > 0)
            np.testing.assert_array_equal(arrays["conditions"][:,:4].argmax(1),
                                          np.repeat(arrays["sleep_stages"], 750))

    def test_default_overlapping_windows_cover_source_time_once_and_preserve_each_slice(self):
        report = self.run_preflight(self.protocol)
        expected = np.asarray([0, 5, 6, 7, 8, 9, 10, 11, 12])*750
        self.assertEqual(report["validation_anchor_windows"], 3)
        self.assertEqual((report["window_stride_seconds"], report["window_overlap_seconds"]), (30, 90))
        self.assertEqual((report["windows"], report["selected_sample_count"], report["total_samples"]),
                         (9, 27000, 27000))
        self.assertEqual((report["unique_covered_samples"], report["duration_seconds"],
                          report["sampled_duration_seconds"]), (11250, 450, 1080))
        self.assertEqual((report["trailing_samples"], report["uncovered_samples"]), (75, 825))
        self.assertTrue(report["generation"]["independent_windows"])
        self.assertEqual((report["generation"]["overlap_samples"],
                          report["generation"]["stitching_overlap_samples"]), (2250, 0))
        with np.load(self.root/"inputs.npz") as arrays:
            np.testing.assert_array_equal(arrays["start_samples"], expected)
            self.assertEqual(int(arrays["window_stride_seconds"]), 30)
            for i, s in enumerate(expected):
                np.testing.assert_array_equal(arrays["reference_ibi_seconds"][i*3000:(i+1)*3000],
                                              self.signal[s:s+3000])
                np.testing.assert_array_equal(arrays["sleep_stages"][i*4:(i+1)*4],
                                              self.stages[s//750:s//750+4])
                np.testing.assert_array_equal(arrays["scoring_mask"][i*3000:(i+1)*3000],
                                              self.signal[s:s+3000] > 0)

    def test_wider_sliding_strides_use_the_original_epoch_grid(self):
        for stride, offsets in [(60, [0, 6, 8, 10, 12]), (90, [0, 6, 9, 12])]:
            with self.subTest(stride=stride):
                report = self.run_preflight(self.protocol, window_stride_seconds=stride)
                self.assertEqual(report["unique_covered_samples"], 10500)
                self.assertEqual(report["window_overlap_seconds"], 120-stride)
                with np.load(self.root/"inputs.npz") as arrays:
                    np.testing.assert_array_equal(arrays["start_samples"], np.asarray(offsets)*750)
                (self.root/"inputs.npz").unlink()

    def test_invalid_stride_is_rejected_before_input_files_are_written(self):
        for stride in (0, -30, 15, 150, True, 30.0):
            with self.subTest(stride=stride), self.assertRaisesRegex(ValueError, "stride"):
                self.run_preflight(self.protocol, window_stride_seconds=stride)
            self.assertFalse((self.root/"inputs.npz").exists())

    def test_incomplete_selection_wrong_coordinate_stage_and_mask_are_rejected(self):
        for error in ("missing", "start", "stages", "hash"):
            p = deepcopy(self.protocol)
            if error == "missing":
                p["windows"].pop()
                p["selection"].update(available_nonoverlapping_windows=2, selected_windows=2)
            elif error == "start":
                p["windows"][1]["start_sample"] = 3000
            elif error == "stages":
                p["windows"][0]["sleep_stages"].reverse()
            else:
                p["windows"][0]["reference_mask"]["sha256"] = "0"*64
            with self.subTest(error=error), self.assertRaises(ValueError):
                self.run_preflight(p)
            self.assertFalse((self.root / "inputs.npz").exists())
