"""Frozen evaluation window dimensions without running a model or sampler."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils import Preprocessor
from scripts.evaluate_denoising import DEFAULT_TRAINING_ARGS, bind_training_data, load_training_run
from scripts.evaluate_original_sampler import select_sampling_window


class ConfigurableEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.records = self.directory / "records"
        self.records.mkdir()
        names, rows = [], []
        for index in range(3):
            sid = f"S{index + 1:03d}"
            names.append(sid + ".npz")
            signal = (.8 + index * .02 + .03 * np.sin(np.arange(7500) / 37.)).astype(np.float32)
            np.savez(self.records / names[-1], data=signal[:, None],
                     stages=np.resize(np.array([0, 1, 2, 3, 4]), 10), fs=np.array(25))
            rows.append(dict(SID=sid, AHI=10 + index, BMI=24 + index, AGE=40 + index, GENDER="F"))
        with (self.records / "participant_info.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (self.records / "splits.json").write_text(
            json.dumps(dict(train=names[:1], val=names[1:2], test=names[2:])), encoding="utf-8")
        self.manifest_path = self.directory / "manifest.json"

    def processor(self, epochs):
        return Preprocessor("DREAMT", data_config={"data_dir": str(self.records), "window_epochs": epochs})

    def manifest(self, epochs=4, *, legacy=False, schema=True):
        processor = self.processor(epochs)
        identity = processor.data_identity()
        if legacy:
            identity.pop("window_epochs")
        if not schema:
            identity.pop("schema")
            identity.pop("schema_version")
        return dict(training_args=dict(DEFAULT_TRAINING_ARGS, dataset="DREAMT", stride=750,
                    window_size=750 * epochs, s4_lmax=max(3000, 750 * epochs)), data_identity=identity)

    def load(self, manifest):
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return load_training_run(self.manifest_path)

    def test_manifest_dimensions_override_historical_model_defaults(self):
        for epochs in (1, 2, 4, 8):
            with self.subTest(window_epochs=epochs):
                args, manifest, information = self.load(self.manifest(epochs))
                self.assertEqual(args.window_size, epochs * 750)
                self.assertGreaterEqual(args.s4_lmax, args.window_size)
                self.assertEqual(information["frozen_window"]["window_epochs"], epochs)
                bound, checks = bind_training_data(args, manifest, self.manifest_path, Preprocessor)
                self.assertEqual(bound.window_size, args.window_size)
                self.assertTrue(checks["all_recorded_identity_fields_match"])

    def test_global_config_cannot_override_frozen_window_length(self):
        args, manifest, _ = self.load(self.manifest(2))
        current_config = self.directory / "current.json"
        current_config.write_text(json.dumps(dict(data_dir="missing-current-data", window_epochs=8)), encoding="utf-8")
        with mock.patch.dict(os.environ, {"DREAMT_CONFIG_PATH": str(current_config), "DREAMT_DATA_DIR": "missing-current-data"}):
            bound, checks = bind_training_data(args, manifest, self.manifest_path, Preprocessor)
        self.assertEqual(bound.window_epochs, 2)
        self.assertEqual(bound.window_size, 1500)
        self.assertTrue(checks["global_config_ignored"])

    def test_legacy_identities_without_window_epochs_mean_four_epochs(self):
        for schema in (True, False):
            with self.subTest(schema=schema):
                args, manifest, _ = self.load(self.manifest(4, legacy=True, schema=schema))
                bound, checks = bind_training_data(args, manifest, self.manifest_path, Preprocessor)
                self.assertEqual(bound.window_size, 3000)
                self.assertTrue(checks["all_recorded_identity_fields_match"])
                manifest["training_args"]["window_size"] = 1500
                with self.assertRaisesRegex(ValueError, "frozen data window_epochs"):
                    self.load(manifest)

    def test_rejects_mismatched_window_and_short_s4_kernel(self):
        cases = [("window_size", 2250), ("window_size", 1500.0), ("window_size", True),
                 ("window_size", 749), ("window_size", 0), ("s4_lmax", 1499),
                 ("s4_lmax", True), ("s4_lmax", 3000.0)]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                manifest = self.manifest(2)
                manifest["training_args"][key] = value
                with self.assertRaises(ValueError):
                    self.load(manifest)

    def test_binding_rejects_preprocessor_with_other_dimensions(self):
        args, manifest, _ = self.load(self.manifest(2))
        wrong = SimpleNamespace(window_size=750, window_epochs=1)
        with mock.patch.object(Preprocessor, "from_identity", return_value=wrong):
            with self.assertRaisesRegex(ValueError, "preprocessor window dimensions"):
                bind_training_data(args, manifest, self.manifest_path, Preprocessor)

    def test_sampling_uses_frozen_window_and_legal_default_stride(self):
        for epochs in (1, 2, 4, 8):
            with self.subTest(window_epochs=epochs):
                processor = self.processor(epochs)
                stride = min(1500, processor.window_size)
                with mock.patch.object(processor, "window_array", wraps=processor.window_array) as reader:
                    filename, start = select_sampling_window(processor, "S002", start_sample=0, windows=2)
                self.assertEqual((filename, start), ("S002.npz", 0))
                reader.assert_called_once_with(filename, 0, processor.window_size + stride)
                self.assertEqual(processor.window_array(filename, start, processor.window_size).shape,
                                 (750 * epochs, 10))

    def test_one_epoch_sampling_rejects_explicit_oversized_or_invalid_stride(self):
        processor = self.processor(1)
        for stride in (1500, 749, 0, -750, True, 750.0):
            with self.subTest(stride=stride), self.assertRaises(ValueError):
                select_sampling_window(processor, "S002", windows=2, stride=stride)
        self.assertEqual(select_sampling_window(processor, windows=2), ("S002.npz", 0))
        self.assertEqual(select_sampling_window(processor, "S002", windows=2, stride=750), ("S002.npz", 0))


if __name__ == "__main__":
    unittest.main()
