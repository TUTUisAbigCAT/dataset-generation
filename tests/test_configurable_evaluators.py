"""Frozen evaluation window dimensions without running a model or sampler."""
from __future__ import annotations

import csv
import hashlib
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
from scripts.verify_upstream_boundary import CORE_FILES, S4_BACKEND_EDITS, expected_core_source


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
                    window_size=750 * epochs, s4_lmax=max(3000, 750 * epochs)), data_identity=identity,
                    source_sha256={"TSImputers/S4Model.py": hashlib.sha256((ROOT / "TSImputers/S4Model.py").read_bytes()).hexdigest()})

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

    def test_missing_s4_source_identity_is_rejected(self):
        for source in (None, {}, {"TSImputers/S4Model.py": ""}):
            manifest = self.manifest()
            manifest["source_sha256"] = source
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "must record source_sha256"):
                self.load(manifest)

    def test_historical_s4_weights_require_their_matching_source(self):
        manifest = self.manifest()
        upstream = ROOT / "third_party/WaveStitch"
        original_hash = hashlib.sha256((upstream / "TSImputers/S4Model.py").read_bytes()).hexdigest()
        manifest["source_sha256"]["TSImputers/S4Model.py"] = original_hash
        with self.assertRaisesRegex(ValueError, "S4 implementation differs"):
            self.load(manifest)
        # Age alone is not an error: the same manifest is valid in a matching
        # historical workspace and still restores its original linear schedule.
        with mock.patch("scripts.evaluate_denoising.ROOT", upstream):
            args, _, information = self.load(manifest)
        self.assertEqual(args.diffusion_schedule, "linear")
        self.assertTrue(information["s4_source_identity"]["matched"])
        self.assertEqual(information["s4_source_identity"]["runtime_sha256"], original_hash)

    def test_windows_source_paths_match_but_conflicting_aliases_are_rejected(self):
        manifest = self.manifest()
        digest = manifest["source_sha256"].pop("TSImputers/S4Model.py")
        manifest["source_sha256"]["TSImputers\\S4Model.py"] = digest
        _, _, information = self.load(manifest)
        self.assertTrue(information["s4_source_identity"]["matched"])
        manifest["source_sha256"]["TSImputers/S4Model.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Conflicting recorded S4 source hashes"):
            self.load(manifest)

    def test_core_allowlist_requires_the_exact_s4_patch_and_no_other_edits(self):
        from scripts.evaluate_original_sampler import extract_original_pipeline
        for name in CORE_FILES:
            original = (ROOT / "third_party/WaveStitch" / name).read_bytes()
            self.assertEqual((ROOT / name).read_bytes(), expected_core_source(name, original))
        source = ROOT / "TSImputers/S4Model.py"
        original_read = Path.read_bytes

        def tampered(path):
            content = original_read(path)
            return content + b"\n# unauthorized change\n" if path == source else content

        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=tampered):
            with self.assertRaisesRegex(RuntimeError, "exact core correction allowlist"):
                extract_original_pipeline()

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


class S4SourceBoundaryTests(unittest.TestCase):
    def test_every_reviewed_backend_block_is_required(self):
        from scripts.evaluate_original_sampler import extract_original_pipeline
        source = ROOT / "TSImputers/S4Model.py"
        actual = source.read_bytes()
        original_read = Path.read_bytes
        for edit in S4_BACKEND_EDITS:
            before, after = edit["before"].encode(), edit["after"].encode()
            self.assertEqual(actual.count(after), 1)
            reverted = actual.replace(after, before, 1)

            def read_with_reverted_block(path):
                return reverted if path == source else original_read(path)

            with self.subTest(block=edit["id"]):
                with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=read_with_reverted_block):
                    with self.assertRaisesRegex(RuntimeError, "exact core correction allowlist"):
                        extract_original_pipeline()

    def test_missing_or_duplicate_upstream_blocks_are_rejected(self):
        name = "TSImputers/S4Model.py"
        original = (ROOT / "third_party/WaveStitch" / name).read_bytes()
        for edit in S4_BACKEND_EDITS:
            before = edit["before"].encode()
            self.assertEqual(original.count(before), 1)
            for altered in (original.replace(before, b"# removed upstream block", 1),
                            original + b"\n" + before):
                with self.subTest(block=edit["id"]):
                    with self.assertRaisesRegex(ValueError, "exactly one upstream block"):
                        expected_core_source(name, altered)

    def test_provenance_separates_core_edits_from_sampler_edits(self):
        from scripts.evaluate_original_sampler import extract_corrected_pipeline, extract_original_pipeline
        original = extract_original_pipeline()[-1]
        corrected = extract_corrected_pipeline()[-1]
        for provenance in (original, corrected):
            self.assertFalse(provenance["imported_core_byte_identical"])
            self.assertTrue(provenance["s4_fallback_conjugate_corrected"])
            self.assertTrue(provenance["s4_device_dtype_dispatch_corrected"])
            self.assertFalse(provenance["random_draws_added_or_removed"])
        self.assertTrue(original["excerpts_byte_identical_to_upstream"])
        self.assertFalse(original["sampler_expressions_edited"])
        self.assertTrue(corrected["upstream_snapshot_byte_identical"])
        self.assertTrue(corrected["sampler_expressions_edited"])


if __name__ == "__main__":
    unittest.main()
