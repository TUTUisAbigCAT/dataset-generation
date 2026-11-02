"""Synthetic plot checks; these fixtures are not trained DREAMT generations."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation_artifacts import write_validation_artifacts
from validation_metrics import window_statistics
from scripts import plot_validation_comparison as plotter


def create_synthetic_pair(output):
    """Three hand-constructed windows, solely for implementation/visual QA."""
    output = Path(output)
    output.mkdir(parents=True)
    seconds = np.arange(3000) / 25.
    smooth = np.stack([center + .055*np.sin(2*np.pi*seconds/22+i)
                      + .012*np.sin(2*np.pi*seconds/3.2) for i, center in enumerate((.8, 1.05, 1.2))])
    reference = smooth.copy()
    for start in (0, 750, 1500, 2250):
        reference[:, start:start+12] = 0.
    reference[1, 100:106] = np.nan
    observed = np.isfinite(reference) & (reference > 0)
    preserve = smooth - .10 + .035*np.cos(2*np.pi*seconds/17)[None, :]
    preserve[~observed] = .04
    preserve[2, 1500:1506] = -.03
    masked = smooth + .025 + .025*np.cos(2*np.pi*seconds/15)[None, :]
    stages = np.array([[0, 1, 1, 2], [1, 3, 3, 1], [2, 0, 1, 1]], dtype=np.int64)
    conditions = np.zeros((3, 3000, 9), dtype=np.float32)
    conditions[:, :, :4] = np.eye(4, dtype=np.float32)[np.repeat(stages, 750, axis=1)]
    conditions[:, :, 4:] = np.array([.2, -.3, .1, 1., 0.], dtype=np.float32)
    columns = ["W", "Light", "REM", "Deep", "AHI", "BMI", "AGE", "F", "M"]
    metadata = [dict(validation_index=i, subject_id=f"SYNTHETIC_FIXTURE_{i}",
                     filename=f"synthetic_fixture_{i}.npz", start_sample=i*6000,
                     start_second=i*240., sleep_stages=stages[i].tolist()) for i in range(3)]
    protocol = output/"synthetic_protocol.json"
    protocol.write_text(json.dumps(dict(implementation_test_only=True, data_synthetic=True,
        trained_model_executed=False, reference_mask_policy="observed_only")), encoding="utf-8")
    for name, generated in (("preserve", preserve), ("masked", masked)):
        stats = window_statistics(torch.from_numpy(generated), torch.from_numpy(reference),
                                   observed_mask=torch.from_numpy(observed))
        write_validation_artifacts(output/name, generated_seconds=generated, reference_seconds=reference,
            scoring_mask=observed, windows_metadata=metadata, conditions=conditions,
            condition_columns=columns, statistics={key: value.numpy() for key, value in stats.items()},
            protocol_path=protocol, epoch=1)
    return output/"preserve", output/"masked"


class PlotValidationComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="plot_validation_test_", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.preserve, self.masked = create_synthetic_pair(self.directory/"inputs")

    def arguments(self, output=None):
        return ["--preserve-dir", str(self.preserve), "--masked-dir", str(self.masked),
                "--output-dir", str(output or self.directory/"plots")]

    def rehash(self, directory, filename, key):
        index = directory/"artifacts.json"
        report = json.loads(index.read_text())
        report[key] = hashlib.sha256((directory/filename).read_bytes()).hexdigest()
        index.write_text(json.dumps(report), encoding="utf-8")

    def rewrite_arrays(self, directory, **changes):
        with np.load(directory/"waveforms.npz", allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
        arrays.update(changes)
        np.savez_compressed(directory/"waveforms.npz", **arrays)
        self.rehash(directory, "waveforms.npz", "waveform_sha256")

    def test_default_plots_three_windows_in_ms_and_marks_stages_as_input_not_prediction(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figures = []
        original = plt.subplots
        def subplots(*args, **kwargs):
            fig, axes = original(*args, **kwargs)
            figures.append(fig)
            return fig, axes
        with mock.patch.object(plt, "subplots", side_effect=subplots), contextlib.redirect_stdout(io.StringIO()):
            plotter.main(self.arguments())
        summary = json.loads((self.directory/"plots"/"summary.json").read_text())
        self.assertFalse(summary["model_executed"])
        self.assertEqual(summary["selected_indices"], [0, 1, 2])
        self.assertEqual(summary["windows"], 3)
        self.assertEqual(len(summary["plots"]), 4)
        self.assertTrue(all(Path(path).is_file() and Path(path).stat().st_size > 10000 for path in summary["plots"]))
        a, _ = plotter.read_artifacts(self.preserve)
        b, _ = plotter.read_artifacts(self.masked)
        for i, fig in enumerate(figures[:3]):
            upper, lower = fig.axes
            self.assertEqual(upper.get_ylabel(), "IBI (ms)")
            np.testing.assert_array_equal(upper.lines[0].get_xdata(), np.arange(3000)/25.)
            np.testing.assert_array_equal(upper.lines[0].get_ydata(),
                np.where(a["reference_observed_mask"][i], a["reference_ibi_seconds"][i]*1000, np.nan))
            np.testing.assert_array_equal(upper.lines[1].get_ydata(), a["generated_ibi_seconds"][i]*1000)
            np.testing.assert_array_equal(upper.lines[2].get_ydata(), b["generated_ibi_seconds"][i]*1000)
            self.assertIn("SYNTHETIC_FIXTURE", upper.get_title(loc="left"))
            self.assertEqual(lower.get_xlim(), (0., 120.))
            self.assertIn("shared input condition, not a generated prediction", fig.texts[0].get_text())
        self.assertLess(float(figures[2].axes[0].lines[1].get_ydata().min()), 0.)

    def test_hash_check_rejects_tampered_csv_or_npz_before_plotting(self):
        for directory, filename in ((self.preserve, "per_window_metrics.csv"), (self.masked, "waveforms.npz")):
            with (directory/filename).open("ab") as stream:
                stream.write(b"tampered")
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "hash"):
                plotter.read_artifacts(directory)

    def test_rehashed_wrong_csv_identity_still_rejected(self):
        table = self.preserve/"per_window_metrics.csv"
        original = table.read_text(encoding="utf-8")
        for field, bad in (("subject_id", "WRONG_SUBJECT"), ("filename", "wrong.npz"),
                           ("start_sample", "999"), ("start_second", "99.5"),
                           ("validation_index", "2")):
            rows = list(csv.DictReader(io.StringIO(original)))
            rows[0][field] = bad
            with table.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            self.rehash(self.preserve, "per_window_metrics.csv", "metrics_sha256")
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity|order"):
                plotter.read_artifacts(self.preserve)

    def test_matching_all_stored_scoring_masks_are_rejected_in_observed_only_comparison(self):
        for directory in (self.preserve, self.masked):
            self.rewrite_arrays(directory, scoring_mask=np.ones((3, 3000), dtype=bool))
            with self.assertRaisesRegex(ValueError, "observed_only"):
                plotter.read_artifacts(directory)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            plotter.main(self.arguments())
        self.assertFalse((self.directory/"plots").exists())

    def test_shapes_mask_dtype_and_false_observation_metadata_are_rejected(self):
        for key, value, expected in (("generated_ibi_seconds", np.zeros((2, 3000)), "shape"),
                                     ("scoring_mask", np.ones((3, 3000)), "boolean"),
                                     ("reference_observed_mask", np.ones((3, 3000), dtype=bool), "observed mask")):
            name = self.directory/("changed_"+key)
            preserve, _ = create_synthetic_pair(name)
            self.rewrite_arrays(preserve, **{key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, expected):
                plotter.read_artifacts(preserve)

    def test_pair_check_rejects_changed_reference_condition_or_window_identity(self):
        a, _ = plotter.read_artifacts(self.preserve)
        b, _ = plotter.read_artifacts(self.masked)
        plotter.verify_pair(a, b)  # Equal missing NaN locations remain comparable.
        for key in ("reference_ibi_seconds", "scoring_mask", "clean_conditions", "start_sample", "sleep_stages"):
            changed = {name: value.copy() for name, value in b.items()}
            value = changed[key]
            value.flat[-1] = not value.flat[-1] if value.dtype == bool else value.flat[-1] + 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                plotter.verify_pair(a, changed)

    def test_invalid_indices_and_existing_output_are_rejected_without_overwrite(self):
        for indices in (["-1"], ["3"], ["1", "1"]):
            with self.subTest(indices=indices), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                plotter.main(self.arguments()+["--indices"]+indices)
        output = self.directory/"existing"
        output.mkdir()
        (output/"keep.txt").write_text("existing", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            plotter.main(self.arguments(output))
        self.assertEqual((output/"keep.txt").read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
