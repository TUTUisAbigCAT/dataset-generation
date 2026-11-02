"""Artificial four-stage output fixtures; no DREAMT data or model inference."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import plot_stage_evaluation as plotter


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synthetic_evaluation(root, *, missing_rem=False):
    """Save 3 artificial 60s windows and 8 paired draws for implementation QA."""
    root = Path(root)
    root.mkdir(parents=True)
    length, samples = 1500, 750
    stages = np.array([[1, 0], [2, 1], [3, 3]], dtype=np.int64)
    if missing_rem:
        stages[1, 0] = 0
    starts = np.array([6000, 0, 3000], dtype=np.int64)
    t = np.arange(length)/25
    signal = np.stack([base+.04*np.sin(t*.7+i)+.01*np.cos(t*2.)
                       for i, base in enumerate((.8, 1.1, .95))])
    real = signal.copy()
    real[:, :12] = 0.
    real[:, 750:762] = 0.
    real[1, 2:6] = np.nan
    mask = np.isfinite(real) & (real > 0)
    conditions = np.zeros((3, length, 9), dtype=np.float32)
    conditions[:, :, :4] = np.eye(4)[np.repeat(stages, samples, axis=1)]
    conditions[:, :, 4:] = [.1, -.2, .3, 1., 0.]
    np.savez_compressed(root/"reference.npz", reference_ibi_seconds=real,
        scoring_mask=mask, clean_conditions=conditions, sleep_stages=stages,
        start_sample=starts, window_index=np.arange(3), subject_id=np.array("SYNTHETIC_S092_TEST"))
    (root/"protocol.json").write_text(json.dumps(dict(implementation_test_only=True,
        data_synthetic=True, model_executed=False, sample_rate_hz=25)), encoding="utf-8")
    summary = dict(status="completed", implementation_test_only=True, arms={})
    metric_keys = [record[0] for record in plotter.METRICS]
    for arm in plotter.ARMS:
        reports = []
        for draw in range(8):
            seed = 42+draw
            directory = root/arm/f"draw_{draw:03d}"
            directory.mkdir(parents=True)
            offset = .10+draw*.01 if arm == "preserve" else .02+draw*.005
            generated = signal+offset
            factor = 1.15+draw*.03 if arm == "preserve" else .8+draw*.02
            for w in range(3):
                for e in range(2):
                    section = slice(e*samples, (e+1)*samples)
                    centered = signal[w, section]-signal[w, section][mask[w, section]].mean()
                    generated[w, section] += (factor-1)*centered
            if arm == "preserve":
                generated[:, :2] = -.01
                generated[:, 750:752] = -.01
            np.savez_compressed(directory/"waveforms.npz", generated_ibi_seconds=generated,
                                seed=np.array(seed), window_index=np.arange(3))
            rows = []
            for w in range(3):
                for e in range(2):
                    section = slice(e*samples, (e+1)*samples)
                    select = mask[w, section]
                    g, r = generated[w, section][select]*1000, real[w, section][select]*1000
                    gv, rv = max(g.var(), 1.), max(r.var(), 1.)
                    kl = .5*(np.log(gv/rv)+(rv+(g.mean()-r.mean())**2)/gv-1)
                    rows.append(dict(window_index=w, epoch_in_window=e, source_epoch_index=int(starts[w]//samples+e),
                        stage_id=int(stages[w, e]), stage_name=plotter.STAGES[stages[w, e]], seed=seed,
                        observed_samples=int(select.sum()), mean_absolute_error_ms=abs(g.mean()-r.mean()),
                        variance_absolute_error_ms2=abs(g.var()-r.var()), gaussian_kl_real_to_generated=kl,
                        generated_nonpositive_fraction=float((generated[w, section]<=0).mean())))
            _write_csv(directory/"per_epoch_metrics.csv", rows)
            stage_rows = []
            for stage_id, stage in enumerate(plotter.STAGES):
                selected = [row for row in rows if row["stage_name"] == stage]
                stage_rows.append(dict(stage_id=stage_id, stage_name=stage, seed=seed,
                    status="available" if selected else "missing", windows=len({row["window_index"] for row in selected}),
                    epochs=len(selected), observed_samples=sum(row["observed_samples"] for row in selected),
                    **{key:float(np.mean([row[key] for row in selected])) if selected else None for key in metric_keys}))
            _write_csv(directory/"per_stage_metrics.csv", stage_rows)
            reports.append({row["stage_name"]:row for row in stage_rows})
        aggregate = {}
        for stage in plotter.STAGES:
            initial = reports[0][stage]
            item = {key:initial[key] for key in ("stage_id", "stage_name", "status", "windows", "epochs", "observed_samples")}
            item["metrics"] = {}
            for key in metric_keys:
                values = [report[stage][key] for report in reports]
                item["metrics"][key] = dict(mean=float(np.mean(values)) if values[0] is not None else None,
                                            std=float(np.std(values, ddof=0)) if values[0] is not None else None)
            aggregate[stage] = item
        summary["arms"][arm] = {"stages":dict(draws=8, seeds=list(range(42, 50)), stages=aggregate)}
    summary["file_hashes"] = {path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in root.rglob("*") if path.is_file()}
    (root/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return root


class PlotStageEvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="plot_stage_test_", dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.inputs = synthetic_evaluation(self.root/"evaluation")

    def summary(self):
        return json.loads((self.inputs/"summary.json").read_text())

    def save_summary(self, summary):
        (self.inputs/"summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def rehash(self, relative):
        summary = self.summary()
        summary["file_hashes"][relative] = hashlib.sha256((self.inputs/relative).read_bytes()).hexdigest()
        self.save_summary(summary)

    def rewrite_npz(self, relative, **updates):
        with np.load(self.inputs/relative, allow_pickle=False) as data:
            arrays = {key:data[key].copy() for key in data.files}
        arrays.update(updates)
        np.savez_compressed(self.inputs/relative, **arrays)
        self.rehash(relative)

    def test_complete_file_hashes_seed_pairing_and_population_std(self):
        data = plotter.load_evaluation(self.inputs)
        self.assertEqual(len(data["arms"]["preserve"]), 8)
        self.assertEqual(len(data["arms"]["masked"]), 8)
        actual = data["aggregates"]["preserve"]["W"]["metrics"]["mean_absolute_error_ms"]
        self.assertAlmostEqual(actual["mean"], 135., places=10)
        self.assertAlmostEqual(actual["std"], float(np.std(np.arange(8)*10., ddof=0)), places=10)
        self.assertNotAlmostEqual(actual["std"], float(np.std(np.arange(8)*10., ddof=1)), places=5)

    def test_examples_use_earliest_source_epoch_not_array_order_or_generated_quality(self):
        data = plotter.load_evaluation(self.inputs)
        choice = plotter.choose_examples(data["reference"])
        self.assertEqual(choice["Light"], dict(window_index=1, epoch_in_window=1, source_start_sample=750, source_start_second=30.))
        self.assertEqual(choice["W"]["window_index"], 0)
        self.assertEqual(choice["REM"]["source_start_second"], 0.)
        for draw in data["arms"]["masked"]:
            draw["generated"][:] = -1000
        self.assertEqual(plotter.choose_examples(data["reference"]), choice)

    def test_actual_stage_helper_csv_and_summary_are_compatible(self):
        import torch
        from stage_evaluation import score_stage_draw, summarize_stage_draws
        reference = plotter._npz(self.inputs/"reference.npz")
        summary = self.summary()
        for arm in plotter.ARMS:
            reports = []
            for draw in range(8):
                prefix = f"{arm}/draw_{draw:03d}"
                arrays = plotter._npz(self.inputs/prefix/"waveforms.npz")
                report = score_stage_draw(torch.from_numpy(arrays["generated_ibi_seconds"]),
                    torch.from_numpy(reference["reference_ibi_seconds"]), torch.from_numpy(reference["sleep_stages"]),
                    start_samples=reference["start_sample"].tolist(), seed=int(arrays["seed"]))
                _write_csv(self.inputs/prefix/"per_epoch_metrics.csv", report["per_epoch"])
                _write_csv(self.inputs/prefix/"per_stage_metrics.csv",
                           [dict(seed=report["seed"], **row) for row in report["stages"].values()])
                reports.append(report)
            summary["arms"][arm]["stages"] = summarize_stage_draws(reports)
        for relative in summary["file_hashes"]:
            summary["file_hashes"][relative] = hashlib.sha256((self.inputs/relative).read_bytes()).hexdigest()
        self.save_summary(summary)
        data = plotter.load_evaluation(self.inputs)
        self.assertEqual(data["aggregates"]["preserve"]["W"]["epochs"], 1)
        self.assertAlmostEqual(data["aggregates"]["preserve"]["W"]["metrics"]["mean_absolute_error_ms"]["mean"], 135., places=10)

    def test_incomplete_task_hash_tampering_or_missing_hash_is_rejected(self):
        original = self.summary()
        changed = json.loads(json.dumps(original));changed["status"] = "running"
        self.save_summary(changed)
        with self.assertRaisesRegex(ValueError, "completed"):
            plotter.load_evaluation(self.inputs)
        changed = json.loads(json.dumps(original));del changed["file_hashes"]["reference.npz"]
        self.save_summary(changed)
        with self.assertRaisesRegex(ValueError, "no recorded"):
            plotter.load_evaluation(self.inputs)
        self.save_summary(original)
        with (self.inputs/"preserve/draw_000/per_stage_metrics.csv").open("ab") as stream:
            stream.write(b"bad")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            plotter.load_evaluation(self.inputs)

    def test_rehashed_window_order_seed_and_nonfinite_generation_are_rejected(self):
        relative = "masked/draw_000/waveforms.npz"
        original = plotter._npz(self.inputs/relative)
        for update, message in ((dict(window_index=np.array([2, 1, 0])), "window order"),
                                (dict(seed=np.array(999)), "seed"),
                                (dict(generated_ibi_seconds=np.full((3, 1500), np.nan)), "finite")):
            self.rewrite_npz(relative, **original, **{})
            self.rewrite_npz(relative, **update)
            with self.subTest(update=list(update)), self.assertRaisesRegex(ValueError, message):
                plotter.load_evaluation(self.inputs)

    def test_rehashed_csv_stage_or_source_epoch_disagreement_is_rejected(self):
        relative = "preserve/draw_000/per_epoch_metrics.csv"
        path = self.inputs/relative
        with path.open(newline="", encoding="utf-8") as stream:
            original = list(csv.DictReader(stream))
        for key, value in (("stage_name", "Deep"), ("source_epoch_index", "999"), ("seed", "999")):
            rows = [row.copy() for row in original]
            rows[0][key] = value
            _write_csv(path, rows);self.rehash(relative)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "differs"):
                plotter.load_evaluation(self.inputs)

    def test_summary_mean_disagreement_is_rejected(self):
        summary = self.summary()
        summary["arms"]["masked"]["stages"]["stages"]["W"]["metrics"]["mean_absolute_error_ms"]["mean"] += 1
        self.save_summary(summary)
        with self.assertRaisesRegex(ValueError, "Summary.*disagree"):
            plotter.load_evaluation(self.inputs)

    def test_csv_must_cover_every_reference_epoch_once_and_match_stage_coverage(self):
        relative = "preserve/draw_000/per_epoch_metrics.csv"
        path = self.inputs/relative
        with path.open(newline="", encoding="utf-8") as stream:
            original = list(csv.DictReader(stream))
        for rows, message in ((original[:-1], "every original"), (original[:-1]+[original[0]], "repeats")):
            _write_csv(path, rows);self.rehash(relative)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                plotter.load_evaluation(self.inputs)
        _write_csv(path, original);self.rehash(relative)
        relative = "preserve/draw_000/per_stage_metrics.csv"
        path = self.inputs/relative
        with path.open(newline="", encoding="utf-8") as stream:
            original = list(csv.DictReader(stream))
        for field in ("windows", "epochs", "observed_samples"):
            rows = [row.copy() for row in original]
            rows[0][field] = str(int(rows[0][field])+1)
            _write_csv(path, rows);self.rehash(relative)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "coverage"):
                plotter.load_evaluation(self.inputs)

    def test_missing_stage_stays_none_and_draws_na_not_zero_bars(self):
        absent = synthetic_evaluation(self.root/"missing_rem", missing_rem=True)
        data = plotter.load_evaluation(absent)
        for arm in plotter.ARMS:
            stage = data["aggregates"][arm]["REM"]
            self.assertEqual(stage["status"], "missing")
            self.assertIsNone(stage["metrics"]["gaussian_kl_real_to_generated"]["mean"])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figures, original = [], plt.subplots
        def subplots(*args, **kwargs):
            fig, axes = original(*args, **kwargs);figures.append(fig);return fig, axes
        with mock.patch.object(plt, "subplots", side_effect=subplots):
            report = plotter.plot_evaluation(data, self.root/"missing_plots")
        self.assertIsNone(report["examples"]["REM"])
        self.assertTrue(any("not confidence intervals" in text.get_text() for text in figures[0].texts))
        for ax in figures[0].axes:
            self.assertEqual(len(ax.patches), 6)  # 3 available stages x 2 arms, no zero REM bars.
            self.assertEqual(sum(text.get_text() == "NA" for text in ax.texts), 2)
        self.assertTrue(any("REM: NA" in text.get_text() for text in figures[3].axes[0].texts))

    def test_actual_figures_show_all_individual_draws_ms_missing_and_negative_values(self):
        data = plotter.load_evaluation(self.inputs)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figures, original = [], plt.subplots
        def subplots(*args, **kwargs):
            fig, axes = original(*args, **kwargs);figures.append(fig);return fig, axes
        before = np.random.get_state()
        with mock.patch.object(plt, "subplots", side_effect=subplots):
            report = plotter.plot_evaluation(data, self.root/"plots")
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1]);self.assertEqual(before[2:], after[2:])
        self.assertFalse(report["model_executed"])
        self.assertEqual(len(report["plots"]), 5)
        self.assertTrue(all(Path(path).stat().st_size > 10000 for path in report["plots"]))
        choice = report["examples"]["W"]
        w, e = choice["window_index"], choice["epoch_in_window"]
        for ax, arm in zip(figures[1].axes, plotter.ARMS):
            self.assertEqual(ax.get_ylabel(), "IBI (ms)")
            self.assertEqual(len(ax.lines), 10)  # 8 draws + real + zero reference line.
            for i in range(8):
                expected = data["arms"][arm][i]["generated"][w, e*750:(e+1)*750]*1000
                np.testing.assert_array_equal(ax.lines[i].get_ydata(), expected)
            self.assertTrue(np.isnan(ax.lines[8].get_ydata()[:12]).all())
            self.assertGreater(len(ax.patches), 0)
        self.assertLess(float(figures[1].axes[0].lines[0].get_ydata().min()), 0.)
        with self.assertRaises(FileExistsError):
            plotter.plot_evaluation(data, self.root/"plots")


if __name__ == "__main__":
    unittest.main()
