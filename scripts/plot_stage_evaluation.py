"""Plot a completed, hash-verified four-stage evaluation without model execution.

Bars show variation across seeded draws, not confidence intervals. Example
epochs are chosen from reference labels and source time, never generated quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np

STAGES = ("W", "Light", "REM", "Deep")
ARMS = ("preserve", "masked")
METRICS = (
    ("mean_absolute_error_ms", "Mean MAE", "Absolute error (ms)", 1.),
    ("variance_absolute_error_ms2", "Variance MAE", "Absolute error (ms$^2$)", 1.),
    ("gaussian_kl_real_to_generated", "Gaussian KL", "KL(real || generated), nats", 1.),
    ("generated_nonpositive_fraction", "Nonpositive generated samples", "Fraction of all generated samples (%)", 100.),
)
FS, EPOCH_SAMPLES = 25, 750


def _sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _verified_files(root, summary, protocol):
    hashes = dict(protocol.get("file_hashes", {}))
    for name, digest in summary.get("file_hashes", {}).items():
        if name in hashes and hashes[name] != digest:
            raise ValueError("Conflicting recorded hashes: " + name)
        hashes[name] = digest
    if not hashes:
        raise ValueError("Completed evaluation has no recorded file_hashes")
    for name, expected in hashes.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected)):
            raise ValueError("Invalid recorded artifact path or SHA256: " + str(name))
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Recorded artifact is missing or outside the evaluation: " + name)
        if _sha256(path) != expected.lower():
            raise ValueError("Artifact SHA256 mismatch: " + name)
    return hashes


def _npz(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def _rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _integer(value, label):
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Expected an integer: " + label) from error
    if isinstance(value, (bool, np.bool_)) or str(number) != str(value):
        raise ValueError("Expected an integer: " + label)
    return number


def _validate_reference(reference):
    real, mask = reference["reference_ibi_seconds"], reference["scoring_mask"]
    if real.ndim != 2 or min(real.shape) < 1 or real.shape[1] % EPOCH_SAMPLES:
        raise ValueError("Reference must contain complete 25 Hz, 30-second epochs")
    windows, length = real.shape
    if mask.dtype != np.bool_ or mask.shape != real.shape:
        raise ValueError("Reference scoring_mask must be bool[windows, samples]")
    if not np.array_equal(mask, np.isfinite(real) & (real > 0)):
        raise ValueError("Four-stage plots require observed-only reference scoring")
    if (mask.reshape(windows, -1, EPOCH_SAMPLES).sum(axis=2) < 2).any():
        raise ValueError("Every original reference epoch needs at least two observations")
    stages = reference["sleep_stages"]
    if (stages.shape != (windows, length // EPOCH_SAMPLES)
            or not np.issubdtype(stages.dtype, np.integer) or not np.isin(stages, np.arange(4)).all()):
        raise ValueError("sleep_stages must use W/Light/REM/Deep codes 0/1/2/3")
    if not np.array_equal(reference["window_index"], np.arange(windows)):
        raise ValueError("Reference window_index must be consecutive in stored order")
    starts = reference["start_sample"]
    if (starts.shape != (windows,) or not np.issubdtype(starts.dtype, np.integer)
            or (starts < 0).any() or (starts % EPOCH_SAMPLES).any()):
        raise ValueError("Reference start_sample must align to complete epochs")
    conditions = reference["clean_conditions"]
    if conditions.shape != (windows, length, 9) or not np.isfinite(conditions).all():
        raise ValueError("Expected finite clean_conditions[windows, samples, 9]")
    expected_stages = np.eye(4)[np.repeat(stages, EPOCH_SAMPLES, axis=1)]
    if not np.array_equal(conditions[:, :, :4], expected_stages):
        raise ValueError("Stored sleep stages and stage conditions disagree")
    if reference["subject_id"].shape != ():
        raise ValueError("subject_id must be a scalar string")


def _stage_rows(rows, seed):
    if len(rows) != 4 or {row["stage_name"] for row in rows} != set(STAGES):
        raise ValueError("Each draw must explicitly record all four stages, including missing stages")
    result = {}
    for row in rows:
        stage = row["stage_name"]
        if _integer(row["seed"], "stage seed") != seed or _integer(row["stage_id"], "stage_id") != STAGES.index(stage):
            raise ValueError("Stage CSV identity or seed differs from its waveform")
        if row["status"] not in ("available", "missing"):
            raise ValueError("Unknown stage availability status")
        item = {"status": row["status"]}
        for field in ("windows", "epochs", "observed_samples"):
            item[field] = _integer(row[field], field)
            if item[field] < 0:
                raise ValueError("Negative stage coverage count")
        for field, _, _, _ in METRICS:
            text = row[field]
            if row["status"] == "missing":
                if text not in ("", "None", "null"):
                    raise ValueError("Missing-stage metrics must be absent, never zero")
                item[field] = None
            else:
                number = float(text)
                if not math.isfinite(number) or number < 0 or (field.endswith("fraction") and number > 1):
                    raise ValueError("Invalid stage metric: " + field)
                item[field] = number
        result[stage] = item
    return result


def load_evaluation(directory):
    """Validate the saved evaluation and return arrays; never run a model."""
    root = Path(directory).resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise ValueError("Only completed evaluation tasks can be plotted")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    hashes = _verified_files(root, summary, protocol)
    def checked(relative):
        if relative not in hashes:
            raise ValueError("Input artifact has no recorded SHA256: " + relative)
        return root / relative
    checked("protocol.json")
    reference = _npz(checked("reference.npz"))
    _validate_reference(reference)
    expected_coverage = {}
    observed_counts = reference["scoring_mask"].reshape(
        len(reference["window_index"]), -1, EPOCH_SAMPLES).sum(axis=2)
    for stage_id, stage in enumerate(STAGES):
        positions = reference["sleep_stages"] == stage_id
        expected_coverage[stage] = dict(status="available" if positions.any() else "missing",
            windows=int(positions.any(axis=1).sum()), epochs=int(positions.sum()),
            observed_samples=int(observed_counts[positions].sum()))
    arms, aggregates = {}, {}
    for arm in ARMS:
        declared = summary["arms"][arm]["stages"]
        directories = sorted(path for path in (root / arm).glob("draw_*") if path.is_dir())
        if not directories or [path.name for path in directories] != [f"draw_{i:03d}" for i in range(len(directories))]:
            raise ValueError("Draw directories must be consecutive from draw_000: " + arm)
        if len(directories) != declared["draws"]:
            raise ValueError("Saved and declared draw counts differ: " + arm)
        draws, seeds = [], []
        for directory in directories:
            relative = directory.relative_to(root).as_posix()
            arrays = _npz(checked(relative + "/waveforms.npz"))
            seed_array = arrays["seed"]
            if seed_array.shape != () or not np.issubdtype(seed_array.dtype, np.integer):
                raise ValueError("Draw seed must be a scalar integer")
            seed = int(seed_array)
            generated = arrays["generated_ibi_seconds"]
            if generated.shape != reference["reference_ibi_seconds"].shape or not np.isfinite(generated).all():
                raise ValueError("Generated waveforms must be finite and match reference dimensions")
            if not np.array_equal(arrays["window_index"], reference["window_index"]):
                raise ValueError("Generated and reference window order differs")
            epoch_rows = _rows(checked(relative + "/per_epoch_metrics.csv"))
            expected_epoch_count = reference["sleep_stages"].size
            if len(epoch_rows) != expected_epoch_count:
                raise ValueError("Per-epoch CSV must record every original reference epoch exactly once")
            seen_epochs = set()
            for row in epoch_rows:
                window = _integer(row["window_index"], "window_index")
                epoch = _integer(row["epoch_in_window"], "epoch_in_window")
                if (not 0 <= window < len(generated) or not 0 <= epoch < reference["sleep_stages"].shape[1]
                        or _integer(row["seed"], "epoch seed") != seed):
                    raise ValueError("Per-epoch CSV identity or seed differs from waveform")
                if (window, epoch) in seen_epochs:
                    raise ValueError("Per-epoch CSV repeats a reference epoch")
                seen_epochs.add((window, epoch))
                expected = int(reference["start_sample"][window]) // EPOCH_SAMPLES + epoch
                if (_integer(row["source_epoch_index"], "source_epoch_index") != expected
                        or row["stage_name"] != STAGES[int(reference["sleep_stages"][window, epoch])]):
                    raise ValueError("Per-epoch CSV stage or source time differs from reference")
            stages = _stage_rows(_rows(checked(relative + "/per_stage_metrics.csv")), seed)
            for stage in STAGES:
                if any(stages[stage][field] != expected for field, expected in expected_coverage[stage].items()):
                    raise ValueError("Stage CSV coverage differs from the reference: " + stage)
            draws.append({"seed": seed, "generated": generated, "stages": stages})
            seeds.append(seed)
        if len(set(seeds)) != len(seeds) or seeds != declared["seeds"]:
            raise ValueError("Seeds must be distinct and agree with the completed summary")
        arms[arm] = draws
        aggregates[arm] = {}
        for stage in STAGES:
            records = [draw["stages"][stage] for draw in draws]
            first = records[0]
            for field in ("status", "windows", "epochs", "observed_samples"):
                if any(record[field] != first[field] for record in records):
                    raise ValueError("Reference stage coverage changed across draws")
                if declared["stages"][stage][field] != first[field]:
                    raise ValueError("Summary stage coverage differs from the reference: " + stage)
            item = {field: first[field] for field in ("status", "windows", "epochs", "observed_samples")}
            item["metrics"] = {}
            for field, _, _, _ in METRICS:
                values = [record[field] for record in records]
                mean, std = (None, None) if first["status"] == "missing" else (float(np.mean(values)), float(np.std(values, ddof=0)))
                item["metrics"][field] = dict(mean=mean, std=std)
                reported = declared["stages"][stage]["metrics"][field]
                for key, value in (("mean", mean), ("std", std)):
                    if (value is None and reported[key] is not None) or (value is not None and
                            (reported[key] is None or not np.isclose(value, reported[key], rtol=1e-10, atol=1e-10))):
                        raise ValueError("Summary and per-draw CSV metrics disagree: " + stage + "/" + field)
            aggregates[arm][stage] = item
    if [draw["seed"] for draw in arms["preserve"]] != [draw["seed"] for draw in arms["masked"]]:
        raise ValueError("The two arms must use the same paired seeds")
    for stage in STAGES:
        if any(aggregates["preserve"][stage][field] != aggregates["masked"][stage][field]
               for field in ("status", "windows", "epochs", "observed_samples")):
            raise ValueError("Stage reference coverage differs between arms")
    return dict(root=root, summary=summary, protocol=protocol, hashes=hashes,
                reference=reference, arms=arms, aggregates=aggregates)


def choose_examples(reference):
    """Earliest original source epoch of each stage, independent of generation."""
    examples = {}
    for stage_id, stage in enumerate(STAGES):
        candidates = [(int(reference["start_sample"][w]) + int(e)*EPOCH_SAMPLES, int(w), int(e))
                      for w, e in np.argwhere(reference["sleep_stages"] == stage_id)]
        if not candidates:
            examples[stage] = None
        else:
            sample, window, epoch = min(candidates)
            examples[stage] = dict(window_index=window, epoch_in_window=epoch,
                                   source_start_sample=sample, source_start_second=sample/FS)
    return examples


def plot_evaluation(data, output_dir):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"preserve": "#2866a6", "masked": "#dc7431"}
    labels = {"preserve": "Preserve zeros", "masked": "Masked missing values"}
    draws = len(data["arms"]["preserve"])
    subject = str(data["reference"]["subject_id"].item())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    x = np.arange(4)
    for ax, (metric, title, unit, scale) in zip(axes.flat, METRICS):
        for offset, arm in zip((-.18, .18), ARMS):
            valid = [i for i, stage in enumerate(STAGES) if data["aggregates"][arm][stage]["status"] == "available"]
            means = [data["aggregates"][arm][STAGES[i]]["metrics"][metric]["mean"]*scale for i in valid]
            stds = [data["aggregates"][arm][STAGES[i]]["metrics"][metric]["std"]*scale for i in valid]
            ax.bar(x[valid]+offset, means, width=.34, yerr=stds, capsize=4, color=colors[arm], alpha=.85, label=labels[arm])
            for i in set(range(4))-set(valid):
                ax.text(x[i]+offset, .025, "NA", transform=ax.get_xaxis_transform(), color=colors[arm], ha="center", weight="bold")
        ax.set_xticks(x, STAGES)
        ax.set_ylabel(unit)
        ax.set_title(title, weight="bold")
        ax.axhline(0, color="#888888", linewidth=.6)
        ax.grid(axis="y", alpha=.15)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{subject} | Four-stage evaluation | {draws} paired seeded draws", fontsize=14)
    fig.text(.06, .018, "Bars: mean across draws. Whiskers: population standard deviation (ddof=0), not confidence intervals.\n"
             "Missing stages are NA. Moment metrics use observed reference positions; nonpositive rates count all generated positions.", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .075, 1, .95))
    path = output/"stage_metrics.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    figures.append(str(path))
    examples = choose_examples(data["reference"])
    for stage, choice in examples.items():
        fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True, sharey=True)
        if choice is None:
            for ax in axes:
                ax.set_axis_off()
            axes[0].text(.5, .45, f"{stage}: NA\nNo reference epoch with this stage in the evaluated windows",
                         transform=axes[0].transAxes, ha="center", va="center", fontsize=15)
            fig.suptitle(f"{subject} | {stage}", fontsize=14)
        else:
            window, epoch = choice["window_index"], choice["epoch_in_window"]
            section = slice(epoch*EPOCH_SAMPLES, (epoch+1)*EPOCH_SAMPLES)
            real = data["reference"]["reference_ibi_seconds"][window, section]*1000
            mask = data["reference"]["scoring_mask"][window, section]
            seconds = np.arange(EPOCH_SAMPLES)/FS
            for ax, arm in zip(axes, ARMS):
                for index, draw in enumerate(data["arms"][arm]):
                    generated = draw["generated"][window, section]*1000
                    ax.plot(seconds, generated, color=colors[arm], linewidth=.8, alpha=.42,
                            label=f"{draws} generated draws (individual curves)" if index == 0 else None)
                ax.plot(seconds, np.where(mask, real, np.nan), color="#202020", linewidth=1.5, label="Real observed IBI", zorder=5)
                finite_missing = ~mask & np.isfinite(real)
                ax.scatter(seconds[finite_missing], real[finite_missing], color="#777777", marker="x", s=15,
                           label="Stored missing values", zorder=6)
                # Shading also makes unknown NaN intervals visible without
                # inventing a zero amplitude for a missing reference value.
                edges = np.diff(np.r_[False, ~mask, False].astype(np.int8))
                for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    ax.axvspan(start/FS, stop/FS, color="#999999", alpha=.12, linewidth=0)
                ax.axhline(0, color="#aaaaaa", linewidth=.6)
                ax.set_title(labels[arm], loc="left", fontsize=11, weight="bold")
                ax.set_ylabel("IBI (ms)")
                ax.grid(alpha=.15)
                ax.legend(loc="best", fontsize=8, ncol=3, frameon=False)
            axes[-1].set_xlim(0, 30)
            axes[-1].set_xlabel("Time within the selected epoch (s)")
            begin = choice["source_start_second"]
            fig.suptitle(f"{subject} | {stage} input condition | NPZ offset {begin:g}-{begin+30:g} s", fontsize=14)
        fig.text(.065, .015, "Selection: earliest source epoch with this reference label; not selected by generation quality.\n"
                 "Every colored curve is one draw, not an average waveform. Gray shading marks reference missing positions; negative values are retained.",
                 fontsize=9, color="#555555")
        fig.subplots_adjust(left=.075, right=.985, top=.88, bottom=.14, hspace=.30)
        path = output/f"stage_{stage.lower()}_example.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        figures.append(str(path))
    report = dict(status="completed", input_directory=str(data["root"]), subject_id=subject,
        draws=draws, seeds=[d["seed"] for d in data["arms"]["preserve"]],
        file_hashes_verified=True, verified_file_count=len(data["hashes"]), model_executed=False,
        uncertainty="Between-seed population standard deviation (ddof=0); not a confidence interval",
        example_selection="Earliest source time per reference stage, independent of generated samples",
        examples=examples, stage_metrics=data["aggregates"], plots=figures,
        limitations="Independent conditional samples need not match the reference point by point. No physiological validity or temporal fidelity is established.")
    (output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Completed stage evaluation directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for figures")
    args = parser.parse_args(argv)
    report = plot_evaluation(load_evaluation(args.input_dir), args.output_dir)
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    main()
