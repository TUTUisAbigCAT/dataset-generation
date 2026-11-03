"""Train one preprocessing factor against a completed, immutable A/B reference.

zero_fill_one changes only raw zeros to one second, retaining z-score.
minmax retains raw zeros and maps [0, 2] seconds to [-1, 1]. No clipping.
Each invocation starts one fresh CUDA training run, never resumes old weights.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_missing_value_ablation import (
    _check_training_args, _epoch_rows, _read_json, arm_config,
    compare_reference_preflights, expected_training_args, load_reference_run,
    save_json, validate_training_budget,
)
from signal_normalization import normalization_spec, signal_affine_parameters

INVARIANT_SOURCES = (
    "training_wavestitch.py", "training_utils.py", "TSImputers/SSSDS4Imputer.py",
    "TSImputers/S4Model.py", "utils/util.py", "watchsleep_loader.py",
    "cuda_runtime.py", "validation_metrics.py", "validation_sampling.py",
    "validation_artifacts.py", "window_config.py", "diffusion_schedule.py", "missing_data.py",
    "third_party/WatchSleepNet/data_setup.py",
)
MINMAX = dict(method="minmax", min_seconds=0.0, max_seconds=2.0)
AFFINE_FIELDS = ("signal_normalization", "signal_center_seconds", "signal_scale_seconds")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_config(base, config_path, experiment):
    if experiment not in ("zero_fill_one", "minmax"):
        raise ValueError("Unknown preprocessing experiment: " + experiment)
    if base.get("missing_value_policy", "preserve") != "preserve" or normalization_spec(base) is not None:
        raise ValueError("Common baseline must use preserve + z-score")
    config = arm_config(base, Path(config_path),
                        "zero_fill_one" if experiment == "zero_fill_one" else "preserve",
                        base.get("validation", {}).get("max_windows"), True)
    config["signal_normalization"] = ({"method": "zscore"} if experiment == "zero_fill_one" else dict(MINMAX))
    return config


def compare_experiment_preflight(reference, current, experiment):
    """Allow only the selected signal transform; keep raw data and scoring fixed."""
    if experiment not in ("zero_fill_one", "minmax"):
        raise ValueError("Unknown preprocessing experiment: " + experiment)
    old = copy.deepcopy(reference["data_identity"])
    new = copy.deepcopy(current["data_identity"])
    old_scaler, new_scaler = old.pop("scaler"), new.pop("scaler")
    old_policy = old.pop("missing_value_policy", "preserve")
    new_policy = new.pop("missing_value_policy", "preserve")
    checks = dict(status=reference.get("status") == current.get("status") == "passed",
                  baseline=old_policy == "preserve" and normalization_spec(old) is None,
                  old_affine=normalization_spec(old_scaler) is None,
                  missing_policy=new_policy == ("zero_fill_one" if experiment == "zero_fill_one" else "preserve"))
    # Validate the full scaler before omitting the explicitly permitted differences.
    signal_affine_parameters(old_scaler)
    signal_affine_parameters(new_scaler)
    if experiment == "zero_fill_one":
        from data_utils import ZERO_FILL_ONE_VALUE_POLICY
        checks["zero_replacement"] = new.get("zero_value_policy") == ZERO_FILL_ONE_VALUE_POLICY
        checks["normalization"] = normalization_spec(new) is None and normalization_spec(new_scaler) is None
        old.pop("zero_value_policy", None)
        new.pop("zero_value_policy", None)
        for key in ("signal_mean_seconds", "signal_std_seconds", "signal_fit_policy"):
            old_scaler.pop(key, None)
            new_scaler.pop(key, None)
    else:
        checks["normalization"] = normalization_spec(new) == MINMAX and normalization_spec(new_scaler) == MINMAX
        checks["affine"] = signal_affine_parameters(new_scaler) == (1.0, 1.0)
        new.pop("signal_normalization", None)
        for key in AFFINE_FIELDS:
            new_scaler.pop(key, None)
    checks.update(data_identity=old == new, scaler_unchanged_fields=old_scaler == new_scaler)
    for key in ("training_windows", "training_input_shape", "window_epochs", "window_size",
                "validation_windows", "validation_selection", "diffusion_identity", "diffusion_settings"):
        checks[key] = key in reference and key in current and reference[key] == current[key]
    checks["validation_options"] = (
        {k: v for k, v in reference["validation_options"].items() if k != "save_artifacts"} ==
        {k: v for k, v in current["validation_options"].items() if k != "save_artifacts"})
    checks["observed_reference"] = current["validation_options"].get("reference_mask_policy") == "observed_only"
    checks["save_artifacts"] = current["validation_options"].get("save_artifacts") is True
    checks["opened_data_hashes"] = (reference["provenance"].get("opened_npz_sha256") is not None and
        reference["provenance"]["opened_npz_sha256"] == current["provenance"].get("opened_npz_sha256"))
    checks["early_stopping_disabled"] = (reference.get("early_stopping", {}).get("enabled") is False and
                                           current.get("early_stopping", {}).get("enabled") is False)
    checks["test_not_opened"] = current.get("test_npz_opened") is False
    if not all(checks.values()):
        raise ValueError("Preprocessing preflight mismatch: " + ", ".join(k for k, v in checks.items() if not v))
    return checks


def verify_signal_scaler(reference, current, experiment):
    """Predict zero->one moments from historical all-point/observed counts."""
    original = reference["arms"]["preserve"]["preflight"]["data_identity"]["scaler"]
    observed = reference["arms"]["masked"]["preflight"]["data_identity"]["scaler"]
    actual = current["data_identity"]["scaler"]
    total, positive = original["signal_fit_samples"], observed["signal_fit_samples"]
    if type(total) is not int or type(positive) is not int or not 0 < positive <= total:
        raise ValueError("Invalid historical signal sample counts")
    zero_fraction = (total - positive) / total
    mean, std = original["signal_mean_seconds"], original["signal_std_seconds"]
    expected_mean = mean + zero_fraction if experiment == "zero_fill_one" else mean
    variance = std**2 + mean**2 + zero_fraction - expected_mean**2 if experiment == "zero_fill_one" else std**2
    if variance <= 0:
        raise ValueError("Expected transformed training variance must be positive")
    expected_std = math.sqrt(variance)
    if (actual.get("signal_fit_samples") != total or
        not math.isclose(actual["signal_mean_seconds"], expected_mean, rel_tol=1e-8, abs_tol=1e-8) or
        not math.isclose(actual["signal_std_seconds"], expected_std, rel_tol=1e-8, abs_tol=1e-8)):
        raise ValueError("Transformed signal scaler differs from historical raw moments/counts")
    return dict(verified=True, raw_zero_fraction=zero_fraction, expected_signal_mean_seconds=expected_mean,
                expected_signal_std_seconds=expected_std, signal_fit_samples=total)


def verify_reference_sources(reference):
    """Verify old frozen bytes and unchanged model/sampler/objective implementations."""
    result = {}
    for policy, arm in reference["arms"].items():
        workspace = reference["root"] / policy / "workspace"
        hashes = {key.replace("\\", "/"): value for key, value in arm["manifest"].get("source_sha256", {}).items()}
        if not hashes or any(key not in hashes for key in INVARIANT_SOURCES):
            raise ValueError("Reference source evidence is incomplete: " + policy)
        changed_adapters = []
        for name, expected in hashes.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or ":" in name:
                raise ValueError("Invalid reference source path: " + name)
            frozen = workspace / name
            if not frozen.is_file() or sha256(frozen) != expected:
                raise ValueError("Reference frozen source hash mismatch: " + policy + "/" + name)
            current = ROOT / name
            same = current.is_file() and sha256(current) == expected
            if name in INVARIANT_SOURCES and not same:
                raise ValueError("Model/training/sampler source changed: " + name)
            if name not in INVARIANT_SOURCES and not same:
                changed_adapters.append(name)
        checkpoint = workspace / "saved_models/DREAMT/model.pth"
        expected_checkpoint = arm["manifest"].get("checkpoint_sha256")
        if not expected_checkpoint or not checkpoint.is_file() or sha256(checkpoint) != expected_checkpoint:
            raise ValueError("Reference final checkpoint hash mismatch: " + policy)
        result[policy] = dict(frozen_files_checked=len(hashes), invariant_files_checked=len(INVARIANT_SOURCES),
                             final_checkpoint_sha256=expected_checkpoint, changed_adapter_sources=changed_adapters)
    return result


def final_score(root, label, preflight, budget):
    rows = _epoch_rows(root / "training/epoch_metrics.csv")
    epoch = budget["completed_epochs"]
    metrics = _read_json(root / f"training/validation/epoch_{epoch:04d}/metrics.json")
    score = float(rows[-1]["gaussian_kl_real_to_generated"])
    if (metrics.get("status") != "completed" or metrics.get("epoch") != epoch or
        not math.isfinite(score) or score < 0 or
        not math.isclose(metrics.get("aggregate", {}).get("gaussian_kl_real_to_generated", math.nan), score)):
        raise ValueError("Final validation result is incomplete or inconsistent: " + label)
    fraction = metrics.get("diagnostics", {}).get("generated_nonpositive_fraction")
    if fraction is None or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Missing/invalid nonpositive generation diagnostic: " + label)
    return dict(experiment=label, epoch=epoch, training_windows=preflight["training_windows"], updates=budget["updates"],
                gaussian_kl_real_to_generated=score, generated_nonpositive_fraction=fraction)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("zero_fill_one", "minmax"), required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true", help="Check data/config/source identity only; no model or GPU")
    cli = parser.parse_args(argv)
    if min(cli.epochs, cli.batch_size, cli.threads) < 1:
        parser.error("epochs, batch-size and threads must be positive")
    reference = load_reference_run(cli.reference_run)
    previous = reference["arms"]["preserve"]
    config = experiment_config(previous["config"], previous["config_path"], cli.experiment)
    root = Path(config.get("output_root", str(ROOT / "runs"))).expanduser()
    if not root.is_absolute():
        root = previous["config_path"].parent / root
    output = (cli.output_dir or root / (cli.experiment + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))).expanduser().resolve()
    if output == reference["root"] or reference["root"] in output.parents:
        raise ValueError("New experiment output must be outside the read-only reference run")
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status="checking", experiment=cli.experiment, output_directory=str(output), training_completed=False,
                  reference_run=str(reference["root"]), common_baseline="preserve + z-score",
                  epochs=cli.epochs, batch_size=cli.batch_size, threads=cli.threads, early_stopping_enabled=False,
                  initialization="fresh model and optimizer; no checkpoint resume",
                  validation_reference="original positive IBI at the same fixed validation positions; no filled truth",
                  config_source=str(previous["config_path"]))
    save_json(output / "summary.json", status)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("DREAMT_")}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        status["source_checks"] = verify_reference_sources(reference)
        for arm in reference["arms"].values():
            _check_training_args(arm["manifest"]["training_args"], expected_training_args(arm["preflight"], cli.epochs, cli.batch_size))
            if arm["manifest"].get("threads") != cli.threads:
                raise ValueError("Reference/current training threads differ")
            if arm["budget"] != previous["budget"]:
                raise ValueError("Historical arms must have equal training budgets")
        config_path = output / "config.json"
        save_json(config_path, config)
        common = [sys.executable, "-B", str(ROOT / "scripts/run_dreamt.py"), "--config", str(config_path),
                  "--epochs", str(cli.epochs), "--batch-size", str(cli.batch_size), "--threads", str(cli.threads)]
        subprocess.run(common + ["--check-only", "--output-dir", str(output / "preflight")],
                       cwd=ROOT, env=environment, check=True)
        preflight = _read_json(output / "preflight/preflight.json")
        status["reference_checks"] = compare_experiment_preflight(previous["preflight"], preflight, cli.experiment)
        status["scaler_check"] = verify_signal_scaler(reference, preflight, cli.experiment)
        status["scaler"] = preflight["data_identity"]["scaler"]
        status["validation_windows"] = len(preflight["validation_windows"])
        status["expected_updates"] = previous["budget"]["updates"]
        status["command"] = common + ["--skip-generation", "--output-dir", str(output / "run")]
        status["status"] = "checked" if cli.check_only else "training"
        save_json(output / "summary.json", status)
        if cli.check_only:
            print(json.dumps(status, ensure_ascii=False), flush=True)
            return status
        print(json.dumps(dict(event="preprocessing_training_started", experiment=cli.experiment,
                              log=str(output / "training.log"))), flush=True)
        with (output / "training.log").open("w", encoding="utf-8") as stream:
            subprocess.run(status["command"], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        compare_reference_preflights(preflight, _read_json(output / "run/preflight.json"))
        manifest = _read_json(output / "run/training/run_manifest.json")
        rows = _epoch_rows(output / "run/training/epoch_metrics.csv")
        budget = validate_training_budget(manifest, rows, preflight, cli.epochs, cli.batch_size)
        _check_training_args(manifest["training_args"], expected_training_args(preflight, cli.epochs, cli.batch_size))
        for key in ("data_identity", "diffusion_identity"):
            if manifest.get(key) != preflight[key]:
                raise ValueError("Completed training/preflight mismatch: " + key)
        checkpoint = output / "run/workspace/saved_models/DREAMT/model.pth"
        if not checkpoint.is_file() or sha256(checkpoint) != manifest.get("checkpoint_sha256"):
            raise ValueError("Completed final checkpoint hash mismatch")
        comparisons = [final_score(reference["root"] / policy, policy, arm["preflight"], arm["budget"])
                       for policy, arm in reference["arms"].items()]
        comparisons.append(final_score(output / "run", cli.experiment, preflight, budget))
        with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        status.update(status="completed", training_completed=True, completed_budget=budget,
                      final_checkpoint=str(checkpoint), final_checkpoint_sha256=manifest["checkpoint_sha256"],
                      comparison_csv=str(output / "comparison.csv"), comparisons=comparisons,
                      limitation="Clean minmax inputs are bounded; diffusion noise and generated outputs are not clipped. "
                                 "Training losses under different scaling are not directly comparable; final-epoch validation uses seconds and identical raw observed references.")
        save_json(output / "summary.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        return status
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        save_json(output / "summary.json", status)
        raise


if __name__ == "__main__":
    main()
