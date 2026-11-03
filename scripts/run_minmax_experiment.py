"""Run one fresh, fixed-budget experiment against a completed [-1,1] minmax run.

minmax01 changes only the clean IBI output range to [0,1]. minmax_dropout
retains [-1,1] and masks 10% of noised IBI inputs during training only.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_missing_value_ablation import (
    _check_training_args, _epoch_rows, _read_json, arm_config,
    compare_reference_preflights, expected_training_args, save_json, validate_training_budget,
)
from scripts.run_preprocessing_experiment import MINMAX, AFFINE_FIELDS, final_score, sha256, verify_reference_sources
from signal_normalization import normalization_spec, signal_affine_parameters
from training_input_dropout import input_dropout_options

EXPERIMENTS = ("minmax01", "minmax_dropout")
MINMAX01 = dict(MINMAX, output_range=[0.0, 1.0])


def experiment_config(base, config_path, experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown minmax experiment: " + experiment)
    if (base.get("missing_value_policy", "preserve") != "preserve" or
        normalization_spec(base) != MINMAX or input_dropout_options(base)["probability"] != 0):
        raise ValueError("Reference must use preserve + [-1,1] minmax without input dropout")
    config = arm_config(base, Path(config_path), "preserve", base.get("validation", {}).get("max_windows"), True)
    config["signal_normalization"] = copy.deepcopy(MINMAX01 if experiment == "minmax01" else MINMAX)
    config["training_input_dropout"] = dict(probability=.1 if experiment == "minmax_dropout" else 0.0, seed=42)
    return config


def compare_minmax_preflight(reference, current, experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown minmax experiment: " + experiment)
    old, new = copy.deepcopy(reference["data_identity"]), copy.deepcopy(current["data_identity"])
    old_scaler, new_scaler = old.pop("scaler"), new.pop("scaler")
    expected_spec = MINMAX01 if experiment == "minmax01" else MINMAX
    expected_dropout = dict(probability=.1 if experiment == "minmax_dropout" else 0.0, seed=42)
    checks = dict(status=reference.get("status") == current.get("status") == "passed",
        baseline_normalization=normalization_spec(old) == normalization_spec(old_scaler) == MINMAX,
        normalization=normalization_spec(new) == normalization_spec(new_scaler) == expected_spec,
        baseline_affine=signal_affine_parameters(old_scaler) == (1.0, 1.0),
        affine=signal_affine_parameters(new_scaler) == ((0.0, 2.0) if experiment == "minmax01" else (1.0, 1.0)),
        baseline_input_dropout=input_dropout_options(reference)["probability"] == 0,
        input_dropout=input_dropout_options(current) == expected_dropout,
        missing_policy=old.pop("missing_value_policy", "preserve") == new.pop("missing_value_policy", "preserve") == "preserve")
    # The population mean/std and every metadata parameter remain identical;
    # only the configured signal affine parameters may change.
    old.pop("signal_normalization", None)
    new.pop("signal_normalization", None)
    for key in AFFINE_FIELDS:
        old_scaler.pop(key, None)
        new_scaler.pop(key, None)
    checks.update(raw_data_identity=old == new, population_and_metadata_scaler=old_scaler == new_scaler)
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
        raise ValueError("Minmax preflight mismatch: " + ", ".join(k for k, v in checks.items() if not v))
    return checks


def load_minmax_reference(path):
    """Read historical metadata and saved bytes, never load checkpoint tensors."""
    root = Path(path).expanduser().resolve()
    summary = _read_json(root / "summary.json")
    if summary.get("status") != "completed" or summary.get("training_completed") is not True or summary.get("experiment") != "minmax":
        raise ValueError("Reference must be a completed minmax experiment root")
    config_path = root / "config.json"
    config = _read_json(config_path)
    experiment_config(config, config_path, "minmax01")  # Check the baseline semantics.
    preflight = _read_json(root / "preflight/preflight.json")
    actual = _read_json(root / "run/preflight.json")
    compare_reference_preflights(preflight, actual)
    if input_dropout_options(preflight) != input_dropout_options(actual):
        raise ValueError("Reference input-dropout preflights differ")
    if preflight.get("status") != "passed" or input_dropout_options(preflight)["probability"] != 0:
        raise ValueError("Reference preflight must pass without training input dropout")
    if (normalization_spec(preflight["data_identity"]) != MINMAX or
        normalization_spec(preflight["data_identity"]["scaler"]) != MINMAX or
        signal_affine_parameters(preflight["data_identity"]["scaler"]) != (1.0, 1.0)):
        raise ValueError("Reference preflight must record [-1,1] minmax")
    manifest = _read_json(root / "run/training/run_manifest.json")
    rows = _epoch_rows(root / "run/training/epoch_metrics.csv")
    epochs, batch = summary.get("epochs"), manifest.get("training_args", {}).get("batch_size")
    if type(epochs) is not int or epochs < 1 or type(batch) is not int or batch < 1:
        raise ValueError("Reference epoch and batch evidence is invalid")
    budget = validate_training_budget(manifest, rows, preflight, epochs, batch)
    if summary.get("completed_budget") != budget:
        raise ValueError("Reference summary and actual completed budget differ")
    for key in ("data_identity", "diffusion_identity"):
        if manifest.get(key) != preflight[key]:
            raise ValueError("Reference manifest/preflight mismatch: " + key)
    if summary.get("final_checkpoint_sha256") != manifest.get("checkpoint_sha256"):
        raise ValueError("Reference checkpoint evidence differs")
    recorded_dropout = manifest.get("training_input_dropout")
    if recorded_dropout is not None and input_dropout_options({"training_input_dropout": recorded_dropout.get("options")})["probability"] != 0:
        raise ValueError("Reference training manifest contains active input dropout")
    return dict(root=root, summary=summary, config=config, config_path=config_path,
                preflight=preflight, manifest=manifest, budget=budget)


def validate_dropout_report(manifest, preflight, budget):
    options = input_dropout_options(preflight)
    report = manifest.get("training_input_dropout")
    if not isinstance(report, dict) or report.get("options") != options:
        raise ValueError("Missing or mismatched completed training input-dropout report")
    active = options["probability"] > 0
    calls = budget["updates"] if active else 0
    eligible = budget["training_windows"] * preflight["window_size"] * budget["completed_epochs"] if active else 0
    if (report.get("active") is not active or report.get("application_calls") != calls or
        report.get("eligible_values") != eligible):
        raise ValueError("Input-dropout application count does not match the full training budget")
    dropped = report.get("dropped_values")
    fraction = report.get("observed_fraction")
    if (type(dropped) is not int or not 0 <= dropped <= eligible or type(fraction) not in (int, float) or
        not math.isfinite(fraction) or not math.isclose(fraction, dropped/eligible if eligible else 0.0, abs_tol=1e-12)):
        raise ValueError("Invalid input-dropout counter/fraction evidence")
    if active and (report.get("counter_dtype") not in ("torch.int64", "int64") or
                   not str(report.get("generator_device", "")).startswith("cuda")):
        raise ValueError("Active dropout must use a CUDA generator and int64 device counters")
    # A wide execution sanity bound catches disabled/all-point masking without
    # mistaking normal Bernoulli sampling variation for an implementation error.
    if active:
        expected_count = eligible * options["probability"]
        tolerance = max(1.0, 8.0 * math.sqrt(eligible * options["probability"] * (1-options["probability"])))
        if abs(dropped - expected_count) > tolerance:
            raise ValueError("Observed input-dropout count is inconsistent with the configured probability")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true", help="Data/config/source checks; no model or GPU")
    cli = parser.parse_args(argv)
    if min(cli.epochs, cli.batch_size, cli.threads) < 1:
        parser.error("epochs, batch-size and threads must be positive")
    reference = load_minmax_reference(cli.reference_run)
    config = experiment_config(reference["config"], reference["config_path"], cli.experiment)
    output_root = Path(config.get("output_root", str(ROOT / "runs"))).expanduser()
    if not output_root.is_absolute():
        output_root = reference["config_path"].parent / output_root
    output = (cli.output_dir or output_root / (cli.experiment + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))).expanduser().resolve()
    if output == reference["root"] or reference["root"] in output.parents:
        raise ValueError("New output must be outside the read-only reference run")
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status="checking", experiment=cli.experiment, output_directory=str(output), training_completed=False,
                  reference_run=str(reference["root"]), common_baseline="preserve + [-1,1] minmax; no input dropout",
                  epochs=cli.epochs, batch_size=cli.batch_size, threads=cli.threads, early_stopping_enabled=False,
                  initialization="fresh model and optimizer; no checkpoint resume",
                  config_source=str(reference["config_path"]), training_input_dropout=input_dropout_options(config),
                  validation_reference="same original positive IBI positions; no dropout during validation or inference")
    save_json(output / "summary.json", status)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("DREAMT_")}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        status["source_checks"] = verify_reference_sources(dict(root=reference["root"], arms={"run": reference}))
        _check_training_args(reference["manifest"]["training_args"], expected_training_args(reference["preflight"], cli.epochs, cli.batch_size))
        if reference["manifest"].get("threads") != cli.threads:
            raise ValueError("Reference/current training threads differ")
        config_path = output / "config.json"
        save_json(config_path, config)
        common = [sys.executable, "-B", str(ROOT / "scripts/run_dreamt.py"), "--config", str(config_path),
                  "--epochs", str(cli.epochs), "--batch-size", str(cli.batch_size), "--threads", str(cli.threads)]
        subprocess.run(common + ["--check-only", "--output-dir", str(output / "preflight")], cwd=ROOT, env=environment, check=True)
        preflight = _read_json(output / "preflight/preflight.json")
        status["reference_checks"] = compare_minmax_preflight(reference["preflight"], preflight, cli.experiment)
        status["scaler"] = preflight["data_identity"]["scaler"]
        status["validation_windows"] = len(preflight["validation_windows"])
        status["expected_updates"] = reference["budget"]["updates"]
        status["command"] = common + ["--skip-generation", "--output-dir", str(output / "run")]
        status["status"] = "checked" if cli.check_only else "training"
        save_json(output / "summary.json", status)
        if cli.check_only:
            print(json.dumps(status, ensure_ascii=False), flush=True)
            return status
        print(json.dumps(dict(event="minmax_training_started", experiment=cli.experiment, log=str(output / "training.log"))), flush=True)
        with (output / "training.log").open("w", encoding="utf-8") as stream:
            subprocess.run(status["command"], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        actual = _read_json(output / "run/preflight.json")
        compare_reference_preflights(preflight, actual)
        if input_dropout_options(actual) != input_dropout_options(preflight):
            raise ValueError("Input dropout changed after preflight")
        manifest = _read_json(output / "run/training/run_manifest.json")
        rows = _epoch_rows(output / "run/training/epoch_metrics.csv")
        budget = validate_training_budget(manifest, rows, preflight, cli.epochs, cli.batch_size)
        _check_training_args(manifest["training_args"], expected_training_args(preflight, cli.epochs, cli.batch_size))
        for key in ("data_identity", "diffusion_identity"):
            if manifest.get(key) != preflight[key]:
                raise ValueError("Completed training/preflight mismatch: " + key)
        status["completed_dropout"] = validate_dropout_report(manifest, preflight, budget)
        checkpoint = output / "run/workspace/saved_models/DREAMT/model.pth"
        if not checkpoint.is_file() or sha256(checkpoint) != manifest.get("checkpoint_sha256"):
            raise ValueError("Completed final checkpoint hash mismatch")
        comparisons = [final_score(reference["root"] / "run", "minmax", reference["preflight"], reference["budget"]),
                       final_score(output / "run", cli.experiment, preflight, budget)]
        with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        status.update(status="completed", training_completed=True, completed_budget=budget,
                      final_checkpoint=str(checkpoint), final_checkpoint_sha256=manifest["checkpoint_sha256"],
                      comparison_csv=str(output / "comparison.csv"), comparisons=comparisons,
                      limitation="Fixed clean input ranges do not bound noisy or generated values. "
                                 "Input dropout is a training-only corruption of x_t; epsilon targets and all-point MSE remain unchanged. "
                                 "Compare validation in original seconds, not training losses across different input scales/corruptions.")
        save_json(output / "summary.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        return status
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        save_json(output / "summary.json", status)
        raise


if __name__ == "__main__":
    main()
