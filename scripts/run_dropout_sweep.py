"""Run one of 16 fresh CUDA dropout/normalization comparisons, with fixed budget.

Two ranges x two independently tested dropout locations x four probabilities.
Only the selected location is active. Reuse the reference data/protocol, never
its weights. Each Slurm array task owns a new config and frozen workspace.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s4_dropout_config import s4_dropout_options
from scripts.run_minmax_experiment import (
    experiment_config, compare_minmax_preflight, load_minmax_reference, validate_dropout_report,
)
from scripts.run_missing_value_ablation import (
    _read_json, _epoch_rows, _check_training_args, compare_reference_preflights,
    expected_training_args, save_json, validate_training_budget,
)
from scripts.run_preprocessing_experiment import final_score, sha256, verify_reference_sources
from training_input_dropout import input_dropout_options

PROBABILITIES = (0.01, 0.03, 0.05, 0.1)
RANGES = ((-1.0, 1.0), (0.0, 1.0))
GRID_PATH = ROOT / "configs/dropout_sweep.json"


def validate_configuration(row):
    fields = {"id", "name", "normalization_range", "dropout_location", "input_probability", "s4_probability"}
    if not isinstance(row, dict) or set(row) != fields:
        raise ValueError("Sweep configuration fields are incomplete or unknown")
    if type(row["id"]) is not int or not 0 <= row["id"] < 16:
        raise ValueError("Configuration id must be an integer from 0 to 15")
    if (not isinstance(row["normalization_range"], list)
            or tuple(row["normalization_range"]) not in RANGES
            or any(isinstance(v, bool) or type(v) not in (int, float) for v in row["normalization_range"])):
        raise ValueError("Expected normalization range [-1,1] or [0,1]")
    p_input = input_dropout_options({"training_input_dropout": {"probability": row["input_probability"]}})["probability"]
    p_s4 = s4_dropout_options({"s4_dropout": row["s4_probability"]})
    location = row["dropout_location"]
    if location == "input" and p_input in PROBABILITIES and p_s4 == 0:
        probability = p_input
    elif location == "s4" and p_s4 in PROBABILITIES and p_input == 0:
        probability = p_s4
    else:
        raise ValueError("Exactly one dropout location must use 0.01, 0.03, 0.05 or 0.1")
    tag = "m11" if row["normalization_range"][0] == -1 else "01"
    expected_name = f"minmax_{tag}_{location}_p{str(probability).replace('.', 'p')}"
    if row["name"] != expected_name:
        raise ValueError("Configuration name must encode its actual range, location and probability")
    return deepcopy(row)


def load_grid(path=GRID_PATH):
    data = _read_json(path)
    if data.get("schema") != "dreamt_dropout_sweep_v1" or not isinstance(data.get("configurations"), list):
        raise ValueError("Invalid dropout sweep JSON schema")
    rows = [validate_configuration(row) for row in data["configurations"]]
    expected = {(r, location, p) for r in RANGES for location in ("input", "s4") for p in PROBABILITIES}
    actual = {(tuple(r["normalization_range"]), r["dropout_location"],
               r["input_probability"] or r["s4_probability"]) for r in rows}
    if (len(rows) != 16 or {r["id"] for r in rows} != set(range(16))
            or len({r["name"] for r in rows}) != 16 or actual != expected):
        raise ValueError("The grid must contain exactly 16 distinct configurations covering all requested combinations")
    return sorted(rows, key=lambda row: row["id"])


def select_configuration(grid, configuration_id):
    if type(configuration_id) is not int or not 0 <= configuration_id < 16:
        raise ValueError("configuration-id must be an integer from 0 to 15")
    rows = [row for row in grid if row["id"] == configuration_id]
    if len(rows) != 1:
        raise ValueError("Expected exactly one selected configuration")
    return validate_configuration(rows[0])


def sweep_config(reference, configuration):
    selected = validate_configuration(configuration)
    if s4_dropout_options(reference["config"]) != 0:
        raise ValueError("Reference must have zero S4 dropout")
    experiment = "minmax01" if selected["normalization_range"][0] == 0 else "minmax_dropout"
    config = experiment_config(reference["config"], reference["config_path"], experiment)
    config["training_input_dropout"] = dict(probability=selected["input_probability"], seed=42)
    config["s4_dropout"] = selected["s4_probability"]
    return config


def compare_sweep_preflight(reference, current, configuration):
    selected = validate_configuration(configuration)
    experiment = "minmax01" if selected["normalization_range"][0] == 0 else "minmax_dropout"
    checks = compare_minmax_preflight(reference, current, experiment,
        expected_input_dropout=dict(probability=selected["input_probability"], seed=42))
    checks["baseline_s4_dropout"] = s4_dropout_options(reference) == 0
    checks["selected_s4_dropout"] = s4_dropout_options(current) == selected["s4_probability"]
    if not all(checks.values()):
        raise ValueError("Dropout sweep preflight differs: " + ", ".join(k for k, v in checks.items() if not v))
    return checks


def validate_s4_report(manifest, probability):
    report = manifest.get("s4_dropout_runtime", {})
    args = manifest.get("training_args", {})
    expected_layers = 2 * args.get("num_res_layers", 0)
    if (s4_dropout_options(manifest) != probability or args.get("s4_dropout") != probability
            or report.get("probability") != probability or report.get("active") is not (probability > 0)
            or report.get("training") is not True or report.get("layers") != expected_layers
            or expected_layers < 1 or report.get("configuration_inspected") is not True
            or report.get("disabled_in_eval") is not True):
        raise ValueError("Missing or inconsistent S4 dropout runtime configuration report")
    modules = report.get("modules", [])
    expected_type = "Dropout2d" if probability > 0 else "Identity"
    if (len(modules) != expected_layers or len({m.get("name") for m in modules}) != expected_layers
            or any(m.get("probability") != probability or m.get("module") != expected_type
                   or m.get("training") is not True for m in modules)):
        raise ValueError("S4 dropout module inventory disagrees with the configured model")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=GRID_PATH)
    parser.add_argument("--configuration-id", type=int, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true", help="Preflight only; no model or CUDA execution")
    cli = parser.parse_args(argv)
    if min(cli.epochs, cli.batch_size, cli.threads) < 1:
        parser.error("epochs, batch-size and threads must be positive")
    configuration = select_configuration(load_grid(cli.grid), cli.configuration_id)
    reference = load_minmax_reference(cli.reference_run)
    for source in (reference["config"], reference["preflight"], reference["manifest"]["training_args"]):
        if s4_dropout_options(source) != 0:
            raise ValueError("Common reference must have zero S4 dropout in config, preflight and training args")
    _check_training_args(reference["manifest"]["training_args"],
                         expected_training_args(reference["preflight"], cli.epochs, cli.batch_size))
    if reference["manifest"].get("threads") != cli.threads:
        raise ValueError("Reference/current training threads differ")
    config = sweep_config(reference, configuration)
    output = cli.output_dir.expanduser().resolve()
    if output == reference["root"] or output.is_relative_to(reference["root"]):
        raise ValueError("Output must be outside the read-only reference run")
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status="checking", experiment=configuration["name"], configuration=configuration,
        reference_run=str(reference["root"]), output_directory=str(output), training_completed=False,
        initialization="fresh model/optimizer with seed 42; no checkpoint resume",
        epochs=cli.epochs, batch_size=cli.batch_size, threads=cli.threads, early_stopping_enabled=False,
        grid_sha256=sha256(cli.grid), expected_updates=reference["budget"]["updates"],
        model_source_changed=False, sampling_formula_changed=False)
    save_json(output / "summary.json", status)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("DREAMT_")}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        status["source_checks"] = verify_reference_sources(dict(root=reference["root"], arms={"run": reference}))
        config_path = output / "config.json"
        save_json(config_path, config)
        common = [sys.executable, "-B", str(ROOT / "scripts/run_dreamt.py"), "--config", str(config_path),
            "--epochs", str(cli.epochs), "--batch-size", str(cli.batch_size), "--threads", str(cli.threads)]
        subprocess.run(common + ["--check-only", "--output-dir", str(output / "preflight")],
                       cwd=ROOT, env=environment, check=True)
        preflight = _read_json(output / "preflight/preflight.json")
        status["reference_checks"] = compare_sweep_preflight(reference["preflight"], preflight, configuration)
        status["scaler"] = preflight["data_identity"]["scaler"]
        status["validation_windows"] = len(preflight["validation_windows"])
        status["command"] = common + ["--skip-generation", "--output-dir", str(output / "run")]
        status["status"] = "checked" if cli.check_only else "training"
        save_json(output / "summary.json", status)
        if cli.check_only:
            print(json.dumps(dict(event="dropout_sweep_checked", configuration=configuration,
                                  output_directory=str(output))), flush=True)
            return status
        print(json.dumps(dict(event="dropout_sweep_training_started", configuration=configuration,
                              log=str(output / "training.log"))), flush=True)
        with (output / "training.log").open("x", encoding="utf-8") as stream:
            subprocess.run(status["command"], cwd=ROOT, env=environment, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        actual = _read_json(output / "run/preflight.json")
        compare_reference_preflights(preflight, actual)
        manifest = _read_json(output / "run/training/run_manifest.json")
        rows = _epoch_rows(output / "run/training/epoch_metrics.csv")
        budget = validate_training_budget(manifest, rows, preflight, cli.epochs, cli.batch_size)
        _check_training_args(manifest["training_args"], expected_training_args(preflight, cli.epochs, cli.batch_size))
        for key in ("data_identity", "diffusion_identity"):
            if manifest.get(key) != preflight[key]:
                raise ValueError("Completed training/preflight identity mismatch: " + key)
        status["completed_input_dropout"] = validate_dropout_report(manifest, preflight, budget)
        status["completed_s4_dropout"] = validate_s4_report(manifest, configuration["s4_probability"])
        checkpoint = output / "run/workspace/saved_models/DREAMT/model.pth"
        if not checkpoint.is_file() or sha256(checkpoint) != manifest.get("checkpoint_sha256"):
            raise ValueError("Completed final checkpoint SHA256 differs")
        comparisons = [final_score(reference["root"] / "run", "minmax_baseline", reference["preflight"], reference["budget"]),
                       final_score(output / "run", configuration["name"], preflight, budget)]
        with (output / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        status.update(status="completed", training_completed=True, completed_budget=budget,
            final_checkpoint=str(checkpoint), final_checkpoint_sha256=manifest["checkpoint_sha256"],
            comparisons=comparisons, best_checkpoints=manifest.get("validation", {}).get("best_checkpoints", {}),
            limitation="Input dropout uses its own RNG; native S4 dropout consumes the training RNG. "
                       "Equal initialization/validation seeds do not imply identical later training random draws.")
        save_json(output / "summary.json", status)
        print(json.dumps(dict(event="dropout_sweep_training_completed", configuration=configuration,
                              output_directory=str(output), final_metrics=comparisons[-1])), flush=True)
        return status
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        save_json(output / "summary.json", status)
        raise


if __name__ == "__main__":
    main()
