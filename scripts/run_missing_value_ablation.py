"""Run legacy-zero and observed-point training with one shared validation protocol.

Data files are read only. New configs, source snapshots and checkpoints live in
a new output folder. Both arms use observed-only reference moments for fair KL
comparison; this changes validation, not the legacy arm's training objective.
"""
from __future__ import annotations
import argparse
import ast
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def arm_config(base, config_path, policy, validation_max_windows, save_validation_artifacts=False):
    config = json.loads(json.dumps(base))
    for key in ("data_dir", "metadata_path", "split_path"):
        value = config.get(key)
        if value is not None:
            path = Path(value).expanduser()
            config[key] = str((path if path.is_absolute() else config_path.parent / path).resolve())
    config["missing_value_policy"] = policy
    config.setdefault("validation", {}).update(reference_mask_policy="observed_only",
                                               max_windows=validation_max_windows)
    if save_validation_artifacts:
        config["validation"]["save_artifacts"] = True
    # Fixed epochs keep the round count comparable; the ordinary entry still
    # supports the user's original early-stopping settings outside this ablation.
    config.setdefault("early_stopping", {})["enabled"] = False
    return config


def compare_preflights(a, b):
    keys = ("data_directory", "metadata_file_sha256", "splits", "split_fingerprint",
            "train_npz_sha256", "window_epochs", "multiplier", "downsample_rate", "max_epochs")
    checks = {key: a["data_identity"].get(key) == b["data_identity"].get(key) for key in keys}
    for key in ("validation_windows", "validation_options", "diffusion_identity", "training_input_shape"):
        checks[key] = a[key] == b[key]
    # The individual preflights hash all opened input records, including validation.
    checks["opened_data_hashes"] = (a["provenance"].get("opened_npz_sha256") ==
                                    b["provenance"].get("opened_npz_sha256"))
    if not all(checks.values()):
        raise ValueError("A/B input or validation mismatch: " + ", ".join(k for k, v in checks.items() if not v))
    return checks


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _epoch_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def expected_training_args(preflight, epochs, batch_size):
    """Read literal parser defaults without importing/executing a training model.

    run_simulation supplies the remaining arguments explicitly. Actual completed
    training_args are checked against this contract again after each arm.
    """
    tree = ast.parse((ROOT / "training_wavestitch.py").read_text(encoding="utf-8"))
    result = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "parser"
                and node.func.attr == "add_argument"):
            continue
        keywords = {item.arg: item.value for item in node.keywords}
        destination = (ast.literal_eval(keywords["dest"]) if "dest" in keywords
                       else ast.literal_eval(node.args[0]).lstrip("-").replace("-", "_"))
        result[destination] = ast.literal_eval(keywords["default"]) if "default" in keywords else None
    if not result or "lr" not in result:
        raise ValueError("Cannot establish the current training parser defaults")
    result.update(dataset="DREAMT", window_size=preflight["window_size"],
                  s4_lmax=preflight["window_size"], stride=750, batch_size=batch_size,
                  epochs=epochs, **preflight["diffusion_settings"])
    from s4_dropout_config import s4_dropout_options
    result["s4_dropout"] = s4_dropout_options(preflight)
    result.pop("validation_output", None)
    return result


def _check_training_args(actual, expected, *, ignore_epochs=False):
    actual = dict(actual)
    expected = dict(expected)
    actual.pop("validation_output", None)  # The new run writes to its own directory.
    if ignore_epochs:
        actual.pop("epochs", None)
        expected.pop("epochs", None)
    differences = [key for key in sorted(set(actual) | set(expected))
                   if key not in actual or key not in expected or actual[key] != expected[key]]
    if differences:
        raise ValueError("Training configuration mismatch: " + ", ".join(differences))


def validate_training_budget(manifest, rows, preflight, epochs, batch_size):
    """Require the requested full epochs and observed optimizer-update budget."""
    windows = preflight["training_windows"]
    batches = (windows + batch_size - 1) // batch_size
    expected = dict(status="completed", train_windows=windows, batches_per_epoch=batches,
                    expected_updates=batches * epochs, updates=batches * epochs,
                    completed_epochs=epochs, requested_epochs=epochs, checkpoint_epoch=epochs,
                    stopped_early=False)
    errors = [key for key, value in expected.items() if key not in manifest or manifest[key] != value]
    if [int(row["epoch"]) for row in rows] != list(range(1, epochs + 1)):
        errors.append("epoch_metrics sequence")
    training_args = manifest.get("training_args") or {}
    if training_args.get("batch_size") != batch_size or training_args.get("epochs") != epochs:
        errors.append("training_args batch_size/epochs")
    if errors:
        raise ValueError("Training budget/completion mismatch: " + ", ".join(errors))
    return dict(training_windows=windows, batch_size=batch_size, batches_per_epoch=batches,
                completed_epochs=epochs, updates=batches * epochs, full_budget_verified=True)


def load_reference_run(path):
    """Read a completed experiment; never load its weights or optimizer state."""
    root = Path(path).expanduser().resolve()
    summary = _read_json(root / "summary.json")
    if summary.get("status") != "completed" or summary.get("training_completed") is not True:
        raise ValueError("Reference A/B run must have completed both training arms")
    arms = {}
    for policy in ("preserve", "masked"):
        preflight = _read_json(root / f"preflight_{policy}" / "preflight.json")
        manifest = _read_json(root / policy / "training" / "run_manifest.json")
        if preflight.get("status") != "passed" or manifest.get("status") != "completed":
            raise ValueError("Reference arm is not complete: " + policy)
        epochs = summary.get("epochs")
        batch_size = (manifest.get("training_args") or {}).get("batch_size")
        if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Reference epoch/batch-size evidence is missing or invalid: " + policy)
        rows = _epoch_rows(root / policy / "training" / "epoch_metrics.csv")
        budget = validate_training_budget(manifest, rows, preflight, epochs, batch_size)
        actual_preflight = _read_json(root / policy / "preflight.json")
        compare_reference_preflights(preflight, actual_preflight)
        if manifest.get("data_identity") != preflight["data_identity"]:
            raise ValueError("Reference manifest/preflight data identity differs: " + policy)
        if manifest.get("diffusion_identity") != preflight["diffusion_identity"]:
            raise ValueError("Reference manifest/preflight schedule differs: " + policy)
        reported = summary.get("arms", {}).get(policy, {})
        if any(reported.get(key) != budget[key] for key in ("training_windows", "completed_epochs", "updates")):
            raise ValueError("Reference summary and completed training evidence differ: " + policy)
        config_path = root / f"config_{policy}.json"
        arms[policy] = dict(config=_read_json(config_path), config_path=config_path,
                            preflight=preflight, manifest=manifest, budget=budget)
    compare_preflights(arms["preserve"]["preflight"], arms["masked"]["preflight"])
    return dict(root=root, summary=summary, arms=arms)


def compare_reference_preflights(reference, current):
    """Compare each arm to its historical self, including its distinct scaler.

    Saving additional validation artifacts is the sole ignored validation option.
    Dataset paths remain strict: an explicit override must still identify the same
    data, split, file hashes and scaler after the loader resolves paths.
    """
    checks = {}
    from s4_dropout_config import s4_dropout_options
    from training_input_dropout import input_dropout_options
    checks["s4_dropout"] = s4_dropout_options(reference) == s4_dropout_options(current)
    checks["training_input_dropout"] = input_dropout_options(reference) == input_dropout_options(current)
    for key in ("data_identity", "training_windows", "training_input_shape", "window_epochs", "window_size",
                "validation_windows", "validation_selection", "diffusion_identity", "diffusion_settings"):
        checks[key] = key in reference and key in current and reference[key] == current[key]
    checks["validation_options"] = (
        {k: v for k, v in reference["validation_options"].items() if k != "save_artifacts"} ==
        {k: v for k, v in current["validation_options"].items() if k != "save_artifacts"})
    checks["opened_data_hashes"] = (reference["provenance"].get("opened_npz_sha256") is not None and
        reference["provenance"]["opened_npz_sha256"] == current["provenance"].get("opened_npz_sha256"))
    checks["early_stopping_disabled"] = (reference.get("early_stopping", {}).get("enabled") is False and
                                          current.get("early_stopping", {}).get("enabled") is False)
    if not all(checks.values()):
        raise ValueError("Reference/current preflight mismatch: " + ", ".join(k for k, v in checks.items() if not v))
    return checks


def require_equal_budgets(arms):
    for key in ("training_windows", "expected_updates"):
        if arms["preserve"][key] != arms["masked"][key]:
            raise ValueError("Equal-update A/B requires identical " + key + " in both arms")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        help="Explicit shared config override; otherwise reuse each reference arm's config, or dataset_config.json")
    parser.add_argument("--reference-run", type=Path,
                        help="Completed old A/B root; verify identical inputs/settings, then train afresh (not resume)")
    parser.add_argument("--require-equal-updates", action="store_true",
                        help="Reject unequal train windows/update budgets and require every requested epoch/update")
    parser.add_argument("--save-validation-artifacts", action="store_true",
                        help="Save per-epoch validation waveforms and window statistics")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--validation-max-windows", type=int, default=64,
                        help="Fixed shared subset for a short comparison; 0 uses all eligible validation windows")
    parser.add_argument("--check-only", action="store_true", help="Data/identity preflights only; no model or CUDA required")
    cli = parser.parse_args(argv)
    if min(cli.epochs, cli.batch_size, cli.threads) < 1 or cli.validation_max_windows < 0:
        parser.error("epochs/batch-size/threads must be positive; validation-max-windows must be >=0")
    reference = load_reference_run(cli.reference_run) if cli.reference_run else None
    if cli.config is not None or reference is None:
        path = (cli.config or ROOT / "dataset_config.json").expanduser().resolve()
        base = _read_json(path)
        bases = {policy: (base, path) for policy in ("preserve", "masked")}
    else:
        bases = {policy: (arm["config"], arm["config_path"]) for policy, arm in reference["arms"].items()}
        base, path = bases["preserve"]
    if cli.output_dir is None:
        root = Path(base.get("output_root", str(ROOT / "runs"))).expanduser()
        root = root if root.is_absolute() else path.parent / root
        output = root / ("missing_value_ablation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    else:
        output = cli.output_dir
    output = output.resolve()
    if reference and (output == reference["root"] or reference["root"] in output.parents):
        raise ValueError("New experiment output must be outside the read-only reference run")
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status="checking", output_directory=str(output), training_completed=False,
                  validation_reference="same observed-positive positions in both arms",
                  epochs=cli.epochs, batch_size=cli.batch_size, threads=cli.threads,
                  require_equal_updates=cli.require_equal_updates,
                  save_validation_artifacts=cli.save_validation_artifacts,
                  initialization="fresh model and optimizer; no checkpoint resume",
                  reference_run=str(reference["root"]) if reference else None,
                  config_source=("explicit shared --config override" if cli.config else
                                 "each reference arm's saved config" if reference else "dataset_config.json"),
                  early_stopping_enabled=False, arms={})
    save_json(output / "summary.json", status)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("DREAMT_")}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    preflights = {}
    try:
        for policy in ("preserve", "masked"):
            arm_base, arm_path = bases[policy]
            config = arm_config(arm_base, arm_path, policy, cli.validation_max_windows or None,
                                cli.save_validation_artifacts)
            config_path = output / f"config_{policy}.json"
            save_json(config_path, config)
            common = [sys.executable, "-B", str(ROOT / "scripts/run_dreamt.py"), "--config", str(config_path),
                      "--epochs", str(cli.epochs), "--batch-size", str(cli.batch_size), "--threads", str(cli.threads)]
            check_path = output / f"preflight_{policy}"
            subprocess.run(common + ["--check-only", "--output-dir", str(check_path)], cwd=ROOT,
                           env=environment, check=True)
            preflights[policy] = json.loads((check_path / "preflight.json").read_text(encoding="utf-8"))
            status["arms"][policy] = dict(config=str(config_path), output=str(output / policy),
                training_windows=preflights[policy]["training_windows"],
                expected_updates=((preflights[policy]["training_windows"]+cli.batch_size-1)//cli.batch_size)*cli.epochs,
                scaler=preflights[policy]["data_identity"]["scaler"],
                command=common + ["--skip-generation", "--output-dir", str(output / policy)])
            if reference:
                previous = reference["arms"][policy]
                checks = compare_reference_preflights(previous["preflight"], preflights[policy])
                _check_training_args(previous["manifest"]["training_args"],
                                     expected_training_args(preflights[policy], cli.epochs, cli.batch_size),
                                     ignore_epochs=True)
                if previous["manifest"].get("threads") != cli.threads:
                    raise ValueError("Reference/current training threads differ: " + policy)
                status["arms"][policy]["reference_checks"] = dict(checks, training_args=True, threads=True)
        status["shared_input_checks"] = compare_preflights(preflights["preserve"], preflights["masked"])
        status["validation_windows"] = len(preflights["masked"]["validation_windows"])
        status["equal_expected_updates"] = (status["arms"]["preserve"]["expected_updates"] ==
                                            status["arms"]["masked"]["expected_updates"])
        if cli.require_equal_updates:
            require_equal_budgets(status["arms"])
        status["status"] = "checked" if cli.check_only else "training"
        save_json(output / "summary.json", status)
        if cli.check_only:
            print(json.dumps(status, ensure_ascii=False), flush=True)
            return
        for policy in ("preserve", "masked"):
            print(json.dumps(dict(event="ablation_arm_started", policy=policy,
                                  log=str(output / f"{policy}.log"))), flush=True)
            with (output / f"{policy}.log").open("w", encoding="utf-8") as stream:
                subprocess.run(status["arms"][policy]["command"], cwd=ROOT, env=environment,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            current = json.loads((output / policy / "preflight.json").read_text(encoding="utf-8"))
            compare_preflights(preflights[policy], current)
            if current["data_identity"] != preflights[policy]["data_identity"]:
                raise ValueError("Training data/scaler identity changed after paired preflight: " + policy)
            manifest = json.loads((output / policy / "training/run_manifest.json").read_text(encoding="utf-8"))
            rows = _epoch_rows(output / policy / "training/epoch_metrics.csv")
            if cli.require_equal_updates or reference:
                status["arms"][policy]["completed_budget"] = validate_training_budget(
                    manifest, rows, preflights[policy], cli.epochs, cli.batch_size)
                for key in ("data_identity", "diffusion_identity"):
                    if manifest.get(key) != preflights[policy][key]:
                        raise ValueError("Completed training manifest/preflight mismatch: " + key)
                _check_training_args(manifest["training_args"],
                                     expected_training_args(preflights[policy], cli.epochs, cli.batch_size))
            scores = [float(row["gaussian_kl_real_to_generated"]) for row in rows]
            last_validation = json.loads((output / policy / "training/validation" /
                f"epoch_{int(rows[-1]['epoch']):04d}" / "metrics.json").read_text(encoding="utf-8"))
            status["arms"][policy].update(completed_epochs=len(rows), updates=manifest["updates"], first_kl=scores[0], last_kl=scores[-1],
                best_kl=min(scores), best_kl_epoch=int(rows[scores.index(min(scores))]["epoch"]),
                generated_nonpositive_fraction=last_validation.get("diagnostics", {}).get("generated_nonpositive_fraction"),
                epoch_metrics=str(output / policy / "training/epoch_metrics.csv"))
            save_json(output / "summary.json", status)
        if cli.require_equal_updates and status["arms"]["preserve"]["updates"] != status["arms"]["masked"]["updates"]:
            raise ValueError("Actual A/B update counts differ")
        with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["policy", "completed_epochs", "training_windows", "updates", "first_kl", "last_kl", "best_kl", "best_kl_epoch", "generated_nonpositive_fraction"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for policy, arm in status["arms"].items():
                writer.writerow(dict(policy=policy, **{k: arm[k] for k in fields if k != "policy"}))
        status.update(status="completed", training_completed=True, comparison_csv=str(output / "comparison.csv"),
                      limitation="Paired Gaussian amplitude KL cannot validate beat detection or temporal fidelity; train losses use different supervision scopes. " +
                      ("Both arms completed the same verified epoch/update budget." if cli.require_equal_updates else
                       "Fixed epochs may give unequal updates if the masked arm drops windows with fewer than two observations."))
        save_json(output / "summary.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        save_json(output / "summary.json", status)
        raise


if __name__ == "__main__":
    main()
