"""Run legacy-zero and observed-point training with one shared validation protocol.

Data files are read only. New configs, source snapshots and checkpoints live in
a new output folder. Both arms use observed-only reference moments for fair KL
comparison; this changes validation, not the legacy arm's training objective.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def arm_config(base, config_path, policy, validation_max_windows):
    config = json.loads(json.dumps(base))
    for key in ("data_dir", "metadata_path", "split_path"):
        value = config.get(key)
        if value is not None:
            path = Path(value).expanduser()
            config[key] = str((path if path.is_absolute() else config_path.parent / path).resolve())
    config["missing_value_policy"] = policy
    config.setdefault("validation", {}).update(reference_mask_policy="observed_only",
                                               max_windows=validation_max_windows)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "dataset_config.json")
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
    path = cli.config.expanduser().resolve()
    base = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(base, dict):
        raise ValueError("Configuration must be an object")
    if cli.output_dir is None:
        root = Path(base.get("output_root", str(ROOT / "runs"))).expanduser()
        root = root if root.is_absolute() else path.parent / root
        output = root / ("missing_value_ablation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    else:
        output = cli.output_dir
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status="checking", output_directory=str(output), training_completed=False,
                  validation_reference="same observed-positive positions in both arms",
                  epochs=cli.epochs, early_stopping_enabled=False, arms={})
    save_json(output / "summary.json", status)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("DREAMT_")}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    preflights = {}
    try:
        for policy in ("preserve", "masked"):
            config = arm_config(base, path, policy, cli.validation_max_windows or None)
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
        status["shared_input_checks"] = compare_preflights(preflights["preserve"], preflights["masked"])
        status["validation_windows"] = len(preflights["masked"]["validation_windows"])
        status["equal_expected_updates"] = (status["arms"]["preserve"]["expected_updates"] ==
                                            status["arms"]["masked"]["expected_updates"])
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
            with (output / policy / "training/epoch_metrics.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            scores = [float(row["gaussian_kl_real_to_generated"]) for row in rows]
            last_validation = json.loads((output / policy / "training/validation" /
                f"epoch_{int(rows[-1]['epoch']):04d}" / "metrics.json").read_text(encoding="utf-8"))
            status["arms"][policy].update(completed_epochs=len(rows), updates=manifest["updates"], first_kl=scores[0], last_kl=scores[-1],
                best_kl=min(scores), best_kl_epoch=int(rows[scores.index(min(scores))]["epoch"]),
                generated_nonpositive_fraction=last_validation.get("diagnostics", {}).get("generated_nonpositive_fraction"),
                epoch_metrics=str(output / policy / "training/epoch_metrics.csv"))
            save_json(output / "summary.json", status)
        with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["policy", "completed_epochs", "training_windows", "updates", "first_kl", "last_kl", "best_kl", "best_kl_epoch", "generated_nonpositive_fraction"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for policy, arm in status["arms"].items():
                writer.writerow(dict(policy=policy, **{k: arm[k] for k in fields if k != "policy"}))
        status.update(status="completed", training_completed=True, comparison_csv=str(output / "comparison.csv"),
                      limitation="Paired Gaussian amplitude KL cannot validate beat detection or temporal fidelity; train losses use different supervision scopes. Fixed epochs may give unequal updates if the masked arm drops windows with fewer than two observations.")
        save_json(output / "summary.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        save_json(output / "summary.json", status)
        raise


if __name__ == "__main__":
    main()
