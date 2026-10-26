"""Validate one DREAMT folder, then run the adapted baseline in an isolated run.

Data/schedule configuration, source copying and subprocess orchestration live here.
Optimizer updates retain the official baseline. Sampling applies the authorized
posterior-variance square-root correction to the preserved upstream excerpt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cuda_runtime import require_cuda, cuda_environment
from diffusion_schedule import diffusion_settings, diffusion_identity


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def resolve_output_directory(config_path, explicit_output=None):
    """Resolve the experiment root from JSON, or use an exact CLI destination."""
    if explicit_output is not None:
        return Path(explicit_output).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("DREAMT configuration must be a JSON object")
    # Old configurations keep the original project-level runs directory.
    output_root = ROOT / "runs"
    if "output_root" in config:
        value = config["output_root"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("output_root must be a non-empty path string")
        output_root = Path(value).expanduser()
        if not output_root.is_absolute():
            output_root = config_path.parent / output_root
    run_name = "dreamt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return (output_root / run_name).resolve()


def snapshot_source(destination):
    destination.mkdir()
    for name in ("training_wavestitch.py", "training_utils.py", "data_utils.py", "watchsleep_loader.py", "cuda_runtime.py",
                 "epoch_validation.py", "validation_metrics.py", "validation_sampling.py", "window_config.py", "diffusion_schedule.py"):
        shutil.copy2(ROOT / name, destination / name)
    for name in ("TSImputers", "utils", "third_party"):
        shutil.copytree(ROOT / name, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    (destination / "scripts").mkdir()
    for name in ("run_simulation.py", "verify_upstream_boundary.py",
                 "evaluate_denoising.py", "evaluate_original_sampler.py"):
        shutil.copy2(ROOT / "scripts" / name, destination / "scripts" / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Override only the NPZ folder; otherwise read dataset_config.json")
    parser.add_argument("--config", type=Path, default=ROOT / "dataset_config.json")
    parser.add_argument("--check-only", action="store_true", help="Validate the complete data interface without training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, help="An exact new run directory; overrides JSON output_root (which receives a timestamped subdirectory)")
    parser.add_argument("--skip-generation", action="store_true", help="Skip final generation; per-epoch generation validation still runs")
    cli = parser.parse_args()
    if min(cli.epochs, cli.batch_size, cli.threads) < 1:
        parser.error("epochs, batch size and threads must be positive")
    cli.config = cli.config.expanduser().resolve()
    output = resolve_output_directory(cli.config, cli.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status="checking", output_directory=str(output), stages=[])
    save_json(output / "status.json", status)
    try:
        # This entry point deliberately uses the selected JSON and optional folder
        # argument, so stale per-process metadata/split overrides cannot redirect it.
        for key in list(os.environ):
            if key.startswith("DREAMT_"):
                del os.environ[key]
        os.environ["DREAMT_CONFIG_PATH"] = str(cli.config.resolve())
        os.environ["OMP_NUM_THREADS"] = str(cli.threads)
        os.environ["MKL_NUM_THREADS"] = str(cli.threads)
        # Data-only validation does not execute a model. All compute stages require CUDA.
        if not cli.check_only:
            device = require_cuda()
            status["cuda_environment"] = cuda_environment(device)
            save_json(output / "status.json", status)
        from data_utils import Preprocessor

        preprocessor = Preprocessor("DREAMT", data_dir=cli.data_dir)
        training = preprocessor.make_training_dataset(window_size=preprocessor.window_size, stride=750)
        if len(training) == 0:
            raise ValueError("No valid configured-length training windows in the selected folder")
        from epoch_validation import select_fixed_windows, validation_options, early_stopping_options
        selected_config = json.loads(cli.config.read_text(encoding="utf-8-sig"))
        schedule_config = diffusion_settings(selected_config, default_schedule="linear_power")
        schedule_identity = diffusion_identity(dict(schedule_config, beta_0=.0001, beta_T=.02, timesteps=200))
        validation_config = validation_options(selected_config)
        stopping_config = early_stopping_options(selected_config)
        validation, selection = select_fixed_windows(preprocessor, validation_config)
        identity = preprocessor.data_identity()
        preflight = dict(status="passed", data_identity=identity,
            training_windows=len(training), training_input_shape=list(training[0].shape),
            window_epochs=preprocessor.window_epochs, window_size=preprocessor.window_size,
            validation_windows=[dict(filename=name, start_sample=start) for name, start in validation],
            validation_selection=selection, validation_options=validation_config,
            early_stopping=stopping_config,
            diffusion_settings=schedule_config,
            diffusion_identity=schedule_identity,
            provenance=preprocessor.provenance(), test_npz_opened=False,
            scope="Original WatchSleepNet preprocessed IBI NPZ interface; no real-data or convergence claim")
        save_json(output / "preflight.json", preflight)
        print(json.dumps(dict(status="data_check_passed", training_windows=len(training),
            validation_windows=len(validation), input_shape=list(training[0].shape),
            output_directory=str(output))), flush=True)
        if cli.check_only:
            status.update(status="checked", preflight=str(output / "preflight.json"))
            save_json(output / "status.json", status)
            return

        workspace = output / "workspace"
        snapshot_source(workspace)
        frozen_config = {key: identity.get(key) for key in
            ("metadata_path", "split_path", "split_seed", "multiplier", "downsample_rate", "max_epochs", "window_epochs")}
        frozen_config["validation"] = validation_config
        frozen_config["early_stopping"] = stopping_config
        frozen_config.update(schedule_config)
        frozen_config["data_dir"] = identity["data_directory"]
        frozen_config["generate_splits"] = identity["split_source"] == "generated_subject_split"
        save_json(workspace / "dataset_config.json", frozen_config)
        env = os.environ.copy()
        env["DREAMT_CONFIG_PATH"] = str(workspace / "dataset_config.json")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        manifest = output / "training" / "run_manifest.json"
        stages = [
            ("training", ["scripts/run_simulation.py", "--dataset", "DREAMT", "--epochs", str(cli.epochs),
                          "--batch-size", str(cli.batch_size), "--output-dir", str(output / "training")]),
            ("denoising", ["scripts/evaluate_denoising.py", "--run-manifest", str(manifest),
                           "--threads", str(cli.threads), "--output-dir", str(output / "denoising")]),
        ]
        if not cli.skip_generation:
            stages.append(("generation", ["scripts/evaluate_original_sampler.py", "--training-args-json", str(manifest),
                "--threads", str(cli.threads), "--output-dir", str(output / "generation")]))
        for name, arguments in stages:
            status["status"] = name
            save_json(output / "status.json", status)
            command = [sys.executable, "-B", *arguments]
            print(json.dumps(dict(status=name, log=str(output / (name + ".log")))), flush=True)
            with (output / (name + ".log")).open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=workspace, env=env, stdout=stream, stderr=subprocess.STDOUT)
            status["stages"].append(dict(stage=name, exit_code=result.returncode, command=command))
            if result.returncode:
                raise RuntimeError(f"{name} failed; see {output / (name + '.log')}")
        status.update(status="completed", run_manifest=str(manifest),
                      source_snapshot=str(workspace), no_quality_or_convergence_claim=True)
        save_json(output / "status.json", status)
        print(json.dumps(status), flush=True)
    except BaseException:
        status.update(status="failed", traceback=traceback.format_exc())
        save_json(output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
