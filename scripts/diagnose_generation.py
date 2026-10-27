"""Audit training identity, repeated conditions, condition response, then night continuation.

This diagnostic reads the existing training run_manifest solely to verify the
current inference setup. It does not change training, the checkpoint format,
the production samplers, their RNG devices, or the normal inference command.
Sampling requires CUDA; --check-only stops before constructing a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from night_checkpoint import load_checkpoint_state, load_inference_config
from scripts.evaluate_denoising import save_json, sha256
from window_config import EPOCH_SAMPLES, FS


def source_hashes():
    names = ["training_wavestitch.py", "data_utils.py", "training_utils.py",
             "watchsleep_loader.py", "window_config.py", "diffusion_schedule.py",
             "cuda_runtime.py", "dataset_config.json", "epoch_validation.py",
             "validation_metrics.py", "validation_sampling.py", "night_checkpoint.py",
             "night_data.py", "night_sampling.py", "diagnostic_identity.py",
             "diagnostic_sampling.py", "scripts/evaluate_original_sampler.py",
             "scripts/evaluate_denoising.py", "scripts/generate_night.py",
             "scripts/diagnose_generation.py", "third_party/WatchSleepNet/data_setup.py"]
    names += [p.relative_to(ROOT).as_posix() for folder in ("TSImputers", "utils")
              for p in (ROOT / folder).glob("*.py")]
    return {name: sha256(ROOT / name) for name in sorted(set(names)) if (ROOT / name).is_file()}


def select_probe_starts(conditions, window_size, requested_epochs=None):
    """Select at most one representative per stage using conditions, never IBI."""
    import numpy as np

    count = len(conditions)
    if count < window_size or count % EPOCH_SAMPLES:
        raise ValueError("The record must contain complete epochs and at least one trained window")
    if requested_epochs is not None:
        starts = [int(epoch) * EPOCH_SAMPLES for epoch in requested_epochs]
        if len(set(starts)) != len(starts) or any(s < 0 or s + window_size > count for s in starts):
            raise ValueError("Probe starts must be unique, nonnegative epochs with a complete window")
        return starts
    stages = np.asarray(conditions)[::EPOCH_SAMPLES, :4].argmax(axis=1)
    width = window_size // EPOCH_SAMPLES
    candidates = np.lib.stride_tricks.sliding_window_view(stages, width)
    selected = []
    for stage in range(4):
        counts = (candidates == stage).sum(axis=1)
        if not counts.max():
            continue
        choices = np.flatnonzero(counts == counts.max())
        position = int(choices[len(choices) // 2]) * EPOCH_SAMPLES
        if position not in selected:
            selected.append(position)
    return sorted(selected or [0])


def _sample(args, checkpoint, prepared, device, *, starts, repeats, comparison_repeats,
            seed, threads, stride, windows_per_group, kl_variance_floor_ms2,
            expected_checkpoint_sha256, progress):
    import numpy as np
    import torch
    from cuda_runtime import assert_cuda_model, cuda_environment
    from diagnostic_sampling import run_sampling_diagnostics
    from diffusion_schedule import build_diffusion_config
    from training_utils import fetchModel

    torch.set_num_threads(threads)
    np.random.seed(42)
    torch.manual_seed(42)
    model = fetchModel(10, 1, args).to(device)
    assert_cuda_model(model)
    checkpoint_bytes = checkpoint.read_bytes()
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    if digest != expected_checkpoint_sha256:
        raise RuntimeError("Checkpoint changed after identity verification; sampling not started")
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True)
    restored = load_checkpoint_state(model, saved)
    del saved, checkpoint_bytes
    model.eval()
    schedule = {key: value.to(device) if torch.is_tensor(value) else value
                for key, value in build_diffusion_config(args).items()}
    conditions = torch.as_tensor(prepared["conditions"], dtype=torch.float32, device=device)
    reference = torch.as_tensor(prepared["reference_seconds"], dtype=torch.float64, device=device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    diagnostics, arrays = run_sampling_diagnostics(model, conditions, reference,
        prepared["scaler"], args, schedule, device, starts=starts, repeats=repeats,
        comparison_repeats=comparison_repeats, seed=seed, stride=stride,
        windows_per_group=windows_per_group, kl_variance_floor_ms2=kl_variance_floor_ms2,
        progress=progress)
    torch.cuda.synchronize(device)
    execution = dict(**cuda_environment(device), loaded_checkpoint_sha256=digest,
        checkpoint_load=restored, seconds=time.perf_counter()-started,
        model_and_reverse_state_device=str(device), sampler_implementation_unchanged=True,
        noise_rng_device_policy="Preserve production factory behavior; see diagnostics.noise_factory",
        real_reference_passed_to_denoiser=False, trained_or_updated_weights=False)
    return diagnostics, arrays, execution


def write_tables(output, diagnostics):
    """Persist small phase tables; nested cells remain explicit JSON strings."""
    tables = []
    for stage, detail in diagnostics.get("stages", {}).items():
        for name, rows in detail.items():
            if not (name == "rows" or name.endswith("_rows")) or not isinstance(rows, list) or not rows:
                continue
            if not all(isinstance(row, dict) for row in rows):
                continue
            path = output / (stage + "_" + name + ".csv")
            columns = list(dict.fromkeys(key for row in rows for key in row))
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                                     if isinstance(value, (dict, list)) else value
                                     for key, value in row.items()})
            tables.append(str(path))
    return tables


def _json_file(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def audit_previous_target(audit, previous, prepared):
    """Bind a previous generation summary to this exact subject and NPZ."""
    if previous is None:
        return
    historical_data = previous.get("data")
    historical_data = historical_data if isinstance(historical_data, dict) else {}
    targets = (
        ("subject_id", previous.get("subject_id"), prepared["subject_id"]),
        ("source_file_sha256", historical_data.get("source_file_sha256"),
         prepared["record"]["file_sha256"]),
    )
    for name, historical, current in targets:
        if not isinstance(historical, str) or not historical:
            status = "incomplete"
        elif name == "source_file_sha256" and (
                len(historical) != 64 or any(c not in "0123456789abcdefABCDEF" for c in historical)):
            status = "incomplete"
        else:
            status = "matched" if historical.lower() == current.lower() else "mismatch"
        audit["blocking_checks"].append(dict(name="previous.target." + name, status=status,
            detail="Previous generation must use the same target subject and source NPZ contents",
            expected=historical, actual=current))
    statuses = {check["status"] for check in audit["blocking_checks"]}
    audit["status"] = "mismatch" if "mismatch" in statuses else (
        "incomplete" if "incomplete" in statuses else "matched")
    audit["can_sample"] = audit["status"] == "matched"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path,
                        help="Audit only; default: checkpoint directory/run_manifest.json")
    parser.add_argument("--validation-summary", type=Path,
                        help="Best checkpoint digest; default: checkpoint directory/validation/summary.json")
    parser.add_argument("--inference-summary", type=Path,
                        help="Optional previous job summary to compare its actual recorded identity")
    parser.add_argument("--config", type=Path, default=ROOT / "dataset_config.json")
    parser.add_argument("--subject", default="S092")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--window-start-epochs", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--comparison-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stride-epochs", type=int, default=1)
    parser.add_argument("--windows-per-group", type=int, default=4)
    cli = parser.parse_args(argv)
    if min(cli.repeats, cli.comparison_repeats, cli.threads, cli.stride_epochs, cli.windows_per_group) < 1:
        parser.error("Repeat counts, threads, stride and windows per group must be positive")
    if not 0 <= cli.seed <= 2**32 - max(cli.repeats, cli.comparison_repeats):
        parser.error("Seed and repeat counts must fit uint32 seeds")
    output = cli.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="dreamt_generation_diagnostics_v1", status="starting",
        output_dir=str(output), subject=cli.subject, check_only=cli.check_only,
        stage_order=["config_identity", "same_conditions", "changed_conditions", "independent_vs_continued"],
        trained_or_updated_weights=False, sampling_performed=False, source_sha256=source_hashes(),
        previous_stage_summaries={}, real_DREAMT_quality_verified=False)

    def save():
        save_json(output / "summary.json", report)

    def progress(event):
        if event.get("event") == "stage_summary":
            report["previous_stage_summaries"][event["stage"]] = event["summary"]
        report["latest_progress"] = event
        save()
        # Stage tables remain on disk; avoid thousands of columns in Slurm logs.
        brief = {key: value for key, value in event.items() if key != "summary"}
        print(json.dumps(dict(event="generation_diagnostic_progress", progress=brief), allow_nan=False), flush=True)

    save()
    try:
        from data_utils import Preprocessor
        from diagnostic_identity import audit_training_identity
        from diffusion_schedule import diffusion_identity
        from night_data import prepare_night
        from night_sampling import plan_night, _night_sampler
        import numpy as np

        checkpoint = cli.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Checkpoint does not exist: " + str(checkpoint))
        training_path = (cli.training_manifest or checkpoint.parent / "run_manifest.json").expanduser().resolve()
        validation_path = (cli.validation_summary or checkpoint.parent / "validation/summary.json").expanduser().resolve()
        report.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                      training_manifest=str(training_path), validation_summary=str(validation_path))
        if not training_path.is_file():
            audit = dict(status="incomplete", can_sample=False, blocking_checks=[dict(
                name="training_manifest", status="incomplete", detail="Training record is absent: " + str(training_path))])
            report.update(status="blocked_config", config_audit=audit)
            save_json(output / "config_audit.json", audit)
            raise ValueError("Training record is required for this diagnostic identity check; sampling not started")
        training = _json_file(training_path)
        validation = _json_file(validation_path) if validation_path.is_file() else None
        if cli.validation_summary and validation is None:
            raise FileNotFoundError("Explicit validation summary does not exist: " + str(validation_path))
        previous = _json_file(cli.inference_summary.expanduser().resolve()) if cli.inference_summary else None
        args, config, config_provenance = load_inference_config(cli.config)
        preprocessor = Preprocessor("DREAMT", data_config=config)
        identity = preprocessor.data_identity()
        audit = audit_training_identity(training, args, identity, source_hashes(),
            report["checkpoint_sha256"], validation_summary=validation, inference_summary=previous)
        report.update(config_audit=audit, model_args=vars(args), config_provenance=config_provenance,
            configured_data_identity=identity, diffusion_identity=diffusion_identity(args),
            training_manifest_sha256=sha256(training_path),
            validation_summary_sha256=sha256(validation_path) if validation is not None else None,
            inference_summary_sha256=sha256(cli.inference_summary.expanduser().resolve()) if previous is not None else None)
        save_json(output / "config_audit.json", audit)
        if not audit["can_sample"]:
            report["status"] = "blocked_config"
            raise ValueError("Training/inference identity is not verified; inspect config_audit.json before sampling")
        prepared = prepare_night(preprocessor, cli.subject)
        prepared["scaler"] = preprocessor.scaler
        audit_previous_target(audit, previous, prepared)
        save_json(output / "config_audit.json", audit)
        if not audit["can_sample"]:
            report["status"] = "blocked_config"
            raise ValueError("Previous generation target identity is not verified; inspect config_audit.json before sampling")
        if not np.isfinite(prepared["reference_seconds"]).all():
            raise ValueError("Reference contains nonfinite samples; diagnostic moments cannot be computed")
        starts = select_probe_starts(prepared["conditions"], args.window_size, cli.window_start_epochs)
        stride = cli.stride_epochs * EPOCH_SAMPLES
        plan = plan_night(len(prepared["conditions"]), args.window_size, stride=stride,
                          windows_per_group=cli.windows_per_group)
        floor = float(config.get("validation", {}).get("kl_variance_floor_ms2", 1.0))
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError("KL variance floor must be finite and positive")
        report.update(subject=prepared["subject_id"], data=prepared["provenance"],
            source_record_sha256=prepared["record"]["file_sha256"],
            input_conditions_sha256=hashlib.sha256(prepared["conditions"].tobytes()).hexdigest(),
            source_record_synthetic=prepared["record"].get("synthetic"),
            scaler=preprocessor.scaler, source_provenance=_night_sampler()[-1],
            probe_start_samples=starts, probe_start_epochs=[s // EPOCH_SAMPLES for s in starts],
            probe_selection="explicit epoch starts" if cli.window_start_epochs else
                            "one representative per present stage, selected using stage conditions only; maximal stage coverage then midpoint",
            samples=len(prepared["conditions"]), sample_rate_hz=FS, window_samples=args.window_size,
            repeats=cli.repeats, comparison_repeats=cli.comparison_repeats, seed=cli.seed,
            kl_variance_floor_ms2=floor, night_groups=len(plan), generation_plan=plan)
        if cli.check_only:
            report["status"] = "checked_only"
            save()
            print(json.dumps(dict(status=report["status"], output=str(output), config_matched=True)), flush=True)
            return report
        from cuda_runtime import require_cuda
        device = require_cuda()
        report["status"] = "sampling"
        save()
        diagnostics, arrays, execution = _sample(args, checkpoint, prepared, device,
            starts=starts, repeats=cli.repeats, comparison_repeats=cli.comparison_repeats,
            seed=cli.seed, threads=cli.threads, stride=stride, windows_per_group=cli.windows_per_group,
            kl_variance_floor_ms2=floor, expected_checkpoint_sha256=report["checkpoint_sha256"], progress=progress)
        if execution["loaded_checkpoint_sha256"] != report["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed after identity verification; discard this diagnostic")
        # This is a diagnostic artifact, not a physiological-validity export.
        # Negative finite generated values are deliberately retained for analysis.
        arrays_path = output / "diagnostics.npz"
        saved_arrays = {}
        for name, value in arrays.items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            saved_arrays[name] = np.asarray(value)
            if saved_arrays[name].dtype.hasobject:
                raise ValueError("Diagnostic arrays cannot require pickle: " + name)
        with arrays_path.open("xb") as stream:
            np.savez_compressed(stream, **saved_arrays)
        report.update(status="completed", sampling_performed=True, diagnostics=diagnostics,
            execution=execution, arrays_path=str(arrays_path), arrays_sha256=sha256(arrays_path),
            csv_files=write_tables(output, diagnostics))
        save()
        print(json.dumps(dict(status="completed", output=str(output), arrays=str(arrays_path))), flush=True)
        return report
    except Exception as error:
        if report["status"] != "blocked_config":
            report["status"] = "failed"
        report.update(error=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
