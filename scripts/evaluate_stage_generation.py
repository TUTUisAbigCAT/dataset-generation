"""Evaluate two final, equal-budget checkpoints on original S092 stage conditions.

Generate independent full model windows, then score their original 30-second
epochs. No real IBI is fed to the model, no stages are overridden, and no test
data are used to fit a scaler, update weights or select a checkpoint. Production
sampling and tensor metrics require CUDA; --check-only performs no sampling.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from cuda_runtime import require_cuda, assert_cuda_model, cuda_environment
from data_utils import Preprocessor
from signal_normalization import signal_affine_parameters
from diffusion_schedule import build_diffusion_config
from night_data import prepare_night, _resolve_subject
from scripts.evaluate_denoising import bind_training_data
from scripts.evaluate_original_sampler import load_original_parameters
from scripts.export_ablation_validation import ARMS, load_reference_pair, sha256, save_json
from stage_evaluation import select_stage_windows, score_stage_draw, summarize_stage_draws
from training_utils import fetchModel
from validation_metrics import preserve_random_state
from validation_sampling import sample_validation_windows
from watchsleep_loader import EPOCH_SAMPLES, STAGE_NAMES, TARGET_FS


def _require_sampling_device(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    return device


def _table(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty metrics table")
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_pair(contexts, subject, windows_per_stage=8, selection_seed=42):
    """Restore both frozen training identities; explicitly read only this test ID."""
    prepared, selections, checks = {}, {}, {}
    for arm in ARMS:
        context = contexts[arm]
        preprocessor, identity_checks = bind_training_data(
            context["args"], context["manifest"], context["manifest_path"], Preprocessor)
        role, filename = _resolve_subject(preprocessor, subject)
        if role != "test":
            raise ValueError("Stage evaluation requires an explicitly held-out test subject: " + filename)
        before = deepcopy(preprocessor.data_identity())
        opened_before = set(preprocessor.accessed_files)
        prepared[arm] = prepare_night(preprocessor, subject)
        item = prepared[arm]
        unchanged = before == preprocessor.data_identity() and opened_before == set(preprocessor.accessed_files)
        provenance = item["provenance"]
        if (not unchanged or provenance["subject_role"] != "test"
                or provenance["subject_in_original_training_scaler"]
                or provenance["scaler_refitted_for_generation"]
                or not provenance["condition_channels_exclude_reference_ibi"]):
            raise ValueError("Test access changed the frozen training data/scaler identity")
        selections[arm] = select_stage_windows(item,
            window_epochs=preprocessor.window_epochs,
            windows_per_stage=windows_per_stage, seed=selection_seed)
        checks[arm] = dict(training_data_identity_checks=identity_checks,
            training_identity_unchanged=unchanged, subject_provenance=provenance,
            test_access="Explicit inference-only prepare_night; training Preprocessor.record remains sealed")
    a, b = (prepared[arm] for arm in ARMS)
    shared = {name: np.array_equal(a[name], b[name], equal_nan=True)
              for name in ("reference_seconds", "conditions", "stage_epochs_four")}
    shared.update(source_file_sha256=a["provenance"]["source_file_sha256"] == b["provenance"]["source_file_sha256"],
                  selection=selections["preserve"] == selections["masked"])
    if not all(shared.values()):
        raise ValueError("A/B test inputs differ: " + ", ".join(key for key, passed in shared.items() if not passed))
    selection = selections["preserve"]
    starts, length = selection["start_samples"], selection["window_size"]
    if not starts:
        raise ValueError("No complete eligible test windows for stage evaluation")
    reference = np.stack([a["reference_seconds"][start:start+length] for start in starts])
    conditions = np.stack([a["conditions"][start:start+length] for start in starts])
    stages = np.array([row["stage_epochs"] for row in selection["windows"]], dtype=np.int64)
    data = dict(reference_seconds=reference, conditions=conditions, stage_epochs_four=stages,
        start_samples=starts, scoring_mask=np.isfinite(reference) & (reference > 0),
        selection=selection, subject_id=a["subject_id"], filename=a["filename"])
    checks["shared_inputs"] = shared
    return data, checks


def sample_arm(context, data, device, *, seeds, batch_size, output_dir, progress=None):
    """Sample full windows with zero IBI input, then score real 30-second labels."""
    device = _require_sampling_device(device)
    if batch_size < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Expected a positive batch size and distinct sampling seeds")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    args = context["args"]
    reference = torch.as_tensor(data["reference_seconds"], dtype=torch.float64, device=device)
    labels = torch.as_tensor(data["stage_epochs_four"], dtype=torch.int64, device=device)
    conditions = torch.as_tensor(data["conditions"], dtype=torch.float32, device=device)
    # Build the input without copying reference IBI, including at missing points.
    inputs = torch.cat((torch.zeros((*conditions.shape[:2], 1), device=device), conditions), dim=2)
    scaler = context["identity"]["scaler"]
    signal_center, signal_scale = signal_affine_parameters(scaler)
    reports, paths = [], []
    with preserve_random_state(device):
        np.random.seed(42)
        torch.manual_seed(42)
        model = fetchModel(10, 1, args).to(device)
        assert_cuda_model(model)
        checkpoint_bytes = Path(context["checkpoint"]).read_bytes()
        loaded_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        if loaded_hash != context["checkpoint_identity"]["sha256"]:
            raise ValueError("Final checkpoint changed after identity verification")
        saved = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True)
        copied = load_original_parameters(model, saved, torch)
        model.eval()
        schedule = {key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in build_diffusion_config(args).items()}
        for draw, seed in enumerate(seeds):
            started = time.perf_counter()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            generated_batches, sampling = [], []
            for start in range(0, len(inputs), batch_size):
                standardized, info = sample_validation_windows(
                    model, inputs[start:start+batch_size], schedule, args, device)
                generated_batches.append(standardized.to(torch.float64) * signal_scale + signal_center)
                sampling.append(info)
            generated = torch.cat(generated_batches, dim=0)
            report = score_stage_draw(generated, reference, labels,
                start_samples=data["start_samples"], seed=seed, kl_variance_floor_ms2=1.0)
            directory = output / f"draw_{draw:03d}"
            directory.mkdir(exist_ok=False)
            waveform_path = directory / "waveforms.npz"
            with waveform_path.open("xb") as stream:
                np.savez_compressed(stream, generated_ibi_seconds=generated.detach().cpu().numpy(),
                    seed=np.array(seed, dtype=np.int64), window_index=np.arange(len(inputs), dtype=np.int64))
            _table(directory / "per_epoch_metrics.csv", report["per_epoch"])
            _table(directory / "per_stage_metrics.csv",
                   [dict(seed=seed, **report["stages"][stage]) for stage in STAGE_NAMES])
            save_json(directory / "metrics.json", dict(status="completed", draw=draw, seed=seed,
                overall=report["overall"], stages=report["stages"], protocol=report["protocol"],
                sampling=sampling, elapsed_seconds=time.perf_counter()-started))
            paths.extend(path for path in directory.iterdir() if path.is_file())
            reports.append(report)
            if progress:
                progress(dict(event="stage_draw_completed", draw=draw, seed=seed,
                              windows=len(inputs), elapsed_seconds=time.perf_counter()-started,
                              gaussian_kl_real_to_generated=report["overall"]["gaussian_kl_real_to_generated"]))
    return dict(status="completed", checkpoint_kind="final", epoch=context["epoch"], updates=context["updates"],
        checkpoint=str(context["checkpoint"]), checkpoint_sha256=loaded_hash, checkpoint_load=copied,
        scaler=scaler, stages=summarize_stage_draws(reports), device=str(device), batch_size=batch_size,
        artifacts=[path.relative_to(output).as_posix() for path in sorted(paths)],
        training_executed=False, best_checkpoint_selected=False)


def _summary_rows(arms):
    rows = []
    for arm in ARMS:
        for stage in STAGE_NAMES:
            item = arms[arm]["stages"]["stages"][stage]
            row = dict(arm=arm, stage_name=stage, status=item["status"], windows=item["windows"],
                       epochs=item["epochs"], observed_samples=item["observed_samples"], draws=item["draws"])
            for name, stats in item["metrics"].items():
                row[name+"_mean"] = stats["mean"]
                row[name+"_std"] = stats["std"]
            rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", default="S092")
    parser.add_argument("--expected-epochs", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--windows-per-stage", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    cli = parser.parse_args(argv)
    if min(cli.expected_epochs, cli.repeats, cli.windows_per_stage, cli.batch_size, cli.threads) < 1:
        parser.error("epochs, repeats, windows-per-stage, batch-size and threads must be positive")
    if not 0 <= cli.seed <= 2**32-cli.repeats or not 0 <= cli.selection_seed < 2**32:
        parser.error("All sampling seeds and selection-seed must fit unsigned 32-bit integers")
    source = cli.ablation_run.expanduser().resolve()
    output = cli.output_dir.expanduser().resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be outside the read-only ablation training directory")
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status="checking", ablation_run=str(source), output_directory=str(output), arms={},
        subject=cli.subject, checkpoint_kind="final", sampling_performed=False, training_executed=False,
        best_checkpoint_selected=False)
    save_json(output / "summary.json", status)
    try:
        contexts, shared_checks = load_reference_pair(source)
        if any(context["epoch"] != cli.expected_epochs for context in contexts.values()):
            raise ValueError("Final checkpoint epoch differs from --expected-epochs")
        print(json.dumps(dict(event="stage_identity_check_passed", epoch=cli.expected_epochs)), flush=True)
        data, data_checks = prepare_pair(contexts, cli.subject, cli.windows_per_stage, cli.selection_seed)
        seeds = list(range(cli.seed, cli.seed+cli.repeats))
        status.update(shared_reference_checks=shared_checks, data_checks=data_checks,
                      selection=data["selection"], subject=data["subject_id"])
        with (output / "reference.npz").open("xb") as stream:
            np.savez_compressed(stream, reference_ibi_seconds=data["reference_seconds"],
                scoring_mask=data["scoring_mask"], clean_conditions=data["conditions"],
                sleep_stages=data["stage_epochs_four"], start_sample=np.array(data["start_samples"], dtype=np.int64),
                window_index=np.arange(len(data["start_samples"]), dtype=np.int64), subject_id=np.array(data["subject_id"]))
        protocol = dict(schema="original_condition_stage_evaluation_v1", subject=data["subject_id"],
            selection=data["selection"], seeds=seeds, repeats=cli.repeats, batch_size=cli.batch_size,
            epoch=cli.expected_epochs, updates={arm:contexts[arm]["updates"] for arm in ARMS},
            source_file_sha256=data_checks["preserve"]["subject_provenance"]["source_file_sha256"],
            sample_rate_hz=TARGET_FS, scoring_epoch_samples=EPOCH_SAMPLES,
            reference_mask="finite_reference_and_reference_greater_than_zero", variance="population; correction=0",
            gaussian_kl=dict(direction="real || generated", variance_floor_ms2=1.0,
                aggregation="Equal weight per original 30-second epoch within each stage"),
            generation=dict(independent_windows=True, overlap_samples=0, real_ibi_prefix_samples=0,
                stage_conditions_overridden=False, reference_ibi_used_as_model_input=False,
                full_recorded_diffusion_schedule=True, clipping_applied=False),
            condition_columns=contexts["preserve"]["identity"]["columns"][1:],
            checkpoints={arm:dict(epoch=contexts[arm]["epoch"], updates=contexts[arm]["updates"],
                identity=contexts[arm]["checkpoint_identity"], configuration=contexts[arm]["config_provenance"],
                runtime_source_checks=contexts[arm]["source_checks"]) for arm in ARMS},
            analysis_source_sha256={name:sha256(ROOT/name) for name in
                ("scripts/evaluate_stage_generation.py", "stage_evaluation.py", "night_data.py",
                 "validation_metrics.py", "signal_normalization.py", "scripts/plot_stage_evaluation.py")},
            limitations=["One held-out subject; sampling standard deviations are not confidence intervals.",
                "Thirty-second Gaussian moment KL is not numerically interchangeable with 120-second training-validation KL.",
                "Moment agreement does not establish temporal, physiological or clinical realism.",
                "A generated sample is not expected to reconstruct an unrelated real sample point by point.",
                "Same seeds do not guarantee bitwise equality across CUDA software or hardware."])
        save_json(output / "protocol.json", protocol)
        print(json.dumps(dict(event="stage_windows_selected", windows=len(data["start_samples"]),
                              stages=data["selection"]["stages"])), flush=True)
        if cli.check_only:
            status.update(status="checked", file_hashes={name:sha256(output/name) for name in ("reference.npz", "protocol.json")})
            save_json(output / "summary.json", status)
            return status
        device = _require_sampling_device(require_cuda())
        torch.set_num_threads(cli.threads)
        status.update(status="sampling", cuda_environment=cuda_environment(device), threads=cli.threads,
                      sampling_performed=True)
        save_json(output / "summary.json", status)
        for arm in ARMS:
            status["arms"][arm] = sample_arm(contexts[arm], data, device, seeds=seeds, batch_size=cli.batch_size,
                output_dir=output/arm, progress=lambda record, arm=arm:
                    print(json.dumps(dict(arm=arm, **record)), flush=True))
            save_json(output / "summary.json", status)
        _table(output / "stage_summary.csv", _summary_rows(status["arms"]))
        status.update(status="completed", file_hashes={path.relative_to(output).as_posix():sha256(path)
            for path in sorted(output.rglob("*")) if path.is_file() and path != output/"summary.json"})
        save_json(output / "summary.json", status)
        print(json.dumps(dict(event="stage_evaluation_completed", output_directory=str(output))), flush=True)
        return status
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        save_json(output / "summary.json", status)
        raise


if __name__ == "__main__":
    main()
