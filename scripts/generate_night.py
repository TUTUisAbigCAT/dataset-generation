"""Generate all complete stored DREAMT epochs with a frozen training run.

The requested participant supplies sleep-stage and demographic conditions.
Reference IBI never enters the sampler. Production sampling requires CUDA;
--check-only validates files, identities and the generation plan without a model.
"""
from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_denoising import (bind_training_data, load_training_run,
                                        resolve_checkpoint, save_json, sha256)
from scripts.evaluate_original_sampler import load_original_parameters
from window_config import EPOCH_SAMPLES, FS


def initialization_threads(manifest):
    """Reconstruct fixed S4 buffers with the recorded constructor thread count."""
    value = manifest.get("threads", 2)
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("Manifest threads must be a positive integer")
    return int(value)


def _sample(model_args, manifest, checkpoint, conditions, scaler, device, *,
            seed, threads, stride, windows_per_group, progress):
    """Construct and sample a model; this boundary accepts no reference IBI."""
    import numpy as np
    import torch
    from cuda_runtime import assert_cuda_model, cuda_environment
    from diffusion_schedule import build_diffusion_config
    from night_sampling import generate_night
    from training_utils import fetchModel

    constructor_threads = initialization_threads(manifest)
    torch.set_num_threads(constructor_threads)
    np.random.seed(42)
    torch.manual_seed(42)
    model = fetchModel(10, 1, model_args).to(device)
    assert_cuda_model(model)
    saved = torch.load(checkpoint, map_location=device)
    checkpoint_load = load_original_parameters(model, saved, torch)
    del saved
    model.eval()
    schedule = {key: value.to(device) if torch.is_tensor(value) else value
                for key, value in build_diffusion_config(model_args).items()}
    condition_tensor = torch.as_tensor(conditions, device=device, dtype=torch.float32)

    # A sampling seed must not change the fixed buffers reconstructed above.
    torch.set_num_threads(threads)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    # The sampler's overlap guidance needs autograd, including during eval().
    with torch.device(device):
        standardized, metadata = generate_night(model, condition_tensor, schedule, model_args,
            device, stride=stride, windows_per_group=windows_per_group, progress=progress)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if standardized.device != device or standardized.shape != (len(conditions),):
        raise RuntimeError("Night sampler returned an unexpected shape or device")
    generated_seconds = (standardized.detach().to(torch.float64) * scaler["signal_std_seconds"]
                         + scaler["signal_mean_seconds"])
    execution = dict(**cuda_environment(device), python=sys.version, executable=sys.executable,
        initialization_seed=42, initialization_threads=constructor_threads,
        initialization_threads_source="manifest.threads" if "threads" in manifest else "legacy default 2",
        sampling_seed=seed, sampling_threads=threads, sampling_seconds=elapsed,
        checkpoint_load=checkpoint_load, reference_ibi_passed_to_sampler=False,
        compute_policy="CUDA required; no CPU fallback",
        host_boundaries=["file I/O and WatchSleepNet preprocessing", "seeded model construction",
                         "recorded diffusion schedule initialization", "final serialization"],
        cpu_cuda_bitwise_equivalence_claim=False)
    return (standardized.detach().cpu().numpy(), generated_seconds.cpu().numpy(),
            metadata, execution)


def _output_statistics(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    result = dict(samples=int(values.size), nonfinite_samples=int((~finite).sum()),
                  negative_samples=int((finite & (values < 0)).sum()),
                  zero_samples=int((values == 0).sum()), units="seconds")
    if finite.all() and values.size:
        with np.errstate(over="ignore", invalid="ignore"):
            for name, value in (("mean_seconds", values.mean()), ("std_seconds", values.std()),
                                ("minimum_seconds", values.min()), ("maximum_seconds", values.max())):
                result[name] = float(value) if np.isfinite(value) else None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", "--training-args-json", dest="run_manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional relocated weights; must match the selected manifest's checkpoint hash")
    parser.add_argument("--subject", default="S092", help="Exact participant ID, case insensitive (default: S092)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed, independent of seed42 model construction")
    parser.add_argument("--threads", type=int, default=4, help="Sampling CPU thread count; constructor uses manifest threads")
    parser.add_argument("--stride-epochs", type=int, default=1, help="Generation stride in 30-second epochs")
    parser.add_argument("--windows-per-group", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing outputs are never overwritten")
    parser.add_argument("--check-only", action="store_true", help="Validate identities, subject and plan without a model or GPU")
    cli = parser.parse_args(argv)
    if min(cli.threads, cli.stride_epochs, cli.windows_per_group) <= 0:
        parser.error("--threads, --stride-epochs and --windows-per-group must be positive")
    if not 0 <= cli.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 2**32 - 1")
    cli.output_dir = cli.output_dir.resolve()
    cli.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(schema="dreamt_whole_night_generation_v1", status="starting",
        requested_subject=cli.subject, check_only=cli.check_only, sampling_seed=cli.seed,
        reference_ibi_passed_to_sampler=False, real_DREAMT_quality_verified=False,
        generated_output_is_synthetic=True, output_dir=str(cli.output_dir))

    def save_report():
        save_json(cli.output_dir / "summary.json", report)

    def progress(event):
        print(json.dumps(dict(event="night_generation_progress", progress=event), allow_nan=False), flush=True)

    save_report()
    try:
        from cuda_runtime import require_cuda
        device = None if cli.check_only else require_cuda()
        from data_utils import Preprocessor
        from diffusion_schedule import diffusion_identity
        from night_data import prepare_night, save_night_npz
        from night_sampling import _night_sampler, plan_night

        args, manifest, config_provenance = load_training_run(cli.run_manifest)
        checkpoint, checkpoint_identity = resolve_checkpoint(cli.run_manifest, manifest, cli.checkpoint)
        constructor_threads = initialization_threads(manifest)
        preprocessor, data_identity_checks = bind_training_data(args, manifest, cli.run_manifest, Preprocessor)
        expected = ("IBI", "stage_W", "stage_Light", "stage_REM", "stage_Deep",
                    "AHI", "BMI", "AGE", "gender_F", "gender_M")
        if (tuple(preprocessor.columns) != expected
                or list(preprocessor.hierarchical_features_cyclic) != list(expected[1:])):
            raise ValueError("Expected one IBI channel and the recorded nine DREAMT conditions")
        prepared = prepare_night(preprocessor, cli.subject)
        stride = cli.stride_epochs * EPOCH_SAMPLES
        plan = plan_night(len(prepared["conditions"]), args.window_size,
                          stride=stride, windows_per_group=cli.windows_per_group)
        # Check the exact grouped adapter even in data-only preflight. Merely
        # verifying its base single-group sampler would miss failed edits here.
        source_provenance = _night_sampler()[-1]
        report.update(subject_id=prepared["subject_id"], source_filename=prepared["filename"],
            checkpoint=str(checkpoint), checkpoint_identity=checkpoint_identity,
            config_provenance=config_provenance, training_args=vars(args),
            training_data_identity_checks=data_identity_checks, data=prepared["provenance"],
            source_record_synthetic=prepared["record"].get("synthetic"),
            scaler=preprocessor.scaler, diffusion_identity=diffusion_identity(args),
            source_provenance=source_provenance, initialization_seed=42,
            initialization_threads=constructor_threads, sampling_threads=cli.threads,
            sample_rate_hz=FS, target_samples=len(prepared["conditions"]),
            target_duration_seconds=len(prepared["conditions"]) / FS,
            window_samples=args.window_size, stride_samples=stride,
            windows_per_group=cli.windows_per_group, groups=len(plan), generation_plan=plan,
            source_sha256={name: sha256(ROOT / name) for name in (
                "scripts/generate_night.py", "night_data.py", "night_sampling.py", "data_utils.py",
                "watchsleep_loader.py", "window_config.py", "diffusion_schedule.py", "TSImputers/S4Model.py",
                "scripts/evaluate_denoising.py", "scripts/evaluate_original_sampler.py")})
        if cli.check_only:
            report.update(status="checked_only", model_constructed=False, sampling_performed=False)
            save_report()
            print(json.dumps(dict(status=report["status"], subject=prepared["subject_id"],
                samples=report["target_samples"], groups=len(plan), output=str(cli.output_dir)), allow_nan=False), flush=True)
            return report

        report.update(status="sampling", model_constructed=False, sampling_performed=False)
        save_report()
        standardized, generated, sampling, execution = _sample(args, manifest, checkpoint,
            prepared["conditions"], preprocessor.scaler, device, seed=cli.seed, threads=cli.threads,
            stride=stride, windows_per_group=cli.windows_per_group, progress=progress)
        report.update(model_constructed=True, sampling_performed=True, sampler=sampling,
                      execution=execution, result=_output_statistics(generated))
        output_path = cli.output_dir / (prepared["subject_id"] + ".npz")
        try:
            report["output_npz"] = save_night_npz(output_path, prepared, generated)
        except Exception:
            import numpy as np

            diagnostic = cli.output_dir / "diagnostic_generated.npz"
            np.savez_compressed(diagnostic, generated_seconds=generated,
                generated_standardized=standardized, conditions=prepared["conditions"],
                sample_rate_hz=np.array(FS), subject=np.array(prepared["subject_id"]))
            report["diagnostic_output"] = str(diagnostic)
            report["invalid_values_clipped_or_repaired"] = False
            raise
        metadata_path = cli.output_dir / "participant_info.csv"
        preprocessor.metadata.loc[[prepared["subject_id"]]].reset_index().to_csv(metadata_path, index=False)
        report.update(status="completed", output_npz_sha256=sha256(output_path),
            participant_metadata=str(metadata_path), participant_metadata_sha256=sha256(metadata_path),
            invalid_values_clipped_or_repaired=False)
        save_report()
        print(json.dumps(dict(status="completed", subject=prepared["subject_id"],
            samples=report["target_samples"], output_npz=str(output_path)), allow_nan=False), flush=True)
        return report
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        save_report()
        raise


if __name__ == "__main__":
    main()
