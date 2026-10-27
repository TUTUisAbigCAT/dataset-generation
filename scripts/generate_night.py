"""Generate a complete stored DREAMT night directly from a .pth checkpoint.

The requested participant supplies sleep-stage and demographic conditions.
Reference IBI never enters the sampler. Production sampling requires CUDA.
No training manifest is read. Dataset JSON selects data, window length and the
diffusion schedule. Its training split is used to reconstruct normalization.
--check-only validates paths, data and the generation plan without a model.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_denoising import save_json, sha256
from window_config import EPOCH_SAMPLES, FS
from night_checkpoint import load_inference_config, load_checkpoint_state


def _sample(model_args, checkpoint, conditions, scaler, device, *,
            seed, threads, stride, windows_per_group, progress):
    """Construct and sample a model; this boundary accepts no reference IBI."""
    import numpy as np
    import torch
    from cuda_runtime import assert_cuda_model, cuda_environment
    from diffusion_schedule import build_diffusion_config
    from night_sampling import generate_night
    from training_utils import fetchModel

    torch.set_num_threads(threads)
    np.random.seed(42)
    torch.manual_seed(42)
    model = fetchModel(10, 1, model_args).to(device)
    assert_cuda_model(model)
    # Load one byte snapshot, so later replacements of a live best.pth cannot
    # change the model halfway through inference or misidentify loaded weights.
    checkpoint_bytes = Path(checkpoint).read_bytes()
    loaded_checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True)
    checkpoint_load = load_checkpoint_state(model, saved)
    del saved
    del checkpoint_bytes
    model.eval()
    schedule = {key: value.to(device) if torch.is_tensor(value) else value
                for key, value in build_diffusion_config(model_args).items()}
    condition_tensor = torch.as_tensor(conditions, device=device, dtype=torch.float32)

    # Parameters AND buffers came from the checkpoint, including the S4 state.
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
        initialization_seed=42, initialization_threads=threads,
        sampling_seed=seed, sampling_threads=threads, sampling_seconds=elapsed,
        checkpoint_load=checkpoint_load, loaded_checkpoint_sha256=loaded_checkpoint_sha256,
        reference_ibi_passed_to_sampler=False,
        compute_policy="CUDA required; no CPU fallback",
        host_boundaries=["file I/O and WatchSleepNet preprocessing", "seeded model construction",
                         "configured diffusion schedule initialization", "final serialization"],
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
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Model state_dict .pth file (for example best_kl_model.pth); no manifest required")
    parser.add_argument("--config", type=Path, default=ROOT / "dataset_config.json",
                        help="Dataset/window/schedule configuration; must match the checkpoint's training setup")
    parser.add_argument("--subject", default="S092", help="Exact participant ID, case insensitive (default: S092)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed, independent of seed42 model construction")
    parser.add_argument("--threads", type=int, default=4, help="Thread count for model construction and sampling")
    parser.add_argument("--stride-epochs", type=int, default=1, help="Generation stride in 30-second epochs")
    parser.add_argument("--windows-per-group", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing outputs are never overwritten")
    parser.add_argument("--check-only", action="store_true", help="Validate checkpoint path, config, subject and plan without loading a model or using GPU")
    cli = parser.parse_args(argv)
    if min(cli.threads, cli.stride_epochs, cli.windows_per_group) <= 0:
        parser.error("--threads, --stride-epochs and --windows-per-group must be positive")
    if not 0 <= cli.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 2**32 - 1")
    cli.output_dir = cli.output_dir.resolve()
    cli.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(schema="dreamt_whole_night_generation_v2", status="starting", input_mode="direct_checkpoint",
        requested_subject=cli.subject, check_only=cli.check_only, sampling_seed=cli.seed,
        reference_ibi_passed_to_sampler=False, real_DREAMT_quality_verified=False,
        generated_output_is_synthetic=True, output_dir=str(cli.output_dir),
        historical_training_identity_verified=False, model_state_validated=False,
        normalization_source="recomputed from the configured training split",
        training_manifest_required=False)

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

        checkpoint = cli.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Checkpoint file does not exist: " + str(checkpoint))
        checkpoint_identity = dict(path=str(checkpoint), sha256=sha256(checkpoint),
                                   hash_scope="preflight_file", historical_digest_verified=False)
        args, data_config, config_provenance = load_inference_config(cli.config)
        preprocessor = Preprocessor("DREAMT", data_config=data_config)
        expected = ("IBI", "stage_W", "stage_Light", "stage_REM", "stage_Deep",
                    "AHI", "BMI", "AGE", "gender_F", "gender_M")
        if (tuple(preprocessor.columns) != expected
                or list(preprocessor.hierarchical_features_cyclic) != list(expected[1:])):
            raise ValueError("Expected one IBI channel and the recorded nine DREAMT conditions")
        prepared = prepare_night(preprocessor, cli.subject)
        prepared["provenance"]["scaler_refitted_for_generation"] = True
        prepared["provenance"]["subject_in_scaler_training_split"] = prepared["provenance"].pop("subject_in_original_training_scaler")
        stride = cli.stride_epochs * EPOCH_SAMPLES
        plan = plan_night(len(prepared["conditions"]), args.window_size,
                          stride=stride, windows_per_group=cli.windows_per_group)
        # Check the exact grouped adapter even in data-only preflight. Merely
        # verifying its base single-group sampler would miss failed edits here.
        source_provenance = _night_sampler()[-1]
        report.update(subject_id=prepared["subject_id"], source_filename=prepared["filename"],
            checkpoint=str(checkpoint), checkpoint_identity=checkpoint_identity,
            config_provenance=config_provenance, model_args=vars(args),
            configured_data_identity=preprocessor.data_identity(), data=prepared["provenance"],
            source_record_synthetic=prepared["record"].get("synthetic"),
            scaler=preprocessor.scaler, diffusion_identity=diffusion_identity(args),
            source_provenance=source_provenance, initialization_seed=42,
            initialization_threads=cli.threads, sampling_threads=cli.threads,
            sample_rate_hz=FS, target_samples=len(prepared["conditions"]),
            target_duration_seconds=len(prepared["conditions"]) / FS,
            window_samples=args.window_size, stride_samples=stride,
            windows_per_group=cli.windows_per_group, groups=len(plan), generation_plan=plan,
            source_sha256={name: sha256(ROOT / name) for name in (
                "scripts/generate_night.py", "night_checkpoint.py", "night_data.py", "night_sampling.py", "data_utils.py",
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
        standardized, generated, sampling, execution = _sample(args, checkpoint,
            prepared["conditions"], preprocessor.scaler, device, seed=cli.seed, threads=cli.threads,
            stride=stride, windows_per_group=cli.windows_per_group, progress=progress)
        report.update(model_constructed=True, sampling_performed=True, sampler=sampling,
                      execution=execution, model_state_validated=True, result=_output_statistics(generated))
        report["checkpoint_identity"]["sha256"] = execution["loaded_checkpoint_sha256"]
        report["checkpoint_identity"]["hash_scope"] = "loaded_checkpoint_bytes"
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
