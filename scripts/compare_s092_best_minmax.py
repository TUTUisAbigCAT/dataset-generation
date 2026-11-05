"""Compare minmax best models on a consecutive S092 segment, CUDA only.

Separate workers import each run's frozen workspace. The new inference adapter
reads only S092 and the hash-verified participant CSV; it never fits a scaler.
--check-only verifies inputs and provenance without constructing a model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("minmax", "minmax01")
S092_SHA256 = "f41e37bd921b39851e82c0075e8c2b4002b73db7329d9e3853f96516f36b08d2"
REQUIRED_SOURCES = (
    "training_utils.py", "TSImputers/SSSDS4Imputer.py", "TSImputers/S4Model.py",
    "utils/util.py", "data_utils.py", "watchsleep_loader.py", "window_config.py",
    "diffusion_schedule.py", "validation_sampling.py", "validation_metrics.py",
    "scripts/evaluate_original_sampler.py", "signal_normalization.py", "cuda_runtime.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + "\n", encoding="utf-8")


def load_arm(run_root, arm):
    """Bind to best, never fall back to the final model or current config."""
    if arm not in ARMS:
        raise ValueError("Unknown comparison arm")
    root = Path(run_root).expanduser().resolve()
    training, workspace = root / "run/training", root / "run/workspace"
    manifest_path = training / "best_kl_manifest.json"
    manifest = read_json(manifest_path)
    checkpoint = training / "best_kl_model.pth"
    selection = manifest.get("checkpoint_selection", {})
    epoch = manifest.get("checkpoint_epoch")
    best = manifest.get("validation", {}).get("best_checkpoints", {}).get("kl", {})
    if (manifest.get("status") != "completed" or not isinstance(epoch, int)
            or isinstance(epoch, bool) or epoch < 1 or selection.get("epoch") != epoch
            or best.get("epoch") != epoch or selection.get("selection_rule") != "smoothed_kl"
            or selection.get("metric") != "gaussian_kl_real_to_generated"
            or selection.get("value") != best.get("value")):
        raise ValueError("Require a completed, consistently selected best KL manifest")
    digest = sha256(checkpoint)
    if digest != manifest.get("checkpoint_sha256") or digest != best.get("checkpoint_sha256"):
        raise ValueError("Best checkpoint SHA256 differs from its manifest")
    recorded = {}
    for key, value in manifest["source_sha256"].items():
        name = key.replace("\\", "/")
        if name in recorded and recorded[name] != value:
            raise ValueError("Conflicting frozen source hash aliases: " + name)
        recorded[name] = value
    if not set(REQUIRED_SOURCES).issubset(recorded):
        raise ValueError("Missing required frozen source identity")
    checks = {}
    for name, expected in recorded.items():
        path = (workspace / name).resolve()
        if not path.is_relative_to(workspace) or not path.is_file():
            raise ValueError("Missing or nonlocal frozen source: " + name)
        actual = sha256(path)
        if actual != expected:
            raise ValueError("Frozen source SHA256 differs: " + name)
        checks[name] = actual
    identity = manifest["data_identity"]
    dropout = manifest.get("training_input_dropout")
    if dropout is not None and (
            dropout.get("options", {}).get("probability") != 0.0
            or dropout.get("active") is not False
            or any(dropout.get(k) != 0 for k in
                   ("application_calls", "eligible_values", "dropped_values", "observed_fraction"))):
        raise ValueError("This comparison requires training input dropout to be inactive")
    scaler = identity["scaler"]
    norm = scaler.get("signal_normalization", {})
    expected_range = [-1.0, 1.0] if arm == "minmax" else [0.0, 1.0]
    if (norm.get("method") != "minmax" or norm.get("min_seconds") != 0.0
            or norm.get("max_seconds") != 2.0
            or norm.get("output_range", [-1.0, 1.0]) != expected_range
            or identity.get("missing_value_policy", "preserve") != "preserve"):
        raise ValueError("Expected the recorded zero-preserving minmax arm")
    splits = identity["splits"]
    flat = [n for names in splits.values() for n in names]
    if (set(splits) != {"train", "val", "test"} or len(flat) != len(set(flat))
            or "S092.npz" not in splits["test"] or "S092" in identity["scaler_subjects"]):
        raise ValueError("S092 must be exclusively held out from training/scaler fitting")
    return dict(arm=arm, root=str(root), workspace=str(workspace),
                manifest_path=str(manifest_path), manifest=manifest, identity=identity,
                epoch=epoch, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                best_manifest_sha256=sha256(manifest_path), source_checks=checks)


def compare_arms(contexts):
    a, b = (contexts[name] for name in ARMS)
    checks = {}
    for key in ("splits", "split_fingerprint", "metadata_file_sha256", "train_npz_sha256",
                "columns", "window_epochs", "sample_rate_hz", "epoch_samples",
                "multiplier", "downsample_rate", "max_epochs", "raw_stage_mapping"):
        checks[key] = key in a["identity"] and a["identity"][key] == b["identity"].get(key)
    for key in ("continuous_columns", "continuous_mean", "continuous_std"):
        checks["scaler." + key] = a["identity"]["scaler"][key] == b["identity"]["scaler"][key]
    checks["diffusion_identity"] = a["manifest"]["diffusion_identity"] == b["manifest"]["diffusion_identity"]
    checks["no_training_input_dropout"] = all(
        c["manifest"].get("training_input_dropout", {}).get("options", {}).get("probability", 0.0) == 0.0
        for c in (a, b))
    ignored_args = {"validation_output"}
    checks["training_args"] = {
        k: v for k, v in a["manifest"]["training_args"].items() if k not in ignored_args
    } == {k: v for k, v in b["manifest"]["training_args"].items() if k not in ignored_args}
    for key in REQUIRED_SOURCES:
        if key not in ("data_utils.py", "signal_normalization.py"):
            checks["source." + key] = a["source_checks"][key] == b["source_checks"][key]
    failed = [k for k, passed in checks.items() if not passed]
    if failed:
        raise ValueError("The comparison changes more than normalization: " + ", ".join(failed))
    return checks


def frozen_imports(context):
    """Only called in a fresh worker, preventing cross-arm module caching."""
    workspace = Path(context["workspace"])
    if any(name in sys.modules for name in ("training_utils", "data_utils", "validation_sampling")):
        raise RuntimeError("Frozen imports require a fresh worker process")
    sys.path[:0] = [str(workspace), str(ROOT)]
    names = ("scripts.evaluate_denoising", "scripts.evaluate_original_sampler", "training_utils",
             "diffusion_schedule", "signal_normalization", "validation_sampling",
             "validation_metrics", "cuda_runtime")
    modules = {name: importlib.import_module(name) for name in names}
    imports = {}
    for name in names + ("data_utils", "watchsleep_loader", "TSImputers.S4Model",
                         "TSImputers.SSSDS4Imputer", "utils.util", "window_config"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(workspace):
            raise RuntimeError("Imported outside the frozen workspace: " + name)
        imports[name] = dict(path=str(path), sha256=sha256(path))
    # Import the inference adapters after their training dependencies have
    # resolved to the selected run. Record their identities for both workers.
    for name in ("night_data", "night_checkpoint"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if path != ROOT / (name + ".py"):
            raise RuntimeError("Unexpected inference adapter path: " + str(path))
        modules[name] = module
        imports[name] = dict(path=str(path), sha256=sha256(path), role="inference adapter")
    loader = modules["scripts.evaluate_denoising"]
    args, manifest, provenance = loader.load_training_run(Path(context["manifest_path"]))
    _, checkpoint = loader.resolve_checkpoint(Path(context["manifest_path"]), manifest,
                                               Path(context["checkpoint"]))
    return modules, args, dict(imported_sources=imports, configuration=provenance,
                              checkpoint_identity=checkpoint)


def _full_inference_starts(prepared, context, window_size, fs, window_stride_seconds):
    """Check the original validation anchors, then choose legal source windows for inference."""
    import numpy as np
    from data_utils import _window_valid

    protocol_path = Path(context["workspace"]).parent / "training/validation/protocol.json"
    contents = protocol_path.read_bytes()
    protocol = json.loads(contents.decode("utf-8-sig"))
    options, selection, dimensions = (protocol[key] for key in ("options", "selection", "window"))
    epoch_size = dimensions["epoch_samples"]
    identity = context["identity"]
    if (protocol.get("evaluation_filenames") != ["S092.npz"]
            or protocol.get("evaluation_protocol") != identity["evaluation_protocol"]
            or identity["train_npz_sha256"].get("S092.npz") != prepared["record"]["file_sha256"]
            or epoch_size != 750 or dimensions["sample_rate_hz"] != fs or dimensions["window_size"] != window_size
            or window_size % epoch_size or dimensions["window_epochs"] != window_size // epoch_size
            or selection["candidate_stride_samples"] != epoch_size or selection["window_size_samples"] != window_size
            or selection["available_subjects"] != 1 or selection["selected_subjects"] != 1
            or options["max_windows"] is not None or options["reference_mask_policy"] != "observed_only"
            or protocol["reference_mask_policy"] != "observed_only"):
        raise ValueError("Require the full, observed-only S092 validation protocol with matching window dimensions")
    record = prepared["record"]
    reference = prepared["reference_seconds"]
    observed = np.isfinite(reference) & (reference > 0)
    if (type(window_stride_seconds) is not int or window_stride_seconds <= 0
            or window_stride_seconds * fs % epoch_size or window_stride_seconds * fs > window_size):
        raise ValueError("Window stride must be a positive whole number of sleep epochs up to the window length")
    stride = window_stride_seconds * fs
    eligible = []
    anchors = []
    for start in range(0, len(reference) - window_size + 1, epoch_size):
        if not _window_valid(record, start // epoch_size, window_size // epoch_size):
            continue
        if np.count_nonzero(observed[start:start + window_size]) >= 2:
            eligible.append(start)
            if not anchors or start >= anchors[-1] + window_size:
                anchors.append(start)
    rows = protocol["windows"]
    if (not anchors or not isinstance(rows, list) or len(rows) != len(anchors)
            or selection["available_nonoverlapping_windows"] != len(anchors)
            or selection["selected_windows"] != len(anchors)):
        raise ValueError("Validation protocol does not cover every eligible nonoverlapping S092 window")
    for index, (row, start) in enumerate(zip(rows, anchors)):
        epoch = start // epoch_size
        stages = prepared["stage_epochs_four"][epoch:epoch + window_size // epoch_size].tolist()
        mask = observed[start:start + window_size]
        mask_meta = row["reference_mask"]
        if (type(row["start_sample"]) is not int or row["start_sample"] != start
                or row["validation_index"] != index or row["start_second"] != start / fs
                or row["subject_id"] != "S092" or row["filename"] != "S092.npz"
                or row["file_sha256"] != record["file_sha256"] or row["sleep_stages"] != stages
                or mask_meta["observed_samples"] != int(mask.sum())
                or mask_meta["missing_samples"] != int((~mask).sum())
                or mask_meta["sha256"] != hashlib.sha256(mask.tobytes()).hexdigest()):
            raise ValueError(f"Frozen validation window {index} differs from S092 source coordinates, stages or mask")
    # A full-length stride reproduces the original greedy validation even
    # when an invalid epoch shifted a later anchor off the initial grid.
    starts = anchors if stride == window_size else [s for s in eligible if s % stride == 0]
    if not starts:
        raise ValueError("No eligible S092 windows for the requested source stride")
    source_samples = len(reference) + record["discarded_tail_samples"]
    unique_samples = window_size + sum(min(window_size, s - prev) for prev, s in zip(starts, starts[1:]))
    selected_samples = len(starts) * window_size
    coverage = dict(all_windows=True, source_samples=source_samples,
                    window_stride_seconds=window_stride_seconds, window_overlap_seconds=(window_size - stride)/fs,
                    validation_anchor_windows=len(anchors), selected_sample_count=selected_samples,
                    unique_covered_samples=unique_samples, sampled_duration_seconds=selected_samples/fs,
                    trailing_samples=source_samples - (starts[-1] + window_size),
                    uncovered_samples=source_samples - unique_samples,
                    validation_protocol_path=str(protocol_path.resolve()),
                    validation_protocol_sha256=hashlib.sha256(contents).hexdigest())
    return np.asarray(starts, dtype=np.int64), coverage


def worker_preflight(context, npz_path, metadata_path, start_second, output_dir,
                     windows=1, require_all_stages=False, *, all_windows=False, window_stride_seconds=30):
    import numpy as np
    import pandas as pd
    if start_second < 0 or start_second % 30:
        raise ValueError("start-second must be a nonnegative multiple of 30")
    if type(windows) is not int or windows < 1:
        raise ValueError("windows must be a positive integer")
    modules, args, provenance = frozen_imports(context)
    identity = context["identity"]
    npz_path, metadata_path = Path(npz_path).resolve(), Path(metadata_path).resolve()
    if npz_path.name != "S092.npz" or sha256(npz_path) != S092_SHA256:
        raise ValueError("S092 NPZ differs from the previously audited record")
    if sha256(metadata_path) != identity["metadata_file_sha256"]:
        raise ValueError("Participant metadata SHA256 differs from training")
    metadata = pd.read_csv(metadata_path, dtype={"SID": str})
    if not {"SID", "AHI", "BMI", "AGE", "GENDER"}.issubset(metadata.columns):
        raise ValueError("Participant metadata columns are incomplete")
    metadata["SID"] = metadata["SID"].str.strip()
    if metadata["SID"].isna().any() or metadata["SID"].eq("").any() or metadata["SID"].duplicated().any():
        raise ValueError("Participant IDs must be nonempty and unique")
    # A frozen inference view, NOT Preprocessor construction or scaler fitting.
    view = SimpleNamespace(data_dir=npz_path.parent, metadata=metadata.set_index("SID"),
        splits=deepcopy(identity["splits"]), scaler=deepcopy(identity["scaler"]),
        scaler_subjects=list(identity["scaler_subjects"]), multiplier=identity["multiplier"],
        downsample_rate=identity["downsample_rate"], max_epochs=identity["max_epochs"])
    prepared = modules["night_data"].prepare_night(view, "S092")
    fs = int(identity["sample_rate_hz"])
    coverage, coordinates = {}, {}
    if all_windows:
        starts, coverage = _full_inference_starts(prepared, context, args.window_size, fs, window_stride_seconds)
        coordinates["start_samples"] = starts
        coordinates["window_stride_seconds"] = np.array(window_stride_seconds, dtype=np.int64)
        start_second, windows = int(starts[0]) // fs, len(starts)
        slices = [slice(start, start + args.window_size) for start in starts]
    else:
        start = start_second * fs
        stop = start + args.window_size * windows
        if stop > len(prepared["reference_seconds"]):
            raise ValueError("Requested S092 window exceeds the record")
        slices = [slice(start, stop)]
    ref = np.concatenate([prepared["reference_seconds"][span] for span in slices])
    stages = np.concatenate([prepared["stage_epochs_four"][span.start//750:span.stop//750] for span in slices])
    if require_all_stages and set(stages.tolist()) != {0, 1, 2, 3}:
        raise ValueError("Selected segment must contain W, Light, REM and Deep")
    conditions = np.concatenate([prepared["conditions"][span] for span in slices])
    mask = np.isfinite(ref) & (ref > 0)
    minimum_block = args.window_size if all_windows else 750
    if not np.isfinite(ref).all() or (ref < 0).any() or (mask.reshape(-1, minimum_block).sum(1) < 2).any():
        raise ValueError("Reference window has invalid IBI or insufficient observed points")
    center, scale = modules["signal_normalization"].signal_affine_parameters(identity["scaler"])
    expected_affine = (1.0, 1.0) if context["arm"] == "minmax" else (0.0, 2.0)
    if (center, scale) != expected_affine:
        raise ValueError("Frozen affine transform disagrees with the requested arm")
    output = Path(output_dir)
    with (output / "inputs.npz").open("xb") as stream:
        np.savez_compressed(stream, reference_ibi_seconds=ref, conditions=conditions,
            scoring_mask=mask, sleep_stages=stages, start_second=np.array(start_second),
            sample_rate_hz=np.array(fs), window_size=np.array(args.window_size), **coordinates)
    report = dict(status="checked", arm=context["arm"], epoch=context["epoch"],
        checkpoint_sha256=context["checkpoint_sha256"], best_manifest_sha256=context["best_manifest_sha256"],
        source_checks=context["source_checks"], **provenance, subject="S092", source_npz=str(npz_path),
        source_npz_sha256=S092_SHA256, metadata_sha256=sha256(metadata_path),
        subject_provenance=prepared["provenance"], scaler=identity["scaler"],
        signal_center_seconds=center, signal_scale_seconds=scale,
        start_second=start_second, window_seconds=args.window_size/fs, window_size=args.window_size,
        windows=windows, duration_seconds=coverage.get("unique_covered_samples", len(ref))/fs,
        total_samples=len(ref), **coverage,
        all_four_stages_present=set(stages.tolist()) == {0, 1, 2, 3},
        stages=stages.tolist(), observed_samples=int(mask.sum()), raw_zero_samples=int((ref == 0).sum()),
        inputs_sha256=sha256(output / "inputs.npz"), model_args=vars(args),
        diffusion_identity=context["manifest"]["diffusion_identity"],
        training_npz_reopened=False, scaler_refitted=False, sampling_performed=False,
        generation=dict(independent_windows=True, overlap_samples=(args.window_size-window_stride_seconds*fs
                            if all_windows else 0), real_ibi_prefix_samples=0,
            reference_ibi_used_as_model_input=False, stage_conditions_overridden=False, clipping_applied=False))
    if all_windows:
        report["generation"]["stitching_overlap_samples"] = 0
    save_json(output / "preflight.json", report)
    return report


def require_sampling_device(runtime):
    device = runtime.require_cuda()
    if device.type != "cuda":
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    return device


def sampling_arrays(arrays, window_size):
    """Recover independent model windows from the packed source slices."""
    import numpy as np
    ref = np.asarray(arrays["reference_ibi_seconds"])
    cond = np.asarray(arrays["conditions"])
    mask = np.asarray(arrays["scoring_mask"])
    if (ref.ndim != 1 or ref.size == 0 or ref.size % window_size
            or cond.shape != (ref.size, 9) or mask.shape != ref.shape
            or mask.dtype != np.bool_):
        raise ValueError("Segment arrays must contain complete matching model windows")
    return (ref.reshape(-1, window_size), cond.reshape(-1, window_size, 9),
            mask.reshape(-1, window_size))


def worker_sample(context, output_dir, seed, threads, *, batch_size=None):
    import numpy as np
    import torch
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("Sampling batch size must be a positive integer")
    modules, args, provenance = frozen_imports(context)
    device = require_sampling_device(modules["cuda_runtime"])
    output = Path(output_dir)
    preflight = read_json(output / "preflight.json")
    if (sha256(output / "inputs.npz") != preflight["inputs_sha256"]
            or context["checkpoint_sha256"] != preflight["checkpoint_sha256"]
            or context["best_manifest_sha256"] != preflight["best_manifest_sha256"]
            or provenance != {k: preflight[k] for k in provenance}):
        raise ValueError("Inputs, checkpoint or runtime identity changed after preflight")
    torch.set_num_threads(threads)
    np.random.seed(42)
    torch.manual_seed(42)
    model = modules["training_utils"].fetchModel(10, 1, args).to(device)
    modules["cuda_runtime"].assert_cuda_model(model)
    checkpoint_bytes = Path(context["checkpoint"]).read_bytes()
    if hashlib.sha256(checkpoint_bytes).hexdigest() != context["checkpoint_sha256"]:
        raise ValueError("Best checkpoint changed before loading")
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True)
    # B/P/w/Q are checkpointed in the eigenbasis used during training.
    # Restore that basis together with the learned parameters.
    loaded = modules["night_checkpoint"].load_checkpoint_state(model, saved)
    model.eval()
    schedule = {k: v.to(device) if torch.is_tensor(v) else v for k, v in
                modules["diffusion_schedule"].build_diffusion_config(args).items()}
    with np.load(output / "inputs.npz", allow_pickle=False) as arrays:
        ref_array, cond_array, mask_array = sampling_arrays(arrays, args.window_size)
        sample_rate = float(arrays["sample_rate_hz"].item())
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("Invalid sampling rate in paired inputs")
        if "start_samples" in arrays:
            starts = arrays["start_samples"]
            if (starts.shape != (len(ref_array),) or starts.dtype.kind not in "iu"
                    or (starts < 0).any() or (starts[1:] <= starts[:-1]).any()):
                raise ValueError("Window starts must be strictly increasing nonnegative sample indices")
        else:
            starts = int(preflight["start_second"] * sample_rate) + np.arange(len(ref_array)) * args.window_size
    windows = len(ref_array)
    if windows != preflight["windows"]:
        raise ValueError("Sampling window count changed after preflight")
    batch_size = min(batch_size or windows, windows)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Same fresh-process seed and same tensor shapes ensure paired random draws.
    rng_before = hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
    started = time.perf_counter()
    seconds_parts, normalized_parts, statistics_parts, batch_counts = [], [], [], []
    sampling = None
    denoiser_calls = 0
    nonpositive = 0
    for first in range(0, windows, batch_size):
        sl = slice(first, min(first + batch_size, windows))
        conditions = torch.as_tensor(cond_array[sl], dtype=torch.float32, device=device)
        inputs = torch.cat([torch.zeros_like(conditions[:, :, :1]), conditions], dim=2)
        normalized, batch_report = modules["validation_sampling"].sample_validation_windows(
            model, inputs, schedule, args, device)
        count = len(conditions)
        if (batch_report["windows"] != count or batch_report["mask_shape"] != [count, args.window_size]
                or normalized.shape != (count, args.window_size)):
            raise ValueError("Sampler returned a different batch shape from the paired inputs")
        common = {key: value for key, value in batch_report.items()
                  if key not in ("windows", "mask_shape", "denoiser_calls")}
        if sampling is None:
            sampling = common
        elif common != sampling:
            raise ValueError("Sampler settings or condition checks changed between batches")
        denoiser_calls += batch_report["denoiser_calls"]
        batch_counts.append(count)
        generated = normalized.to(torch.float64) * preflight["signal_scale_seconds"] + preflight["signal_center_seconds"]
        reference = torch.as_tensor(ref_array[sl], dtype=torch.float64, device=device)
        mask = torch.as_tensor(mask_array[sl], dtype=torch.bool, device=device)
        statistics_parts.append(modules["validation_metrics"].window_statistics(
            generated, reference, observed_mask=mask))
        nonpositive += int((generated <= 0).sum().item())
        seconds_parts.append(generated.detach().cpu().numpy())
        normalized_parts.append(normalized.detach().cpu().numpy())
    rng_after = hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
    sampling.update(windows=windows, mask_shape=[windows, args.window_size], denoiser_calls=denoiser_calls,
                    batch_window_counts=batch_counts)
    # Concatenate the one-scalar-per-window results before reducing. This
    # weights the last (possibly shorter) batch by its actual window count.
    statistics = {key: torch.cat([batch[key] for batch in statistics_parts])
                  for key in statistics_parts[0]}
    metrics = {key: statistics[key].mean().item() for key in
               ("reference_mean_ms", "generated_mean_ms", "reference_variance_ms2", "generated_variance_ms2",
                "mean_error_ms", "mean_absolute_error_ms", "variance_error_ms2", "variance_absolute_error_ms2",
                "gaussian_kl_real_to_generated")}
    columns = {key: values.tolist() for key, values in statistics.items()}
    per_window_metrics = [dict(window_index=i, start_sample=int(start), start_second=int(start) / sample_rate,
        **{key: values[i] for key, values in columns.items()}) for i, start in enumerate(starts)]
    with (output / "waveforms.npz").open("xb") as stream:
        np.savez_compressed(stream, generated_ibi_seconds=np.concatenate(seconds_parts).reshape(-1),
                            generated_normalized=np.concatenate(normalized_parts).reshape(-1), seed=np.array(seed),
                            window_size=np.array(args.window_size), start_samples=starts)
    report = dict(status="completed", epoch=context["epoch"], checkpoint_kind="best_kl",
        checkpoint_sha256=context["checkpoint_sha256"], best_manifest_sha256=context["best_manifest_sha256"],
        checkpoint_load=loaded, seed=seed, batch_size=batch_size, windows=windows,
        sampling=sampling, metrics=metrics, metric_aggregation="equal-weight mean over model windows",
        per_window_metrics=per_window_metrics,
        generated_nonpositive_samples=nonpositive, generated_nonpositive_fraction=nonpositive/ref_array.size,
        cuda_environment=modules["cuda_runtime"].cuda_environment(device),
        cuda_rng_before=rng_before, cuda_rng_after=rng_after, elapsed_seconds=time.perf_counter()-started,
        waveform_sha256=sha256(output / "waveforms.npz"), scaler=preflight["scaler"],
        training_executed=False, device=str(device), runtime=provenance)
    save_json(output / "summary.json", report)
    return report


def matching_inputs(output):
    import numpy as np
    arrays = []
    for arm in ARMS:
        with np.load(Path(output) / arm / "inputs.npz", allow_pickle=False) as saved:
            arrays.append({k: saved[k].copy() for k in saved.files})
    if set(arrays[0]) != set(arrays[1]):
        raise ValueError("Reference input schemas differ")
    checks = {k: np.array_equal(v, arrays[1][k]) for k, v in arrays[0].items()}
    if not all(checks.values()):
        raise ValueError("Paired S092 reference, conditions or stages differ")
    with (Path(output) / "reference.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays[0])
    return checks


def launch_worker(cli, context, phase, destination):
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker", phase,
        "--arm", context["arm"], "--run-root", context["root"], "--output-dir", str(destination),
        "--start-second", str(cli.start_second), "--windows", str(cli.windows),
        "--seed", str(cli.seed), "--threads", str(cli.threads)]
    if cli.require_all_stages:
        command.append("--require-all-stages")
    if phase == "preflight":
        identity = context["identity"]
        command += ["--npz", str(cli.npz or Path(identity["data_directory"]) / "S092.npz"),
                    "--metadata", str(cli.metadata or identity["metadata_path"])]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with (destination / (phase + ".log")).open("x", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f"{context['arm']} {phase} failed. See {destination / (phase + '.log')}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minmax-run", type=Path)
    parser.add_argument("--minmax01-run", type=Path)
    parser.add_argument("--npz", type=Path, help="Defaults to frozen data directory / S092.npz")
    parser.add_argument("--metadata", type=Path, help="Defaults to frozen participant_info.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-second", type=int, default=840)
    parser.add_argument("--windows", type=int, default=1,
                        help="Consecutive nonoverlapping model windows, generated independently")
    parser.add_argument("--require-all-stages", action="store_true")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--worker", choices=("preflight", "sample"), help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--run-root", type=Path, help=argparse.SUPPRESS)
    cli = parser.parse_args(argv)
    if cli.threads < 1 or cli.windows < 1 or cli.start_second < 0 or cli.start_second % 30 or not 0 <= cli.seed < 2**32:
        parser.error("Require positive threads, unsigned seed and nonnegative 30s-aligned start")
    if cli.worker:
        context = load_arm(cli.run_root, cli.arm)
        if cli.worker == "preflight":
            worker_preflight(context, cli.npz, cli.metadata, cli.start_second, cli.output_dir,
                             cli.windows, cli.require_all_stages)
        else:
            worker_sample(context, cli.output_dir, cli.seed, cli.threads)
        return
    if cli.minmax_run is None or cli.minmax01_run is None:
        parser.error("Both --minmax-run and --minmax01-run are required")
    output = cli.output_dir.expanduser().resolve()
    roots = {arm: getattr(cli, arm + "_run").expanduser().resolve() for arm in ARMS}
    if any(output == r or output.is_relative_to(r) for r in roots.values()):
        raise ValueError("Output must be outside the read-only training runs")
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status="checking", output_directory=str(output), arms={}, sampling_performed=False,
        training_executed=False, independent_windows=True, real_ibi_used_as_input=False,
        overlap_samples=0, seed=cli.seed, start_second=cli.start_second, windows=cli.windows,
        require_all_stages=cli.require_all_stages,
        script_sha256=sha256(__file__), check_only=cli.check_only)
    try:
        save_json(output / "summary.json", status)
        contexts = {arm: load_arm(roots[arm], arm) for arm in ARMS}
        status["shared_training_checks"] = compare_arms(contexts)
        for arm, context in contexts.items():
            destination = output / arm
            destination.mkdir()
            print(json.dumps(dict(event="best_preflight_started", arm=arm, epoch=context["epoch"])), flush=True)
            launch_worker(cli, context, "preflight", destination)
            status["arms"][arm] = dict(epoch=context["epoch"], checkpoint=context["checkpoint"],
                checkpoint_sha256=context["checkpoint_sha256"], preflight=str(destination / "preflight.json"))
        status["shared_input_checks"] = matching_inputs(output)
        if cli.check_only:
            status["status"] = "checked"
        else:
            for arm, context in contexts.items():
                print(json.dumps(dict(event="best_sampling_started", arm=arm, epoch=context["epoch"],
                                      start_second=cli.start_second, seed=cli.seed)), flush=True)
                launch_worker(cli, context, "sample", output / arm)
                status["arms"][arm].update(read_json(output / arm / "summary.json"))
                status["sampling_performed"] = True
                save_json(output / "summary.json", status)
            a, b = (status["arms"][arm] for arm in ARMS)
            status["paired_cuda_rng_states_match"] = all(a[k] == b[k] for k in ("cuda_rng_before", "cuda_rng_after"))
            if not status["paired_cuda_rng_states_match"]:
                raise RuntimeError("CUDA random streams differ between the paired samples")
            subprocess.run([sys.executable, str(ROOT / "scripts/plot_s092_best_minmax.py"),
                            "--result-dir", str(output)], check=True)
            status["status"] = "completed"
            status["plots_directory"] = str(output / "plots")
        save_json(output / "summary.json", status)
        print(json.dumps(dict(event="s092_best_comparison_finished", status=status["status"],
                              output_directory=str(output))), flush=True)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        save_json(output / "summary.json", status)
        raise


if __name__ == "__main__":
    main()
