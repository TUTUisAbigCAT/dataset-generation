"""Controlled sampling diagnostics without changing the production samplers.

Reference IBI is used only after generation for statistics. Public execution is
CUDA-only; the private implementation supports small numerical CPU tests.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from numbers import Integral

import numpy as np
import torch

from night_sampling import generate_night, _generate_night
from validation_metrics import summarize_distribution
from validation_sampling import sample_validation_windows, _sample_windows_from_original
from window_config import FS, EPOCH_SAMPLES

STAGES = ("W", "Light", "REM", "Deep")
CONTINUOUS = ("AHI", "BMI", "AGE")
VARIANTS = ("baseline",) + tuple("stage_"+name for name in STAGES) + tuple(
    f"{name}_{direction}1std" for name in CONTINUOUS for direction in ("minus", "plus")) + ("gender_swap",)


def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _state_hash(model):
    digest = hashlib.sha256()
    state = [("parameter:"+name, value) for name, value in model.named_parameters()]
    state += [("buffer:"+name, value) for name, value in model.named_buffers()]
    for name, value in sorted(state):
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        digest.update(raw.tobytes())
    return digest.hexdigest()


def _cuda_ids(device):
    return [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []


@contextmanager
def _seeded(seed, device):
    """Reset and restore the complete CPU and selected-CUDA RNG streams."""
    ids = _cuda_ids(device)
    with torch.random.fork_rng(devices=ids):
        torch.random.default_generator.manual_seed(int(seed))
        if ids:
            torch.cuda.default_generators[ids[0]].manual_seed(int(seed))
        yield


def _rng_hash(device):
    digest = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    for index in _cuda_ids(device):
        digest.update(torch.cuda.get_rng_state(index).cpu().numpy().tobytes())
    return digest.hexdigest()


def _statistics(seconds):
    values = seconds.detach().to(torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Nonfinite diagnostic signal; no clipping or replacement is allowed")
    variance, mean = torch.var_mean(values, unbiased=False)
    return dict(samples=values.numel(), mean_ms=float(mean*1000), std_ms=float(variance.sqrt()*1000),
                variance_ms2=float(variance*1e6), minimum_ms=float(values.min()*1000),
                maximum_ms=float(values.max()*1000), negative_fraction=float((values < 0).double().mean()),
                below100ms_fraction=float((values < .1).double().mean()),
                at_least1500ms_fraction=float((values >= 1.5).double().mean()))


def _pair_statistics(generated, reference, floor):
    result = {"generated_"+key: value for key, value in _statistics(generated).items()}
    result.update({"reference_"+key: value for key, value in _statistics(reference).items()})
    result["mean_error_ms"] = result["generated_mean_ms"]-result["reference_mean_ms"]
    result["variance_error_ms2"] = result["generated_variance_ms2"]-result["reference_variance_ms2"]
    result["gaussian_kl_real_to_generated"] = summarize_distribution(
        generated.reshape(1, -1), reference.reshape(1, -1),
        kl_variance_floor_ms2=floor)["aggregate"]["gaussian_kl_real_to_generated"]
    return result


def _covering_starts(total, length, stride):
    starts = list(range(0, total-length+1, stride))
    if starts[-1] != total-length:
        starts.append(total-length)
    return starts


def _noise_factory_probe(device):
    with _seeded(0, device), torch.device(device):
        sampled = torch.normal(0, 1, (12, 10))
        view = sampled.unfold(0, 8, 2).transpose(1, 2)
        transferred = view.to(device)
        return dict(context_device=str(device), actual_normal_factory_device=str(sampled.device),
                    factory_on_selected_device=sampled.device == device,
                    raw_view_stride=list(view.stride()), transferred_stride=list(transferred.stride()),
                    transfer_copies_storage=sampled.device != device or view.data_ptr() != transferred.data_ptr(),
                    rng_device_path_changed=False,
                    note="Probe preserves RNG. torch.normal may ignore torch.device; this diagnostic preserves the existing sampler's factory and transfer behavior.")


def _variant(base, name, scaler):
    changed = base.clone()
    detail = dict(variant=name)
    if name == "baseline":
        return changed, detail
    if name.startswith("stage_"):
        index = STAGES.index(name[len("stage_"):])
        changed[:, :4] = 0
        changed[:, index] = 1
        detail["change"] = "all samples set to "+STAGES[index]
    elif name == "gender_swap":
        changed[:, 7:9] = base[:, [8, 7]]
        detail["change"] = "swap F/M one-hot columns"
    else:
        field, direction = name.split("_")
        index = CONTINUOUS.index(field)
        delta = -1 if direction.startswith("minus") else 1
        changed[:, 4+index] += delta
        raw_before = base[:, 4+index].double()*scaler["continuous_std"][index]+scaler["continuous_mean"][index]
        raw_after = changed[:, 4+index].double()*scaler["continuous_std"][index]+scaler["continuous_mean"][index]
        detail.update(change=f"{field} normalized value {delta:+d}", raw_before_min=float(raw_before.min()),
                      raw_before_max=float(raw_before.max()), raw_after_min=float(raw_after.min()),
                      raw_after_max=float(raw_after.max()))
        invalid = raw_after < 0 if field == "AHI" else raw_after <= 0
        if bool(invalid.any()):
            detail.update(status="skipped", reason=f"Counterfactual {field} violates raw {'>= 0' if field == 'AHI' else '> 0'}; no clipping")
            return None, detail
    return changed, detail


def _check_inputs(model, conditions, reference, scaler, args, device, starts, repeats,
                  comparison_repeats, seed, stride, windows_per_group, floor):
    if model.training:
        raise ValueError("Sampling diagnostics require model.eval(); mode is not changed implicitly")
    if (not torch.is_tensor(conditions) or conditions.ndim != 2 or conditions.shape[1] != 9
            or conditions.dtype != torch.float32 or conditions.device != device):
        raise ValueError("conditions must be float32 [N,9] on the selected device")
    if (not torch.is_tensor(reference) or reference.shape != (len(conditions),)
            or not reference.is_floating_point() or reference.device != device):
        raise ValueError("reference_seconds must be a floating [N] tensor on the selected device")
    if not bool(torch.isfinite(conditions).all() & torch.isfinite(reference).all()):
        raise ValueError("Finite conditions and reference are required for diagnostic metrics")
    for subset in (conditions[:, :4], conditions[:, 7:9]):
        if not bool(((subset == 0) | (subset == 1)).all() & (subset.sum(1) == 1).all()):
            raise ValueError("Stage and gender conditions must be valid one-hot encodings")
    length = _integer("window_size", args.window_size, 2)
    if len(conditions) < length:
        raise ValueError("Diagnostic conditions are shorter than one trained window")
    for name, value in (("repeats", repeats), ("comparison_repeats", comparison_repeats),
                        ("stride", stride), ("windows_per_group", windows_per_group)):
        _integer(name, value)
    _integer("seed", seed, 0)
    if seed+max(repeats, comparison_repeats)-1 > np.iinfo(np.int64).max:
        raise ValueError("Diagnostic seeds must fit signed int64")
    if stride > length:
        raise ValueError("stride must not exceed window_size")
    if not isinstance(starts, (list, tuple)) or not starts:
        raise ValueError("starts must contain at least one window start")
    for start in starts:
        _integer("start", start, 0)
        if start+length > len(conditions):
            raise ValueError("A selected window exceeds the condition sequence")
    if len(set(starts)) != len(starts):
        raise ValueError("starts must be unique")
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor <= 0:
        raise ValueError("kl_variance_floor_ms2 must be finite and positive")
    for name in ("signal_mean_seconds", "signal_std_seconds"):
        if not math.isfinite(float(scaler[name])):
            raise ValueError("Invalid signal scaler")
    if scaler["signal_std_seconds"] <= 0:
        raise ValueError("Signal standard deviation must be positive")
    for key in ("continuous_mean", "continuous_std"):
        if len(scaler[key]) != 3 or not all(math.isfinite(float(v)) for v in scaler[key]):
            raise ValueError("AHI/BMI/AGE training scaler is required")
    if any(value <= 0 for value in scaler["continuous_std"]):
        raise ValueError("Continuous standard deviations must be positive")


def _run_sampling_diagnostics(model, conditions, reference_seconds, scaler, args, schedule, device, *,
                              starts, repeats=8, comparison_repeats=3, seed=42, stride=EPOCH_SAMPLES,
                              windows_per_group=4, kl_variance_floor_ms2=1.0, progress=None):
    """Private numerical implementation; CPU is reserved for small toy tests."""
    device = torch.device(device)
    _check_inputs(model, conditions, reference_seconds, scaler, args, device, starts, repeats,
                  comparison_repeats, seed, stride, windows_per_group, kl_variance_floor_ms2)
    length, total = args.window_size, len(conditions)
    reference = reference_seconds.detach().to(torch.float64)
    independent_sampler = sample_validation_windows if device.type == "cuda" else _sample_windows_from_original
    night_sampler = generate_night if device.type == "cuda" else _generate_night
    original_hash = _state_hash(model)
    eval_stride = min(EPOCH_SAMPLES, length)
    evaluation_starts = _covering_starts(total, length, eval_stride)
    independent_starts = _covering_starts(total, length, length)
    report = dict(schema="sampling_diagnostics_v1", protocol=dict(
        device=str(device), sample_rate_hz=FS, signal_array_units="seconds", population_variance=True,
        window_size=length, starts=list(starts), repeats=repeats, comparison_repeats=comparison_repeats,
        seed=seed, stride=stride, windows_per_group=windows_per_group,
        kl_direction="real_to_generated", kl_variance_floor_ms2=float(kl_variance_floor_ms2),
        kl_definition="One univariate Gaussian per window using within-window time-axis moments; equal-weight mean across evaluation windows.",
        evaluation_starts=evaluation_starts, evaluation_stride=eval_stride,
        reference_used_by_sampler=False, clipping_applied=False,
        condition_comparison_rng="Reset complete CPU and selected CUDA streams to the same seed before every variant; not just initial x_T.",
        continuation_comparison_rng="Same seed per branch; different shapes/call counts mean reverse-noise trajectories are not paired.",
        interpretation="Sampling sensitivity and distribution diagnostics, not a requirement to reconstruct the paired reference waveform."),
        noise_factory=_noise_factory_probe(device), model_state=dict(before_sha256=original_hash,
            hash_scope="All named parameters and registered buffers, including nonpersistent buffers."),
        reference=_statistics(reference), stages={})
    arrays = dict(reference_seconds=reference.cpu().numpy(), conditions=conditions.detach().cpu().numpy(),
                  selected_start_samples=np.asarray(starts, dtype=np.int64),
                  repeat_seeds=np.arange(seed, seed+repeats, dtype=np.int64),
                  comparison_seeds=np.arange(seed, seed+comparison_repeats, dtype=np.int64),
                  variant_names=np.asarray(VARIANTS), evaluation_start_samples=np.asarray(evaluation_starts, dtype=np.int64),
                  repeat_generated_seconds=np.empty((len(starts), repeats, length), dtype=np.float64),
                  condition_generated_seconds=np.full((len(starts), len(VARIANTS), length), np.nan),
                  condition_variant_valid=np.zeros((len(starts), len(VARIANTS)), dtype=bool),
                  independent_night_seconds=np.empty((comparison_repeats, total), dtype=np.float64),
                  continued_night_seconds=np.empty((comparison_repeats, total), dtype=np.float64))

    def emit(event, **fields):
        if progress:
            progress(dict(event=event, **fields))

    def stage_done(name, result):
        if _state_hash(model) != original_hash:
            raise RuntimeError("Model parameters or buffers changed during diagnostics: "+name)
        report["stages"][name] = result
        emit("stage_summary", stage=name, summary=result)

    def independent(condition):
        clean = torch.zeros(1, length, 10, dtype=torch.float32, device=device)
        clean[0, :, 1:] = condition
        standardized, _ = independent_sampler(model, clean, schedule, args, device)
        return standardized[0].double()*scaler["signal_std_seconds"]+scaler["signal_mean_seconds"]

    def seeded_independent(condition, draw_seed):
        with _seeded(draw_seed, device):
            result = independent(condition)
            ending_rng = _rng_hash(device)
        return result, ending_rng

    rows, between = [], []
    for index, start in enumerate(starts):
        means, stds, variances = [], [], []
        for repeat in range(repeats):
            generated, _ = seeded_independent(conditions[start:start+length], seed+repeat)
            metrics = _pair_statistics(generated, reference[start:start+length], kl_variance_floor_ms2)
            rows.append(dict(window_index=index, start_sample=start, repeat=repeat, seed=seed+repeat, **metrics))
            means.append(metrics["generated_mean_ms"])
            stds.append(metrics["generated_std_ms"])
            variances.append(metrics["generated_variance_ms2"])
            arrays["repeat_generated_seconds"][index, repeat] = generated.detach().cpu().numpy()
            emit("diagnostic_progress", stage="same_conditions", window_index=index, repeat=repeat)
        between.append(dict(window_index=index, start_sample=start, draws=repeats,
                            mean_of_draw_means_ms=float(np.mean(means)), std_of_draw_means_ms=float(np.std(means)),
                            min_draw_mean_ms=min(means), max_draw_mean_ms=max(means),
                            mean_of_draw_stds_ms=float(np.mean(stds)), std_of_draw_stds_ms=float(np.std(stds)),
                            mean_of_draw_variances_ms2=float(np.mean(variances)),
                            std_of_draw_variances_ms2=float(np.std(variances))))
    stage_done("same_conditions", dict(rows=rows, between_draw_rows=between))

    rows, determinism, epsilon_rows = [], [], []
    probe_steps = sorted(set((0, schedule["T"]//2, schedule["T"]-1)))
    for index, start in enumerate(starts):
        base = conditions[start:start+length]
        baseline, baseline_rng = seeded_independent(base, seed)
        repeated, repeated_rng = seeded_independent(base, seed)
        equal = bool(torch.equal(baseline, repeated)) and baseline_rng == repeated_rng
        determinism.append(dict(window_index=index, start_sample=start, seed=seed,
                                baseline_repeat_bitwise_equal=bool(torch.equal(baseline, repeated)),
                                baseline_repeat_end_rng_equal=baseline_rng == repeated_rng))
        if not equal:
            raise RuntimeError("Baseline rerun is not bitwise deterministic under restored full RNG state")
        with _seeded(seed, device):
            epsilon_noise = torch.randn(1, length, device=device)
        baseline_eps = {}
        for step in probe_steps:
            x = torch.zeros(1, length, 10, device=device)
            x[:, :, 0] = (1-schedule["alpha_bars"][step]).sqrt()*epsilon_noise
            x[:, :, 1:] = base
            with _seeded(seed, device), torch.no_grad():
                baseline_eps[step] = model(x, torch.full((1, 1), step, device=device)).detach().clone()
        for variant_index, name in enumerate(VARIANTS):
            changed, detail = _variant(base, name, scaler)
            row = dict(window_index=index, start_sample=start, seed=seed, **detail)
            if changed is None:
                rows.append(row)
                continue
            generated, ending_rng = (baseline, baseline_rng) if name == "baseline" else seeded_independent(changed, seed)
            if ending_rng != baseline_rng:
                raise RuntimeError("Condition variant consumed a different RNG trajectory")
            diff = generated-baseline
            row.update(status="completed", end_rng_matches_baseline=True,
                       generation_rmse_vs_baseline_ms=float(diff.square().mean().sqrt()*1000),
                       generation_mean_change_ms=float(diff.mean()*1000),
                       generation_variance_change_ms2=float((generated.var(unbiased=False)-baseline.var(unbiased=False))*1e6),
                       **_pair_statistics(generated, reference[start:start+length], kl_variance_floor_ms2))
            rows.append(row)
            arrays["condition_generated_seconds"][index, variant_index] = generated.detach().cpu().numpy()
            arrays["condition_variant_valid"][index, variant_index] = True
            for step in probe_steps:
                x = torch.zeros(1, length, 10, device=device)
                x[:, :, 0] = (1-schedule["alpha_bars"][step]).sqrt()*epsilon_noise
                x[:, :, 1:] = changed
                with _seeded(seed, device), torch.no_grad():
                    epsilon = model(x, torch.full((1, 1), step, device=device))
                delta = epsilon.double()-baseline_eps[step].double()
                epsilon_rows.append(dict(window_index=index, start_sample=start, variant=name, timestep=step,
                                         epsilon_rmse_vs_baseline=float(delta.square().mean().sqrt()),
                                         epsilon_max_difference=float(delta.abs().max())))
            emit("diagnostic_progress", stage="changed_conditions", window_index=index, variant=name)
    stage_done("changed_conditions", dict(rows=rows, determinism_rows=determinism, epsilon_rows=epsilon_rows,
        epsilon_probe="Same x_t=sqrt(1-alpha_bar_t)*fixed_noise and raw timestep for every variant; no reference IBI input; epsilon metrics are dimensionless."))

    rows, comparison_rows, window_rows = [], [], []
    for repeat in range(comparison_repeats):
        draw_seed = seed+repeat
        with _seeded(draw_seed, device):
            independent_signal = torch.empty(total, dtype=torch.float64, device=device)
            covered = 0
            for start in independent_starts:
                generated = independent(conditions[start:start+length])
                independent_signal[covered:start+length] = generated[covered-start:]
                covered = start+length
                emit("diagnostic_progress", stage="independent_vs_continued", method="independent", repeat=repeat, generated_samples=covered)
        with _seeded(draw_seed, device):
            standardized, _ = night_sampler(model, conditions, schedule, args, device, stride=stride,
                windows_per_group=windows_per_group,
                progress=lambda event: emit("diagnostic_progress", stage="independent_vs_continued", method="continued", repeat=repeat, sampler_progress=event))
            continued_signal = standardized.double()*scaler["signal_std_seconds"]+scaler["signal_mean_seconds"]
        for method, signal in (("independent", independent_signal), ("continued", continued_signal)):
            kl_values = []
            for start in evaluation_starts:
                metrics = _pair_statistics(signal[start:start+length], reference[start:start+length], kl_variance_floor_ms2)
                kl_values.append(metrics["gaussian_kl_real_to_generated"])
                window_rows.append(dict(method=method, repeat=repeat, seed=draw_seed, start_sample=start, **metrics))
            rows.append(dict(method=method, repeat=repeat, seed=draw_seed, **_statistics(signal),
                             mean_window_gaussian_kl_real_to_generated=float(np.mean(kl_values)), evaluation_windows=len(kl_values)))
            arrays["independent_night_seconds" if method == "independent" else "continued_night_seconds"][repeat] = signal.detach().cpu().numpy()
        comparison_rows.append(dict(repeat=repeat, seed=draw_seed,
            continued_minus_independent_mean_ms=float((continued_signal.mean()-independent_signal.mean())*1000),
            continued_minus_independent_variance_ms2=float((continued_signal.var(unbiased=False)-independent_signal.var(unbiased=False))*1e6),
            waveform_difference_rmse_ms=float((continued_signal-independent_signal).square().mean().sqrt()*1000),
            noise_trajectories_paired=False))
    stage_done("independent_vs_continued", dict(rows=rows, comparison_rows=comparison_rows, window_rows=window_rows,
        coverage=dict(total_samples=total, independent_window_starts=independent_starts,
                      final_independent_window_overlap_samples=(independent_starts[-2]+length-independent_starts[-1]) if len(independent_starts)>1 else 0,
                      tail_policy="Independent full windows; final end-aligned window contributes only uncovered suffix. No padding, averaging, clipping or discarded samples.",
                      identical_evaluation_windows=True)))
    report["model_state"].update(after_sha256=_state_hash(model), unchanged=True)
    return report, arrays


def run_sampling_diagnostics(model, conditions, reference_seconds, scaler, args, schedule, device, *,
                             starts, repeats=8, comparison_repeats=3, seed=42, stride=EPOCH_SAMPLES,
                             windows_per_group=4, kl_variance_floor_ms2=1.0, progress=None):
    """Run three controlled diagnostic stages; CUDA is mandatory publicly."""
    from cuda_runtime import assert_cuda_model, require_cuda

    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for sampling diagnostics; CPU fallback is disabled")
    selected = require_cuda()
    if device.index is None:
        device = selected
    assert_cuda_model(model)
    if (isinstance(args.window_size, bool) or not isinstance(args.window_size, int)
            or args.window_size < EPOCH_SAMPLES or args.window_size % EPOCH_SAMPLES
            or args.s4_lmax < args.window_size):
        raise ValueError("Production windows must contain complete 750-point epochs and fit the S4 kernel")
    if isinstance(stride, bool) or not isinstance(stride, Integral) or stride < EPOCH_SAMPLES or stride % EPOCH_SAMPLES:
        raise ValueError("Production diagnostic stride must be a positive multiple of 750")
    if len(conditions) % EPOCH_SAMPLES or any(start % EPOCH_SAMPLES for start in starts):
        raise ValueError("Production diagnostic sequence and selected starts must align to 750-sample epochs")
    return _run_sampling_diagnostics(model, conditions, reference_seconds, scaler, args, schedule, device,
        starts=starts, repeats=repeats, comparison_repeats=comparison_repeats, seed=seed, stride=stride,
        windows_per_group=windows_per_group, kl_variance_floor_ms2=kl_variance_floor_ms2, progress=progress)
