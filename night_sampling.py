"""Bounded-group WaveStitch inference with previously generated overlap context.

The fixed denoiser, training schedule, corrected posterior standard deviation,
and within-group stitching loss are reused. Known IBI comes only from completed
generation. At each diffusion step it is projected to q(x_t | known prefix),
using the existing forward-noise draw. This replaces the upstream soft self
loss, whose x0 inversion is ill-conditioned at the low-SNR schedule endpoint.
"""
from __future__ import annotations

import contextlib
from copy import deepcopy
from functools import lru_cache
import hashlib
import io
from numbers import Integral
import time
from types import SimpleNamespace

import torch

from validation_sampling import _arguments, _validate_inputs
from window_config import EPOCH_SAMPLES


def plan_night(total_samples, window_size, *, stride=EPOCH_SAMPLES, windows_per_group=4):
    """Cover the complete signal; align a final short group to its true end."""
    for name, value in (("total_samples", total_samples), ("window_size", window_size),
                        ("stride", stride), ("windows_per_group", windows_per_group)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if total_samples < window_size:
        raise ValueError("The night must contain at least one complete trained window")
    if stride > window_size:
        raise ValueError("stride cannot exceed the trained window size")
    starts = list(range(0, total_samples - window_size + 1, stride))
    groups = []
    covered = 0

    def add(start, windows):
        nonlocal covered
        end = start + window_size + stride * (windows - 1)
        overlap = covered - start
        if overlap < 0 or end <= covered or end > total_samples:
            raise RuntimeError("Invalid generation coverage plan")
        groups.append(dict(index=len(groups), start_sample=start, end_sample=end,
                           windows=windows, known_prefix_samples=overlap,
                           append_samples=end - covered))
        covered = end

    for index in range(0, len(starts), windows_per_group):
        add(starts[index], min(windows_per_group, len(starts) - index))
    if covered < total_samples:
        # No label padding or discarded final epochs when stride does not divide
        # the available range. This last single window has a larger known prefix.
        add(total_samples - window_size, 1)
    if covered != total_samples:
        raise RuntimeError("Generation plan did not cover the complete night")
    return groups


@lru_cache(maxsize=1)
def _night_sampler():
    from scripts.evaluate_original_sampler import SOURCE, extract_corrected_pipeline

    noise, generation, _, original_provenance = extract_corrected_pipeline()
    newline = "\r\n" if "\r\n" in generation else "\n"
    edits = [
        ("project_known_prefix_each_step",
         "                    if step == diffusion_config['T'] - 1:\n"
         "                        x[~mask_expanded] = conditional_fwd[~mask_expanded]",
         "                    x[~mask_expanded] = conditional_fwd[~mask_expanded]"),
        ("replace_soft_self_loss_with_hard_projection",
         "                        loss2 = torch.sum(~mask_expanded[:, :, non_hier_cols] * ((x[:, :, non_hier_cols]-torch.sqrt(1-alpha_bar_t)*epsilon_pred)/(torch.sqrt(alpha_bar_t)) - test_batch[:, :, non_hier_cols])**2, dim=(1, 2))",
         "                        loss2 = torch.zeros_like(loss1)"),
    ]
    for name, before, after in edits:
        before, after = before.replace("\n", newline), after.replace("\n", newline)
        if generation.count(before) != 1:
            raise RuntimeError("Night sampler adaptation does not match the reviewed source: " + name)
        generation = generation.replace(before, after, 1)
    compiled = "if True:\n    if True:\n" + generation
    provenance = deepcopy(original_provenance)
    executed_hash = hashlib.sha256(compiled.encode()).hexdigest()
    provenance.update(
        night_sampler="wavestitch_grouped_hard_prefix",
        base_corrected_sampler_sha256=original_provenance["executed_sampler_sha256"],
        executed_sampler_sha256=executed_hash,
        night_sampler_sha256=executed_hash,
        night_adaptations=[name for name, _, _ in edits],
        stitching_loss_and_guidance_coefficient_unchanged=True,
        soft_self_loss_enabled=False,
        context_projection="sqrt(alpha_bar_t)*generated_prefix + sqrt(1-alpha_bar_t)*existing_pipelined_noise at every step",
        context_noise="fresh existing shared overlap noise each step; q_t marginals, not a stored forward trajectory",
        known_prefix_final_restoration="exact copy of previously generated values",
        real_ibi_used_as_context=False,
    )
    return compile(noise, str(SOURCE), "exec"), compile(compiled, str(SOURCE), "exec"), provenance


def _sample_group(model, clean_span, known_prefix_samples, diffusion_config, training_args,
                  device, *, stride):
    """Private mathematical path for CPU tests; public inference requires CUDA."""
    device = torch.device(device)
    args = _arguments(training_args)
    if not torch.is_tensor(clean_span) or clean_span.ndim != 2 or clean_span.shape[1] != 10:
        raise ValueError("A generation span must have shape [samples, 10]")
    if (isinstance(stride, bool) or not isinstance(stride, Integral)
            or not 1 <= stride <= args.window_size):
        raise ValueError("Invalid inference stride")
    if (len(clean_span) < args.window_size
            or (len(clean_span) - args.window_size) % stride):
        raise ValueError("Generation span does not form complete uniform-stride windows")
    if (isinstance(known_prefix_samples, bool) or not isinstance(known_prefix_samples, Integral)
            or not 0 <= known_prefix_samples < len(clean_span)):
        raise ValueError("Invalid known prefix length")
    training_stride = args.stride
    args.stride = int(stride)
    span = clean_span.detach().clone()
    # Caller-provided values in unknown IBI cannot leak into the sampler.
    span[known_prefix_samples:, 0] = 0.0
    windows = span.unfold(0, args.window_size, args.stride).transpose(1, 2)
    _validate_inputs(model, windows, diffusion_config, args, device)
    unknown = torch.arange(len(span), device=device) >= known_prefix_samples
    masks = unknown.unfold(0, args.window_size, args.stride)
    noise_code, generation_code, provenance = _night_sampler()
    scope = dict(torch=torch, sqrt=torch.sqrt, args=args, device=device, model=model,
                 diffusion_config=diffusion_config, test_dataset=SimpleNamespace(inputs=windows),
                 test_dataloader=[windows], mask_dataloader=[masks],
                 hierarchical_column_indices=torch.arange(1, 10, device=device),
                 non_hier_cols=torch.tensor([0], dtype=torch.long, device=device), num_ops=0, trial=0)
    try:
        exec(noise_code, scope)
        with torch.device(device), contextlib.redirect_stdout(io.StringIO()):
            exec(generation_code, scope)
        generated = scope["synth_tensor"].detach()
        if generated.shape != span.shape or generated.device != device:
            raise RuntimeError("Night sampler changed shape or device")
        if not bool(torch.isfinite(generated).all()):
            raise FloatingPointError("Night sampler produced nonfinite values; no clipping applied")
        if not torch.equal(generated[:, 1:], span[:, 1:]):
            raise RuntimeError("Night sampler changed clean conditions")
        if not torch.equal(generated[:known_prefix_samples, 0], span[:known_prefix_samples, 0]):
            raise RuntimeError("Night sampler changed previously generated prefix")
        if scope["num_ops"] != diffusion_config["T"]:
            raise RuntimeError("Night sampler did not execute the complete training schedule")
        metadata = dict(windows=len(windows), window_size=args.window_size,
            known_prefix_samples=known_prefix_samples,
            known_samples_per_window=(~masks).sum(dim=1).tolist(),
            training_stride=training_stride, inference_stride=args.stride,
            denoiser_calls=int(scope["num_ops"]), all_finite=True,
            condition_columns_bitwise_preserved=True, known_prefix_bitwise_preserved=True,
            signal_unknown_region_zero=True, real_ibi_used_as_context=False,
            clipping_applied=False, device=str(device))
        return generated, metadata
    finally:
        scope.clear()


@torch.no_grad()
def _generate_night(model, conditions, diffusion_config, training_args, device, *,
                    stride=EPOCH_SAMPLES, windows_per_group=4, progress=None):
    """Generate with bounded denoiser batches and append-only overlap handling."""
    device = torch.device(device)
    args = _arguments(training_args)
    if (not torch.is_tensor(conditions) or conditions.ndim != 2 or conditions.shape[1] != 9
            or conditions.dtype != torch.float32 or conditions.device != device):
        raise ValueError("Night conditions must be float32 [samples, 9] on the selected device")
    if not bool(torch.isfinite(conditions).all()):
        raise ValueError("Nonfinite night conditions")
    groups = plan_night(len(conditions), args.window_size, stride=stride,
                        windows_per_group=windows_per_group)
    generated = torch.empty(len(conditions), dtype=torch.float32, device=device)
    records = []
    covered = 0
    started = time.perf_counter()
    for group in groups:
        start, end, known = group["start_sample"], group["end_sample"], group["known_prefix_samples"]
        if start + known != covered:
            raise RuntimeError("Context does not match the previously generated coverage")
        span = torch.zeros(end - start, 10, dtype=torch.float32, device=device)
        span[:, 1:] = conditions[start:end]
        if known:
            span[:known, 0] = generated[start:covered]
        result, metadata = _sample_group(model, span, known, diffusion_config, args, device, stride=stride)
        # Previously emitted samples are never overwritten or averaged.
        generated[covered:end] = result[known:, 0]
        covered = end
        records.append(dict(**group, sampling=metadata))
        if progress is not None:
            progress(dict(status="night_group_completed", completed_groups=len(records),
                          total_groups=len(groups), generated_samples=covered,
                          total_samples=len(conditions), elapsed_seconds=time.perf_counter() - started))
    _, _, provenance = _night_sampler()
    return generated, dict(sampler="wavestitch_grouped_hard_prefix", groups=records,
        total_samples=covered, window_size=args.window_size, inference_stride=stride,
        windows_per_group=windows_per_group, denoiser_calls=sum(r["sampling"]["denoiser_calls"] for r in records),
        source_provenance=deepcopy(provenance), device=str(device),
        all_finite=True, condition_columns_bitwise_preserved=True,
        all_known_prefixes_bitwise_preserved=True, real_ibi_used_as_context=False,
        append_only=True, clipping_applied=False,
        limitations="Exact context preservation does not guarantee physiological realism or small boundary jumps.")


def generate_night(model, conditions, diffusion_config, training_args, device, *,
                   stride=EPOCH_SAMPLES, windows_per_group=4, progress=None):
    """CUDA-only whole-night inference; model activations span at most one group."""
    from cuda_runtime import assert_cuda_model, require_cuda

    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for whole-night generation; CPU fallback is disabled")
    selected = require_cuda()
    if device.index is None:
        device = selected
    assert_cuda_model(model)
    args = _arguments(training_args)
    if (not isinstance(args.window_size, int) or args.window_size < EPOCH_SAMPLES
            or args.window_size % EPOCH_SAMPLES or args.s4_lmax < args.window_size):
        raise ValueError("The training window must contain complete 30-second epochs and fit S4 l_max")
    if (isinstance(stride, bool) or not isinstance(stride, Integral)
            or stride < EPOCH_SAMPLES or stride % EPOCH_SAMPLES):
        raise ValueError("Inference stride must be a positive multiple of 750 samples")
    if len(conditions) % EPOCH_SAMPLES:
        raise ValueError("Night conditions must contain complete 30-second epochs")
    return _generate_night(model, conditions, diffusion_config, args, device, stride=stride,
                           windows_per_group=windows_per_group, progress=progress)
