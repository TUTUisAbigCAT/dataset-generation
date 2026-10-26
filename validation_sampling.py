"""Independent validation windows through the corrected WaveStitch sampler.

The adapter supplies clean conditions and an entirely unknown IBI channel to
the verified upstream block with the posterior-variance square-root fix.
A full-window inference stride makes
the overlap slices empty, so unrelated validation windows are never stitched.
Training arguments, schedules, model parameters and guidance stay unchanged.
The caller owns the separate evaluation model and RNG isolation.
"""
from __future__ import annotations

from argparse import Namespace
import contextlib
from copy import deepcopy
from functools import lru_cache
import io
from types import SimpleNamespace

import torch

from window_config import EPOCH_SAMPLES
from diffusion_schedule import validate_diffusion_config, diffusion_identity


@lru_cache(maxsize=1)
def _corrected_sampler():
    """Verify the upstream snapshot and compile the corrected runtime excerpt."""
    from scripts.evaluate_original_sampler import SOURCE, extract_corrected_pipeline

    noise, _, generation, provenance = extract_corrected_pipeline()
    return (compile(noise, str(SOURCE), "exec"),
            compile(generation, str(SOURCE), "exec"), provenance)


def _arguments(training_args):
    values = training_args if isinstance(training_args, dict) else vars(training_args)
    return Namespace(**deepcopy(values))


def _validate_inputs(model, clean_windows, diffusion_config, training_args, device):
    if not torch.is_tensor(clean_windows) or clean_windows.ndim != 3:
        raise ValueError("Validation windows must have shape [batch, length, 10]")
    args = _arguments(training_args)
    if (clean_windows.shape[0] < 1 or clean_windows.shape[1] != args.window_size
            or clean_windows.shape[2] != 10):
        raise ValueError("Validation requires nonempty, complete 10-channel windows")
    if clean_windows.dtype != torch.float32 or clean_windows.device != device:
        raise ValueError("Validation input must be float32 on the selected device")
    if not bool(torch.isfinite(clean_windows).all()):
        raise ValueError("Validation input contains nonfinite values")
    if model.training:
        raise ValueError("Use a separate model in eval mode for validation")
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    if any(value.device != device for _, value in tensors):
        raise ValueError("All validation model parameters and buffers must use the selected device")
    steps = diffusion_config.get("T")
    if not isinstance(steps, int) or steps < 1 or steps != args.timesteps:
        raise ValueError("Validation must use the complete training diffusion schedule")
    for name in ("betas", "alphas", "alpha_bars"):
        value = diffusion_config.get(name)
        if (not torch.is_tensor(value) or value.shape != (steps, 1)
                or value.dtype != torch.float32 or value.device != device
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Invalid training diffusion schedule tensor: {name}")
    # Production arguments always carry both beta endpoints. Tiny test-only
    # denoisers may supply a complete schedule directly without CLI metadata.
    if hasattr(args, "beta_0") and hasattr(args, "beta_T"):
        validate_diffusion_config(args, diffusion_config)
    if torch.is_inference_mode_enabled():
        raise RuntimeError("The original sampler requires autograd for its unchanged guidance expression")
    return args


def _sample_windows_from_original(model, clean_windows, diffusion_config, training_args, device):
    """Shared mathematical implementation; CPU use is restricted to unit tests.

    Production callers must use sample_validation_windows, whose CUDA contract
    cannot be bypassed by supplying a CPU device or by lacking CUDA hardware.
    """
    device = torch.device(device)
    args = _validate_inputs(model, clean_windows, diffusion_config, training_args, device)
    training_stride = args.stride
    args.stride = args.window_size
    windows = clean_windows.detach().clone()
    windows[:, :, 0] = 0.0
    masks = torch.ones(windows.shape[:2], dtype=torch.bool, device=device)
    noise_code, generation_code, provenance = _corrected_sampler()
    scope = dict(
        torch=torch, sqrt=torch.sqrt, args=args, device=device, model=model,
        diffusion_config=diffusion_config,
        test_dataset=SimpleNamespace(inputs=windows),
        test_dataloader=[windows], mask_dataloader=[masks],
        hierarchical_column_indices=torch.arange(1, 10, device=device),
        non_hier_cols=torch.tensor([0], device=device), num_ops=0, trial=0,
    )
    try:
        exec(noise_code, scope)
        # Preserve the exact upstream factory calls and random draw order. Its
        # per-step console output is suppressed; the caller reports epoch metrics.
        with torch.device(device), contextlib.redirect_stdout(io.StringIO()):
            exec(generation_code, scope)
        generated = scope["synth_tensor"].detach().reshape_as(windows)
        conditions_preserved = bool(torch.equal(generated[:, :, 1:], clean_windows[:, :, 1:]))
        finite = bool(torch.isfinite(generated).all())
        zero_guidance = bool(torch.count_nonzero(scope["grad"]) == 0)
        if generated.device != device or not conditions_preserved:
            raise RuntimeError("Original validation sampler changed device or clean conditions")
        if not finite:
            raise RuntimeError("Original validation sampler produced nonfinite values; no clipping is permitted")
        if not zero_guidance:
            raise RuntimeError("Independent, all-unknown windows unexpectedly produced guidance")
        if scope["num_ops"] != diffusion_config["T"]:
            raise RuntimeError("Original validation sampler did not execute every training diffusion step")
        metadata = {
            "source_provenance": deepcopy(provenance),
            "sampler": "upstream_pipeline_posterior_std_corrected",
            "diffusion_steps": diffusion_config["T"],
            "denoiser_calls": int(scope["num_ops"]),
            "windows": len(windows),
            "window_size": args.window_size,
            "mask_shape": list(masks.shape),
            "training_stride": training_stride,
            "inference_stride": args.stride,
            "independent_windows": True,
            "signal_input_zero": True,
            "signal_all_unknown": True,
            "condition_columns_bitwise_preserved": conditions_preserved,
            "last_step_guidance_exactly_zero": zero_guidance,
            "all_finite": finite,
            "clipping_applied": False,
            "device": str(device),
        }
        if hasattr(args, "beta_0") and hasattr(args, "beta_T"):
            metadata["diffusion_identity"] = diffusion_identity(args, diffusion_config)
        return generated[:, :, 0], metadata
    finally:
        # The exec-defined function refers to this scope; break that cycle so
        # intermediate GPU tensors cannot survive until Python cyclic GC.
        scope.clear()


def sample_validation_windows(model, clean_windows, diffusion_config, training_args, device):
    """Return (standardized IBI [B, window_epochs * 750], provenance) using CUDA only.

    clean_windows contains the real validation IBI and nine clean condition
    columns. Its IBI values are removed before the corrected sampler sees the
    input. Supply the *training* diffusion_config without rescheduling it.
    """
    from cuda_runtime import assert_cuda_model, require_cuda

    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for validation sampling; CPU fallback is disabled")
    selected = require_cuda()
    if device.index is None:
        device = selected
    assert_cuda_model(model)
    args = _arguments(training_args)
    window_size = args.window_size
    if (isinstance(window_size, bool) or not isinstance(window_size, int)
            or window_size < EPOCH_SAMPLES or window_size % EPOCH_SAMPLES):
        raise ValueError("Production validation requires a positive whole number of 750-point sleep epochs")
    if args.s4_lmax < window_size:
        raise ValueError("Production validation window_size must fit within the S4 kernel length")
    return _sample_windows_from_original(model, clean_windows, diffusion_config, training_args, device)
