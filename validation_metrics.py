"""Mean and population-variance checks on generated IBI in physical units.

These are distribution diagnostics, not FID or paired waveform errors. Inputs
are windows in seconds. Arithmetic remains on the input device in float64;
only the final scalars are converted for JSON logging. Production callers must
require CUDA before calling these helpers. CPU inputs support unit tests only.
"""
from __future__ import annotations

from contextlib import contextmanager
import random

import numpy as np
import torch


def summarize_distribution(generated_seconds, reference_seconds):
    """Compare equal-shaped ``[windows, samples]`` IBI tensors without clipping.

    Variances use the population definition (``unbiased=False``). Aggregate
    statistics cover all samples, while the two window-error summaries average
    absolute differences of each window's mean or variance. A variance ratio
    with a zero reference variance is undefined and is serialized as ``None``.
    """
    for name, value in (("generated", generated_seconds), ("reference", reference_seconds)):
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise TypeError(f"{name} must be a real floating-point tensor")
        if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 2:
            raise ValueError(f"{name} must have shape [windows >= 1, samples >= 2]")
    if generated_seconds.shape != reference_seconds.shape:
        raise ValueError("Generated and reference shapes must match")
    if generated_seconds.device != reference_seconds.device:
        raise ValueError("Generated and reference tensors must be on the same device")

    with torch.no_grad():
        generated = generated_seconds.detach().to(dtype=torch.float64) * 1000.0
        reference = reference_seconds.detach().to(dtype=torch.float64) * 1000.0
        if not bool(torch.isfinite(generated).all() & torch.isfinite(reference).all()):
            raise ValueError("Generated and reference IBI must be finite in milliseconds")
        generated_variance, generated_mean = torch.var_mean(generated, dim=1, unbiased=False)
        reference_variance, reference_mean = torch.var_mean(reference, dim=1, unbiased=False)
        global_generated_variance, global_generated_mean = torch.var_mean(generated, unbiased=False)
        global_reference_variance, global_reference_mean = torch.var_mean(reference, unbiased=False)

        def row(gen_mean, ref_mean, gen_var, ref_var):
            values = torch.stack((gen_mean, ref_mean, gen_mean - ref_mean,
                                  (gen_mean - ref_mean).abs(), gen_var, ref_var,
                                  gen_var - ref_var, (gen_var - ref_var).abs()))
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError("Nonfinite IBI distribution statistic")
            result = dict(zip(("generated_mean_ms", "reference_mean_ms", "mean_error_ms",
                               "mean_absolute_error_ms", "generated_variance_ms2",
                               "reference_variance_ms2", "variance_error_ms2",
                               "variance_absolute_error_ms2"), values.tolist()))
            if bool(ref_var == 0):
                result["variance_ratio"] = None
            else:
                ratio = gen_var / ref_var
                if not bool(torch.isfinite(ratio)):
                    raise FloatingPointError("Nonfinite IBI variance ratio")
                result["variance_ratio"] = float(ratio)
            return result

        aggregate = row(global_generated_mean, global_reference_mean,
                        global_generated_variance, global_reference_variance)
        window_errors = torch.stack(((generated_mean - reference_mean).abs().mean(),
                                     (generated_variance - reference_variance).abs().mean()))
        if not bool(torch.isfinite(window_errors).all()):
            raise FloatingPointError("Nonfinite mean per-window distribution error")
        aggregate["mean_window_absolute_error_ms"] = float(window_errors[0])
        aggregate["variance_window_absolute_error_ms2"] = float(window_errors[1])
        per_window = [dict(window_index=index, **row(generated_mean[index], reference_mean[index],
                                                    generated_variance[index], reference_variance[index]))
                      for index in range(generated.shape[0])]
    return {"aggregate": aggregate, "per_window": per_window}


@contextmanager
def preserve_random_state(device):
    """Restore Python, NumPy, CPU torch and all visible CUDA generators on exit.

    Validation may choose its own fixed seed inside the context without changing
    subsequent training noise or data-loader shuffling. Model state/mode is not
    changed here. CUDA contexts preserve every visible generator because
    ``torch.manual_seed`` seeds all of them. CPU test contexts do not initialize
    CUDA. Passing CUDA never falls back to CPU when CUDA is unavailable.
    """
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("Expected a CPU test device or the production CUDA device")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA random-state preservation requires an available CUDA device")
    if selected.type == "cuda" and selected.index is None:
        selected = torch.device("cuda", torch.cuda.current_device())
    if selected.type == "cuda" and selected.index >= torch.cuda.device_count():
        raise ValueError("Requested CUDA device is not visible")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if selected.type == "cuda" else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
