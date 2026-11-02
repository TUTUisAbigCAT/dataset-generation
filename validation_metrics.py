"""Mean paired-window Gaussian KL for validation IBI sequences.

Each row is one independent window. Its mean and population variance are
estimated over time (dim=1), then paired with the corresponding reference row.
The aggregate is the mean of these univariate Gaussian KL values over windows.
All tensor arithmetic stays on the input device in float64. Production callers
require CUDA; CPU supports unit tests.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import random

import numpy as np
import torch


def _prepare_pair(generated_seconds, reference_seconds, observed_mask=None):
    for name, value in (("generated", generated_seconds), ("reference", reference_seconds)):
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise TypeError(f"{name} must be a real floating-point tensor")
        if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 2:
            raise ValueError(f"{name} must have shape [windows >= 1, samples >= 2]")
    if generated_seconds.shape != reference_seconds.shape:
        raise ValueError("Generated and reference shapes must match")
    if generated_seconds.device != reference_seconds.device:
        raise ValueError("Generated and reference tensors must use the same device")
    if observed_mask is not None:
        if not torch.is_tensor(observed_mask) or observed_mask.dtype != torch.bool:
            raise TypeError("observed_mask must be a torch.bool tensor")
        if observed_mask.shape != generated_seconds.shape:
            raise ValueError("observed_mask must match the [windows, samples] shape")
        if observed_mask.device != generated_seconds.device:
            raise ValueError("observed_mask and IBI tensors must use the same device")
        if not bool((observed_mask.sum(dim=1) >= 2).all()):
            raise ValueError("Every window must contain at least two observed samples")
    generated = generated_seconds.detach().to(dtype=torch.float64) * 1000.0
    reference = reference_seconds.detach().to(dtype=torch.float64) * 1000.0
    if observed_mask is None:
        if not bool(torch.isfinite(generated).all() & torch.isfinite(reference).all()):
            raise ValueError("Generated and reference IBI must be finite in milliseconds")
    else:
        # A reference mask never excuses a failed generated trajectory, even
        # outside scored positions. Missing reference values may be nonfinite.
        if not bool(torch.isfinite(generated).all()):
            raise ValueError("Generated IBI must be finite in milliseconds at every position")
        if not bool(torch.isfinite(reference[observed_mask]).all()):
            raise ValueError("Reference IBI must be finite in milliseconds at observed positions")
    return generated, reference


def _masked_var_mean(values, observed_mask):
    """Population moments at a fixed reference-selected set of time samples."""
    count = observed_mask.sum(dim=1).to(dtype=torch.float64)
    # Do not multiply unknown NaN/Inf reference values by zero: 0*NaN is NaN.
    selected = torch.where(observed_mask, values, 0.)
    mean = selected.sum(dim=1) / count
    centered = torch.where(observed_mask, selected - mean[:, None], 0.)
    variance = centered.square().sum(dim=1) / count
    return variance, mean


def _validated_variance_floor(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError("kl_variance_floor_ms2 must be a finite positive number")
    return float(value)


@torch.no_grad()
def window_statistics(generated_seconds, reference_seconds, *, kl_variance_floor_ms2=1.0,
                      observed_mask=None):
    """Return paired within-window moments and KL as device-resident [B] tensors.

    Inputs have shape [B, L] in seconds. Means and signed differences use ms;
    raw population variances and their differences use ms^2. Differences are
    generated minus reference. The variance floor affects KL only, never the
    returned raw variances. Each KL component already includes its factor 0.5
    and their sum is KL(real || generated), in nats per window.

    A supplied reference-defined bool mask selects the same positions for both
    fits, with at least two observations per row. Unselected reference values
    may be nonfinite, but every generated value must be finite. Negative
    generated values are retained. Without a mask, all positions are observed.
    The returned observed_counts is int64; every other tensor is float64.
    All outputs stay on the input device without gradients or RNG changes.
    """
    floor = _validated_variance_floor(kl_variance_floor_ms2)
    generated, reference = _prepare_pair(generated_seconds, reference_seconds, observed_mask)
    if observed_mask is None:
        # Keep the legacy unmasked reduction and arithmetic order unchanged.
        gen_var, gen_mean = torch.var_mean(generated, dim=1, unbiased=False)
        ref_var, ref_mean = torch.var_mean(reference, dim=1, unbiased=False)
        observed_counts = torch.full((len(generated),), generated.shape[1],
                                     dtype=torch.int64, device=generated.device)
    else:
        gen_var, gen_mean = _masked_var_mean(generated, observed_mask)
        ref_var, ref_mean = _masked_var_mean(reference, observed_mask)
        observed_counts = observed_mask.sum(dim=1)
    if not bool(torch.isfinite(gen_var).all() & torch.isfinite(ref_var).all()
                & torch.isfinite(gen_mean).all() & torch.isfinite(ref_mean).all()):
        raise FloatingPointError("Nonfinite within-window moments")
    # One KL(real || generated) per corresponding pair of windows. The
    # lower bound regularizes the fitted variances, not the waveforms.
    kl_gen_var = gen_var.clamp_min(floor)
    kl_ref_var = ref_var.clamp_min(floor)
    relative_change = (kl_ref_var - kl_gen_var) / kl_gen_var
    # For small d, d - log1p(d) loses significant digits. Its expansion
    # through d^5 has sub-float64 relative truncation error at |d| <= 1e-4.
    # Clamp unused polynomial arguments because torch evaluates both branches.
    small = relative_change.clamp(-1e-4, 1e-4)
    near_equal = small.square() * (0.5 + small * (-1.0 / 3.0 + small * (0.25 - small / 5.0)))
    log_ratio = kl_ref_var.log() - kl_gen_var.log()
    general = kl_ref_var / kl_gen_var - 1.0 - log_ratio
    variance_term = torch.where(relative_change.abs() <= 1e-4, near_equal, general).clamp_min(0.0)
    mean_term = ((ref_mean - gen_mean) / kl_gen_var.sqrt()).square()
    kl_variance_term = 0.5 * variance_term
    kl_mean_term = 0.5 * mean_term
    window_kl = kl_variance_term + kl_mean_term
    if not bool(torch.isfinite(window_kl).all()):
        raise FloatingPointError("Nonfinite paired-window Gaussian KL divergence")
    mean_error = gen_mean - ref_mean
    variance_error = gen_var - ref_var
    return dict(
        generated_mean_ms=gen_mean, reference_mean_ms=ref_mean,
        generated_variance_ms2=gen_var, reference_variance_ms2=ref_var,
        mean_error_ms=mean_error, mean_absolute_error_ms=mean_error.abs(),
        variance_error_ms2=variance_error, variance_absolute_error_ms2=variance_error.abs(),
        observed_counts=observed_counts, gaussian_kl_real_to_generated=window_kl,
        kl_mean_term=kl_mean_term, kl_variance_term=kl_variance_term)


class DistributionAccumulator:
    """Accumulate paired-window KL with equal weight per validation window.

    Compute moments within each window, never across windows. Accumulating a
    scalar KL sum and a window count correctly weights unequal batches. No
    waveform or moment vector is retained after update. This compares each
    pair's amplitude moments, not its temporal ordering or joint distribution.
    """
    def __init__(self, *, kl_variance_floor_ms2=1.0):
        self.kl_variance_floor_ms2 = _validated_variance_floor(kl_variance_floor_ms2)
        self.count = 0
        self.window_size = self.device = None
        self.kl_sum = None

    @torch.no_grad()
    def update(self, generated_seconds, reference_seconds, observed_mask=None):
        """Accumulate this batch and return its statistics without retaining them."""
        statistics = window_statistics(
            generated_seconds, reference_seconds, observed_mask=observed_mask,
            kl_variance_floor_ms2=self.kl_variance_floor_ms2)
        if self.count and (generated_seconds.shape[1] != self.window_size
                           or generated_seconds.device != self.device):
            raise ValueError("Batches must have the same feature dimension and device")
        batch_count = len(generated_seconds)
        window_kl = statistics["gaussian_kl_real_to_generated"]
        batch_kl_sum = window_kl.sum()
        total_kl = batch_kl_sum if self.count == 0 else self.kl_sum + batch_kl_sum
        if not bool(torch.isfinite(window_kl).all() & torch.isfinite(total_kl)):
            raise FloatingPointError("Nonfinite paired-window Gaussian KL divergence")
        self.kl_sum = total_kl
        self.window_size, self.device = generated_seconds.shape[1], generated_seconds.device
        self.count += batch_count
        return statistics

    @torch.no_grad()
    def finalize(self):
        if self.count == 0:
            raise ValueError("Cannot summarize empty validation data")
        mean_window_kl = self.kl_sum / self.count
        return {"aggregate": {"gaussian_kl_real_to_generated": float(mean_window_kl)}}


def summarize_distribution(generated_seconds, reference_seconds, *, kl_variance_floor_ms2=1.0,
                           observed_mask=None):
    """Mean KL of same-index window pairs in [windows, time samples] tensors.

    Inputs are seconds. Each window's time samples determine its mean and
    population variance (correction=0), even when there is just one window.
    KL(real || generated) is computed for each pair of fitted univariate
    Gaussians, then averaged equally over windows. Stored zeros and negative
    generated values are not clipped; the variance floor is in ms squared.
    An optional bool mask selects the same reference-observed time positions
    for both fits. Each row needs at least two selected samples. Missing
    reference values outside that mask may be nonfinite; generated values must
    always be finite everywhere. Masking never depends on generated amplitudes.
    """
    accumulator = DistributionAccumulator(kl_variance_floor_ms2=kl_variance_floor_ms2)
    accumulator.update(generated_seconds, reference_seconds, observed_mask=observed_mask)
    return accumulator.finalize()


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
