"""Diagonal Gaussian KL on all validation sequence samples.

One row is a sequence sample; dim=0 estimates each time coordinate's mean and
population variance across windows. All tensor arithmetic stays on the input
device in float64. Production callers require CUDA; CPU supports unit tests.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import random

import numpy as np
import torch


def _prepare_pair(generated_seconds, reference_seconds):
    for name, value in (("generated", generated_seconds), ("reference", reference_seconds)):
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise TypeError(f"{name} must be a real floating-point tensor")
        if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 2:
            raise ValueError(f"{name} must have shape [windows >= 1, samples >= 2]")
    if generated_seconds.shape != reference_seconds.shape:
        raise ValueError("Generated and reference shapes must match")
    if generated_seconds.device != reference_seconds.device:
        raise ValueError("Generated and reference tensors must use the same device")
    generated = generated_seconds.detach().to(dtype=torch.float64) * 1000.0
    reference = reference_seconds.detach().to(dtype=torch.float64) * 1000.0
    if not bool(torch.isfinite(generated).all() & torch.isfinite(reference).all()):
        raise ValueError("Generated and reference IBI must be finite in milliseconds")
    return generated, reference


class DistributionAccumulator:
    """Merge batch moments into moments of ALL sequence samples, on device.

    Chan's centered-moment merge weights unequal batches by their window counts.
    No generated waveform is retained after update. Only O(sequence length)
    running tensors remain. Gaussian KL fits independent coordinate Gaussians to these ALL-window
    moments; it does not estimate the full joint sequence distribution.
    """
    def __init__(self, *, kl_variance_floor_ms2=1.0):
        if (isinstance(kl_variance_floor_ms2, bool)
                or not isinstance(kl_variance_floor_ms2, (int, float))
                or not math.isfinite(kl_variance_floor_ms2)
                or kl_variance_floor_ms2 <= 0):
            raise ValueError("kl_variance_floor_ms2 must be a finite positive number")
        self.kl_variance_floor_ms2 = float(kl_variance_floor_ms2)
        self.count = 0
        self.generated_mean = self.reference_mean = None
        self.generated_m2 = self.reference_m2 = None

    @torch.no_grad()
    def update(self, generated_seconds, reference_seconds):
        generated, reference = _prepare_pair(generated_seconds, reference_seconds)
        if self.count and (generated.shape[1] != self.generated_mean.numel()
                           or generated.device != self.generated_mean.device):
            raise ValueError("Batches must have the same feature dimension and device")
        batch_count = len(generated)
        gen_var, gen_mean = torch.var_mean(generated, dim=0, unbiased=False)
        ref_var, ref_mean = torch.var_mean(reference, dim=0, unbiased=False)
        if not bool(torch.isfinite(gen_var).all() & torch.isfinite(ref_var).all()
                    & torch.isfinite(gen_mean).all() & torch.isfinite(ref_mean).all()):
            raise FloatingPointError("Nonfinite sequence-coordinate moments")
        if self.count == 0:
            self.generated_mean, self.reference_mean = gen_mean, ref_mean
            self.generated_m2, self.reference_m2 = gen_var * batch_count, ref_var * batch_count
        else:
            total = self.count + batch_count
            for prefix, mean, variance in (("generated", gen_mean, gen_var), ("reference", ref_mean, ref_var)):
                old_mean = getattr(self, prefix + "_mean")
                delta = mean - old_mean
                m2 = (getattr(self, prefix + "_m2") + variance * batch_count
                      + delta.square() * (self.count * batch_count / total))
                setattr(self, prefix + "_mean", old_mean + delta * (batch_count / total))
                setattr(self, prefix + "_m2", m2)
        self.count += batch_count

    @torch.no_grad()
    def finalize(self):
        if self.count == 0:
            raise ValueError("Cannot summarize empty validation data")
        gen_var = self.generated_m2 / self.count
        ref_var = self.reference_m2 / self.count
        # KL(real || generated), in nats, for fitted diagonal Gaussians. The
        # lower bound changes only the KL approximation, never the waveforms
        # or the stored moments.
        kl_gen_var = gen_var.clamp_min(self.kl_variance_floor_ms2)
        kl_ref_var = ref_var.clamp_min(self.kl_variance_floor_ms2)
        relative_change = (kl_ref_var - kl_gen_var) / kl_gen_var
        # For small d, d - log1p(d) loses significant digits. Its expansion
        # through d^5 has sub-float64 relative truncation error at |d| <= 1e-4.
        # Clamp the unused polynomial arguments to avoid overflow from torch's
        # eager evaluation of both where branches when variance ratios are huge.
        small = relative_change.clamp(-1e-4, 1e-4)
        near_equal = small.square() * (0.5 + small * (-1.0 / 3.0 + small * (0.25 - small / 5.0)))
        log_ratio = kl_ref_var.log() - kl_gen_var.log()
        general = kl_ref_var / kl_gen_var - 1.0 - log_ratio
        variance_term = torch.where(relative_change.abs() <= 1e-4, near_equal, general).clamp_min(0.0)
        mean_term = ((self.reference_mean - self.generated_mean) / kl_gen_var.sqrt()).square()
        gaussian_kl = (0.5 * variance_term + 0.5 * mean_term).sum()
        if not bool(torch.isfinite(gaussian_kl)):
            raise FloatingPointError("Nonfinite diagonal Gaussian KL divergence")
        return {"aggregate": {"gaussian_kl_real_to_generated": float(gaussian_kl)}}


def summarize_distribution(generated_seconds, reference_seconds, *, kl_variance_floor_ms2=1.0):
    """Compare [windows, time coordinates] using the same all-window estimator.

    Inputs are seconds. Coordinate population variances use correction=0; for
    one window they are zero. Stored zeros and negative generated values are not clipped.
    Diagonal Gaussian KL(real || generated) sums over coordinates after fitting
    their moments across all windows. The fixed variance floor is in ms squared.
    """
    accumulator = DistributionAccumulator(kl_variance_floor_ms2=kl_variance_floor_ms2)
    accumulator.update(generated_seconds, reference_seconds)
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
