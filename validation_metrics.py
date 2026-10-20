"""Sequence-coordinate moments and legacy IBI distribution diagnostics.

One row is a sequence sample; dim=0 estimates each time coordinate's mean and
population variance across windows. All tensor arithmetic stays on the input
device in float64. Production callers require CUDA; CPU supports unit tests.
"""
from __future__ import annotations

from contextlib import contextmanager
import random

import numpy as np
import torch


_ROW_FIELDS = ("generated_mean_ms", "reference_mean_ms", "mean_error_ms",
               "mean_absolute_error_ms", "generated_variance_ms2",
               "reference_variance_ms2", "variance_error_ms2", "variance_absolute_error_ms2")


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


def _moment_rows(gen_mean, ref_mean, gen_var, ref_var):
    """Transfer a batch of final diagnostics together, avoiding per-row CUDA sync."""
    values = torch.stack((gen_mean, ref_mean, gen_mean - ref_mean,
                          (gen_mean - ref_mean).abs(), gen_var, ref_var,
                          gen_var - ref_var, (gen_var - ref_var).abs()), dim=-1).reshape(-1, 8)
    valid = ref_var != 0
    ratio = gen_var / torch.where(valid, ref_var, torch.ones_like(ref_var))
    if not bool(torch.isfinite(values).all() & torch.isfinite(ratio[valid]).all()):
        raise FloatingPointError("Nonfinite IBI distribution statistic")
    packed = torch.cat((values, ratio.reshape(-1, 1), valid.to(torch.float64).reshape(-1, 1)), dim=1)
    return [dict(zip(_ROW_FIELDS, row[:8]), variance_ratio=row[8] if row[9] else None)
            for row in packed.tolist()]


class DistributionAccumulator:
    """Merge batch moments into moments of ALL sequence samples, on device.

    Chan's centered-moment merge weights unequal batches by their window counts.
    No generated waveform is retained after update. Only O(sequence length)
    running tensors and serialized per-window diagnostics remain. The L2 metrics
    are neither per-window waveform distances nor RMS (no division by sqrt(D)).
    """
    def __init__(self):
        self.count = 0
        self.generated_mean = self.reference_mean = None
        self.generated_m2 = self.reference_m2 = None
        self.window_errors = None
        self.per_window = []

    @torch.no_grad()
    def update(self, generated_seconds, reference_seconds):
        generated, reference = _prepare_pair(generated_seconds, reference_seconds)
        if self.count and (generated.shape[1] != self.generated_mean.numel()
                           or generated.device != self.generated_mean.device):
            raise ValueError("Batches must have the same feature dimension and device")
        batch_count = len(generated)
        gen_window_var, gen_window_mean = torch.var_mean(generated, dim=1, unbiased=False)
        ref_window_var, ref_window_mean = torch.var_mean(reference, dim=1, unbiased=False)
        rows = _moment_rows(gen_window_mean, ref_window_mean, gen_window_var, ref_window_var)
        errors = torch.stack(((gen_window_mean - ref_window_mean).abs().sum(),
                              (gen_window_var - ref_window_var).abs().sum()))
        gen_var, gen_mean = torch.var_mean(generated, dim=0, unbiased=False)
        ref_var, ref_mean = torch.var_mean(reference, dim=0, unbiased=False)
        if not bool(torch.isfinite(gen_var).all() & torch.isfinite(ref_var).all()
                    & torch.isfinite(gen_mean).all() & torch.isfinite(ref_mean).all()
                    & torch.isfinite(errors).all()):
            raise FloatingPointError("Nonfinite sequence-coordinate moments")
        if self.count == 0:
            self.generated_mean, self.reference_mean = gen_mean, ref_mean
            self.generated_m2, self.reference_m2 = gen_var * batch_count, ref_var * batch_count
            self.window_errors = errors
        else:
            total = self.count + batch_count
            for prefix, mean, variance in (("generated", gen_mean, gen_var), ("reference", ref_mean, ref_var)):
                old_mean = getattr(self, prefix + "_mean")
                delta = mean - old_mean
                m2 = (getattr(self, prefix + "_m2") + variance * batch_count
                      + delta.square() * (self.count * batch_count / total))
                setattr(self, prefix + "_mean", old_mean + delta * (batch_count / total))
                setattr(self, prefix + "_m2", m2)
            self.window_errors = self.window_errors + errors
        self.per_window.extend(dict(window_index=self.count + index, **row) for index, row in enumerate(rows))
        self.count += batch_count

    @torch.no_grad()
    def finalize(self):
        if self.count == 0:
            raise ValueError("Cannot summarize empty validation data")
        gen_var = self.generated_m2 / self.count
        ref_var = self.reference_m2 / self.count
        gen_mean = self.generated_mean.mean()
        ref_mean = self.reference_mean.mean()
        # Law of total variance recovers the original pooled scalar diagnostic.
        pooled_gen_var = (gen_var + (self.generated_mean - gen_mean).square()).mean()
        pooled_ref_var = (ref_var + (self.reference_mean - ref_mean).square()).mean()
        aggregate = _moment_rows(gen_mean, ref_mean, pooled_gen_var, pooled_ref_var)[0]
        distances = torch.stack((torch.linalg.vector_norm(self.generated_mean - self.reference_mean),
                                 torch.linalg.vector_norm(gen_var - ref_var)))
        window_errors = self.window_errors / self.count
        if not bool(torch.isfinite(distances).all() & torch.isfinite(window_errors).all()):
            raise FloatingPointError("Nonfinite validation error")
        aggregate.update(zip(("mean_window_absolute_error_ms", "variance_window_absolute_error_ms2"),
                             window_errors.tolist()))
        aggregate.update(zip(("mean_vector_l2_ms", "variance_vector_l2_ms2"), distances.tolist()))
        return {"aggregate": aggregate, "per_window": [row.copy() for row in self.per_window]}


def summarize_distribution(generated_seconds, reference_seconds):
    """Compare [windows, time coordinates] using the same all-window estimator.

    Inputs are seconds. Coordinate population variances use correction=0; for
    one window they are zero. Legacy pooled/window diagnostics retain their
    definitions. Stored zeros and negative generated values are not clipped.
    """
    accumulator = DistributionAccumulator()
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
