"""Explicit diffusion schedules outside the immutable WaveStitch core.

Missing schedule fields always mean the legacy linear schedule. New training
launchers explicitly select linear_power; checkpoint readers never consult the
current dataset JSON when restoring a historical run.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from numbers import Integral, Real

import numpy as np


SCHEDULES = ("linear", "linear_power")
DEFAULT_TERMINAL_ALPHA_BAR = 1e-8
IDENTITY_SCHEMA = "diffusion_schedule_v1"


def _values(args):
    return args if isinstance(args, Mapping) else vars(args)


def _unit_float(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be a finite number strictly between 0 and 1")
    return float(value)


def diffusion_settings(config, default_schedule="linear"):
    """Read only schedule selectors; callers must explicitly choose new defaults."""
    values = _values(config)
    schedule = values.get("diffusion_schedule", default_schedule)
    if schedule not in SCHEDULES:
        raise ValueError(f"diffusion_schedule must be one of {SCHEDULES}")
    target = _unit_float("terminal_alpha_bar", values.get("terminal_alpha_bar", DEFAULT_TERMINAL_ALPHA_BAR))
    return dict(diffusion_schedule=schedule, terminal_alpha_bar=target)


def _arrays(args):
    values = _values(args)
    settings = diffusion_settings(values)
    steps = values.get("timesteps")
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("timesteps must be a positive integer")
    beta_0 = _unit_float("beta_0", values.get("beta_0"))
    beta_T = _unit_float("beta_T", values.get("beta_T"))
    if beta_0 > beta_T:
        raise ValueError("beta_0 must not exceed beta_T")
    # Preserve every legacy NumPy operation and the final float32 cast.
    betas = np.linspace(beta_0, beta_T, int(steps)).reshape((-1, 1))
    alphas = 1 - betas
    power = 1.0
    if settings["diffusion_schedule"] == "linear_power":
        power = math.log(settings["terminal_alpha_bar"]) / float(np.log(alphas).sum())
        alphas = np.exp(np.log(alphas) * power)
        betas = 1 - alphas
    alpha_bars = np.cumprod(alphas).reshape((-1, 1))
    arrays = {"betas": betas.astype(np.float32), "alphas": alphas.astype(np.float32),
              "alpha_bars": alpha_bars.astype(np.float32)}
    for name, array in arrays.items():
        if not np.isfinite(array).all() or not ((array > 0) & (array < 1)).all():
            raise ValueError(f"{name} must remain finite and strictly between 0 and 1 in float32")
    if not np.all(np.diff(arrays["alpha_bars"].reshape(-1)) < 0):
        raise ValueError("alpha_bars must strictly decrease in float32")
    parameters = dict(**settings, timesteps=int(steps), beta_0=beta_0, beta_T=beta_T,
                      alpha_power=power)
    return arrays, parameters


def build_diffusion_config(args):
    """Return the original four-key tensor contract, with a recorded selection."""
    import torch

    arrays, parameters = _arrays(args)
    return dict(T=parameters["timesteps"],
                **{name: torch.from_numpy(value) for name, value in arrays.items()})


def validate_diffusion_config(args, config):
    """Reject a same-length schedule that differs from the recorded arguments."""
    import torch

    expected = build_diffusion_config(args)
    if isinstance(config.get("T"), bool) or config.get("T") != expected["T"]:
        raise ValueError("Diffusion schedule T differs from recorded training arguments")
    for name in ("betas", "alphas", "alpha_bars"):
        actual = config.get(name)
        reference = expected[name]
        if (not torch.is_tensor(actual) or actual.dtype != torch.float32 or actual.shape != reference.shape
                or not torch.equal(actual, reference.to(actual.device))):
            raise ValueError(f"Diffusion schedule {name} differs from recorded training arguments")
    return True


def diffusion_identity(args, config=None):
    """Store the complete effective beta array and fingerprints of all tensors."""
    arrays, parameters = _arrays(args)
    if config is not None:
        validate_diffusion_config(args, config)
    terminal = float(arrays["alpha_bars"][-1, 0])
    return dict(
        schema=IDENTITY_SCHEMA, **parameters,
        terminal_target_applied=parameters["diffusion_schedule"] == "linear_power",
        effective_terminal_alpha_bar=terminal, terminal_snr=terminal / (1 - terminal),
        tensor_dtype="float32", tensor_shape=[parameters["timesteps"], 1],
        betas_float32=arrays["betas"].reshape(-1).tolist(),
        tensor_sha256={name: hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest()
                       for name, value in arrays.items()},
    )


def verify_diffusion_identity(args, identity):
    """Legacy runs may omit identity only when their restored schedule is linear."""
    expected = diffusion_identity(args)
    if identity is None:
        if expected["diffusion_schedule"] != "linear":
            raise ValueError("Nonlinear schedule requires its recorded diffusion_identity")
        return dict(legacy_linear=True, recorded_identity_verified=False, identity=expected)
    if not isinstance(identity, dict) or identity != expected:
        raise ValueError("Recorded diffusion_identity differs from training schedule arguments")
    return dict(legacy_linear=False, recorded_identity_verified=True, identity=expected)
