"""Explicit, invertible IBI affine transforms without waveform clipping.

Stored signal_mean_seconds/signal_std_seconds always describe the actual
training population. Fixed minmax bounds have separate affine parameters.
Only clean inputs are range-checked; diffusion noise and generated outputs
must not be clipped to the clean-input interval.
"""
from __future__ import annotations

import math
from numbers import Real


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite real number')
    return float(value)


def normalization_spec(config):
    """Return canonical minmax settings, or None for legacy/default z-score.

    config is the full dataset configuration. The optional nested object is
    either {method: zscore} or {method: minmax, min_seconds: lo, max_seconds: hi}.
    Bounds are fixed configuration values, never fitted on held-out data.
    """
    if not isinstance(config, dict):
        raise ValueError('Dataset configuration must be a dictionary')
    if 'signal_normalization' not in config:
        return None
    spec = config['signal_normalization']
    if not isinstance(spec, dict) or 'method' not in spec:
        raise ValueError('signal_normalization must be an object with an explicit method')
    if spec['method'] == 'zscore':
        if set(spec) != {'method'}:
            raise ValueError('zscore signal_normalization accepts only method')
        return None
    if spec['method'] != 'minmax' or set(spec) != {'method', 'min_seconds', 'max_seconds'}:
        raise ValueError('minmax signal_normalization requires exactly method, min_seconds and max_seconds')
    low = _finite_number(spec['min_seconds'], 'min_seconds')
    high = _finite_number(spec['max_seconds'], 'max_seconds')
    scale = (high-low)/2.
    center = low+scale
    if high <= low or not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('minmax bounds must define a finite positive-width interval')
    return dict(method='minmax', min_seconds=low, max_seconds=high)


def signal_affine_parameters(scaler):
    """Return (center_seconds, scale_seconds) for x=(normalized*scale)+center.

    Legacy scalers use population mean/std. A minmax scaler must record its
    complete specification and exact matching center/scale; incomplete or
    conflicting metadata are rejected rather than silently using z-score.
    """
    if not isinstance(scaler, dict):
        raise ValueError('Signal scaler must be a dictionary')
    try:
        mean = _finite_number(scaler['signal_mean_seconds'], 'signal_mean_seconds')
        std = _finite_number(scaler['signal_std_seconds'], 'signal_std_seconds')
    except KeyError as error:
        raise ValueError('Signal scaler requires recorded training mean and standard deviation') from error
    if std <= 0:
        raise ValueError('signal_std_seconds must be positive')
    spec = normalization_spec(scaler)
    if spec is None:
        if 'signal_center_seconds' in scaler or 'signal_scale_seconds' in scaler:
            raise ValueError('Separate affine parameters require an explicit minmax specification')
        return mean, std
    expected_scale = (spec['max_seconds']-spec['min_seconds'])/2.
    expected_center = spec['min_seconds']+expected_scale
    try:
        center = _finite_number(scaler['signal_center_seconds'], 'signal_center_seconds')
        scale = _finite_number(scaler['signal_scale_seconds'], 'signal_scale_seconds')
    except KeyError as error:
        raise ValueError('minmax scaler requires both signal_center_seconds and signal_scale_seconds') from error
    if center != expected_center or scale != expected_scale:
        raise ValueError('minmax affine center/scale disagree with the frozen bounds')
    return center, scale


def validate_signal_bounds(signal, spec, *, context='IBI'):
    """Reject out-of-range clean values; never alter, clip, or fit the data."""
    if spec is None:
        return
    import numpy as np

    canonical = normalization_spec({'signal_normalization': spec})
    if canonical is None:
        return
    values = np.asarray(signal)
    if values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
        raise ValueError(f'{context}: minmax input must contain finite real values')
    if np.any(values < canonical['min_seconds']) or np.any(values > canonical['max_seconds']):
        raise ValueError(f"{context}: clean IBI lies outside fixed minmax bounds "
                         f"[{canonical['min_seconds']}, {canonical['max_seconds']}] seconds; no clipping is applied")
