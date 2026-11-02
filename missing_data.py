"""Explicit observed-sample loss for the optional missing-data experiment."""
from __future__ import annotations

import torch


def masked_noise_mse(predicted, target, observed_mask):
    """Return (observed-sample mean squared error, detached sum, valid count).

    ``predicted`` and ``target`` must be finite floating tensors with matching
    dtype/device and shape [B, 1, L]. ``observed_mask`` is a Boolean [B, L]
    tensor on that same device. Every input position must be finite, including
    masked-out positions; the caller owns any finite placeholder policy.

    The denominator is the total number of observed samples across the batch,
    not the number of windows. Masked-out values receive zero gradient. Valid
    values are selected before subtraction, preventing invalid-position
    arithmetic from contaminating the loss or backward pass. Half/bfloat16
    errors accumulate in float32; float32/float64 keep their input precision.
    All tensor computation stays on the input device. The count is explicitly
    synchronized once to a Python int; the detached sum stays on the device.
    """
    for name, value in (("predicted", predicted), ("target", target), ("observed_mask", observed_mask)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if predicted.ndim != 3 or predicted.shape[1] != 1:
        raise ValueError("predicted must have shape [B, 1, L]")
    if target.shape != predicted.shape:
        raise ValueError("target must have the same [B, 1, L] shape as predicted")
    if observed_mask.shape != (predicted.shape[0], predicted.shape[2]):
        raise ValueError("observed_mask must have shape [B, L]")
    if observed_mask.dtype != torch.bool:
        raise TypeError("observed_mask must be Boolean; numeric/negative masks are not accepted")
    if not predicted.is_floating_point() or not target.is_floating_point():
        raise TypeError("predicted and target must be real floating tensors")
    if predicted.dtype != target.dtype:
        raise TypeError("predicted and target must have the same dtype")
    if target.device != predicted.device or observed_mask.device != predicted.device:
        raise ValueError("predicted, target and observed_mask must be on the same device")
    all_finite = torch.isfinite(predicted).all() & torch.isfinite(target).all()
    if not bool(all_finite.item()):
        raise ValueError("predicted and target must be finite at every position, including masked-out positions")
    count = int(observed_mask.sum().item())
    if count == 0:
        raise ValueError("observed_mask must contain at least one observed sample")
    valid_predicted = predicted[:, 0, :].masked_select(observed_mask)
    valid_target = target[:, 0, :].masked_select(observed_mask)
    if predicted.dtype in (torch.float16, torch.bfloat16):
        valid_predicted = valid_predicted.float()
        valid_target = valid_target.float()
    squared_sum = (valid_predicted-valid_target).square().sum()
    return squared_sum/count, squared_sum.detach(), count
