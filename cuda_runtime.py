"""CUDA-only execution policy for the adapted WaveStitch entry points.

Host-side file loading, NumPy metadata, and serialization are intentionally not
model computation. Training and reverse-diffusion tensors must reside on CUDA.
"""
from __future__ import annotations

import torch


def require_cuda() -> torch.device:
    """Return the assigned CUDA device or fail before model execution."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required; no usable CUDA device is available. "
            "Run inside the Fir GPU allocation with a CUDA-enabled PyTorch build "
            "and check CUDA_VISIBLE_DEVICES. CPU fallback is disabled. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}"
        )
    try:
        selected = torch.device("cuda", torch.cuda.current_device())
        torch.empty(0, device=selected)
    except Exception as error:
        raise RuntimeError(
            "CUDA is required but device initialization failed; "
            "CPU fallback is disabled. Check the GPU allocation and CUDA driver."
        ) from error
    return selected


def assert_cuda_model(model: torch.nn.Module) -> None:
    """Reject a model containing any parameter or registered buffer off CUDA."""
    misplaced = [f"parameter {name}: {tensor.device}" for name, tensor in model.named_parameters()
                 if tensor.device.type != "cuda"]
    misplaced.extend(f"buffer {name}: {tensor.device}" for name, tensor in model.named_buffers()
                     if tensor.device.type != "cuda")
    if misplaced:
        raise RuntimeError("CUDA-only model check failed: " + "; ".join(misplaced[:8]))


def cuda_environment(device: torch.device | None = None) -> dict:
    """Record the selected GPU and runtime without changing execution settings."""
    selected = require_cuda() if device is None else torch.device(device)
    if selected.type != "cuda":
        raise RuntimeError("CUDA-only runtime does not accept a non-CUDA device")
    return {
        "device": str(selected),
        "device_name": torch.cuda.get_device_name(selected),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cpu_fallback_enabled": False,
    }
