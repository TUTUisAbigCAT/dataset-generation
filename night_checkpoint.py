"""Explicit current-config inference and complete raw state-dict restoration.

A bare checkpoint does not identify its training data, scaler, diffusion
schedule or source revision. This path reports those limits and never reads a
run manifest to infer missing training provenance.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from diffusion_schedule import diffusion_settings
from scripts.evaluate_denoising import DEFAULT_TRAINING_ARGS
from window_config import EPOCH_SAMPLES, window_settings


def load_inference_config(config_path):
    """Read an explicit JSON independently of DREAMT_* environment overrides."""
    path = Path(config_path).expanduser().resolve()
    contents = path.read_bytes()
    config = json.loads(contents.decode("utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Inference configuration must be a JSON object")
    dimensions = window_settings(config)
    schedule = diffusion_settings(config, default_schedule="linear_power")
    args = SimpleNamespace(**dict(DEFAULT_TRAINING_ARGS, dataset="DREAMT",
        window_size=dimensions["window_size"], s4_lmax=dimensions["window_size"],
        stride=EPOCH_SAMPLES, **schedule))
    data_config = deepcopy(config)
    for key in ("data_dir", "metadata_path", "split_path"):
        value = data_config.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path string or null")
        selected = Path(value).expanduser()
        data_config[key] = str((selected if selected.is_absolute() else path.parent / selected).resolve())
    provenance = dict(source="explicit inference JSON and fixed S4 baseline template",
        config_path=str(path), config_sha256=hashlib.sha256(contents).hexdigest(),
        window=dimensions, diffusion_settings=schedule,
        environment_overrides_used=False, training_manifest_read=False,
        training_identity_verified=False, training_source_identity_verified=False,
        training_schedule_identity_verified=False, training_scaler_identity_verified=False,
        scaler_source="training partition reconstructed from the explicit current data configuration",
        limitations="A raw state_dict does not record the training schedule, scaler, split or source. "
                    "The supplied configuration must match training; tensor compatibility alone cannot verify that.")
    return args, data_config, provenance


def load_checkpoint_state(model, saved):
    """Strictly restore every parameter/buffer, including complex components.

    The original S4 constructor broadcasts some fixed buffers with stride zero.
    Materialize their registered storage before PyTorch's in-place strict copy;
    no checkpoint values, parameterization or production S4 source are changed.
    """
    import torch

    expected = model.state_dict()
    if not isinstance(saved, Mapping) or set(saved) != set(expected):
        raise ValueError("Checkpoint must be a raw state_dict with exactly the configured model keys")
    # Complete validation precedes every mutation: never silently cast a saved
    # dtype, discard imaginary components or partially load incompatible states.
    for name, reference in expected.items():
        value = saved[name]
        if not torch.is_tensor(value) or not torch.is_tensor(reference):
            raise ValueError(f"Checkpoint state must contain only tensors: {name}")
        if value.shape != reference.shape:
            raise ValueError(f"Checkpoint shape mismatch for {name}: {tuple(value.shape)} != "
                             f"{tuple(reference.shape)}. Check window_epochs and the model configuration.")
        if value.dtype != reference.dtype:
            raise ValueError(f"Checkpoint dtype mismatch for {name}: {value.dtype} != {reference.dtype}")
        if value.layout != torch.strided or reference.layout != torch.strided:
            raise ValueError(f"Checkpoint requires dense strided state tensors: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Checkpoint contains nonfinite values: {name}")
    cached_lengths = {}
    for module_name, module in model.named_modules():
        if hasattr(module, "L") and {"omega", "z"}.issubset(module._buffers):
            length = module.L
            if not isinstance(length, int) or isinstance(length, bool) or length < 1:
                raise ValueError("Invalid S4 cached length; check window_epochs")
            for key in ("omega", "z"):
                if tuple(module._buffers[key].shape) != (length // 2 + 1, 2):
                    raise ValueError("S4 cached length disagrees with omega/z buffers; check window_epochs")
            cached_lengths[module_name] = length

    materialized = []
    for module_name, module in model.named_modules():
        for name, buffer in list(module._buffers.items()):
            full_name = f"{module_name}.{name}" if module_name else name
            if buffer is not None and full_name in expected:
                module._buffers[name] = buffer.detach().clone(memory_format=torch.contiguous_format)
                materialized.append(full_name)
    model.load_state_dict(saved, strict=True)
    restored = model.state_dict()
    for name, value in restored.items():
        if not torch.equal(value.detach(), saved[name].detach().to(device=value.device)):
            raise RuntimeError(f"Checkpoint state was not restored bitwise: {name}")
    return dict(method="model.load_state_dict(strict=True), restoring parameters and persistent buffers",
        all_state_bitwise_equal=True, parameters_and_buffers_restored=True,
        restored_state_tensors=len(restored),
        buffer_storage_materialized=materialized,
        buffer_storage_materialization_reason="S4 constructor buffers may share stride-zero storage; "
                                              "cloning permits an exact in-place state copy",
        s4_cached_lengths=cached_lengths,
        training_source_identity_verified=False,
        initialization_seed_match_required=False,
        limitations="Matching state keys, shapes and values does not verify the training source, "
                    "diffusion schedule or data scaler. Recurrent step caches must be set up again before step().")
