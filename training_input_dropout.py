"""Opt-in, training-only IBI input erasure after diffusion forward noising.

This is not inverted dropout: kept values are never scaled. Epsilon targets,
condition channels and the loss are unchanged. The private random stream and
counters live in this adapter, never in the network or its checkpoint.
"""
from __future__ import annotations

import json
import math
from numbers import Integral, Real
import os
from pathlib import Path


def input_dropout_options(config):
    """Canonical configuration; absent settings retain exact inactive behavior."""
    if not isinstance(config, dict):
        raise ValueError("Dataset configuration must be a JSON object")
    configured = config.get("training_input_dropout", {})
    if not isinstance(configured, dict) or set(configured) - {"probability", "seed"}:
        raise ValueError("training_input_dropout accepts only probability and seed")
    probability = configured.get("probability", 0.)
    seed = configured.get("seed", 42)
    if (isinstance(probability, bool) or not isinstance(probability, Real)
            or not math.isfinite(probability) or probability < 0 or probability > 1):
        raise ValueError("training_input_dropout.probability must be finite and within [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 or seed > 2**63-1:
        raise ValueError("training_input_dropout.seed must be an integer in [0, 2**63-1]")
    return dict(probability=float(probability), seed=int(seed))


class TrainingInputDropout:
    """Erase individual channel-zero values using an explicit device generator.

    Production passes the required CUDA device. Explicit CPU construction is
    supported for numerical unit tests; no device fallback occurs here.
    """
    def __init__(self, options, device):
        self.options = input_dropout_options({"training_input_dropout": options})
        self.probability = self.options["probability"]
        self.application_calls = 0
        self._generator = None
        self._eligible_values = None
        self._dropped_values = None
        self.device = None
        # Do not create a generator/tensor, copy data, or consume any random
        # stream for the legacy p=0 path.
        if self.probability == 0:
            return
        import torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Training input dropout requires an explicit CPU test or CUDA device")
        self._generator = torch.Generator(device=self.device).manual_seed(self.options["seed"])
        self._eligible_values = torch.zeros((), dtype=torch.int64, device=self.device)
        self._dropped_values = torch.zeros((), dtype=torch.int64, device=self.device)

    def __call__(self, batch_noised):
        if self.probability == 0:
            return batch_noised
        import torch
        if (not torch.is_tensor(batch_noised) or batch_noised.ndim != 3
                or min(batch_noised.shape) < 1 or not batch_noised.is_floating_point()):
            raise ValueError("Input dropout expects a floating [batch, time, channels] tensor")
        if batch_noised.device != self.device:
            raise ValueError("Input dropout and its private generator must use the input device")
        dropped = torch.rand(batch_noised.shape[:2], device=self.device,
                             generator=self._generator, dtype=torch.float32) < self.probability
        result = batch_noised.clone()
        result[:, :, 0].masked_fill_(dropped, 0.)
        self.application_calls += 1
        # A 30-epoch real run has over 2**32 eligible values: never int32.
        # These are device reductions only; there is no per-batch scalar read.
        self._eligible_values.add_(dropped.numel())
        self._dropped_values.add_(dropped.sum(dtype=torch.int64))
        return result

    def report(self):
        """Transfer cumulative device counters only when producing the final report."""
        eligible = dropped = 0
        if self.probability > 0:
            import torch
            eligible, dropped = torch.stack((self._eligible_values, self._dropped_values)).detach().cpu().tolist()
        return dict(options=dict(self.options), active=self.probability > 0,
            application_calls=self.application_calls, eligible_values=int(eligible), dropped_values=int(dropped),
            observed_fraction=dropped/eligible if eligible else 0., counter_dtype="int64",
            generator_device=str(self.device) if self._generator is not None else None,
            protocol=dict(training_only=True, stage="after forward noising and clean-condition restoration, before model",
                channels=[0], channel_name="IBI", mask_shape="[batch, time]", fill_value=0.,
                kept_values_rescaled=False, epsilon_targets_changed=False, loss_changed=False,
                condition_channels_changed=False, generator="dedicated device generator; seeded once per training run",
                global_rng_streams_consumed=False, validation_or_inference_applied=False,
                observed_fraction_definition="dropped_values / eligible_values; 0 when inactive or no applications",
                count_scope="Active training applications only; p=0 creates no generator/counters and reports zero counts"))


def create_training_input_dropout(device):
    """Read the same selected JSON as Preprocessor; no independent overrides."""
    selected = os.environ.get("DREAMT_CONFIG_PATH")
    path = (Path(selected).expanduser().resolve() if selected
            else Path(__file__).resolve().with_name("dataset_config.json"))
    if selected and not path.is_file():
        raise FileNotFoundError(f"Selected DREAMT configuration does not exist: {path}")
    config = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    return TrainingInputDropout(input_dropout_options(config), device)
