"""Configure the existing S4Layer dropout without changing the model source."""
from __future__ import annotations

import math
from numbers import Real


def s4_dropout_options(config):
    if not isinstance(config, dict):
        raise ValueError("S4 dropout configuration must be a JSON object")
    value = config.get("s4_dropout", 0.0)
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value < 1):
        raise ValueError("s4_dropout must be a finite number in [0, 1)")
    return float(value)


def s4_dropout_report(model, probability, expected_layers=None):
    """Inspect module configuration/modes only; never run a forward or draw RNG."""
    import torch.nn as nn
    probability = s4_dropout_options({"s4_dropout": probability})
    records = []
    for name, module in model.named_modules():
        if type(module).__name__ != "S4Layer":
            continue
        dropout = module.dropout
        if probability > 0:
            if not isinstance(dropout, nn.Dropout2d) or dropout.p != probability:
                raise ValueError("S4 layer dropout differs from configuration: " + name)
        elif not isinstance(dropout, nn.Identity):
            raise ValueError("Zero S4 dropout must retain the original Identity module: " + name)
        if dropout.training != model.training:
            raise ValueError("S4 dropout train/eval mode differs from its model: " + name)
        records.append(dict(name=name, module=type(dropout).__name__, probability=probability,
                            training=dropout.training))
    if not records or (expected_layers is not None and len(records) != expected_layers):
        raise ValueError("Unexpected number of configured S4Layer dropout modules")
    return dict(probability=probability, active=probability > 0, training=model.training,
        layers=len(records), modules=records,
        location="S4Layer output, before its residual addition and LayerNorm",
        masking="native nn.Dropout2d on [batch, channels, time]; channel mask shared over time",
        retained_activation_scaling="1 / (1 - p) during training",
        random_stream="native PyTorch training RNG", disabled_in_eval=True,
        configuration_inspected=True, forward_executed_by_report=False)
