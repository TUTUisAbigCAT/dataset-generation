"""Single definition of the epoch-aligned DREAMT sequence dimensions."""
from __future__ import annotations

FS = 25
EPOCH_SECONDS = 30
EPOCH_SAMPLES = FS * EPOCH_SECONDS
DEFAULT_WINDOW_EPOCHS = 4


def window_settings(config):
    """Resolve a strict positive sleep-epoch count; unrelated config is allowed."""
    if not isinstance(config, dict):
        raise ValueError("DREAMT configuration must be a JSON object")
    epochs = config.get("window_epochs", DEFAULT_WINDOW_EPOCHS)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("window_epochs must be a positive integer number of 30-second sleep epochs")
    return dict(window_epochs=epochs, window_size=epochs * EPOCH_SAMPLES,
                epoch_samples=EPOCH_SAMPLES, sample_rate_hz=FS,
                window_seconds=epochs * EPOCH_SECONDS)
