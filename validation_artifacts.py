"""Serialize already computed validation samples; never generate or resample."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def write_validation_artifacts(output, *, generated_seconds, reference_seconds,
                               scoring_mask, windows_metadata, conditions,
                               condition_columns, statistics, protocol_path, epoch):
    """Host-side I/O only. Moments/KL were computed on the model device.

    Physical waveforms retain zeros, negatives and all missing positions. The
    scoring mask selects the same positions in both fits; the original-positive
    mask is also saved so plots can show missing values explicitly.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    generated = np.asarray(generated_seconds)
    reference = np.asarray(reference_seconds)
    mask = np.asarray(scoring_mask)
    if generated.ndim != 2 or generated.shape != reference.shape or mask.shape != generated.shape:
        raise ValueError("Artifact waveforms and scoring mask must share [windows, samples] shape")
    n, length = generated.shape
    if n < 1 or mask.dtype != np.bool_ or len(windows_metadata) != n:
        raise ValueError("Artifact metadata and boolean scoring mask must match the windows")
    if conditions.shape != (n, length, len(condition_columns)):
        raise ValueError("Artifact condition dimensions do not match the validation windows")
    if any(np.asarray(values).shape != (n,) for values in statistics.values()):
        raise ValueError("Artifact statistics must contain one scalar per window")
    if [m['validation_index'] for m in windows_metadata] != list(range(n)):
        raise ValueError("Artifact order must follow the fixed validation indices")
    waveforms = output / 'waveforms.npz'
    table = output / 'per_window_metrics.csv'
    index = output / 'artifacts.json'
    if any(path.exists() for path in (waveforms, table, index)):
        raise FileExistsError("Refusing to overwrite existing validation artifacts")
    # NumPy serialization is lossless and contains no object arrays/pickle.
    with (output / 'waveforms.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream,
            generated_ibi_seconds=generated, reference_ibi_seconds=reference,
            scoring_mask=mask,
            reference_observed_mask=np.isfinite(reference) & (reference > 0),
            clean_conditions=conditions, condition_columns=np.asarray(condition_columns, dtype=str),
            validation_index=np.arange(n, dtype=np.int64),
            subject_id=np.asarray([m['subject_id'] for m in windows_metadata], dtype=str),
            filename=np.asarray([m['filename'] for m in windows_metadata], dtype=str),
            start_sample=np.asarray([m['start_sample'] for m in windows_metadata], dtype=np.int64),
            start_second=np.asarray([m['start_second'] for m in windows_metadata], dtype=np.float64),
            sleep_stages=np.asarray([m['sleep_stages'] for m in windows_metadata], dtype=np.int64))
    (output / 'waveforms.npz.tmp').replace(waveforms)
    identity_fields = ['validation_index', 'subject_id', 'filename', 'start_sample', 'start_second']
    with (output / 'per_window_metrics.csv.tmp').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=identity_fields + list(statistics))
        writer.writeheader()
        for i, metadata in enumerate(windows_metadata):
            row = {key: metadata[key] for key in identity_fields}
            row.update({key: np.asarray(values)[i].item() for key, values in statistics.items()})
            writer.writerow(row)
    (output / 'per_window_metrics.csv.tmp').replace(table)
    report = dict(schema='ibi_validation_artifacts_v1', epoch=epoch, windows=n,
        samples_per_window=length, waveforms=str(waveforms.resolve()),
        per_window_metrics=str(table.resolve()), protocol=str(Path(protocol_path).resolve()),
        waveform_unit='seconds', mean_unit='milliseconds', variance_unit='milliseconds_squared',
        variance_correction=0, waveform_clipping=False,
        moments='Same scoring_mask time positions in real and generated; raw variances before KL floor',
        kl_terms='kl_mean_term + kl_variance_term = gaussian_kl_real_to_generated; natural logarithm',
        generation='Existing independent-window validation samples; no extra sampling or RNG draws',
        waveform_sha256=hashlib.sha256(waveforms.read_bytes()).hexdigest(),
        metrics_sha256=hashlib.sha256(table.read_bytes()).hexdigest())
    index.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return dict(**report, index=str(index.resolve()))
