"""Whole-record generation conditions using an already verified training scaler.

This inference-only adapter may open the requested test record. It neither
changes training partitions nor fits a scaler. Reference IBI is returned only
for optional evaluation; the model conditions contain no reference IBI values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_utils import CONTINUOUS
from watchsleep_loader import EPOCH_SAMPLES, STAGE_NAMES, TARGET_FS, SSDataset


def _resolve_subject(preprocessor, subject):
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError("subject must be a nonempty participant ID")
    requested = subject.strip()
    if any(character in requested for character in "/\\:"):
        raise ValueError("subject must be a participant ID, not a path")
    if requested.casefold().endswith(".npz"):
        requested = requested[:-4]
    matches = [(role, name) for role, names in preprocessor.splits.items()
               for name in names if Path(name).stem.casefold() == requested.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Expected one case-insensitive exact subject match for {subject!r}; found {len(matches)}")
    return matches[0]


def _check_stage_conditions(path, record, downsample_rate):
    """Separate label validity from the loader's combined signal/label flag."""
    epochs = record["stored_complete_epochs"]
    with np.load(path, allow_pickle=False) as saved:
        original_samples = np.asarray(saved["data"]).size
        labels = np.asarray(saved["stages"]).reshape(-1)
    if labels.size == original_samples:
        raw = labels[::downsample_rate][:epochs * EPOCH_SAMPLES].reshape(epochs, EPOCH_SAMPLES)
        raw_first = raw[:, 0]
        consistent = np.all(raw == raw[:, :1], axis=1)
    elif labels.size == epochs or (record["discarded_tail_samples"] and labels.size == epochs + 1):
        raw_first = labels[:epochs]
        consistent = np.ones(epochs, dtype=bool)
    else:
        raise ValueError("stages do not align with the complete stored epochs")
    stages = SSDataset.remap_labels(raw_first, "sleep_staging")
    invalid = np.flatnonzero((stages < 0) | ~consistent)
    if invalid.size:
        raise ValueError("Unknown or inconsistent sleep-stage conditions at epoch indices "
                         + repr(invalid[:20].tolist()) + "; no epochs were removed")
    if not np.array_equal(stages, record["stage_epochs"]):
        raise ValueError("Generation stage parsing disagrees with the source loader")
    return stages.astype(np.int64, copy=False), raw_first.astype(np.int64, copy=False)


def prepare_night(preprocessor, subject="S092"):
    """Prepare all complete stored epochs without applying training max_epochs.

    Callers supply the selected training preprocessor and its fitted scaler.
    This function preserves that object's scaler, caches and data identity.
    Unknown/inconsistent stage labels or known source-time gaps are errors.
    Invalid reference IBI is reported but is not used to restrict conditions.
    """
    role, filename = _resolve_subject(preprocessor, subject)
    sid = Path(filename).stem
    record = SSDataset(preprocessor.data_dir, "dreamt_ibi", file_list=[filename],
                       multiplier=preprocessor.multiplier,
                       downsample_rate=preprocessor.downsample_rate,
                       task="sleep_staging", max_epochs=None).read_record(0)
    if not record["stored_complete_epochs"]:
        raise ValueError("Subject contains no complete 30-second epochs")
    stages, raw_stages = _check_stage_conditions(Path(preprocessor.data_dir) / filename, record,
                                                preprocessor.downsample_rate)
    gaps = np.flatnonzero(~record["adjacent"])
    if gaps.size:
        raise ValueError("Known source-time gaps after epoch indices "
                         + repr(gaps[:20].tolist()) + "; a continuous whole night cannot be generated")

    row = preprocessor.metadata.loc[sid]
    continuous = row[list(CONTINUOUS)].to_numpy(dtype=np.float64)
    gender = str(row["GENDER"]).strip().upper()
    if (not np.isfinite(continuous).all() or continuous[0] < 0
            or np.any(continuous[1:] <= 0) or gender not in ("F", "M")):
        raise ValueError(f"Invalid participant conditions for {sid}")
    if record["ahi"] is not None and not np.isclose(record["ahi"], continuous[0], rtol=1e-6, atol=1e-7):
        raise ValueError(f"NPZ and participant AHI disagree: {filename}")
    scaler = preprocessor.scaler
    if list(scaler["continuous_columns"]) != list(CONTINUOUS):
        raise ValueError("Training scaler has an incompatible continuous-condition order")
    meta_mean = np.asarray(scaler["continuous_mean"], dtype=np.float64)
    meta_std = np.asarray(scaler["continuous_std"], dtype=np.float64)
    if (meta_mean.shape != (3,) or meta_std.shape != (3,)
            or not np.isfinite(meta_mean).all() or not np.isfinite(meta_std).all()
            or np.any(meta_std <= 0)):
        raise ValueError("Training metadata scaler must contain three finite means and positive standard deviations")

    samples = len(stages) * EPOCH_SAMPLES
    conditions = np.empty((samples, 9), dtype=np.float32)
    conditions[:, :4] = np.repeat(np.eye(4, dtype=np.float32)[stages], EPOCH_SAMPLES, axis=0)
    conditions[:, 4:7] = (continuous - meta_mean) / meta_std
    conditions[:, 7] = float(gender == "F")
    conditions[:, 8] = float(gender == "M")
    if not np.isfinite(conditions).all():
        raise ValueError("Participant conditions cannot be represented as finite float32 values")
    reference = record["ibi_epochs"].astype(np.float64).reshape(-1)
    finite = np.isfinite(reference)
    provenance = dict(
        subject_role=role, purpose="conditional_generation_reference",
        scaler_refitted_for_generation=False,
        subject_in_original_training_scaler=sid in preprocessor.scaler_subjects,
        condition_channels_exclude_reference_ibi=True,
        condition_columns=["stage_" + value for value in STAGE_NAMES]
                          + list(CONTINUOUS) + ["gender_F", "gender_M"],
        training_max_epochs=preprocessor.max_epochs, generation_max_epochs=None,
        stored_complete_epochs=int(record["stored_complete_epochs"]),
        target_samples=samples, coverage="all_complete_stored_epochs",
        discarded_tail_samples=int(record["discarded_tail_samples"]),
        discarded_tail_seconds=record["discarded_tail_samples"] / TARGET_FS,
        continuity_source=record["continuity_source"],
        source_time_continuity_verified=record["continuity_source"] != "stored_order_only",
        reference_nonfinite_samples=int((~finite).sum()),
        reference_negative_samples=int((finite & (reference < 0)).sum()),
        reference_valid_epochs=int(record["valid_epochs"].sum()),
        source_file_sha256=record["file_sha256"], source_fs=record["source_fs"],
        source_multiplier=preprocessor.multiplier, source_downsample_rate=preprocessor.downsample_rate,
        output_fs=TARGET_FS, output_units="seconds", output_multiplier=1.0, output_downsample_rate=1,
        raw_stage_export_policy="copy source raw labels: W=0, N1=1, N2=2, Deep=3, REM=4; model merges N1/N2 as Light",
    )
    return dict(conditions=conditions, reference_seconds=reference, stage_epochs_four=stages,
                raw_stage_epochs=raw_stages, ahi=float(continuous[0]),
                subject_id=sid, filename=filename, record=record, provenance=provenance)


def save_night_npz(path, prepared, generated_seconds):
    """Save canonical 25 Hz seconds and sample-level raw labels for SSDataset.

    Generated values are never clipped or repaired. Invalid values fail before
    creating the file. Read this output with multiplier=1 and downsample_rate=1.
    """
    path = Path(path)
    if path.suffix.casefold() != ".npz":
        raise ValueError("Generated night output must have an .npz suffix")
    generated = np.asarray(generated_seconds)
    expected = len(prepared["conditions"])
    if generated.ndim != 1 or generated.shape != (expected,) or generated.dtype.kind != "f":
        raise ValueError(f"generated_seconds must be a floating-point vector of length {expected}")
    if not np.isfinite(generated).all() or np.any(generated < 0):
        raise ValueError("Generated IBI must be finite and nonnegative; values are not clipped")
    with np.errstate(over="ignore", invalid="ignore"):
        stored = generated.astype(np.float32)
    if not np.isfinite(stored).all():
        raise ValueError("Generated IBI cannot be represented as finite float32 values")
    raw_stages = np.repeat(prepared["raw_stage_epochs"], EPOCH_SAMPLES)
    if raw_stages.shape != stored.shape:
        raise ValueError("Generated IBI and stage conditions differ in length")
    arrays = dict(data=stored[:, None], stages=raw_stages, fs=np.array(TARGET_FS),
                  ahi=np.array(prepared["ahi"], dtype=np.float64), synthetic=np.array(True),
                  stage_mapping=np.array(json.dumps(dict(W=0, N1=1, N2=2, REM=4, Deep=3), sort_keys=True)))
    for key in ("epoch_index", "epoch_start_seconds"):
        if prepared["record"][key] is not None:
            arrays[key] = prepared["record"][key]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return dict(path=str(path.resolve()), samples=expected, epochs=len(prepared["raw_stage_epochs"]),
                sample_rate_hz=TARGET_FS, units="seconds", zero_samples=int((stored == 0).sum()),
                loader_multiplier=1.0, loader_downsample_rate=1)
