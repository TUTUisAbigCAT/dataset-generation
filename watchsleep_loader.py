"""DREAMT IBI loading aligned with WatchSleepNet's SSDataset.

The source loader flattens the signal, applies an integer sample stride, and
cuts complete 30-second epochs. This adapter keeps that order and its tuple
API, while exposing invalid epochs and optional source-time gaps separately.
It never fills gaps, removes invalid epochs, or reconstructs deleted time.
"""
from __future__ import annotations

import hashlib
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


UPSTREAM_SOURCE_PATH = Path(__file__).resolve().parent / "third_party/WatchSleepNet/data_setup.py"
UPSTREAM_SOURCE_SHA256 = "47936ea03412afcacb98121c3239cf19a240ee0b700b14381a19c93cbd87504e"
WATCHSLEEPNET_SOURCE_SHA256 = UPSTREAM_SOURCE_SHA256
TARGET_FS = 25
EPOCH_SECONDS = 30
EPOCH_SAMPLES = TARGET_FS * EPOCH_SECONDS
STAGE_NAMES = ("W", "Light", "REM", "Deep")
STAGE_MAPPING = {0: 0, 1: 1, 2: 1, 3: 3, 4: 2}
TIMESTAMP_ADJACENCY_ATOL_SECONDS = 1e-4


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_number(value, name):
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be one real numeric scalar")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _source_axis(saved, key, epochs, *, integer):
    """Validate an optional axis for complete epochs, before max_epochs limits."""
    if key not in saved:
        return None
    value = np.asarray(saved[key])
    if value.ndim != 1 or value.shape != (epochs,) or value.dtype.kind not in "biuf":
        raise ValueError(f"{key} must be a numeric vector with one value per complete epoch ({epochs})")
    if not np.isfinite(value).all():
        raise ValueError(f"{key} contains a nonfinite value")
    if integer:
        if np.any(value != np.round(value)):
            raise ValueError("epoch_index values must be integers")
        # Python integers preserve already-integer indices beyond float64's
        # exact range, and let us check bounds without an overflowing cast.
        integers = [int(item) for item in value]
        if any(item < -(2 ** 63) or item >= 2 ** 63 for item in integers):
            raise ValueError("epoch_index lies outside int64 range")
        result = np.asarray(integers, dtype=np.int64)
    else:
        result = value.astype(np.float64)
    if len(result) > 1 and np.any(result[1:] <= result[:-1]):
        raise ValueError(f"{key} must be strictly increasing; duplicate/reversed epochs are not reordered")
    return result


class SSDataset(Dataset):
    """Load one complete stored record per item, with original tuple semantics.

    ``dataset`` must identify DREAMT. ``downsample_rate`` is an integer stride,
    not a resampling algorithm, and fs / downsample_rate must equal 25 Hz.
    Default ``max_epochs=None`` retains the complete stored night; explicitly
    passing 1100 reproduces the original loader's length cap.

    ``read_record`` returns epoch arrays without removing invalid positions.
    ``valid_epochs`` tests finite nonnegative float32 IBI, a known mapped stage,
    and (for point-level annotations) one raw stage throughout each epoch.
    Zero values are retained as stored missing-value markers, including wholly
    zero epochs, as in the original loader. Validity here is structural and does
    not claim that every stored sample is a measured physiological IBI.
    ``adjacent`` describes source-time continuity separately from validity.
    When original time is absent, it is True for neighboring stored epochs but
    ``continuity_source='stored_order_only'`` explicitly limits that claim.
    """

    def __init__(self, dir, dataset, file_list=None, multiplier=1.0,
                 downsample_rate=1, task="sleep_staging", return_file_name=False,
                 max_epochs=None):
        self.dir = Path(dir)
        self.dataset = dataset
        if "dreamt" not in str(dataset).lower():
            raise ValueError("This adapter supports DREAMT-format records only")
        if not isinstance(multiplier, Real) or isinstance(multiplier, (bool, np.bool_)) \
                or not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("multiplier must be finite and positive")
        if not isinstance(downsample_rate, Integral) or isinstance(downsample_rate, (bool, np.bool_)) \
                or downsample_rate <= 0:
            raise ValueError("downsample_rate must be a positive integer stride")
        if max_epochs is not None and (not isinstance(max_epochs, Integral)
                or isinstance(max_epochs, (bool, np.bool_)) or max_epochs <= 0):
            raise ValueError("max_epochs must be None or a positive integer")
        if task not in ("sleep_staging", "sleep_wake"):
            raise ValueError("task must be sleep_staging or sleep_wake")
        self.multiplier = float(multiplier)
        self.downsample_rate = int(downsample_rate)
        self.task = task
        self.return_file_name = bool(return_file_name)
        self.max_epochs = None if max_epochs is None else int(max_epochs)
        self.files = sorted(self.dir.glob("*.npz")) if file_list is None else [self.dir / name for name in file_list]

    def __len__(self):
        return len(self.files)

    @staticmethod
    def remap_labels(labels, task):
        """Original four-class mapping, with all unknown values kept as -1."""
        if task not in ("sleep_staging", "sleep_wake"):
            raise ValueError("Unknown sleep task")
        values = np.asarray(labels)
        result = np.full(values.shape, -1, dtype=np.int64)
        if values.dtype.kind not in "biuf":
            return result
        mapping = STAGE_MAPPING if task == "sleep_staging" else {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
        for source, target in mapping.items():
            result[values == source] = target
        return result

    def read_record(self, idx):
        """Return arrays and provenance for a stored whole-night record.

        Required keys: data, stages, fs. AHI may be absent/invalid; this API
        returns None so the caller can consult participant_info.csv. Only the
        legacy tuple API falls back to AHI=0.

        Labels are preferably sample-level, matching the original loader. An
        explicit epoch-level branch accepts one label per complete epoch (or an
        extra label for an incomplete tail). Optional epoch_index and
        epoch_start_seconds must describe exactly the complete epochs before
        max_epochs truncation. ``discarded_tail_samples`` counts 25 Hz samples,
        not original-rate samples or epochs excluded by max_epochs.
        """
        file_path = self.files[idx]
        with np.load(file_path, allow_pickle=False) as saved:
            fs = _scalar_number(saved["fs"], "fs")
            if fs <= 0 or not np.isclose(fs / self.downsample_rate, TARGET_FS, rtol=0.0, atol=1e-10):
                raise ValueError("fs / downsample_rate must equal 25 Hz; interpolation is not performed")
            signal = np.asarray(saved["data"])
            if signal.dtype.kind not in "biuf":
                raise ValueError("data must contain a real numeric IBI series")
            signal = signal.flatten()
            original_samples = signal.size
            # Preserve the original loader's flatten -> multiplier -> stride.
            with np.errstate(over="ignore", invalid="ignore"):
                downsampled = (signal * self.multiplier)[::self.downsample_rate]
                ibi = np.asarray(downsampled, dtype=np.float32)
            full_epochs, tail = divmod(len(ibi), EPOCH_SAMPLES)
            ibi_epochs = ibi[:full_epochs * EPOCH_SAMPLES].reshape(full_epochs, EPOCH_SAMPLES).copy()
            labels = np.asarray(saved["stages"]).reshape(-1)
            if labels.size == original_samples:
                sampled_labels = labels[::self.downsample_rate][:full_epochs * EPOCH_SAMPLES]
                label_epochs = sampled_labels.reshape(full_epochs, EPOCH_SAMPLES)
                epoch_first = label_epochs[:, 0]
                stage_consistent = np.all(label_epochs == label_epochs[:, :1], axis=1)
                label_granularity = "sample_level"
            elif labels.size == full_epochs or (tail > 0 and labels.size == full_epochs + 1):
                epoch_first = labels[:full_epochs]
                stage_consistent = np.ones(full_epochs, dtype=bool)
                label_granularity = "epoch_level"
            else:
                raise ValueError("stages must match original signal samples or complete 30-second epochs")
            stage_epochs = self.remap_labels(epoch_first, self.task)
            valid_epochs = (np.isfinite(ibi_epochs).all(axis=1) & (ibi_epochs >= 0).all(axis=1)
                            & (stage_epochs >= 0) & stage_consistent)
            epoch_index = _source_axis(saved, "epoch_index", full_epochs, integer=True)
            epoch_start_seconds = _source_axis(saved, "epoch_start_seconds", full_epochs, integer=False)
            adjacent = np.ones(max(0, full_epochs - 1), dtype=bool)
            if epoch_index is not None:
                adjacent &= np.fromiter((int(right) - int(left) == 1 for left, right in
                                         zip(epoch_index[:-1], epoch_index[1:])), dtype=bool,
                                        count=max(0, full_epochs - 1))
            if epoch_start_seconds is not None:
                adjacent &= np.isclose(np.diff(epoch_start_seconds), EPOCH_SECONDS,
                                       rtol=0.0, atol=TIMESTAMP_ADJACENCY_ATOL_SECONDS)
            if epoch_index is not None and epoch_start_seconds is not None:
                continuity_source = "epoch_index_and_epoch_start_seconds"
            elif epoch_index is not None:
                continuity_source = "epoch_index"
            elif epoch_start_seconds is not None:
                continuity_source = "epoch_start_seconds"
            else:
                continuity_source = "stored_order_only"
            ahi = None
            synthetic = None
            if "synthetic" in saved:
                flag = np.asarray(saved["synthetic"])
                if flag.shape == () and flag.dtype.kind == "b":
                    synthetic = bool(flag.item())
            if "ahi" in saved:
                try:
                    candidate = _scalar_number(saved["ahi"], "ahi")
                    if candidate >= 0:
                        ahi = candidate
                except (TypeError, ValueError, OverflowError):
                    pass
        kept = full_epochs if self.max_epochs is None else min(full_epochs, self.max_epochs)
        zero_mask = ibi_epochs[:kept] == 0
        return {"ibi_epochs": ibi_epochs[:kept], "stage_epochs": stage_epochs[:kept],
                "valid_epochs": valid_epochs[:kept], "adjacent": adjacent[:max(0, kept - 1)],
                "zero_samples": int(zero_mask.sum()),
                "zero_epochs": int(zero_mask.all(axis=1).sum()),
                "epochs_with_zero": int(zero_mask.any(axis=1).sum()),
                "ahi": ahi, "synthetic": synthetic, "subject_id": file_path.stem, "file_path": file_path,
                "file_sha256": _sha256(file_path), "continuity_source": continuity_source,
                "epoch_index": None if epoch_index is None else epoch_index[:kept],
                "epoch_start_seconds": None if epoch_start_seconds is None else epoch_start_seconds[:kept],
                "discarded_tail_samples": int(tail), "discarded_by_max_epochs": full_epochs - kept,
                "stored_complete_epochs": full_epochs, "source_fs": fs, "effective_fs": TARGET_FS,
                "downsample_rate": self.downsample_rate, "multiplier": self.multiplier,
                "label_granularity": label_granularity, "source_loader_sha256": UPSTREAM_SOURCE_SHA256}

    def __getitem__(self, idx):
        record = self.read_record(idx)
        result = (torch.from_numpy(record["ibi_epochs"]).float(),
                  torch.from_numpy(record["stage_epochs"]).long(),
                  len(record["stage_epochs"]), 0.0 if record["ahi"] is None else record["ahi"])
        return result + (record["subject_id"],) if self.return_file_name else result

    @staticmethod
    def collate_fn(batch):
        """Pad variable whole nights; retain the original AHI tuple return."""
        if not batch or len(batch[0]) not in (4, 5) or any(len(item) != len(batch[0]) for item in batch):
            raise ValueError("Expected a nonempty homogeneous batch of four- or five-element SSDataset tuples")
        ibis, labels, lengths, ahis = zip(*(item[:4] for item in batch))
        result = (pad_sequence(ibis, batch_first=True, padding_value=0),
                  pad_sequence(labels, batch_first=True, padding_value=-1),
                  torch.tensor(lengths, dtype=torch.long), ahis)
        return result + (tuple(item[4] for item in batch),) if len(batch[0]) == 5 else result
