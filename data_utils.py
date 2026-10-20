"""WatchSleepNet epoch loading adapted to conditional WaveStitch windows.

Only data handling changes: four original sleep classes, configurable 30-second epochs,
participant conditions, and training-only population scaling. Optional subject
splits use a local NumPy generator and never consume global training RNG state.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from watchsleep_loader import SSDataset
from window_config import FS, EPOCH_SAMPLES, window_settings

ROOT = Path(__file__).resolve().parent
datasets = {"DREAMT_SIM": None, "DREAMT": None}
STAGE_NAMES = ("W", "Light", "REM", "Deep")
CONTINUOUS = ("AHI", "BMI", "AGE")
COLUMNS = ("IBI",) + tuple("stage_" + name for name in STAGE_NAMES) + CONTINUOUS + ("gender_F", "gender_M")
ZERO_VALUE_POLICY = "preserve stored zero missing-value markers in scaler and unmodified MSE; no imputation"


class CyclicEncoder:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DREAMT stages use four-class one-hot encoding")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve_path(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _window_valid(record, first, count):
    last = first + count
    return (first >= 0 and last <= len(record["valid_epochs"])
            and bool(record["valid_epochs"][first:last].all())
            and bool(record["adjacent"][first:last - 1].all()))


class WindowDataset(Dataset):
    """Configured consecutive epochs, excluding invalid labels and temporal gaps."""
    def __init__(self, preprocessor, split, window_size, stride):
        if split not in ("train", "val"):
            raise ValueError("Only train and validation are available; test is sealed")
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size != preprocessor.window_size:
            raise ValueError("Window size must match configured window_epochs * epoch_samples")
        if stride <= 0 or stride % EPOCH_SAMPLES:
            raise ValueError("Stride must be a positive whole number of 30-second epochs")
        self.preprocessor, self.split = preprocessor, split
        self.window_size, self.stride = int(window_size), int(stride)
        self.window_index, self.excluded_windows = [], 0
        for filename in preprocessor.splits[split]:
            record = preprocessor.record(filename)
            for start in range(0, len(record["ibi"]) - window_size + 1, stride):
                if _window_valid(record, start // EPOCH_SAMPLES, preprocessor.window_epochs):
                    self.window_index.append((filename, start))
                else:
                    self.excluded_windows += 1
        if not self.window_index:
            raise ValueError("No complete valid configured-length windows in selected split")

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, index):
        filename, start = self.window_index[index]
        return torch.from_numpy(self.preprocessor.window_array(filename, start, self.window_size))


class Preprocessor:
    def __init__(self, dataset, propCycEnc=False, *, data_dir=None, metadata_path=None,
                 split_path=None, multiplier=None, downsample_rate=None, max_epochs=None,
                 split_seed=None, generate_splits=None, config_override=None, data_config=None, window_epochs=None):
        if dataset not in datasets:
            raise ValueError("Expected dataset DREAMT or DREAMT_SIM")
        if propCycEnc:
            raise ValueError("Cyclic encoding is not defined for unordered sleep stages")
        if config_override is not None and data_config is not None:
            raise ValueError("Use only one of config_override and data_config")
        override = config_override if config_override is not None else data_config
        if override is not None:
            if not isinstance(override, dict):
                raise ValueError("config_override must be a dictionary")
            config, config_base, environment = dict(override), Path.cwd(), {}
        else:
            environment = os.environ
            selected_config = environment.get("DREAMT_CONFIG_PATH")
            config_path = (_resolve_path(selected_config, Path.cwd()) if selected_config
                           else ROOT / "dataset_config.json")
            if selected_config and not config_path.is_file():
                raise FileNotFoundError(f"Selected DREAMT configuration does not exist: {config_path}")
            config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else {}
            config_base = config_path.parent
        if not isinstance(config, dict):
            raise ValueError("DREAMT configuration must be a JSON object")

        def option(explicit, key, env_key, default=None):
            if explicit is not None:
                return explicit, Path.cwd()
            if env_key in environment:
                return environment[env_key], Path.cwd()
            return config.get(key, default), config_base

        self.dataset = dataset
        dimensions = window_settings(config if window_epochs is None else {"window_epochs": window_epochs})
        self.window_epochs = dimensions["window_epochs"]
        self.window_size = dimensions["window_size"]
        data_value, data_base = option(data_dir, "data_dir", "DREAMT_DATA_DIR")
        if not data_value:
            raise ValueError("Set data_dir in dataset_config.json, DREAMT_DATA_DIR, or the constructor; no dataset fallback is used")
        self.data_dir = _resolve_path(data_value, data_base)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"DREAMT data directory does not exist: {self.data_dir}")
        metadata_value, metadata_base = option(metadata_path, "metadata_path", "DREAMT_METADATA_PATH")
        if metadata_value:
            self.metadata_path = _resolve_path(metadata_value, metadata_base)
        else:
            candidates = (self.data_dir / "participant_info.csv", self.data_dir.parent / "participant_info.csv")
            self.metadata_path = next((path for path in candidates if path.is_file()), candidates[0])
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Missing participant_info.csv metadata with SID, AHI, BMI, AGE, GENDER: {self.metadata_path}")
        self.metadata_file_sha256 = file_sha256(self.metadata_path)
        self.multiplier = float(option(multiplier, "multiplier", "DREAMT_MULTIPLIER", 1.)[0])
        downsample_value = option(downsample_rate, "downsample_rate", "DREAMT_DOWNSAMPLE_RATE", 1)[0]
        if isinstance(downsample_value, bool) or not float(downsample_value).is_integer() or float(downsample_value) < 1:
            raise ValueError("downsample_rate must be a positive integer stride")
        self.downsample_rate = int(downsample_value)
        if not np.isfinite(self.multiplier) or self.multiplier <= 0:
            raise ValueError("multiplier must be finite and positive")
        self.max_epochs = max_epochs if max_epochs is not None else config.get("max_epochs")
        seed_value = option(split_seed, "split_seed", "DREAMT_SPLIT_SEED", 42)[0]
        if isinstance(seed_value, bool) or not float(seed_value).is_integer() or float(seed_value) < 0:
            raise ValueError("split_seed must be a nonnegative integer")
        self.split_seed = int(seed_value)
        if generate_splits is None:
            generate_splits = config.get("generate_splits")
        if generate_splits is not None and not isinstance(generate_splits, bool):
            raise ValueError("generate_splits must be true, false, or null")
        split_value, split_base = option(split_path, "split_path", "DREAMT_SPLIT_PATH")
        candidate_split = (_resolve_path(split_value, split_base) if split_value
                           else self.data_dir / "splits.json")
        filenames = sorted(path.name for path in self.data_dir.glob("*.npz") if path.is_file())
        if generate_splits is True or (generate_splits is None and not split_value and not candidate_split.exists()):
            if len(filenames) < 3:
                raise ValueError("At least three subjects are required to generate train/val/test splits")
            permutation = np.random.default_rng(self.split_seed).permutation(filenames).tolist()
            n_train = min(max(1, int(.75 * len(filenames))), len(filenames) - 2)
            n_val = min(max(1, int(.125 * len(filenames))), len(filenames) - n_train - 1)
            self.splits = dict(train=sorted(permutation[:n_train]),
                               val=sorted(permutation[n_train:n_train + n_val]),
                               test=sorted(permutation[n_train + n_val:]))
            self.split_source, self.split_path, self.split_file_sha256 = "generated_subject_split", None, None
        else:
            self.split_path = candidate_split
            self.splits = json.loads(self.split_path.read_text(encoding="utf-8-sig"))
            self.split_source, self.split_file_sha256 = "split_file", file_sha256(self.split_path)
        if not isinstance(self.splits, dict):
            raise ValueError("splits.json must contain a JSON object")
        if set(self.splits) != {"train", "val", "test"} or any(not isinstance(v, list) for v in self.splits.values()):
            raise ValueError("splits.json must contain train/val/test filename lists")
        flat = [name for names in self.splits.values() for name in names]
        if not self.splits["train"] or any(not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz") for name in flat):
            raise ValueError("Expected nonempty train and local NPZ filenames")
        if len(set(flat)) != len(flat) or set(flat) != set(filenames):
            raise ValueError("Subject splits must assign each NPZ exactly once without overlap")
        if len({Path(name).stem for name in flat}) != len(flat):
            raise ValueError("Each NPZ must identify one unique participant SID")
        self.split_fingerprint = _fingerprint(dict(filenames=filenames, seed=self.split_seed,
            source=self.split_source, policy="subject_75_12.5_rest_minimum_one_v1" if self.split_source == "generated_subject_split" else "explicit_file",
            splits=self.splits))
        self.columns = pd.Index(COLUMNS)
        self.schema_frame = pd.DataFrame(columns=self.columns, dtype=np.float32)
        self.hierarchical_features_cyclic = list(COLUMNS[1:])
        self.hierarchical_features_uncyclic = list(COLUMNS[1:])
        metadata = pd.read_csv(self.metadata_path, dtype={"SID": str})
        if not {"SID", "AHI", "BMI", "AGE", "GENDER"}.issubset(metadata.columns):
            raise ValueError("participant_info.csv requires SID, AHI, BMI, AGE, GENDER")
        metadata["SID"] = metadata["SID"].str.strip()
        if metadata["SID"].isna().any() or metadata["SID"].eq("").any() or metadata["SID"].duplicated().any():
            raise ValueError("Participant SID must be present and unique")
        self.metadata = metadata.set_index("SID")
        # Validate conditions for all partitions before opening any signal file.
        # Reading the CSV is permitted; held-out test NPZ contents remain sealed.
        self._conditions = {}
        for filename in flat:
            sid = Path(filename).stem
            if sid not in self.metadata.index:
                raise ValueError(f"Missing participant_info.csv metadata for {sid}")
            row = self.metadata.loc[sid]
            try:
                continuous = row[list(CONTINUOUS)].to_numpy(dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Incomplete or invalid participant conditions for {sid}") from error
            gender = str(row["GENDER"]).strip().upper()
            if not np.isfinite(continuous).all() or continuous[0] < 0 or np.any(continuous[1:] <= 0) or gender not in ("F", "M"):
                raise ValueError(f"Incomplete or invalid participant conditions for {sid}")
            self._conditions[sid] = (continuous, gender)
        self._records, self.accessed_files = {}, []
        self.validation_selection = None
        total, summed, squares, values = 0, 0., 0., []
        signal_minimum, signal_maximum = np.inf, -np.inf
        self.scaler_subjects = []
        for filename in self.splits["train"]:
            record = self.record(filename)
            signal = record["ibi"].reshape(-1, EPOCH_SAMPLES)[record["valid_epochs"]].astype(np.float64)
            if not signal.size:
                continue
            total += signal.size
            summed += float(signal.sum())
            squares += float(np.square(signal).sum())
            signal_minimum = min(signal_minimum, float(signal.min()))
            signal_maximum = max(signal_maximum, float(signal.max()))
            values.append(record["continuous"])
            self.scaler_subjects.append(record["subject_id"])
        if not total:
            raise ValueError("Training subjects contain no valid IBI epochs")
        mean = summed / total
        std = float(np.sqrt(max(0., squares / total - mean * mean)))
        meta = np.asarray(values, dtype=np.float64)
        meta_mean, meta_raw_std = meta.mean(axis=0), meta.std(axis=0)
        constant = np.ptp(meta, axis=0) == 0
        meta_raw_std[constant] = 0.
        meta_mean[constant] = meta[0, constant]
        meta_std = np.where(constant, 1., meta_raw_std)
        if std <= 0 or signal_minimum == signal_maximum:
            raise ValueError("Training IBI signal must have nonzero variance")
        self.scaler = dict(signal_mean_seconds=mean, signal_std_seconds=std,
            continuous_columns=list(CONTINUOUS), continuous_mean=meta_mean.tolist(),
            continuous_std=meta_std.tolist(), continuous_raw_std=meta_raw_std.tolist(),
            constant_columns=[column for column, flag in zip(CONTINUOUS, constant) if flag], signal_fit_samples=total,
            metadata_fit_subjects=len(values), fit_split="train",
            continuous_transform="raw values, population z-score; no logarithm",
            signal_fit_policy="structurally valid training epochs, including stored zero missing-value markers; no interpolation or clipping")
        self._meta_mean, self._meta_std = meta_mean, meta_std

    def record(self, filename):
        if filename not in self.splits["train"] + self.splits["val"]:
            raise ValueError("Only train/validation records may be opened; test is sealed")
        if filename in self._records:
            return self._records[filename]
        sid = Path(filename).stem
        if sid not in self.metadata.index:
            raise ValueError(f"Missing metadata for {sid}")
        continuous, gender = self._conditions[sid]
        loaded = SSDataset(self.data_dir, "dreamt_ibi", file_list=[filename],
            multiplier=self.multiplier, downsample_rate=self.downsample_rate,
            task="sleep_staging", max_epochs=self.max_epochs).read_record(0)
        if loaded["ahi"] is not None and not np.isclose(loaded["ahi"], continuous[0], rtol=1e-6, atol=1e-7):
            raise ValueError(f"NPZ and participant AHI disagree: {filename}")
        record = dict(loaded)
        record.update(subject_id=sid, ibi=loaded["ibi_epochs"].reshape(-1),
            stages=np.repeat(loaded["stage_epochs"], EPOCH_SAMPLES).astype(np.int8),
            continuous=continuous, gender=gender,
            ahi_source="participant_info.csv" if loaded["ahi"] is None else "NPZ verified against participant_info.csv")
        self._records[filename] = record
        self.accessed_files.append(filename)
        return record

    def window_array(self, filename, start, length):
        record = self.record(filename)
        if start < 0 or length <= 0 or start % EPOCH_SAMPLES or length % EPOCH_SAMPLES or start + length > len(record["ibi"]):
            raise ValueError("Window must align with complete epochs within one subject")
        if not _window_valid(record, start // EPOCH_SAMPLES, length // EPOCH_SAMPLES):
            raise ValueError("Window crosses an invalid epoch or known time gap")
        end = start + length
        array = np.empty((length, len(COLUMNS)), dtype=np.float32)
        array[:, 0] = (record["ibi"][start:end].astype(np.float64) - self.scaler["signal_mean_seconds"]) / self.scaler["signal_std_seconds"]
        array[:, 1:5] = np.eye(4, dtype=np.float32)[record["stages"][start:end]]
        array[:, 5:8] = (record["continuous"] - self._meta_mean) / self._meta_std
        array[:, 8] = float(record["gender"] == "F")
        array[:, 9] = float(record["gender"] == "M")
        return array

    def make_training_dataset(self, window_size, stride):
        return WindowDataset(self, "train", window_size, stride)

    def make_validation_dataset(self, window_size=None, stride=1500):
        return WindowDataset(self, "val", self.window_size if window_size is None else window_size, stride)

    def select_validation_windows(self, windows_per_subject=2):
        """Select legal windows without assuming cohort size or recording length.

        The first legal window and the distinct window nearest 30 minutes are
        selected per subject. Requests beyond two add remaining windows in
        increasing start order; no candidate is ever repeated.
        """
        if isinstance(windows_per_subject, bool) or int(windows_per_subject) != windows_per_subject or windows_per_subject < 1:
            raise ValueError("windows_per_subject must be a positive integer")
        windows_per_subject = int(windows_per_subject)
        selected, excluded, subjects = [], [], []
        stride = min(2 * EPOCH_SAMPLES, self.window_size)
        for filename in self.splits["val"]:
            record = self.record(filename)
            candidates = [start for start in range(0, len(record["ibi"]) - self.window_size + 1, stride)
                          if _window_valid(record, start // EPOCH_SAMPLES, self.window_epochs)]
            if not candidates:
                excluded.append(dict(filename=filename, subject_id=Path(filename).stem,
                                     reason="no complete valid configured-length window"))
                continue
            starts = [candidates[0]]
            remaining = candidates[1:]
            if windows_per_subject > 1 and remaining:
                starts.append(min(remaining, key=lambda start: (abs(start - 45000), start)))
            if windows_per_subject > 2:
                starts.extend(start for start in candidates if start not in starts)
            starts = starts[:windows_per_subject]
            selected.extend((filename, start) for start in starts)
            subjects.append(dict(filename=filename, candidates=len(candidates), selected_starts=starts))
        self.validation_selection = dict(window_size=self.window_size, stride=stride,
            windows_per_subject=int(windows_per_subject), preferred_second_start=45000,
            policy="first legal; distinct nearest 45000; additional starts in order",
            selected=[dict(filename=name, start=start) for name, start in selected],
            subjects=subjects, excluded=excluded)
        if not selected:
            raise ValueError("No complete valid configured-length windows across validation subjects")
        return selected

    def inverse_signal(self, standardized):
        return np.asarray(standardized) * self.scaler["signal_std_seconds"] + self.scaler["signal_mean_seconds"]

    def data_identity(self):
        """Frozen training inputs; opening validation records cannot change it."""
        train_hashes = {name: self._records[name]["file_sha256"] for name in self.splits["train"]}
        return _json_copy(dict(schema="dreamt_data_identity_v1", schema_version=1,
            dataset=self.dataset, data_directory=str(self.data_dir),
            metadata_path=str(self.metadata_path), metadata_file_sha256=self.metadata_file_sha256,
            split_path=str(self.split_path) if self.split_path is not None else None,
            split_source=self.split_source, split_file_sha256=self.split_file_sha256,
            split_seed=self.split_seed, split_fingerprint=self.split_fingerprint, splits=self.splits,
            sample_rate_hz=FS, epoch_samples=EPOCH_SAMPLES, window_epochs=self.window_epochs,
            multiplier=self.multiplier, downsample_rate=self.downsample_rate, max_epochs=self.max_epochs,
            columns=list(COLUMNS), stage_names=list(STAGE_NAMES),
            zero_value_policy=ZERO_VALUE_POLICY,
            raw_stage_mapping={"0": 0, "1": 1, "2": 1, "3": 3, "4": 2, "other": -1},
            scaler=self.scaler, scaler_subjects=self.scaler_subjects,
            train_npz_sha256=train_hashes, training_npz_sha256=train_hashes))

    @classmethod
    def from_identity(cls, identity, dataset="DREAMT"):
        """Reconstruct training inputs without consulting environment/config.

        A generated split stays generated even if splits.json appears later.
        Every identity field is verified after refitting the training scaler;
        changed metadata, training files, partitions or schema fail explicitly.
        """
        if not isinstance(identity, dict) or identity.get("schema") != "dreamt_data_identity_v1" or identity.get("schema_version") != 1:
            raise ValueError("Expected dreamt_data_identity_v1 training identity")
        required = ("data_directory", "metadata_path", "metadata_file_sha256", "split_path",
                    "split_source", "split_seed", "split_fingerprint", "splits", "scaler",
                    "train_npz_sha256", "multiplier", "downsample_rate", "max_epochs")
        missing = [key for key in required if key not in identity]
        if missing:
            raise ValueError(f"Training data identity is missing fields: {missing}")
        if identity["split_source"] not in ("generated_subject_split", "split_file"):
            raise ValueError("Unknown training split source")
        generated = identity["split_source"] == "generated_subject_split"
        if (generated and identity["split_path"] is not None) or (not generated and not identity["split_path"]):
            raise ValueError("Training split source and split path disagree")
        for key in ("data_directory", "metadata_path") + (() if generated else ("split_path",)):
            if not Path(identity[key]).is_absolute():
                raise ValueError(f"Training data identity must record absolute {key}")
        processor = cls(identity.get("dataset", dataset), data_dir=identity["data_directory"],
            metadata_path=identity["metadata_path"], split_path=identity["split_path"],
            multiplier=identity["multiplier"], downsample_rate=identity["downsample_rate"],
            max_epochs=identity["max_epochs"], split_seed=identity["split_seed"],
            generate_splits=generated, config_override={}, window_epochs=identity.get("window_epochs", 4))
        actual = processor.data_identity()
        expected = dict(identity)
        expected.setdefault("window_epochs", 4)  # Legacy manifests used four sleep epochs.
        mismatches = [key for key in actual if key not in expected or actual[key] != expected[key]]
        if mismatches:
            raise ValueError(f"Training data identity mismatch: {', '.join(mismatches)}")
        return processor

    def provenance(self):
        flags = [self._records[name].get("synthetic") for name in self.splits["train"]]
        synthetic = True if all(flag is True for flag in flags) else (False if all(flag is False for flag in flags) else None)
        return dict(dataset=self.dataset, data_directory=str(self.data_dir.resolve()),
            data_synthetic=synthetic,
            columns=list(COLUMNS), non_hierarchical_columns=["IBI"],
            hierarchical_columns=self.hierarchical_features_cyclic,
            sample_rate_hz=FS, epoch_samples=EPOCH_SAMPLES, window_epochs=self.window_epochs, window_size=self.window_size, stage_names=list(STAGE_NAMES),
            zero_value_policy=ZERO_VALUE_POLICY,
            raw_stage_mapping={"0": 0, "1": 1, "2": 1, "3": 3, "4": 2, "other": -1},
            multiplier=self.multiplier, downsample_rate=self.downsample_rate,
            max_epochs=self.max_epochs, scaler=self.scaler, scaler_subjects=self.scaler_subjects,
            splits=self.splits, split_source=self.split_source,
            split_path=str(self.split_path) if self.split_path is not None else None,
            split_file_sha256=self.split_file_sha256, split_seed=self.split_seed,
            split_fingerprint=self.split_fingerprint, metadata_path=str(self.metadata_path),
            metadata_file_sha256=self.metadata_file_sha256,
            validation_selection=_json_copy(self.validation_selection),
            data_identity=self.data_identity(),
            opened_npz_sha256={name: self._records[name]["file_sha256"] for name in self.accessed_files},
            record_alignment={name: dict(continuity_source=self._records[name]["continuity_source"],
                invalid_epochs=int((~self._records[name]["valid_epochs"]).sum()),
                known_gaps=int((~self._records[name]["adjacent"]).sum()),
                zero_samples=self._records[name]["zero_samples"],
                zero_epochs=self._records[name]["zero_epochs"],
                epochs_with_zero=self._records[name]["epochs_with_zero"],
                discarded_tail_samples=self._records[name]["discarded_tail_samples"])
                for name in self.accessed_files},
            test_npz_opened=False, global_rng_used=False,
            loader_source_sha256=file_sha256(ROOT / "watchsleep_loader.py"),
            upstream_loader_sha256=file_sha256(ROOT / "third_party/WatchSleepNet/data_setup.py"))
