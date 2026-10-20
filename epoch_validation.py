"""Epoch-end distribution checks without changing the WaveStitch updates."""
from __future__ import annotations

from collections import defaultdict, deque
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from cuda_runtime import assert_cuda_model, require_cuda
from validation_metrics import DistributionAccumulator, preserve_random_state
from validation_sampling import sample_validation_windows

DEFAULT_VALIDATION = {"max_windows": None, "batch_size": 8, "seed": 20261020}
DEFAULT_EARLY_STOPPING = {
    "enabled": True, "patience": 10, "smoothing_window": 5,
    "min_delta_mean_ms": 0.0, "min_delta_variance_ms2": 0.0,
}
EARLY_STOP_METRICS = {
    "mean": ("mean_vector_l2_ms", "min_delta_mean_ms"),
    "variance": ("variance_vector_l2_ms2", "min_delta_variance_ms2"),
}


def early_stopping_options(config):
    values = config.get("early_stopping", {})
    if not isinstance(values, dict) or set(values) - set(DEFAULT_EARLY_STOPPING):
        raise ValueError("early_stopping must be an object with only supported options")
    options = {**DEFAULT_EARLY_STOPPING, **values}
    if not isinstance(options["enabled"], bool):
        raise ValueError("early_stopping.enabled must be boolean")
    for key in ("patience", "smoothing_window"):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"early_stopping.{key} must be a positive integer")
    for key in ("min_delta_mean_ms", "min_delta_variance_ms2"):
        value = options[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"early_stopping.{key} must be finite and nonnegative")
        options[key] = float(value)
    return options


class EarlyStopping:
    """Reset patience only when both full-window moving averages improve.

    Raw minima select individual checkpoints; joint checkpoints use smoothed
    improvements. Separate threshold anchors allow several
    small improvements to accumulate past min_delta, without resetting patience
    on numerical jitter. No mean/variance sum mixes their different units.
    """
    def __init__(self, config):
        self.options = early_stopping_options(config)
        self.best, self.anchors = {}, {}
        self.history = deque(maxlen=self.options["smoothing_window"])
        self.smoothed = {name: None for name in EARLY_STOP_METRICS}
        self.completed_epochs = self.bad_epochs = 0
        self.stopped_epoch = None

    def update(self, epoch, aggregate):
        if self.stopped_epoch is not None or epoch != self.completed_epochs + 1:
            raise ValueError("Early stopping requires consecutive, non-stopped epochs")
        values = {name: float(aggregate[key]) for name, (key, _) in EARLY_STOP_METRICS.items()}
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("Early-stopping metrics must be finite nonnegative absolute errors")
        self.history.append(values)
        ready = len(self.history) == self.options["smoothing_window"]
        if ready:
            self.smoothed = {name: math.fsum(row[name] for row in self.history) / len(self.history)
                             for name in EARLY_STOP_METRICS}
        improved, new_best = [], []
        for name, value in values.items():
            metric, delta_key = EARLY_STOP_METRICS[name]
            if name not in self.best or value < self.best[name]["value"]:
                self.best[name] = dict(epoch=epoch, metric=metric, value=value)
                new_best.append(name)
            if ready and (name not in self.anchors
                          or self.smoothed[name] < self.anchors[name] - self.options[delta_key]):
                self.anchors[name] = self.smoothed[name]
                improved.append(name)
        joint_improvement = len(improved) == len(EARLY_STOP_METRICS)
        if ready:
            self.bad_epochs = 0 if joint_improvement else self.bad_epochs + 1
        self.completed_epochs = epoch
        if self.options["enabled"] and self.bad_epochs >= self.options["patience"]:
            self.stopped_epoch = epoch
        return dict(**self.summary(), improved_metrics=improved, new_best_metrics=new_best,
                    joint_improvement=joint_improvement)

    def summary(self):
        return dict(options=self.options.copy(), improvement_rule="both_smoothed_metrics_same_epoch",
                    smoothing_method="simple_moving_average",
                    smoothing_ready=len(self.history) == self.options["smoothing_window"],
                    smoothed_metrics={EARLY_STOP_METRICS[name][0]: value for name, value in self.smoothed.items()},
                    bad_epochs=self.bad_epochs,
                    stopped_early=self.stopped_epoch is not None, stopped_epoch=self.stopped_epoch,
                    completed_epochs=self.completed_epochs, threshold_anchors=self.anchors.copy(),
                    best_metrics={name: value.copy() for name, value in self.best.items()})


def validation_options(config):
    values = config.get("validation", {})
    if not isinstance(values, dict):
        raise ValueError("validation must be a JSON object")
    unknown = set(values) - set(DEFAULT_VALIDATION)
    if unknown:
        raise ValueError(f"Unknown validation options: {sorted(unknown)}")
    result = {**DEFAULT_VALIDATION, **values}
    for key in ("max_windows", "batch_size", "seed"):
        value = result[key]
        if key == "max_windows" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"validation.{key} must be an integer")
        if (key == "seed" and not 0 <= value < 2**32) or (key != "seed" and value < 1):
            raise ValueError(f"Invalid validation.{key}: {value}")
    return result


def select_fixed_windows(preprocessor, options):
    """Greedy nonoverlapping candidates, then seeded subject-round-robin sampling.

    Search at every 30-second boundary so a missing first epoch cannot discard
    all legal windows in a short validation record. Selection never opens test.
    """
    dataset = preprocessor.make_validation_dataset(window_size=3000, stride=750)
    groups = defaultdict(list)
    end_by_subject = {}
    for name, start in dataset.window_index:
        if name not in preprocessor.splits["val"]:
            raise RuntimeError("Validation selection escaped the validation split")
        if start >= end_by_subject.get(name, 0):
            groups[name].append((name, start))
            end_by_subject[name] = start + 3000
    available = sum(map(len, groups.values()))
    if not available:
        raise ValueError("No complete valid two-minute validation windows")
    rng = np.random.default_rng(options["seed"])
    names = sorted(groups)
    rng.shuffle(names)
    for name in names:
        rng.shuffle(groups[name])
    count = available if options["max_windows"] is None else min(available, options["max_windows"])
    selected = []
    position = 0
    while len(selected) < count:
        for name in names:
            if position < len(groups[name]):
                selected.append(groups[name][position])
                if len(selected) == count:
                    break
        position += 1
    return sorted(selected), {
        "available_nonoverlapping_windows": available,
        "available_subjects": len(groups),
        "selected_windows": count,
        "selected_subjects": len({name for name, _ in selected}),
        "candidate_stride_samples": 750,
        "window_size_samples": 3000,
        "policy": "greedy nonoverlap per subject; seeded round-robin subjects and shuffled starts; sorted fixed output",
    }


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _require_device(device, model):
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for epoch validation; CPU fallback is disabled")
    selected = require_cuda()
    device = selected if device.index is None else device
    assert_cuda_model(model)
    if any(t.device != device for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("Training and validation must use the same CUDA device")
    return device


class EpochValidator:
    """Own a frozen validation set and an independent, fresh evaluation model."""
    def __init__(self, preprocessor, training_args, training_model, diffusion_config,
                 device, output_dir, config):
        self.device = _require_device(device, training_model)
        self.options = validation_options(config)
        self.early_stopping = EarlyStopping(config)
        self.best_checkpoints = {}
        self.should_stop = False
        self.args = training_args
        if training_args.window_size != 3000 or training_args.s4_lmax < training_args.window_size:
            raise ValueError("Validation requires a complete 3000-point window within the original S4 kernel length")
        self.diffusion_config = diffusion_config
        self.output = Path(output_dir).resolve()
        self.training_output = self.output.parent
        self.output.mkdir(parents=True, exist_ok=False)
        self.reports = []
        with preserve_random_state(self.device):
            selected, selection = select_fixed_windows(preprocessor, self.options)
            # Data loading is on the host; generation and all statistics use CUDA.
            self.windows = torch.stack([torch.from_numpy(preprocessor.window_array(name, start, 3000))
                                        for name, start in selected])
            self.reference = torch.stack([
                torch.from_numpy(np.asarray(preprocessor.record(name)["ibi"][start:start + 3000],
                                            dtype=np.float64).copy())
                for name, start in selected])
            self.scale = float(preprocessor.scaler["signal_std_seconds"])
            self.center = float(preprocessor.scaler["signal_mean_seconds"])
            if not math.isfinite(self.scale) or self.scale <= 0 or not math.isfinite(self.center):
                raise ValueError("Invalid training-only IBI scaler")
            self.windows_metadata = [
                dict(filename=name, subject_id=Path(name).stem, start_sample=start,
                     start_second=start / 25, sleep_stages=preprocessor.record(name)["stages"][start:start+3000:750].tolist(),
                     file_sha256=preprocessor.record(name)["file_sha256"])
                for name, start in selected]
        self.protocol = dict(
            schema="ibi_epoch_coordinate_moments_v2", metric="Euclidean distances between 3000-dimensional coordinate mean and variance vectors",
            options=self.options, selection=selection, windows=self.windows_metadata,
            early_stopping=self.early_stopping.options,
            early_stopping_improvement_rule="both_smoothed_metrics_same_epoch",
            early_stopping_smoothing_method="simple_moving_average",
            scaler=preprocessor.scaler, split_fingerprint=preprocessor.split_fingerprint,
            data_synthetic=preprocessor.provenance()["data_synthetic"],
            condition_columns=preprocessor.columns.tolist()[1:],
            generation="full original DDPM schedule; unknown IBI; clean conditions; independent windows",
            diffusion_steps=diffusion_config["T"], initialization_seed=42,
            validation_seed_reset_each_epoch=True, training_rng_restored=True,
            signal_unit="milliseconds", variance_unit="milliseconds_squared", variance_correction=0,
            aggregation="one 3000-dimensional sample per window; coordinate moments over all selected windows; stored zeros included; no clipping",
            vector_metrics=dict(mean="mean_vector_l2_ms", variance="variance_vector_l2_ms2",
                                dimension=3000, moment_axis="windows (dim=0)", distance="L2, no dimension normalization",
                                covariance="diagonal only", accumulation="CUDA batched centered-moment merge",
                                single_window_variance="zero under population definition"),
            diagnostic_aggregation="legacy pooled and per-window scalars retained; excluded from early stopping",
            metric_dtype="float64", device=str(self.device), test_npz_opened=False,
            limitations="Coordinate means and variances exclude cross-coordinate covariance and do not establish temporal or conditional fidelity.")
        _save_json(self.output / "protocol.json", self.protocol)

    def _generate(self, training_model):
        # Reconstruct afresh to keep S4 caches and any future eval-state mutation
        # away from the live training model. Preserve its original seed-42 buffers.
        from training_utils import fetchModel
        from scripts.evaluate_original_sampler import load_original_parameters
        np.random.seed(42)
        torch.manual_seed(42)
        model = fetchModel(self.windows.shape[2], 1, self.args).to(self.device)
        copied = load_original_parameters(model, training_model.state_dict(), torch)
        model.eval()
        torch.manual_seed(self.options["seed"])
        np.random.seed(self.options["seed"])
        accumulator, sampler_reports = DistributionAccumulator(), []
        for start in range(0, len(self.windows), self.options["batch_size"]):
            clean = self.windows[start:start+self.options["batch_size"]].to(self.device)
            ibi, metadata = sample_validation_windows(model, clean, self.diffusion_config, self.args, self.device)
            generated_seconds = ibi.to(dtype=torch.float64) * self.scale + self.center
            reference_seconds = self.reference[start:start+self.options["batch_size"]].to(self.device)
            accumulator.update(generated_seconds, reference_seconds)
            sampler_reports.append(metadata)
        return accumulator.finalize(), copied, sampler_reports

    def run(self, epoch, training_model, train_mean_loss):
        if self.should_stop or epoch != len(self.reports) + 1:
            raise ValueError("Validation must run exactly once at the end of every consecutive epoch")
        if not math.isfinite(train_mean_loss):
            raise ValueError("Nonfinite epoch mean loss")
        print(json.dumps(dict(status="epoch_validation_started", epoch=epoch,
                              windows=len(self.windows), diffusion_steps=self.diffusion_config["T"])), flush=True)
        started = time.perf_counter()
        epoch_dir = self.output / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(exist_ok=False)
        with preserve_random_state(self.device):
            metrics, copied, sampler_reports = self._generate(training_model)
            for row, metadata in zip(metrics["per_window"], self.windows_metadata):
                row.update(metadata)
        aggregate = metrics["aggregate"]
        stopping = self.early_stopping.update(epoch, aggregate)
        self.should_stop = stopping["stopped_early"]
        checkpoint_updates = {
            name: stopping["best_metrics"][name] for name in stopping["new_best_metrics"]}
        if stopping["joint_improvement"]:
            checkpoint_updates["joint"] = dict(
                epoch=epoch, metric="both_smoothed_metrics_same_epoch",
                value=stopping["smoothed_metrics"].copy(),
                raw_metrics={key: aggregate[key] for key, _ in EARLY_STOP_METRICS.values()},
                smoothing_window=self.early_stopping.options["smoothing_window"],
                window_start_epoch=epoch-self.early_stopping.options["smoothing_window"]+1)
        for name, checkpoint_metrics in checkpoint_updates.items():
            checkpoint = self.training_output / f"best_{name}_model.pth"
            # Save only state_dict, matching the baseline checkpoint format.
            # Temporary-file replacement preserves the previous best on I/O failure.
            temporary = checkpoint.with_suffix(".pth.tmp")
            torch.save(training_model.state_dict(), temporary)
            temporary.replace(checkpoint)
            self.best_checkpoints[name] = dict(
                **checkpoint_metrics, checkpoint=str(checkpoint),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        report = dict(status="completed", epoch=epoch, train_mean_loss=float(train_mean_loss),
                      aggregate=aggregate, per_window=metrics["per_window"],
                      validation_seconds=time.perf_counter()-started, device=str(self.device),
                      protocol=str(self.output / "protocol.json"), model_copy=copied,
                      sampling=sampler_reports, early_stopping=stopping)
        _save_json(epoch_dir / "metrics.json", report)
        self.reports.append(report)
        # Rewriting a short history avoids partial CSV rows after a failed append.
        csv_path = self.training_output / "epoch_metrics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            fields = ["epoch", "train_mean_loss", *aggregate, "validation_seconds",
                      *("smoothed_" + key for key, _ in EARLY_STOP_METRICS.values()),
                      "smoothing_ready",
                      "joint_improvement", "early_stopping_bad_epochs", "stopped_early"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in self.reports:
                writer.writerow(dict(epoch=row["epoch"], train_mean_loss=row["train_mean_loss"],
                                     **row["aggregate"], validation_seconds=row["validation_seconds"],
                                     **{"smoothed_" + key: value for key, value in row["early_stopping"]["smoothed_metrics"].items()},
                                     smoothing_ready=row["early_stopping"]["smoothing_ready"],
                                     joint_improvement=row["early_stopping"]["joint_improvement"],
                                     early_stopping_bad_epochs=row["early_stopping"]["bad_epochs"],
                                     stopped_early=row["early_stopping"]["stopped_early"]))
        _save_json(self.output / "summary.json", self.summary())
        print(json.dumps(dict(status="epoch_validation_completed", epoch=epoch,
                              train_mean_loss=float(train_mean_loss), **aggregate,
                              early_stopping=stopping)), flush=True)
        return report

    def summary(self):
        return dict(metric="Euclidean distances between 3000-dimensional coordinate mean and variance vectors",
                    completed_epochs=len(self.reports), options=self.options,
                    early_stopping=self.early_stopping.summary(),
                    best_checkpoints={name: value.copy() for name, value in self.best_checkpoints.items()},
                    protocol=str(self.output / "protocol.json"),
                    epoch_metrics_csv=str(self.training_output / "epoch_metrics.csv"),
                    latest_metrics=str(self.output / f"epoch_{len(self.reports):04d}" / "metrics.json")
                    if self.reports else None)


def create_epoch_validator(preprocessor, args, model, diffusion_config, device):
    config_path = Path(os.environ.get("DREAMT_CONFIG_PATH", Path(__file__).parent / "dataset_config.json"))
    config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else {}
    output = args.validation_output or f"saved_models/{args.dataset}/validation"
    return EpochValidator(preprocessor, args, model, diffusion_config, device, output, config)


def weighted_loss(rows):
    """Average per-batch MSE by the number of equally sized training windows."""
    count = sum(row["batch_samples"] for row in rows)
    if count <= 0:
        raise ValueError("Cannot average an empty training run")
    return math.fsum(row["epsilon_mse"] * row["batch_samples"] for row in rows) / count
