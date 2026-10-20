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
from window_config import EPOCH_SAMPLES, window_settings

DEFAULT_VALIDATION = {"max_windows": None, "batch_size": 8, "seed": 20261020,
                      "kl_variance_floor_ms2": 1.0}
DEFAULT_EARLY_STOPPING = {
    "enabled": True, "patience": 10, "smoothing_window": 5,
    "min_delta_kl": 0.0,
}
EARLY_STOP_METRICS = {"kl": ("gaussian_kl_real_to_generated", "min_delta_kl")}
EPOCH_METRIC_FIELDS = (
    "epoch", "train_mean_loss", "gaussian_kl_real_to_generated",
    "smoothed_gaussian_kl_real_to_generated", "validation_seconds",
    "smoothing_ready", "metric_improvement", "early_stopping_bad_epochs", "stopped_early",
)


def early_stopping_options(config):
    values = config.get("early_stopping", {})
    # Zero-valued historical thresholds have an unambiguous migration. A
    # nonzero threshold in ms or ms^2 cannot be reused for dimensionless KL.
    legacy = {"min_delta_mean_ms", "min_delta_variance_ms2"}
    if not isinstance(values, dict) or set(values) - (set(DEFAULT_EARLY_STOPPING) | legacy):
        raise ValueError("early_stopping must be an object with only supported options")
    for key in legacy & values.keys():
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0:
            raise ValueError(f"Remove legacy early_stopping.{key} and configure min_delta_kl in nats")
    options = {**DEFAULT_EARLY_STOPPING, **{key: value for key, value in values.items() if key not in legacy}}
    if not isinstance(options["enabled"], bool):
        raise ValueError("early_stopping.enabled must be boolean")
    for key in ("patience", "smoothing_window"):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"early_stopping.{key} must be a positive integer")
    for key in ("min_delta_kl",):
        value = options[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"early_stopping.{key} must be finite and nonnegative")
        options[key] = float(value)
    return options


class EarlyStopping:
    """Track full-window SMA of KL(real || generated), lower is better.

    The first complete smoothing window establishes the anchor. Only a
    strict decrease exceeding min_delta_kl resets patience and selects the
    current weights. Raw minima are diagnostics, not checkpoint triggers.
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
            raise ValueError("Early-stopping KL must be finite and nonnegative")
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
        metric_improvement = "kl" in improved
        if ready:
            self.bad_epochs = 0 if metric_improvement else self.bad_epochs + 1
        self.completed_epochs = epoch
        if self.options["enabled"] and self.bad_epochs >= self.options["patience"]:
            self.stopped_epoch = epoch
        return dict(**self.summary(), improved_metrics=improved, new_best_metrics=new_best,
                    metric_improvement=metric_improvement)

    def summary(self):
        return dict(options=self.options.copy(), improvement_rule="smoothed_kl",
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
    floor = result["kl_variance_floor_ms2"]
    if (isinstance(floor, bool) or not isinstance(floor, (int, float))
            or not math.isfinite(floor) or floor <= 0):
        raise ValueError("validation.kl_variance_floor_ms2 must be finite and positive")
    result["kl_variance_floor_ms2"] = float(floor)
    return result


def select_fixed_windows(preprocessor, options):
    """Greedy nonoverlapping candidates, then seeded subject-round-robin sampling.

    Search at every 30-second boundary so a missing first epoch cannot discard
    all legal windows in a short validation record. Selection never opens test.
    """
    window_size = preprocessor.window_size
    if (isinstance(window_size, bool) or not isinstance(window_size, int)
            or window_size < EPOCH_SAMPLES or window_size % EPOCH_SAMPLES):
        raise ValueError("Validation requires a positive whole number of 30-second epochs")
    dataset = preprocessor.make_validation_dataset(window_size=window_size, stride=EPOCH_SAMPLES)
    groups = defaultdict(list)
    end_by_subject = {}
    for name, start in dataset.window_index:
        if name not in preprocessor.splits["val"]:
            raise RuntimeError("Validation selection escaped the validation split")
        if start >= end_by_subject.get(name, 0):
            groups[name].append((name, start))
            end_by_subject[name] = start + window_size
    available = sum(map(len, groups.values()))
    if not available:
        raise ValueError("No complete valid validation windows for the configured window_epochs")
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
        "candidate_stride_samples": EPOCH_SAMPLES,
        "window_size_samples": window_size,
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
        self.window = window_settings(config)
        self.window_size = self.window["window_size"]
        self.window_epochs = self.window["window_epochs"]
        if (training_args.window_size != self.window_size
                or preprocessor.window_size != self.window_size
                or preprocessor.window_epochs != self.window_epochs):
            raise ValueError("Validation window_epochs must match the configuration, preprocessor and training window_size")
        if training_args.s4_lmax < self.window_size:
            raise ValueError("Validation window_size must fit within the original S4 kernel length")
        self.metric_description = (f"KL(real || generated) between diagonal Gaussian fits to "
                                   f"all {self.window_size}-dimensional validation windows; sum in nats")
        self.diffusion_config = diffusion_config
        self.output = Path(output_dir).resolve()
        self.training_output = self.output.parent
        self.output.mkdir(parents=True, exist_ok=False)
        self.reports = []
        with preserve_random_state(self.device):
            selected, selection = select_fixed_windows(preprocessor, self.options)
            # Data loading is on the host; generation and all statistics use CUDA.
            self.windows = torch.stack([torch.from_numpy(preprocessor.window_array(name, start, self.window_size))
                                        for name, start in selected])
            self.reference = torch.stack([
                torch.from_numpy(np.asarray(preprocessor.record(name)["ibi"][start:start + self.window_size],
                                            dtype=np.float64).copy())
                for name, start in selected])
            self.scale = float(preprocessor.scaler["signal_std_seconds"])
            self.center = float(preprocessor.scaler["signal_mean_seconds"])
            if not math.isfinite(self.scale) or self.scale <= 0 or not math.isfinite(self.center):
                raise ValueError("Invalid training-only IBI scaler")
            self.windows_metadata = [
                dict(filename=name, subject_id=Path(name).stem, start_sample=start,
                     start_second=start / self.window["sample_rate_hz"],
                     sleep_stages=preprocessor.record(name)["stages"][start:start+self.window_size:EPOCH_SAMPLES].tolist(),
                     file_sha256=preprocessor.record(name)["file_sha256"])
                for name, start in selected]
        self.protocol = dict(
            schema="ibi_epoch_diagonal_gaussian_kl_v3", metric=self.metric_description,
            window=self.window,
            options=self.options, selection=selection, windows=self.windows_metadata,
            early_stopping=self.early_stopping.options,
            early_stopping_improvement_rule="smoothed_kl",
            early_stopping_smoothing_method="simple_moving_average",
            scaler=preprocessor.scaler, split_fingerprint=preprocessor.split_fingerprint,
            data_synthetic=preprocessor.provenance()["data_synthetic"],
            condition_columns=preprocessor.columns.tolist()[1:],
            generation="full original DDPM schedule; unknown IBI; clean conditions; independent windows",
            diffusion_steps=diffusion_config["T"], initialization_seed=42,
            validation_seed_reset_each_epoch=True, training_rng_restored=True,
            signal_unit="milliseconds", variance_unit="milliseconds_squared", variance_correction=0,
            aggregation=f"one {self.window_size}-dimensional sample per window; coordinate moments over all selected windows; stored zeros included; no clipping",
            gaussian_kl=dict(metric="gaussian_kl_real_to_generated", direction="real || generated",
                             dimension=self.window_size, moment_axis="windows (dim=0)",
                             reduction="sum over coordinates; no dimension normalization",
                             logarithm="natural", unit="nats", covariance="diagonal only",
                             variance_floor_ms2=self.options["kl_variance_floor_ms2"],
                             variance_regularization="clamp each real and generated coordinate variance to the fixed floor",
                             accumulation="CUDA batched centered-moment merge",
                             single_window_variance="zero population estimate; floor applies; insufficient for distribution assessment"),
            metric_dtype="float64", device=str(self.device), test_npz_opened=False,
            limitations="KL of moment-matched diagonal Gaussian approximations, not the true sequence distributions. "
                        "Excludes cross-coordinate covariance and non-Gaussian structure; does not establish temporal or conditional fidelity.")
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
        accumulator = DistributionAccumulator(kl_variance_floor_ms2=self.options["kl_variance_floor_ms2"])
        sampler_reports = []
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
                              windows=len(self.windows), window_size=self.window_size,
                              window_epochs=self.window_epochs,
                              diffusion_steps=self.diffusion_config["T"])), flush=True)
        started = time.perf_counter()
        epoch_dir = self.output / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(exist_ok=False)
        with preserve_random_state(self.device):
            metrics, copied, sampler_reports = self._generate(training_model)
        aggregate = metrics["aggregate"]
        stopping = self.early_stopping.update(epoch, aggregate)
        self.should_stop = stopping["stopped_early"]
        checkpoint_updates = {}
        if stopping["metric_improvement"]:
            checkpoint_updates["kl"] = dict(
                epoch=epoch, metric="gaussian_kl_real_to_generated", selection_rule="smoothed_kl",
                value=stopping["smoothed_metrics"]["gaussian_kl_real_to_generated"],
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
                      aggregate=aggregate,
                      validation_seconds=time.perf_counter()-started, device=str(self.device),
                      protocol=str(self.output / "protocol.json"), model_copy=copied,
                      sampling=sampler_reports, early_stopping=stopping)
        _save_json(epoch_dir / "metrics.json", report)
        self.reports.append(report)
        # Rewriting a short history avoids partial CSV rows after a failed append.
        csv_path = self.training_output / "epoch_metrics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EPOCH_METRIC_FIELDS)
            writer.writeheader()
            for row in self.reports:
                writer.writerow(dict(epoch=row["epoch"], train_mean_loss=row["train_mean_loss"],
                                     **row["aggregate"], validation_seconds=row["validation_seconds"],
                                     **{"smoothed_" + key: value for key, value in row["early_stopping"]["smoothed_metrics"].items()},
                                     smoothing_ready=row["early_stopping"]["smoothing_ready"],
                                     metric_improvement=row["early_stopping"]["metric_improvement"],
                                     early_stopping_bad_epochs=row["early_stopping"]["bad_epochs"],
                                     stopped_early=row["early_stopping"]["stopped_early"]))
        _save_json(self.output / "summary.json", self.summary())
        print(json.dumps(dict(status="epoch_validation_completed", epoch=epoch,
                              train_mean_loss=float(train_mean_loss), **aggregate,
                              early_stopping=stopping)), flush=True)
        return report

    def summary(self):
        return dict(metric=self.metric_description, window=self.window.copy(),
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
