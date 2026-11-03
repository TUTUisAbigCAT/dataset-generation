"""Epoch-end distribution checks without changing the WaveStitch updates."""
from __future__ import annotations

from collections import defaultdict, deque
import csv
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import time

import numpy as np
import torch

from cuda_runtime import assert_cuda_model, require_cuda
from validation_metrics import DistributionAccumulator, preserve_random_state
from validation_sampling import sample_validation_windows
from signal_normalization import signal_affine_parameters
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
CURRENT_EPOCH_METRIC_FIELDS = tuple(
    field for field in EPOCH_METRIC_FIELDS
    if field not in ("smoothed_gaussian_kl_real_to_generated", "smoothing_ready"))


def early_stopping_options(config):
    values = config.get("early_stopping", {})
    # Zero-valued historical thresholds have an unambiguous migration. A
    # nonzero threshold in ms or ms^2 cannot be reused for dimensionless KL.
    legacy = {"min_delta_mean_ms", "min_delta_variance_ms2"}
    if not isinstance(values, dict) or set(values) - (set(DEFAULT_EARLY_STOPPING) | legacy | {"selection_mode"}):
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
    selection_mode = options.get("selection_mode", "smoothed")
    if not isinstance(selection_mode, str) or selection_mode not in ("smoothed", "current"):
        raise ValueError("early_stopping.selection_mode must be smoothed or current")
    if selection_mode == "current" and options["smoothing_window"] != 1:
        raise ValueError("early_stopping.selection_mode=current requires smoothing_window=1")
    if selection_mode == "smoothed":
        # Preserve historical serialized options and run identities.
        options.pop("selection_mode", None)
    for key in ("min_delta_kl",):
        value = options[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"early_stopping.{key} must be finite and nonnegative")
        options[key] = float(value)
    return options


class EarlyStopping:
    """Track KL(real || generated), lower is better.

    Legacy mode selects from a full-window SMA. Current mode uses each epoch's
    KL directly, beginning at epoch one, without accumulating or averaging a
    history. A strict decrease exceeding min_delta_kl selects current weights.
    Disabling early termination leaves checkpoint selection active.
    """
    def __init__(self, config):
        self.options = early_stopping_options(config)
        self.selection_mode = self.options.get("selection_mode", "smoothed")
        self.best, self.anchors = {}, {}
        self.history = deque(maxlen=self.options["smoothing_window"])
        self.smoothed = {name: None for name in EARLY_STOP_METRICS}
        self.current = {name: None for name in EARLY_STOP_METRICS}
        self.completed_epochs = self.bad_epochs = 0
        self.stopped_epoch = None

    def update(self, epoch, aggregate):
        if self.stopped_epoch is not None or epoch != self.completed_epochs + 1:
            raise ValueError("Early stopping requires consecutive, non-stopped epochs")
        values = {name: float(aggregate[key]) for name, (key, _) in EARLY_STOP_METRICS.items()}
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("Early-stopping KL must be finite and nonnegative")
        if self.selection_mode == "current":
            self.current = values.copy()
            ready, selection_values = True, values
        else:
            self.history.append(values)
            ready = len(self.history) == self.options["smoothing_window"]
            if ready:
                self.smoothed = {name: math.fsum(row[name] for row in self.history) / len(self.history)
                                 for name in EARLY_STOP_METRICS}
            selection_values = self.smoothed
        improved, new_best = [], []
        for name, value in values.items():
            metric, delta_key = EARLY_STOP_METRICS[name]
            if name not in self.best or value < self.best[name]["value"]:
                self.best[name] = dict(epoch=epoch, metric=metric, value=value)
                new_best.append(name)
            if ready and (name not in self.anchors
                          or selection_values[name] < self.anchors[name] - self.options[delta_key]):
                self.anchors[name] = selection_values[name]
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
        summary = dict(options=self.options.copy(), bad_epochs=self.bad_epochs,
                       stopped_early=self.stopped_epoch is not None, stopped_epoch=self.stopped_epoch,
                       completed_epochs=self.completed_epochs, threshold_anchors=self.anchors.copy(),
                       best_metrics={name: value.copy() for name, value in self.best.items()})
        if self.selection_mode == "current":
            summary.update(improvement_rule="current_kl", smoothing_method="none",
                           selection_ready=self.completed_epochs > 0,
                           current_metrics={EARLY_STOP_METRICS[name][0]: value for name, value in self.current.items()})
        else:
            summary.update(improvement_rule="smoothed_kl", smoothing_method="simple_moving_average",
                           smoothing_ready=len(self.history) == self.options["smoothing_window"],
                           smoothed_metrics={EARLY_STOP_METRICS[name][0]: value for name, value in self.smoothed.items()})
        return summary


def validation_options(config):
    values = config.get("validation", {})
    if not isinstance(values, dict):
        raise ValueError("validation must be a JSON object")
    unknown = set(values) - (set(DEFAULT_VALIDATION) | {"reference_mask_policy", "save_artifacts"})
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
    mask_policy = result.get("reference_mask_policy", "training")
    if not isinstance(mask_policy, str) or mask_policy not in ("training", "observed_only"):
        raise ValueError("validation.reference_mask_policy must be training or observed_only")
    # Omit the default to preserve the existing protocol and options payload.
    if mask_policy == "training":
        result.pop("reference_mask_policy", None)
    if 'save_artifacts' in result and not isinstance(result['save_artifacts'], bool):
        raise ValueError('validation.save_artifacts must be boolean')
    return result


def _uses_reference_mask(preprocessor, options):
    return (getattr(preprocessor, "missing_value_policy", "preserve") in ("masked", "zero_fill_one")
            or options.get("reference_mask_policy", "training") == "observed_only")


def _window_reference_mask(preprocessor, name, start, length):
    observed = np.asarray(preprocessor.window_observed_mask(name, start, length))
    if observed.dtype != np.bool_:
        raise TypeError("window_observed_mask must return a boolean array")
    if observed.shape != (length,):
        raise ValueError("window_observed_mask must match the validation window length")
    return observed.copy()


def select_fixed_windows(preprocessor, options):
    """Greedy nonoverlapping candidates, then seeded subject-round-robin sampling.

    Search at every 30-second boundary so a missing first epoch cannot discard
    all legal windows in a short evaluation record. Selection is restricted to
    the preprocessor's explicit evaluation filenames (the legacy val split).
    """
    window_size = preprocessor.window_size
    if (isinstance(window_size, bool) or not isinstance(window_size, int)
            or window_size < EPOCH_SAMPLES or window_size % EPOCH_SAMPLES):
        raise ValueError("Validation requires a positive whole number of 30-second epochs")
    dataset = preprocessor.make_validation_dataset(window_size=window_size, stride=EPOCH_SAMPLES)
    evaluation_filenames = getattr(preprocessor, "evaluation_filenames", preprocessor.splits["val"])
    groups = defaultdict(list)
    end_by_subject = {}
    use_mask = _uses_reference_mask(preprocessor, options)
    excluded_insufficient_observations = 0
    for name, start in dataset.window_index:
        if name not in evaluation_filenames:
            raise RuntimeError("Validation selection escaped the configured evaluation filenames")
        if use_mask and int(_window_reference_mask(preprocessor, name, start, window_size).sum()) < 2:
            # Filter before greedy nonoverlap: a rejected window must not block
            # a later usable start, or participate in the seeded truncation.
            excluded_insufficient_observations += 1
            continue
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
    metadata = {
        "available_nonoverlapping_windows": available,
        "available_subjects": len(groups),
        "selected_windows": count,
        "selected_subjects": len({name for name, _ in selected}),
        "candidate_stride_samples": EPOCH_SAMPLES,
        "window_size_samples": window_size,
        "policy": "greedy nonoverlap per subject; seeded round-robin subjects and shuffled starts; sorted fixed output",
    }
    if use_mask:
        metadata["excluded_insufficient_observation_candidates"] = excluded_insufficient_observations
        metadata["minimum_observed_samples_per_window"] = 2
    return sorted(selected), metadata


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
        self.save_artifacts = self.options.get('save_artifacts', False)
        self.early_stopping = EarlyStopping(config)
        self.best_checkpoints = {}
        self.should_stop = False
        self.args = training_args
        self.window = window_settings(config)
        self.window_size = self.window["window_size"]
        self.window_epochs = self.window["window_epochs"]
        self.missing_value_policy = getattr(preprocessor, "missing_value_policy", "preserve")
        if self.missing_value_policy not in ("preserve", "masked", "zero_fill_one"):
            raise ValueError("Validation missing_value_policy must be preserve, masked or zero_fill_one")
        self.reference_mask = None
        self.reference_mask_policy = self.options.get("reference_mask_policy", "training")
        use_reference_mask = _uses_reference_mask(preprocessor, self.options)
        if (training_args.window_size != self.window_size
                or preprocessor.window_size != self.window_size
                or preprocessor.window_epochs != self.window_epochs):
            raise ValueError("Validation window_epochs must match the configuration, preprocessor and training window_size")
        if training_args.s4_lmax < self.window_size:
            raise ValueError("Validation window_size must fit within the original S4 kernel length")
        self.metric_description = (f"Mean paired-window KL(real || generated) between univariate Gaussian fits "
                                   f"to the {self.window_size} time samples within each window; nats per window")
        if use_reference_mask:
            self.metric_description = (f"Mean paired-window KL(real || generated) between univariate Gaussian fits "
                                       f"at the same reference-observed positions within each {self.window_size}-sample window; nats per window")
        self.diffusion_config = diffusion_config
        from diffusion_schedule import diffusion_identity
        self.diffusion_identity = diffusion_identity(training_args, diffusion_config)
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
            if use_reference_mask:
                masks = []
                for name, start in selected:
                    observed = _window_reference_mask(preprocessor, name, start, self.window_size)
                    if int(observed.sum()) < 2:
                        raise ValueError("Every masked validation window needs at least two observed samples")
                    masks.append(torch.from_numpy(observed.copy()))
                self.reference_mask = torch.stack(masks)
            self.center, self.scale = signal_affine_parameters(preprocessor.scaler)
            if not math.isfinite(self.scale) or self.scale <= 0 or not math.isfinite(self.center):
                raise ValueError("Invalid training-only IBI scaler")
            self.windows_metadata = [
                dict(validation_index=index, filename=name, subject_id=Path(name).stem, start_sample=start,
                     start_second=start / self.window["sample_rate_hz"],
                     sleep_stages=preprocessor.record(name)["stages"][start:start+self.window_size:EPOCH_SAMPLES].tolist(),
                     file_sha256=preprocessor.record(name)["file_sha256"])
                for index, (name, start) in enumerate(selected)]
            if self.reference_mask is not None:
                for metadata, observed in zip(self.windows_metadata, self.reference_mask):
                    count = int(observed.sum())
                    metadata["reference_mask"] = dict(
                        observed_samples=count, missing_samples=self.window_size-count,
                        observed_fraction=count/self.window_size,
                        sha256=hashlib.sha256(observed.numpy().tobytes()).hexdigest())
        self.protocol = dict(
            schema="ibi_epoch_window_gaussian_kl_v4", metric=self.metric_description,
            window=self.window,
            options=self.options, selection=selection, windows=self.windows_metadata,
            early_stopping=self.early_stopping.options,
            early_stopping_improvement_rule=self.early_stopping.summary()["improvement_rule"],
            early_stopping_smoothing_method=self.early_stopping.summary()["smoothing_method"],
            scaler=preprocessor.scaler, split_fingerprint=preprocessor.split_fingerprint,
            data_synthetic=preprocessor.provenance()["data_synthetic"],
            condition_columns=preprocessor.columns.tolist()[1:],
            generation="full recorded training DDPM schedule; unknown IBI; clean conditions; independent windows",
            diffusion_identity=self.diffusion_identity,
            diffusion_steps=diffusion_config["T"], initialization_seed=42,
            validation_seed_reset_each_epoch=True, training_rng_restored=True,
            signal_unit="milliseconds", variance_unit="milliseconds_squared", variance_correction=0,
            aggregation="paired-window temporal population moments; mean KL over all selected windows; stored zeros included; no clipping",
            gaussian_kl=dict(metric="gaussian_kl_real_to_generated", direction="real || generated",
                             dimension=1, window_samples=self.window_size,
                             moment_axis="time within each window (dim=1)",
                             pairing="real[i] and generated[i] share validation_index, source window and conditions",
                             reduction="mean over windows; equal weight per window, not batch means",
                             logarithm="natural", unit="nats per window", distribution="univariate Gaussian per window",
                             variance_floor_ms2=self.options["kl_variance_floor_ms2"],
                             variance_regularization="clamp each real and generated within-window variance to the fixed floor",
                             accumulation="CUDA sum of per-window KL divided by total window count; short batches weighted by window count",
                             single_window_variance="population variance over time samples within that window; nonconstant windows may have positive variance"),
            metric_dtype="float64", device=str(self.device), test_npz_opened=False,
            limitations="KL of paired moment-matched univariate Gaussian approximations, not the joint sequence distributions. "
                        "Invariant to temporal permutations within each window; does not assess temporal order, autocorrelation "
                        "or non-Gaussian structure and does not establish conditional fidelity.")
        evaluation_protocol = getattr(preprocessor, "evaluation_protocol", None)
        if evaluation_protocol is not None:
            self.protocol["evaluation_protocol"] = evaluation_protocol
            self.protocol["evaluation_filenames"] = list(preprocessor.evaluation_filenames)
        if self.reference_mask is not None:
            self.protocol.update(
                schema="ibi_epoch_masked_window_gaussian_kl_v5",
                missing_value_policy=self.missing_value_policy,
                reference_mask_policy=self.reference_mask_policy,
                aggregation="paired-window population moments on reference-observed time positions; mean KL over windows; missing positions excluded from both fits; no clipping",
                reference_mask=dict(
                    source="preprocessor.window_observed_mask(filename, start_sample, window_size)",
                    dtype="bool", shape=[len(self.reference), self.window_size],
                    frozen=True, minimum_observed_samples_per_window=2,
                    pairing="The same fixed real-data mask selects positions in real[i] and generated[i]",
                    generation_dependency="None: the mask is used only for scoring, never supplied as an IBI observation to the sampler",
                    generated_value_policy="All generated positions must be finite, including unobserved positions; negative values remain in the fit when selected",
                    reference_value_policy="Only observed positions must be finite; missing zero/nonfinite placeholders do not enter reference moments",
                    mask_sha256_encoding="Contiguous NumPy bool bytes in window sample order"))
            self.protocol["gaussian_kl"].update(
                moment_axis="reference-observed time positions within each window (dim=1)",
                single_window_variance="Population variance over at least two reference-observed positions; no missing-value padding enters either fit")
            self.protocol["limitations"] += (
                " Masked KL does not assess generated values at unobserved reference positions, beyond requiring finiteness. "
                "Scores from different reference masks or window sets are not directly comparable.")
        if self.save_artifacts:
            self.protocol['artifacts'] = dict(enabled=True, waveforms='waveforms.npz',
                per_window_metrics='per_window_metrics.csv', waveform_unit='seconds',
                moments='raw population moments in ms/ms^2 at the same scoring positions',
                extra_sampling=False, directory='epoch_NNNN')
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
        generated_samples = 0
        generated_nonpositive = torch.zeros((), dtype=torch.int64, device=self.device)
        sampler_reports = []
        artifact_waveforms, artifact_statistics = [], defaultdict(list)
        for start in range(0, len(self.windows), self.options["batch_size"]):
            clean = self.windows[start:start+self.options["batch_size"]].to(self.device)
            ibi, metadata = sample_validation_windows(model, clean, self.diffusion_config, self.args, self.device)
            generated_seconds = ibi.to(dtype=torch.float64) * self.scale + self.center
            # Preserve the same frozen window order in both sides of each pair.
            reference_seconds = self.reference[start:start+self.options["batch_size"]].to(self.device)
            if self.reference_mask is None:
                statistics = accumulator.update(generated_seconds, reference_seconds)
            else:
                observed_mask = self.reference_mask[start:start+self.options["batch_size"]].to(self.device)
                statistics = accumulator.update(generated_seconds, reference_seconds, observed_mask=observed_mask)
            if self.save_artifacts:
                # Copy only for file I/O; moments and KL remain CUDA operations.
                artifact_waveforms.append(generated_seconds.detach().cpu().numpy())
                for key, value in statistics.items():
                    artifact_statistics[key].append(value.detach().cpu().numpy())
                artifact_statistics['generated_nonpositive_samples_all'].append(
                    (generated_seconds <= 0).sum(dim=1).detach().cpu().numpy())
            # Independent all-position diagnostics: a reference mask must not
            # hide nonpositive generated values. These do not select checkpoints.
            generated_samples += generated_seconds.numel()
            generated_nonpositive += (generated_seconds <= 0).sum()
            sampler_reports.append(metadata)
        metrics = accumulator.finalize()
        nonpositive_samples = int(generated_nonpositive)
        metrics["diagnostics"] = dict(
            generated_samples=generated_samples,
            generated_nonpositive_samples=nonpositive_samples,
            generated_nonpositive_fraction=nonpositive_samples/generated_samples)
        if self.save_artifacts:
            metrics['_artifact_payload'] = dict(
                generated_seconds=np.concatenate(artifact_waveforms, axis=0),
                statistics={key: np.concatenate(value) for key, value in artifact_statistics.items()})
        return metrics, copied, sampler_reports

    def save_generation_artifacts(self, epoch_dir, metrics, epoch):
        """Serialize this exact generation; also usable for checkpoint audits."""
        if not self.save_artifacts:
            return None
        from validation_artifacts import write_validation_artifacts
        payload = metrics.pop('_artifact_payload')
        reference = self.reference.numpy()
        mask = (np.ones(reference.shape, dtype=bool) if self.reference_mask is None
                else self.reference_mask.numpy())
        return write_validation_artifacts(epoch_dir, **payload,
            reference_seconds=reference, scoring_mask=mask,
            windows_metadata=self.windows_metadata, conditions=self.windows[:, :, 1:].numpy(),
            condition_columns=self.protocol['condition_columns'],
            protocol_path=self.output / 'protocol.json', epoch=epoch)

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
        artifacts = self.save_generation_artifacts(epoch_dir, metrics, epoch)
        aggregate = metrics["aggregate"]
        stopping = self.early_stopping.update(epoch, aggregate)
        self.should_stop = stopping["stopped_early"]
        checkpoint_updates = {}
        if stopping["metric_improvement"]:
            checkpoint_updates["kl"] = dict(
                epoch=epoch, metric="gaussian_kl_real_to_generated",
                selection_rule=stopping["improvement_rule"],
                value=self.early_stopping.anchors["kl"],
                raw_metrics={key: aggregate[key] for key, _ in EARLY_STOP_METRICS.values()})
            if self.early_stopping.selection_mode == "smoothed":
                checkpoint_updates["kl"].update(
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
                      diagnostics=metrics.get("diagnostics", {}),
                      validation_seconds=time.perf_counter()-started, device=str(self.device),
                      protocol=str(self.output / "protocol.json"), model_copy=copied,
                      sampling=sampler_reports, early_stopping=stopping)
        if artifacts is not None:
            report['artifacts'] = artifacts
        _save_json(epoch_dir / "metrics.json", report)
        self.reports.append(report)
        # Rewriting a short history avoids partial CSV rows after a failed append.
        csv_path = self.training_output / "epoch_metrics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            current_mode = self.early_stopping.selection_mode == "current"
            writer = csv.DictWriter(stream, fieldnames=CURRENT_EPOCH_METRIC_FIELDS if current_mode else EPOCH_METRIC_FIELDS)
            writer.writeheader()
            for row in self.reports:
                csv_row = dict(epoch=row["epoch"], train_mean_loss=row["train_mean_loss"],
                               **row["aggregate"], validation_seconds=row["validation_seconds"],
                               metric_improvement=row["early_stopping"]["metric_improvement"],
                               early_stopping_bad_epochs=row["early_stopping"]["bad_epochs"],
                               stopped_early=row["early_stopping"]["stopped_early"])
                if not current_mode:
                    csv_row.update(
                        {"smoothed_" + key: value for key, value in row["early_stopping"]["smoothed_metrics"].items()},
                        smoothing_ready=row["early_stopping"]["smoothing_ready"])
                writer.writerow(csv_row)
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
    """Average batch losses by their actual reduction weight.

    Masked runs supply observed-point counts as loss_weight. Existing rows
    without it retain the original equally sized window-count weighting.
    """
    rows = list(rows)
    weights = [row.get("loss_weight", row["batch_samples"]) for row in rows]
    if any(isinstance(weight, (bool, np.bool_)) or not isinstance(weight, Real)
           or not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("Loss weights must be finite positive numbers, not bool")
    count = sum(weights)
    if not math.isfinite(count) or count <= 0:
        raise ValueError("Cannot average an empty training run")
    return math.fsum(row["epsilon_mse"] * weight for row, weight in zip(rows, weights)) / count
