"""Epoch-end distribution checks without changing the WaveStitch updates."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from cuda_runtime import assert_cuda_model, require_cuda
from validation_metrics import preserve_random_state, summarize_distribution
from validation_sampling import sample_validation_windows

DEFAULT_VALIDATION = {"max_windows": 64, "batch_size": 8, "seed": 20261020}


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
            schema="ibi_epoch_mean_variance_v1", metric="IBI distribution mean and population variance",
            options=self.options, selection=selection, windows=self.windows_metadata,
            scaler=preprocessor.scaler, split_fingerprint=preprocessor.split_fingerprint,
            data_synthetic=preprocessor.provenance()["data_synthetic"],
            condition_columns=preprocessor.columns.tolist()[1:],
            generation="full original DDPM schedule; unknown IBI; clean conditions; independent windows",
            diffusion_steps=diffusion_config["T"], initialization_seed=42,
            validation_seed_reset_each_epoch=True, training_rng_restored=True,
            signal_unit="milliseconds", variance_unit="milliseconds_squared", variance_correction=0,
            aggregation="all selected IBI samples pooled; conditions excluded; stored zeros included; no clipping",
            metric_dtype="float64", device=str(self.device), test_npz_opened=False,
            limitations="Mean and variance alone do not establish waveform, temporal, or conditional fidelity.")
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
        generated, sampler_reports = [], []
        for start in range(0, len(self.windows), self.options["batch_size"]):
            clean = self.windows[start:start+self.options["batch_size"]].to(self.device)
            ibi, metadata = sample_validation_windows(model, clean, self.diffusion_config, self.args, self.device)
            generated.append(ibi.to(dtype=torch.float64) * self.scale + self.center)
            sampler_reports.append(metadata)
        return torch.cat(generated), copied, sampler_reports

    def run(self, epoch, training_model, train_mean_loss):
        if epoch != len(self.reports) + 1:
            raise ValueError("Validation must run exactly once at the end of every consecutive epoch")
        if not math.isfinite(train_mean_loss):
            raise ValueError("Nonfinite epoch mean loss")
        print(json.dumps(dict(status="epoch_validation_started", epoch=epoch,
                              windows=len(self.windows), diffusion_steps=self.diffusion_config["T"])), flush=True)
        started = time.perf_counter()
        epoch_dir = self.output / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(exist_ok=False)
        with preserve_random_state(self.device):
            generated, copied, sampler_reports = self._generate(training_model)
            reference = self.reference.to(self.device)
            metrics = summarize_distribution(generated, reference)
            for row, metadata in zip(metrics["per_window"], self.windows_metadata):
                row.update(metadata)
        aggregate = metrics["aggregate"]
        report = dict(status="completed", epoch=epoch, train_mean_loss=float(train_mean_loss),
                      aggregate=aggregate, per_window=metrics["per_window"],
                      validation_seconds=time.perf_counter()-started, device=str(self.device),
                      protocol=str(self.output / "protocol.json"), model_copy=copied,
                      sampling=sampler_reports)
        _save_json(epoch_dir / "metrics.json", report)
        self.reports.append(report)
        # Rewriting a short history avoids partial CSV rows after a failed append.
        csv_path = self.training_output / "epoch_metrics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            fields = ["epoch", "train_mean_loss", *aggregate, "validation_seconds"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in self.reports:
                writer.writerow(dict(epoch=row["epoch"], train_mean_loss=row["train_mean_loss"],
                                     **row["aggregate"], validation_seconds=row["validation_seconds"]))
        _save_json(self.output / "summary.json", self.summary())
        print(json.dumps(dict(status="epoch_validation_completed", epoch=epoch,
                              train_mean_loss=float(train_mean_loss), **aggregate)), flush=True)
        return report

    def summary(self):
        return dict(metric="IBI distribution mean and population variance",
                    completed_epochs=len(self.reports), options=self.options,
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
