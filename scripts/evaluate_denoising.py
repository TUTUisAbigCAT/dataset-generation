"""Read-only teacher-forced denoising audit of the strict upstream model.

This is an external observation: no sampling, training, loss modification,
missing-value masking, output clipping, or state-dict loading repair. It uses
the official conditional forward-noising expression and ordinary MSELoss.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_TRAINING_ARGS = dict(dataset="DREAMT_SIM", backbone="S4", beta_0=.0001, beta_T=.02,
    timesteps=200, hdim=64, lr=.0001, batch_size=1024, epochs=1000, layers=4,
    window_size=3000, stride=1500, num_res_layers=4, res_channels=64, skip_channels=64,
    diff_step_embed_in=32, diff_step_embed_mid=64, diff_step_embed_out=64,
    s4_lmax=3000, s4_dstate=64, s4_dropout=0., s4_bidirectional=True, s4_layernorm=True,
    propCycEnc=False)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def table_files(output, stem, rows):
    save_json(output / f"{stem}.json", rows)
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_training_run(manifest_path):
    """Require explicit training arguments and data provenance; never infer a run."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    config = manifest.get("training_args")
    if not isinstance(config, dict) or not any(isinstance(manifest.get(key), dict) for key in ("data_identity", "data")):
        raise ValueError("Provide a run manifest (or explicit specification) with complete training_args and data")
    missing = sorted(set(DEFAULT_TRAINING_ARGS) - set(config))
    if missing:
        raise ValueError("Incomplete training_args; missing: " + ", ".join(missing))
    if config["dataset"] not in ("DREAMT", "DREAMT_SIM"):
        raise ValueError("Expected dataset DREAMT or DREAMT_SIM")
    for key in ("backbone", "beta_0", "beta_T", "timesteps", "window_size", "num_res_layers",
                "res_channels", "skip_channels", "diff_step_embed_in", "diff_step_embed_mid", "diff_step_embed_out",
                "s4_lmax", "s4_dstate", "s4_dropout", "s4_bidirectional", "s4_layernorm", "propCycEnc"):
        if config[key] != DEFAULT_TRAINING_ARGS[key]:
            raise ValueError(f"This audit expects the planned strict model; {key} differs: {config[key]!r}")
    info = dict(source="explicit run_manifest.training_args and data",
        manifest=str(manifest_path.resolve()), manifest_sha256=sha256(manifest_path),
        supplied_training_args=config, actual_training_duration_verified=False)
    return SimpleNamespace(**config), manifest, info


def training_arguments(manifest_path):
    model_args, _, info = load_training_run(manifest_path)
    return model_args, info


def _recorded_path(value, manifest_path):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(manifest_path).resolve().parent / path).resolve()


def bind_training_data(model_args, manifest, manifest_path, preprocessor_class):
    """Rebuild only the recorded training source and reject changed identities."""
    data = manifest.get("data", {})
    expected = manifest.get("data_identity", data.get("data_identity", data))
    if not isinstance(expected, dict):
        raise ValueError("Training data identity must be a dictionary")
    if expected.get("dataset", model_args.dataset) != model_args.dataset:
        raise ValueError("Training arguments and data identity name different datasets")
    required = {"data_directory", "splits", "scaler", "metadata_file_sha256"}
    if not required.issubset(expected):
        raise ValueError("Training data identity is incomplete: " + ", ".join(sorted(required - set(expected))))
    data_dir = _recorded_path(expected["data_directory"], manifest_path)
    metadata_path = _recorded_path(expected.get("metadata_path") or data_dir / "participant_info.csv", manifest_path)
    generated = expected.get("split_source") == "generated_subject_split"
    if generated and "split_seed" not in expected:
        raise ValueError("Generated training split must record its split_seed")
    split_value = expected.get("split_path")
    split_path = None if generated else _recorded_path(split_value or data_dir / "splits.json", manifest_path)
    config = dict(data_dir=str(data_dir), metadata_path=str(metadata_path), generate_splits=generated,
        split_path=None if generated else str(split_path), split_seed=expected.get("split_seed", 42),
        multiplier=expected.get("multiplier", 1.0), downsample_rate=expected.get("downsample_rate", 1),
        max_epochs=expected.get("max_epochs"))
    if "schema" in expected:
        if expected["schema"] != "dreamt_data_identity_v1":
            raise ValueError("Unsupported training data identity schema")
        preprocessor = preprocessor_class.from_identity(expected, dataset=model_args.dataset)
    else:
        preprocessor = preprocessor_class(model_args.dataset, model_args.propCycEnc, data_config=config)
    actual = preprocessor.data_identity()
    checks = {}
    for key in ("dataset", "data_directory", "metadata_path", "split_source", "split_path", "split_seed",
                "split_file_sha256", "split_fingerprint", "metadata_file_sha256", "splits", "scaler",
                "scaler_subjects", "multiplier", "downsample_rate", "max_epochs", "sample_rate_hz",
                "epoch_samples", "columns", "stage_names", "raw_stage_mapping"):
        if key in expected:
            reference = expected[key]
            if key in ("data_directory", "metadata_path", "split_path") and reference is not None:
                reference = str(_recorded_path(reference, manifest_path))
            checks[key] = actual.get(key) == reference
    train_names = expected["splits"]["train"]
    stored_hashes = expected.get("train_npz_sha256", expected.get("training_npz_sha256", expected.get("opened_npz_sha256", {})))
    if not all(name in stored_hashes for name in train_names):
        raise ValueError("Training data identity must include every training NPZ hash")
    checks["training_npz_sha256"] = actual.get("train_npz_sha256", actual.get("training_npz_sha256")) == {
        name: stored_hashes[name] for name in train_names}
    failed = [key for key, matches in checks.items() if not matches]
    if failed:
        raise ValueError("Training data/scaler identity changed: " + ", ".join(failed))
    return preprocessor, dict(all_recorded_identity_fields_match=True, checks=checks,
        global_config_ignored=True, data_identity=actual)


def resolve_checkpoint(manifest_path, manifest, override=None):
    """Allow explicit relocated weights, while enforcing any recorded digest."""
    recorded = manifest.get("checkpoint")
    recorded_path = _recorded_path(recorded, manifest_path) if recorded else None
    path = Path(override).resolve() if override is not None else recorded_path
    if path is None:
        raise ValueError("Manifest has no checkpoint; specify --checkpoint explicitly")
    if not path.is_file():
        raise FileNotFoundError("Training checkpoint is not available: " + str(path))
    expected_sha = manifest.get("checkpoint_sha256")
    digest_source = "manifest" if expected_sha else None
    report_path = Path(manifest_path).resolve().parent / "training_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        report_checkpoint = report.get("checkpoint")
        if recorded_path is not None and report_checkpoint and _recorded_path(report_checkpoint, manifest_path) == recorded_path:
            report_sha = report.get("checkpoint_sha256")
            if expected_sha and report_sha and expected_sha != report_sha:
                raise ValueError("Manifest and training report checkpoint digests disagree")
            if report_sha:
                expected_sha, digest_source = report_sha, "training_report.json"
    if expected_sha is None and recorded_path is not None and recorded_path.is_file():
        expected_sha, digest_source = sha256(recorded_path), "current recorded checkpoint file"
    actual_sha = sha256(path)
    if expected_sha is not None and actual_sha != expected_sha:
        raise ValueError("Selected checkpoint does not match the recorded training checkpoint digest")
    return path, dict(path=str(path), sha256=actual_sha, explicit_override=override is not None,
        recorded_path=None if recorded_path is None else str(recorded_path),
        recorded_digest_source=digest_source, recorded_digest_verified=expected_sha is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Explicit weights override; default: checkpoint recorded in the manifest")
    parser.add_argument("--run-manifest", type=Path, required=True,
                        help="Training manifest with complete training_args and recorded data identity")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/denoising_evaluation")
    parser.add_argument("--batch-size", type=int, default=2, help="Evaluation batch size only; does not change training")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("Positive evaluation batch size and thread count required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a fresh output directory; existing evaluations are not overwritten")
    from cuda_runtime import require_cuda, assert_cuda_model, cuda_environment
    device = require_cuda()
    model_args, manifest, config_provenance = load_training_run(args.run_manifest)
    args.checkpoint, checkpoint_identity = resolve_checkpoint(args.run_manifest, manifest, args.checkpoint)

    import numpy as np
    import torch
    from torch import from_numpy, nn, sqrt
    from data_utils import Preprocessor
    from training_utils import fetchModel, fetchDiffusionConfig

    torch.set_num_threads(args.threads)
    # Same seeds and construction order as official training. Preprocessor is
    # deterministic and draws no global NumPy/Torch random numbers.
    np.random.seed(42)
    torch.manual_seed(42)
    preprocessor, data_identity_checks = bind_training_data(model_args, manifest, args.run_manifest, Preprocessor)
    columns = preprocessor.columns
    input_features = len(columns)
    expected_columns = ("IBI", "stage_W", "stage_Light", "stage_REM", "stage_Deep",
                        "AHI", "BMI", "AGE", "gender_F", "gender_M")
    hierarchy = columns.get_indexer(preprocessor.hierarchical_features_cyclic)
    non_hier_cols = np.setdiff1d(np.arange(input_features), hierarchy)
    if (tuple(columns) != expected_columns or hierarchy.tolist() != list(range(1, input_features))
            or non_hier_cols.tolist() != [0]):
        raise ValueError("Expected the four-stage W/Light/REM/Deep schema and one IBI output")
    # Preserve the original seeded constructor, then transfer every model tensor.
    model = fetchModel(input_features, 1, model_args).to(device)
    assert_cuda_model(model)
    # Preserve upstream NumPy schedule values exactly; transfer once before use.
    diffusion = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in fetchDiffusionConfig(model_args).items()}
    model.eval()
    checkpoint_before = sha256(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location=device)
    current = model.state_dict()
    if set(saved) != set(current):
        raise ValueError("Saved state keys differ from the original freshly constructed model")
    for name in current:
        if saved[name].shape != current[name].shape or saved[name].dtype != current[name].dtype:
            raise ValueError(f"Saved tensor shape/dtype mismatch: {name}")
    buffer_checks = []
    for name, value in model.named_buffers():
        equal = torch.equal(value, saved[name])
        buffer_checks.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                                  equal_before_any_weight_loading=equal))
        if not equal:
            raise ValueError(f"Fixed upstream buffer differs from the seed-42 constructor: {name}")

    windows, identities = [], []
    selected_windows = preprocessor.select_validation_windows(windows_per_subject=2)
    for filename, start in selected_windows:
        record = preprocessor.record(filename)
        windows.append(preprocessor.window_array(filename, start, 3000))
        identities.append(dict(subject_id=record["subject_id"], filename=filename,
            split="val", start_sample=start, start_epoch=start // 750, start_seconds=start / 25., samples=3000))
    window_count = len(windows)
    subject_count = len({row["subject_id"] for row in identities})
    clean = from_numpy(np.stack(windows)).to(device=device, dtype=torch.float32)
    hierarchy_gpu = torch.as_tensor(hierarchy, device=device, dtype=torch.long)
    non_hier_gpu = torch.as_tensor(non_hier_cols, device=device, dtype=torch.long)
    if clean.shape != (window_count, 3000, input_features) or not torch.isfinite(clean).all():
        raise ValueError("Expected fully finite 120-second validation windows")
    if set(preprocessor.accessed_files) & set(preprocessor.splits["test"]):
        raise AssertionError("Test NPZ files were accessed")
    generator = torch.Generator(device=device).manual_seed(20261017)
    random_timesteps = torch.randint(diffusion["T"], size=(window_count,), generator=generator, device=device)
    # Match the upstream sigma shape, including clean metadata channels; their
    # draws are retained even though only signal-channel noise is supervised.
    sigmas = torch.normal(0., 1., size=clean.shape, generator=generator, device=device)
    scenarios = {"random_t": random_timesteps,
                 "fixed_t199": torch.full((window_count,), 199, dtype=torch.long, device=device)}
    criterion = nn.MSELoss()
    # Keep predictions and metrics on CUDA until final file serialization.
    raw = {"clean_channels": clean, "sigma_all_channels": sigmas,
           "random_timesteps": random_timesteps}
    rows, load_checks = [], {}

    def evaluate(state_name):
        for scenario, all_timesteps in scenarios.items():
            predictions = []
            for offset in range(0, len(clean), args.batch_size):
                batch = clean[offset:offset + args.batch_size]
                sigma = sigmas[offset:offset + args.batch_size]
                timesteps = all_timesteps[offset:offset + args.batch_size]
                alpha_bars = diffusion["alpha_bars"]
                coeff_1 = sqrt(alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
                coeff_2 = sqrt(1 - alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
                # Verbatim upstream forward-noising algebra. This conditional
                # channel selector is not a validity mask or a loss weighting.
                conditional_mask = torch.ones_like(batch)
                conditional_mask[:, :, non_hier_gpu] = 0
                batch_noised = (1 - conditional_mask) * (coeff_1 * batch + coeff_2 * sigma) + conditional_mask * batch
                if not torch.equal(batch_noised[:, :, hierarchy_gpu], batch[:, :, hierarchy_gpu]):
                    raise AssertionError("Metadata channels were noised")
                # Frozen upstream embedding factories inherit this CUDA device.
                with torch.no_grad(), torch.device(device):
                    predicted = model(batch_noised, timesteps.reshape((-1, 1)))
                target = sigma[:, :, non_hier_gpu].permute((0, 2, 1))
                batch_mse = criterion(predicted, target)
                noisy_signal = batch_noised[:, :, non_hier_gpu].permute((0, 2, 1))
                estimate = (noisy_signal - coeff_2 * predicted) / coeff_1
                if not torch.isfinite(predicted).all() or not torch.isfinite(estimate).all() or not torch.isfinite(batch_mse):
                    raise FloatingPointError("Nonfinite teacher-forced denoising result")
                if state_name == "initial_zero_head" and bool((predicted != 0).any()):
                    raise AssertionError("The original initialized zero output head did not predict exactly zero")
                for i in range(len(batch)):
                    truth_seconds = (batch[i, :, 0].double() * preprocessor.scaler["signal_std_seconds"]
                                     + preprocessor.scaler["signal_mean_seconds"])
                    estimate_seconds = (estimate[i, 0].double() * preprocessor.scaler["signal_std_seconds"]
                                        + preprocessor.scaler["signal_mean_seconds"])
                    delta_ms = (estimate_seconds.mean() - truth_seconds.mean()) * 1000
                    rows.append(dict(model_state=state_name, scenario=scenario, **identities[offset + i],
                        timestep=int(timesteps[i]), alpha_bar=float(alpha_bars[timesteps[i]]),
                        epsilon_mse=criterion(predicted[i], target[i]),
                        predicted_epsilon_mean=predicted[i].double().mean(),
                        target_epsilon_mean=target[i].double().mean(),
                        epsilon_mean_error=(predicted[i].double() - target[i].double()).mean(),
                        reference_mean_ms=truth_seconds.mean() * 1000,
                        reconstructed_mean_ms=estimate_seconds.mean() * 1000,
                        x0_window_mean_absolute_error_ms=delta_ms.abs(), x0_window_mean_signed_bias_ms=delta_ms))
                predictions.append(predicted)
            raw[f"epsilon_prediction_{state_name}_{scenario}"] = torch.cat(predictions)
            print(json.dumps(dict(model_state=state_name, scenario=scenario, windows=window_count)), flush=True)

    evaluate("initial_zero_head")
    # Check again immediately before loading. Do not copy buffers, invoke
    # load_state_dict, materialize tensor layouts, or patch the upstream model.
    if any(not torch.equal(value, saved[name]) for name, value in model.named_buffers()):
        raise ValueError("A fixed buffer changed while evaluating the initial model")
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(saved[name])
            param.requires_grad = True
    load_checks["all_named_parameters_equal_saved_after_copy"] = all(torch.equal(param, saved[name]) for name, param in model.named_parameters())
    load_checks["all_buffers_equal_saved_without_copy"] = all(torch.equal(value, saved[name]) for name, value in model.named_buffers())
    if not all(load_checks.values()):
        raise AssertionError("Original parameter-copy checkpoint loading did not reproduce the saved tensors")
    evaluate("trained")
    if any(not torch.equal(param, saved[name]) for name, param in model.named_parameters()):
        raise AssertionError("Trained parameters changed during read-only evaluation")
    if any(not torch.equal(value, saved[name]) for name, value in model.named_buffers()):
        raise AssertionError("Fixed buffers changed during read-only evaluation")
    if any(param.grad is not None for param in model.parameters()) or sha256(args.checkpoint) != checkpoint_before:
        raise AssertionError("Evaluation changed gradients or the saved checkpoint")

    metric_keys = ("epsilon_mse", "epsilon_mean_error", "reference_mean_ms", "reconstructed_mean_ms",
                   "x0_window_mean_absolute_error_ms", "x0_window_mean_signed_bias_ms")
    subjects, totals = [], []
    for state in ("initial_zero_head", "trained"):
        for scenario in scenarios:
            selected = [row for row in rows if row["model_state"] == state and row["scenario"] == scenario]
            for sid in sorted({row["subject_id"] for row in selected}):
                individual = [row for row in selected if row["subject_id"] == sid]
                subjects.append(dict(model_state=state, scenario=scenario, subject_id=sid, split="val", windows=len(individual),
                    **{key: torch.stack([row[key] for row in individual]).double().mean() for key in metric_keys}))
            per_subject = [row for row in subjects if row["model_state"] == state and row["scenario"] == scenario]
            totals.append(dict(model_state=state, scenario=scenario, subjects=subject_count, windows=window_count,
                **{key: torch.stack([row[key] for row in per_subject]).double().mean() for key in metric_keys}))
    # Scalar transfer is solely the final JSON/CSV serialization boundary.
    for table in (rows, subjects, totals):
        for row in table:
            for key, value in row.items():
                if torch.is_tensor(value):
                    row[key] = float(value)
    sources = {name: sha256(ROOT / name) for name in ("training_wavestitch.py", "training_utils.py", "data_utils.py", "watchsleep_loader.py",
        "TSImputers/SSSDS4Imputer.py", "TSImputers/S4Model.py", "utils/util.py", "scripts/evaluate_denoising.py")}
    report = dict(status="passed", evaluation_type="teacher-forced conditional epsilon denoising; not free generation",
        protocol=dict(initialization_seed=42, evaluation_noise_seed=20261017, scenarios=list(scenarios),
            same_epsilon_across_scenarios=True, initial_and_trained_share_noised_inputs=True,
            random_timesteps=random_timesteps.tolist(), evaluation_batch_size=args.batch_size,
            input_features=input_features, output_features=1, window_length=3000, validation_subjects=subject_count, windows=window_count,
            validation_selection=preprocessor.validation_selection,
            stage_columns=list(columns[1:5]), condition_columns=list(columns[hierarchy]),
            observations_or_inpainting=False, output_clipping=False, extra_loss_terms=False,
            loss="original nn.MSELoss over predicted IBI epsilon", noise="original full input-shape normal, clean metadata restored",
            test_npz_opened=False, generation_quality_claim=False),
        runtime={**cuda_environment(device), "python": sys.version, "executable": sys.executable,
                 "torch": torch.__version__, "numpy": np.__version__, "threads": args.threads,
                 "backend_overrides": False, "monkeypatch": False,
                 "compute_policy": "CUDA required; no CPU fallback",
                 "host_boundaries": ["file I/O and WatchSleepNet preprocessing",
                     "original seeded constructor and exact diffusion schedule initialization",
                     "final result serialization"]},
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_before,
        training_args=vars(model_args), config_provenance=config_provenance,
        checkpoint_identity=checkpoint_identity, training_data_identity_checks=data_identity_checks,
        windows=identities, columns=list(columns), scaler=preprocessor.scaler,
        data_provenance=preprocessor.provenance(), source_sha256=sources,
        checkpoint_loading="official named_parameter.copy_; no load_state_dict and no buffer copies",
        buffer_checks=buffer_checks, load_checks=load_checks, all_parameters_and_buffers_unchanged_after_evaluation=True,
        parameter_gradients_absent=True, checkpoint_unchanged=True, summaries=totals)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_files(args.output_dir, "per_window", rows)
    table_files(args.output_dir, "per_subject", subjects)
    table_files(args.output_dir, "summary", totals)
    np.savez_compressed(args.output_dir / "raw_predictions.npz",
                        **{key: value.detach().cpu().numpy() for key, value in raw.items()})
    save_json(args.output_dir / "evaluation_report.json", report)
    lines = ["# 固定验证窗口的一步去噪检查", "",
        "这是外部只读观测：对记录中的目标 x0 正向加噪后，比较原模型初始化零输出头与训练后 epsilon 预测。"
        "不运行反向采样，不修改训练算法；这些是 teacher-forced 去噪指标，不是自由生成质量、FID 或整夜评价。", "",
        f"本次 {subject_count} 位 val 主体、{window_count} 个合法 120 秒窗口；每人选择首个及最接近30分钟的不同合法窗口，只有一个时保留一个。"
        f"使用训练manifest记录的数据和train-only scaler；test NPZ未打开。两种模型状态使用相同x_t/epsilon；B={args.batch_size}。", "",
        "| 模型状态 | 情景 | epsilon MSE | x0 窗口均值 MAE (ms) | x0 均值有符号偏差 (ms) |", "| --- | --- | --- | --- | --- |"]
    for row in totals:
        lines.append(f"| {row['model_state']} | {row['scenario']} | {row['epsilon_mse']:.6g} | "
            f"{row['x0_window_mean_absolute_error_ms']:.4f} | {row['x0_window_mean_signed_bias_ms']:+.4f} |")
    lines += ["", "先平均每人的已选窗口，再对实际参与验证的主体等权平均。x0 均值误差是辅助观测，不是新增训练损失；"
        "整体 epsilon MSE 与均值误差可以朝不同方向变化。零输出头重建使用带有真实目标信息的 x_t，不能将其均值表现解读为零模型能自由生成。", "",
        "模型由 np/torch seed42 和原 fetchModel 构造；加载前逐项验证固定 buffers 与保存值一致，"
        "随后严格采用原采样的 named_parameter.copy_ 加载参数，没有 load_state_dict 或布局修补。", "",
        f"配置来源：{config_provenance['source']}；数据身份、scaler与检查点核验详见JSON。", "",
        "逐窗口、逐主体 JSON/CSV、原始 epsilon 预测、固定输入/噪声、源码 SHA 和完整审计保存在同目录。", ""]
    (args.output_dir / "DENOISING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(status="passed", output=str(args.output_dir.resolve()), windows=window_count, rows=len(rows))), flush=True)


if __name__ == "__main__":
    main()
