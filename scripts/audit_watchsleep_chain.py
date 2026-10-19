"""Audit the data-only WatchSleepNet -> conditional-window numerical chain.

The frozen original SSDataset is the epoch-level numerical oracle. Temporary
fixtures exercise composed transformations and intentional task differences;
this does not run a model or claim equivalence of classification/generation
training pipelines, real-data quality, or physiological validity.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils import Preprocessor
from watchsleep_loader import SSDataset, UPSTREAM_SOURCE_SHA256

SOURCE = ROOT / "third_party/WatchSleepNet/data_setup.py"
COLUMNS = ["IBI", "stage_W", "stage_Light", "stage_REM", "stage_Deep",
           "AHI", "BMI", "AGE", "gender_F", "gender_M"]
INVERSE_ATOL_SECONDS = 2e-7
STANDARDIZED_ATOL = 2e-6


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_ssdataset(source_path=SOURCE):
    """Extract only the immutable class; do not import unrelated sklearn code."""
    source_path = Path(source_path)
    require(sha256(source_path) == UPSTREAM_SOURCE_SHA256, "WatchSleepNet oracle source hash mismatch")
    tree = ast.parse(source_path.read_text(encoding="utf-8-sig"), filename=str(source_path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SSDataset"]
    require(len(classes) == 1, "Expected exactly one original SSDataset class")
    scope = dict(np=np, torch=torch, Dataset=Dataset, pad_sequence=pad_sequence, json=json, os=os, Path=Path)
    code = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    exec(compile(code, str(source_path), "exec"), scope)
    return scope["SSDataset"]


def file_snapshot(directory):
    return {path.relative_to(directory).as_posix(): sha256(path)
            for path in directory.rglob("*") if path.is_file()}


def write_fixture(directory, epoch_counts, *, fs, multiplier, raw_tail, missing_ahi=None):
    """Create independent ordinary files, fixed subject splits, and CSV values."""
    directory.mkdir(parents=True, exist_ok=True)
    names = ["S101.npz", "S102.npz", "S103.npz", "S104.npz"]
    require(len(epoch_counts) == len(names), "Fixture requires four distinct subjects")
    splits = dict(train=names[:2], val=[names[2]], test=[names[3]])
    rows, specs = [], {}
    for index, (name, epochs) in enumerate(zip(names, epoch_counts)):
        row = dict(SID=Path(name).stem, AHI=2.123456789 + 7.5*index,
                   BMI=21.25 + 4.*index, AGE=31. + 13.*index, GENDER="F" if index % 2 == 0 else "M")
        rows.append(row)
        n = epochs*30*fs + raw_tail
        t = np.arange(n, dtype=np.float64)
        # Source units are deliberately different in the stride/unit test.
        values = ((.72 + .035*index + .021*np.sin(t/53.) + .002*((t//(30*fs)) % 5))/multiplier).astype(np.float32)
        complete = values[:epochs*30*fs].reshape(epochs, 30*fs)
        complete[:, :8*int(fs/25)] = 0.
        complete[:, -11*int(fs/25):] = 0.
        complete[:, 210*int(fs/25):214*int(fs/25)] = 0.
        if index == 0 and epochs > 2:
            complete[2] = 0.
        raw_labels = np.resize(np.asarray([0, 1, 2, 4, 3], dtype=np.int8), epochs)
        stages = np.repeat(raw_labels, 30*fs)
        if raw_tail:
            stages = np.concatenate((stages, np.full(raw_tail, raw_labels[-1], dtype=np.int8)))
        payload = dict(data=values if index % 2 else values[:, None], stages=stages, fs=np.asarray(fs))
        if name != missing_ahi:
            payload["ahi"] = np.asarray(row["AHI"], dtype=np.float32)
        np.savez(directory / name, **payload)
        specs[name] = dict(epochs=epochs, raw_samples=n, raw_stage_epochs=raw_labels.tolist(),
                           ahi_omitted=name == missing_ahi)
    with (directory / "participant_info.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("SID", "AHI", "BMI", "AGE", "GENDER"))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    return dict(splits=splits, rows={row["SID"] + ".npz": row for row in rows}, records=specs)


def preprocessor(directory, downsample_rate, multiplier):
    return Preprocessor("DREAMT", config_override=dict(data_dir=str(directory.resolve()),
        metadata_path=str((directory / "participant_info.csv").resolve()),
        split_path=str((directory / "splits.json").resolve()),
        multiplier=multiplier, downsample_rate=downsample_rate))


def max_error(actual, expected):
    require(actual.shape == expected.shape, f"Array shape mismatch: {actual.shape} vs {expected.shape}")
    return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if actual.size else 0.


def audit_numeric_case(directory, *, fs=25, downsample_rate=1, multiplier=1., missing_ahi=None):
    """Compare every legal train/val window to independently composed epochs."""
    require(fs/downsample_rate == 25, "Audit case must yield 25 Hz by an integer stride")
    tail = 29 if fs == 25 else 173
    fixture = write_fixture(directory, [7, 5, 6, 4], fs=fs, multiplier=multiplier,
                            raw_tail=tail, missing_ahi=missing_ahi)
    before = file_snapshot(directory)
    original = original_ssdataset()
    names = fixture["splits"]["train"] + fixture["splits"]["val"]
    oracle = {name: original(directory, "dreamt", [name], multiplier=multiplier,
                             downsample_rate=downsample_rate)[0] for name in names}
    processor = preprocessor(directory, downsample_rate, multiplier)
    require(list(processor.columns) == COLUMNS, "Conditional channel order differs from contract")
    reference_signal = np.concatenate([oracle[name][0].numpy().reshape(-1)
                                       for name in fixture["splits"]["train"]]).astype(np.float64)
    ref_mean, ref_std = float(reference_signal.mean()), float(reference_signal.std())
    train_metadata = np.asarray([[fixture["rows"][name][key] for key in ("AHI", "BMI", "AGE")]
                                 for name in fixture["splits"]["train"]], dtype=np.float64)
    meta_mean, meta_std = train_metadata.mean(axis=0), train_metadata.std(axis=0)
    require(abs(processor.scaler["signal_mean_seconds"] - ref_mean) <= 1e-12, "Scaler mean differs from original train epochs")
    require(abs(processor.scaler["signal_std_seconds"] - ref_std) <= 1e-12, "Scaler std differs from original train epochs")
    require(processor.scaler["signal_fit_samples"] == reference_signal.size, "Padding, tail or held-out samples entered scaler")
    require(np.allclose(processor.scaler["continuous_mean"], meta_mean, atol=1e-12, rtol=0)
            and np.allclose(processor.scaler["continuous_std"], meta_std, atol=1e-12, rtol=0), "Metadata scaler differs from training CSV rows")
    epoch_rows, window_rows = [], []
    for name in names:
        signal, labels, count, legacy_ahi = oracle[name]
        record = processor.record(name)
        expected_signal = signal.numpy()
        expected_labels = labels.numpy()
        require(np.array_equal(record["ibi_epochs"], expected_signal), f"Epoch signal differs from original: {name}")
        require(np.array_equal(record["stage_epochs"], expected_labels), f"Epoch labels differ from original: {name}")
        require(record["valid_epochs"].all(), "Valid stored zero markers were rejected")
        require(record["continuity_source"] == "stored_order_only", "Plain NPZ must not imply verified source-time continuity")
        expected_tail = ((fixture["records"][name]["raw_samples"] + downsample_rate - 1)//downsample_rate) % 750
        require(record["discarded_tail_samples"] == expected_tail, "Tail accounting differs after integer stride")
        if fixture["records"][name]["ahi_omitted"]:
            require(legacy_ahi == 0 and record["ahi"] is None, "Expected explicit legacy missing-AHI distinction")
            require(record["continuous"][0] == fixture["rows"][name]["AHI"], "CSV AHI fallback was not used")
        else:
            require(record["ahi"] == legacy_ahi, "Present NPZ AHI differs from original tuple")
        epoch_rows.append(dict(filename=name, split="train" if name in fixture["splits"]["train"] else "val",
            epochs=int(count), epoch_signal_bitwise_equal=True, labels_bitwise_equal=True,
            zero_samples=int((expected_signal == 0).sum()), all_zero_epochs=int((expected_signal == 0).all(axis=1).sum()),
            discarded_25hz_tail_samples=expected_tail, legacy_tuple_ahi=float(legacy_ahi),
            condition_ahi=float(record["continuous"][0]), ahi_source=record["ahi_source"]))
    for split in ("train", "val"):
        dataset = (processor.make_training_dataset(3000, 1500) if split == "train"
                   else processor.make_validation_dataset(3000, 1500))
        expected_index = [(name, first*750) for name in fixture["splits"][split]
                          for first in range(0, oracle[name][2] - 4 + 1, 2)]
        require(dataset.window_index == expected_index, "Windows cross subjects, use padding, or shift the epoch grid")
        for index, (name, start) in enumerate(expected_index):
            first = start//750
            signal, labels, _, _ = oracle[name]
            # Construct each expected channel from the original epoch tensors
            # and independent CSV statistics, not the adapter's window output.
            raw_window = signal[first:first+4].numpy().reshape(-1)
            epoch_labels = labels[first:first+4].numpy()
            expected = np.empty((3000, 10), dtype=np.float32)
            expected[:, 0] = (raw_window.astype(np.float64) - ref_mean)/ref_std
            expected[:, 1:5] = np.eye(4, dtype=np.float32)[np.repeat(epoch_labels, 750)]
            metadata = np.asarray([fixture["rows"][name][key] for key in ("AHI", "BMI", "AGE")])
            expected[:, 5:8] = (metadata-meta_mean)/meta_std
            expected[:, 8] = fixture["rows"][name]["GENDER"] == "F"
            expected[:, 9] = fixture["rows"][name]["GENDER"] == "M"
            actual = dataset[index].numpy()
            require(actual.shape == (3000, 10), "Unexpected generation input shape")
            require(np.array_equal(actual[:, 1:], expected[:, 1:]), "Stage/metadata/gender channel mismatch")
            standardized_error = max_error(actual[:, :1], expected[:, :1])
            restored = processor.inverse_signal(actual[:, 0])
            inverse_error = max_error(restored, raw_window)
            require(standardized_error <= STANDARDIZED_ATOL, "Standardized IBI differs from independently composed oracle")
            require(inverse_error <= INVERSE_ATOL_SECONDS, "Inverse scaling does not recover original IBI window")
            window_rows.append(dict(split=split, filename=name, first_epoch=first, start_sample=start,
                raw_four_epoch_points=3000, condition_channels_exact=True,
                standardized_ibi_max_absolute_error=standardized_error,
                inverse_ibi_max_absolute_error_seconds=inverse_error,
                zero_markers_in_oracle_window=int((raw_window == 0).sum())))
    try:
        processor.record(fixture["splits"]["test"][0])
    except ValueError:
        pass
    else:
        raise AssertionError("Test subject was not sealed")
    require(set(processor.accessed_files) == set(names), "Unexpected subject accessed")
    require(before == file_snapshot(directory), "Audit mutated a data file")
    return dict(status="passed", source_fs=fs, effective_fs=25, downsample_rate=downsample_rate,
        multiplier=multiplier, records=epoch_rows, windows=window_rows, splits=fixture["splits"],
        train_window_count=sum(row["split"] == "train" for row in window_rows),
        val_window_count=sum(row["split"] == "val" for row in window_rows),
        train_only_scaler_verified=True, original_epoch_signal_bitwise_equal=True,
        original_epoch_labels_bitwise_equal=True, condition_channels_exact=True,
        maximum_standardized_ibi_error=max(row["standardized_ibi_max_absolute_error"] for row in window_rows),
        maximum_inverse_ibi_error_seconds=max(row["inverse_ibi_max_absolute_error_seconds"] for row in window_rows),
        test_npz_opened=False, data_files_unchanged=True,
        independent_signal_mean_seconds=ref_mean, independent_signal_std_seconds=ref_std,
        scaler_fit_sample_count=int(reference_signal.size))


def audit_boundary_case(directory):
    """Quantify actual-length/padding and original-cap differences together."""
    fixture = write_fixture(directory, [1105, 3, 4, 4], fs=25, multiplier=1., raw_tail=37)
    before = file_snapshot(directory)
    original = original_ssdataset()
    old_long = original(directory, "dreamt", ["S101.npz"])[0]
    old_short = original(directory, "dreamt", ["S102.npz"])[0]
    padded_ibi, padded_labels, lengths, _ = original.collate_fn([old_long, old_short])
    require(tuple(padded_ibi.shape) == (2, 1100, 750), "Unexpected original collate shape")
    require(lengths.tolist() == [1100, 3], "Original collate lost actual lengths")
    require(torch.all(padded_ibi[1, 3:] == 0).item() and torch.all(padded_labels[1, 3:] == -1).item(),
            "Unexpected original padding sentinel values")
    require(torch.equal(padded_ibi[1, :3], old_short[0]), "Original padding altered real short-record epochs")
    adapter_long = SSDataset(directory, "dreamt", ["S101.npz"])[0]
    limited = SSDataset(directory, "dreamt", ["S101.npz"], max_epochs=1100)[0]
    require(adapter_long[2] == 1105 and old_long[2] == 1100, "Original/default length-cap distinction is wrong")
    require(torch.equal(limited[0], old_long[0]) and torch.equal(limited[1], old_long[1])
            and limited[2:] == old_long[2:], "Explicit max_epochs=1100 does not reproduce original tuple")
    require(torch.equal(adapter_long[0][:1100], old_long[0]), "Original retained prefix differs")
    processor = preprocessor(directory, 1, 1.)
    dataset = processor.make_training_dataset(3000, 1500)
    expected_index = [("S101.npz", first*750) for first in range(0, 1105-4+1, 2)]
    require(dataset.window_index == expected_index, "Padding, subject boundary, or truncated epochs entered windows")
    try:
        processor.window_array("S102.npz", 0, 3000)
    except ValueError:
        pass
    else:
        raise AssertionError("A three-epoch record was padded into a generation window")
    require(processor.scaler["signal_fit_samples"] == (1105+3)*750, "Collate padding or source tail entered scaler")
    last_start = expected_index[-1][1]
    last_window = dataset[-1].numpy()
    expected_raw = adapter_long[0][last_start//750:last_start//750+4].numpy().reshape(-1)
    inverse_error = max_error(processor.inverse_signal(last_window[:, 0]), expected_raw)
    require(inverse_error <= INVERSE_ATOL_SECONDS, "Last retained window no longer recovers original sample order")
    require(before == file_snapshot(directory), "Boundary audit mutated data")
    return dict(status="passed", original_long_epochs=1100, adapter_default_long_epochs=1105,
        explicit_cap_1100_tuple_exact=True, common_1100_epoch_prefix_exact=True,
        original_padded_batch_shape=list(padded_ibi.shape), original_actual_lengths=lengths.tolist(),
        original_short_padding_epochs=1097, original_padding_ibi=0, original_padding_label=-1,
        short_record_epochs=3, generation_windows_from_short_record=0,
        generated_training_windows=len(dataset), windows_if_original_1100_cap_were_used=len(range(0, 1100-4+1, 2)),
        final_window_start_epoch=last_start//750, final_window_inverse_max_error_seconds=inverse_error,
        signal_scaler_points=(1105+3)*750, padding_in_signal_scaler=False, padding_in_generation_windows=False,
        data_files_unchanged=True, test_npz_opened=False)


def report_markdown(report):
    cases = report["numeric_cases"]
    lines = ["# WatchSleepNet → 四分类条件窗口：独立数值链路验收", "",
        "结果：临时同格式数据上的数值链路检查通过。原 SSDataset 的 30 秒 IBI 和标签是独立参考；"
        "没有运行模型、训练或生成评估。本结论不代表真实 DREAMT 数据质量或两个任务的训练 dataloader 完全相同。", "",
        "| 输入与转换 | 对照窗口数 | 原 epoch IBI / 标签 | 9 条件通道 | 最大反归一化误差（秒） |",
        "| --- | --- | --- | --- | --- |"]
    for case in cases:
        lines.append(f"| {case['source_fs']} Hz，stride={case['downsample_rate']}，multiplier={case['multiplier']:g} | "
                     f"{len(case['windows'])} | 逐值相等 | 逐值相等 | {case['maximum_inverse_ibi_error_seconds']:.12g} |")
    boundary = report["boundary_case"]
    lines += ["", "检查从原 SSDataset 完整输出开始，独立拼接同一人的四个 epoch，并独立计算训练集总体均值/标准差。"
        "逐窗核对 3000×10 输入：IBI、W/Light/REM/Deep one-hot、AHI/BMI/AGE、F/M。"
        "N1 和 N2 均映射到 Light；原标签 3 为 Deep，4 为 REM。25 Hz 和 100 Hz 按原顺序执行 flatten → multiplier → 整数 stride，未添加插值。", "",
        "局部零点和全零 epoch 在数值对照中原样保留，反归一化恢复零标记；它们不是插补值，也不能解释成有效生理 IBI。"
        "下采样后的不足 30 秒尾段未进入统计或训练窗。AHI 存在时与原 tuple 相同并核对 CSV；缺失时，"
        "原分类 tuple 返回 0，而条件适配器使用必需的 CSV AHI，这是明确的条件接口扩展。", "",
        "以下是生成任务有意保留的区别，不能称为完全复刻原分类训练流程：", "",
        "- 原 SSDataset 返回一整条记录的 epoch 张量；生成训练使用同一个人的四个合法 epoch，步长两个 epoch。",
        "- 原 collate 按 batch 最长记录填充 IBI=0、标签=-1，并返回实际长度。生成训练只使用真实完整窗口，不从 padding 构造样本。",
        f"- 本边界案例中原 batch 为 {boundary['original_padded_batch_shape']}，实际长度 {boundary['original_actual_lengths']}；"
        f"短记录的 {boundary['original_short_padding_epochs']} 个 padding epoch 不参与 scaler，也没有被拼成训练窗口。",
        f"- 原 loader 强制最多 1100 epoch；适配器默认保留 {boundary['adapter_default_long_epochs']} 个完整 epoch。"
        "显式 max_epochs=1100 时，tuple 与原实现逐值一致。",
        f"- 长记录默认产生 {boundary['generated_training_windows']} 个窗口；按原 1100 上限会产生 "
        f"{boundary['windows_if_original_1100_cap_were_used']} 个窗口。最后一个实际窗口起点为 epoch {boundary['final_window_start_epoch']}。",
        "- 生成任务增加 CSV 个人条件和训练集专用 scaler，固定受试者划分后再取窗口；未复刻原 create_dataloaders/random_split 的随机划分过程。",
        "- 适配器会排除无效标签、负/非有限 IBI、混合标签 epoch，以及已知时间缺口跨越窗。普通四字段 NPZ 只保证存储顺序，不能据此还原删除过的原始时间。", "",
        f"反归一化容差为 {INVERSE_ATOL_SECONDS:g} 秒，用于 float32 标准化往返；未裁剪或修补误差。"
        "审计仅创建临时数据，按人分为两个 train、一个 val、一个 test；test NPZ 未被 loader 打开。输入文件哈希前后保持一致。", "",
        "冻结原加载器 SHA256：`" + report["source_sha256"]["third_party/WatchSleepNet/data_setup.py"] + "`。"
        "完整逐窗误差、尾段数、AHI 来源、边界计数和源码哈希见同目录 JSON。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/recheck_export")
    parser.add_argument("--fixture-parent", type=Path, default=ROOT / "work/loader_chain_audit")
    parser.add_argument("--overwrite", action="store_true", help="Replace only this audit's JSON and Markdown reports")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    json_path, md_path = output / "watchsleep_chain_audit.json", output / "WATCHSLEEP_CHAIN_AUDIT.md"
    require(args.overwrite or not (json_path.exists() or md_path.exists()), "Audit reports already exist; use --overwrite")
    source_names = ("third_party/WatchSleepNet/data_setup.py", "watchsleep_loader.py", "data_utils.py",
                    "scripts/audit_watchsleep_chain.py", "tests/test_loader_chain.py")
    source_hashes = {name: sha256(ROOT / name) for name in source_names}
    args.fixture_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chain_", dir=args.fixture_parent) as temporary:
        directory = Path(temporary)
        numeric = [audit_numeric_case(directory / "fs25", fs=25, downsample_rate=1, multiplier=1.),
                   audit_numeric_case(directory / "fs100", fs=100, downsample_rate=4, multiplier=.001, missing_ahi="S102.npz")]
        boundary = audit_boundary_case(directory / "boundaries")
    require(source_hashes == {name: sha256(ROOT / name) for name in source_names}, "Source changed during audit")
    loaded_model_modules = [name for name in sys.modules if name == "training_utils" or name.startswith("TSImputers")]
    require(not loaded_model_modules, "Model modules were imported by a data-only audit")
    report = dict(status="passed", audit_type="temporary-fixture independent numerical chain",
        original_loader_oracle="Frozen SSDataset class extracted by AST; no original classification training loop",
        source_sha256=source_hashes, source_files_unchanged=True, numeric_cases=numeric, boundary_case=boundary,
        tolerances=dict(standardized_absolute=STANDARDIZED_ATOL, inverse_seconds_absolute=INVERSE_ATOL_SECONDS),
        model_imported=False, model_executed=False, real_dataset_evaluated=False,
        classifier_and_generator_training_dataloaders_identical=False)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    markdown = report_markdown(report)
    output.mkdir(parents=True, exist_ok=True)
    json_path.write_text(serialized + "\n", encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    print(json.dumps(dict(status="passed", json=str(json_path), markdown=str(md_path),
                         maximum_inverse_error_seconds=max(case["maximum_inverse_ibi_error_seconds"] for case in numeric))))


if __name__ == "__main__":
    main()
