"""Create small artificial four-field DREAMT-format records for interface checks.

These waveforms are generated here; they are not observations from DREAMT or a
physiological simulator. This utility does not load a model or train anything.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def create_example_data(output_dir, *, fixed_splits=False):
    """Write a fresh 4-subject, 8-epoch example; return its generation metadata.

    Default examples omit splits.json so the normal adapter demonstrates its
    deterministic subject split. The standalone 3-update smoke test requests
    fixed_splits=True to guarantee two train, one val and one test subject.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261019)
    fs, epoch_samples, epochs = 25, 750, 8
    samples = epoch_samples * epochs
    seconds = np.arange(samples, dtype=np.float64) / fs
    subjects, rows, records = [], [], []
    stage_epochs = np.array([0, 1, 2, 3, 4, 2, 1, 0], dtype=np.int8)
    for index in range(4):
        sid = f"EXAMPLE{index + 1:03d}"
        subjects.append(sid)
        row = dict(SID=sid, AHI=5. + index * 4., BMI=22. + index * 1.5,
                   AGE=30. + index * 8., GENDER="F" if index % 2 == 0 else "M")
        rows.append(row)
        signal = (.72 + index * .035 + .025 * np.sin(2*np.pi*seconds/19. + index)
                  + .008 * np.sin(2*np.pi*seconds/4.7) + rng.normal(0., .001, samples)).astype(np.float32)
        # Deliberate stored missing-value markers, as found in the original
        # preprocessing interface. Zero is not a physiological IBI measurement.
        for epoch in range(epochs):
            signal[epoch*epoch_samples:epoch*epoch_samples + 5] = 0.
            signal[(epoch + 1)*epoch_samples - 5:(epoch + 1)*epoch_samples] = 0.
        signal[2*epoch_samples + 200:2*epoch_samples + 215] = 0.
        path = output / (sid + ".npz")
        np.savez_compressed(path, data=signal[:, None],
            stages=np.repeat(stage_epochs, epoch_samples), fs=np.asarray(fs, dtype=np.int32),
            ahi=np.asarray(row["AHI"], dtype=np.float32))
        records.append(dict(filename=path.name, samples=samples, epochs=epochs,
            zero_samples=int((signal == 0).sum()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with (output / "participant_info.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("SID", "AHI", "BMI", "AGE", "GENDER"))
        writer.writeheader()
        writer.writerows(rows)
    if fixed_splits:
        splits = dict(train=[sid + ".npz" for sid in subjects[:2]],
                      val=[subjects[2] + ".npz"], test=[subjects[3] + ".npz"])
        (output / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    info = dict(artificial_example=True, real_DREAMT_data=False,
        intended_use="Data-interface and execution checks only; no physiological or model-quality claim",
        seed=20261019, subjects=4, epochs_per_subject=epochs, sample_rate_hz=fs,
        ibi_units="seconds", raw_stage_codes=dict(W=0, N1=1, N2=2, N3=3, REM=4),
        npz_fields=["data", "stages", "fs", "ahi"],
        zero_value_policy="Deliberate missing-value markers retained in the stored target signal",
        fixed_splits=bool(fixed_splits), records=records)
    (output / "EXAMPLE_DATA.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# 人工接口样例（非真实 DREAMT 数据）\n\n"
        "本目录由 scripts/create_example_data.py 现场生成：4 位虚构受试者，每人 8 个 30 秒 epoch，"
        "25 Hz、4 分钟、IBI 单位为秒。波形仅用于文件接口和程序执行测试，不模拟真实生理分布。\n\n"
        "每个 NPZ 仅含 data [6000,1]、stages [6000]、fs、ahi 四字段；"
        "原始阶段编码为 W=0、N1=1、N2=2、N3=3、REM=4。CSV 提供 SID/AHI/BMI/AGE/GENDER。"
        "局部 0 是刻意加入的缺测标记，不是真实生理 IBI。\n\n"
        + ("本短测样例附带固定 train=2 / val=1 / test=1 划分。\n" if fixed_splits else
           "未附带 splits.json，用于演示适配器的固定种子自动受试者划分。\n")
        + "\n训练或生成结果只能说明接口可执行，不能用于真实 DREAMT 效果或生成质量结论。\n",
        encoding="utf-8")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="A new directory; existing paths are never overwritten")
    args = parser.parse_args()
    info = create_example_data(args.output_dir)
    print(json.dumps(dict(output_directory=str(args.output_dir.resolve()), **info), indent=2))


if __name__ == "__main__":
    main()
