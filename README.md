# DREAMT 条件 IBI 生成（CUDA 工作流）

基于固定版本 WaveStitch，适配 WatchSleepNet 的 DREAMT IBI 数据接口。每个输入窗口为 `[B, 3000, 10]`，对应 25 Hz、120 秒、四个连续的 30 秒睡眠阶段；生成目标为 IBI。九个条件通道依次为 W / Light / REM / Deep、AHI、BMI、AGE、gender_F、gender_M。

原模型主体、优化器、损失和 DDPM 采样公式保持固定上游版本。在 `de30ac2` 的数据与 CUDA 设备适配基础上，增加每轮生成验证、loss 数值平均及基于验证指标的早停；这些改动纳入源码检查器的逐项白名单。上游来源和许可证说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 安装

```bash
git clone https://github.com/TUTUisAbigCAT/dataset-generation.git
cd dataset-generation
```

使用 Python 3.12。`requirement.txt` 与 `requirements_cuda.txt` 固定 PyTorch 2.2.2 + CUDA 12.1 构建及其余依赖；来源参照 [PyTorch 官方历史版本说明](https://docs.pytorch.org/get-started/previous-versions/#v2-2-2)。需要可用的 NVIDIA GPU 与兼容驱动。若采用集群提供的 CUDA wheel，应记录实际版本并重新运行检查。

Linux / Fir（在克隆后的项目根目录）：

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirement.txt
python scripts/run_dreamt.py --check-only
```

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirement.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 --check-only
```

`run.ps1` 优先使用 `DATASET_GENERATION_PYTHON` 指定的解释器，然后使用项目内 `.venv`，最后读取可选的私人 `runtime_config.json`。公共仓库不包含任何本机解释器路径；没有可用环境时会说明如何安装。Linux 直接使用已激活环境中的 `python scripts/run_dreamt.py`。

## Fir 提交

先按上面的 Linux 命令准备 `./venv`，并在 `dataset_config.json` 中设置集群上的真实 IBI NPZ 路径。在项目根目录执行：

```bash
sbatch submit.sh
```

脚本除 Python 命令外，与提供的 WatchSleepNet `train_transfer_sleepconvnet.sh` 逐字一致：`def-plago`、4 CPU、60G、1 个 H100、24 小时，保留原 venv 激活和 pip 命令。使用该模板需具有对应账户权限。原 pip 行只列出 scikit-learn / numpy / pandas / torch，其余依赖需提前通过 `requirement.txt` 安装。

提交命令运行完整的 **训练 → 去噪评估 → 原始 200 步生成**，默认 10 个 epoch、batch size 8、4 个主机线程。各次运行单独保存源码、配置、checkpoint 和日志。

在**已经分配 GPU 的计算节点**上验证环境和完整流程：

```bash
python scripts/check_cuda.py
python scripts/smoke_cuda.py --threads 4
```

第二条命令自动创建四个人的人工短记录，运行 1 个 epoch、去噪评估和完整采样，并保存 `cuda_smoke_report.json`。它不修改真实数据，也不代表收敛或生成质量已合格。

## 数据目录与训练

默认 `dataset_config.json` 的 `data_dir` 为 `example_data`。其中四个人的短记录是**人工格式样例**，仅用于检查安装和入口，不是 DREAMT 观测数据。真实数据需用户自行准备，仓库不包含真实受试者数据或训练 checkpoint。

训练真实数据时，只需将 `data_dir` 改成兼容的 IBI NPZ 文件夹；`metadata_path` 和 `split_path` 可以保持 null。配置中的相对路径以配置文件所在目录为基准。

```text
DREAMT/
  participant_info.csv
  DREAMT_PIBI_SE_new/
    S001.npz
    S002.npz
    ...
```

- `data_dir` 读取原项目已提取的 IBI，默认 25 Hz、秒单位，不直接读取原始 BVP/ECG CSV。
- NPZ 使用原字段 `data, stages, fs, ahi`。不要求模拟器字段、时间索引或额外 subject_id。IBI 可以是向量或单列数组。
- CSV 位于 NPZ 文件夹或其直接上一层，必须有 `SID, AHI, BMI, AGE, GENDER`；SID 对应 NPZ 文件名，GENDER 为 F/M。原四字段 NPZ 不足以提供全部个人条件。
- 原始 stage 编码为 W=0、N1=1、N2=2、N3=3、REM=4；读取后合并为 W=0、Light=1、REM=2、Deep=3。不要传入已经合并的标签再映射一次。
- `splits.json` 可选；未提供时按固定种子 42 按受试者划分，目标比例 75% / 12.5% / 其余，至少需要三个人，且训练集和验证集各有合法两分钟窗口。已有 split 文件会严格校验。
- 100 Hz 输入可设置 `downsample_rate=4`；其他单位通过 `multiplier` 明确换算。

```bash
python scripts/run_dreamt.py --check-only
python scripts/run_dreamt.py --epochs 10 --batch-size 8 --threads 4
```

也可通过 `--data-dir /path/to/DREAMT_PIBI_SE_new` 覆盖 NPZ 目录，或把私人路径写进已忽略的 `dataset_config.local.json` 并传入 `--config dataset_config.local.json`。Windows 使用 `run.ps1` 时传入相同参数即可。

默认每次训练创建独立的 `runs/dreamt_<UTC时间戳>`，保存源码、配置、按人划分、训练集统计、checkpoint 和日志。每个 epoch 结束执行生成验证，训练结束后再执行去噪评估及原始 200 步采样。`--skip-generation` 只跳过最后的独立采样任务，每轮 validation 仍执行。旧模型评估绑定原训练 manifest，不跟随后来修改的数据目录。

## 每轮 validation 和平均 loss

每个 epoch 完成全部 optimizer 更新后（包括最后一轮），用当前权重在验证集条件下从噪声生成两分钟 IBI。执行与训练相同的完整扩散日程，默认 200 步；直接复用经过哈希检查的原始采样片段。验证窗口独立生成，因此推理 stride 为完整的 3000 点，不将不同片段或受试者互相拼接；训练 stride 为 750 点（30 秒），相邻两分钟窗口重叠 90 秒，即 75%。

在 `dataset_config.json` 中设置：

```json
"validation": {
  "max_windows": 64,
  "batch_size": 8,
  "seed": 20261020
}
```

默认从按受试者隔离的 val split 中固定选择最多 64 个互不重叠的合法窗口；不足时使用全部可用窗口，`max_windows: null` 使用全部不重叠窗口。候选搜索对齐 30 秒边界，按固定种子在受试者间轮流选取。训练期间窗口、顺序、采样种子不变；修改 batch_size 也可能改变采样随机流，跨实验比较应保持整个 validation 配置一致。所选窗口和实际数量记录在 `training/validation/protocol.json`，不读取 test split，不重新拟合标准化参数。

指标只统计 IBI，反归一化到毫秒后用 float64 计算，总体方差采用 `correction=0`：

| 字段 | 定义 |
|---|---|
| `mean_error_ms` | 所有验证样本合并后的生成均值 − 参考均值 |
| `variance_error_ms2` | 所有验证样本合并后的生成方差 − 参考方差 |
| `mean_absolute_error_ms` / `variance_absolute_error_ms2` | 上述整体差值的绝对值 |
| `mean_window_absolute_error_ms` | 每个两分钟窗口均值差的绝对值，再对窗口取平均 |
| `variance_window_absolute_error_ms2` | 每个两分钟窗口方差差的绝对值，再对窗口取平均 |

整体指标和逐窗口指标同时记录，避免不同窗口的正负偏差被相互抵消。原存储零值仍计入统计，不裁剪生成值。每轮使用独立的 eval 模型副本；退出验证时恢复随机数状态，不改变训练模型的模式、梯度、buffer 或后续训练随机流。

训练 `train_mean_loss = Σ(batch_MSE × batch样本数) / Σ(batch样本数)`，输出普通数值，不累积带计算图的 loss tensor。最后一个短 batch 按实际样本数加权；`training_report.json` 中的 `mean_loss` 为整个训练过程的同口径平均。反向传播仍使用原始 MSE，没有增加均值或方差损失项。

每轮结果见 `training/epoch_metrics.csv`，详细逐窗口结果见 `training/validation/epoch_0001/metrics.json` 等文件，最新摘要见 `training/validation/summary.json`。默认每轮最多增加 `ceil(64/8) × 200 = 1600` 次采样模型前向调用，并需要一个额外的模型副本；显存不足时可调小 validation.batch_size。均值、方差指标不能单独证明整夜连续性或条件生成质量。

## 基于均值和方差误差的早停

`dataset_config.json` 默认启用以下配置；旧配置未填写时也使用这些默认值：

```json
"early_stopping": {
  "enabled": true,
  "patience": 5,
  "min_delta_mean_ms": 0.0,
  "min_delta_variance_ms2": 0.0
}
```

监控 `mean_window_absolute_error_ms` 和 `variance_window_absolute_error_ms2`，越低越好。分别保存两项指标的最佳值，只有同一轮两项都改善才重置等待计数（AND 规则）；仅均值改善、仅方差改善或两项均未改善，都将等待计数加 1。连续 5 个 epoch 没有出现双指标同时改善，在该轮 validation 完成后停止。第一轮建立参考值，因此最早在第 6 轮结束。`--epochs 10` 是最大轮数，触发早停后不会继续训练剩余轮数；设置 `enabled: false` 可关闭早停。

改善要求误差下降幅度严格大于对应的 `min_delta`；默认 0 表示任何严格下降都算单项改善。需要忽略微小波动时，可分别设置毫秒和平方毫秒单位的正阈值。两项各自维护比较基准，单项改善也会更新该项基准，但不能单独清零等待计数；先后两轮分别改善不算同一轮整体改善。小幅改善可以累计超过各自阈值；最佳权重始终按实际最小误差保存。此处判断相对历史表现的改善，没有设置绝对误差达标线。均值与方差单位不同，不直接相加形成总分。

结果和权重：

- `training/epoch_metrics.csv` 记录 `joint_improvement`（本轮两项是否同时改善）、`early_stopping_bad_epochs` 和 `stopped_early`；每轮 JSON 同时记录最佳值、改善项和阈值参考值。
- `training/training_report.json` 的 `epochs` 是实际完成轮数，`requested_epochs` 是指定上限；`mean_loss` 只平均实际执行的训练样本。`stop_reason` 区分 `early_stopping` 与 `max_epochs`。
- `training/best_joint_model.pth` 保存双指标同时改善的整体最佳权重，与早停计数清零同步更新；第一轮保存初始参考 checkpoint。仅一项改善时不会覆盖此文件。`best_joint_manifest.json` 记录该轮两个指标、epoch 和权重哈希，供评估入口使用。
- `training/best_mean_model.pth` 和 `training/best_variance_model.pth` 分别保存两项指标最佳的权重，可能来自不同 epoch。相应的 `best_mean_manifest.json`、`best_variance_manifest.json` 记录权重哈希和所选 epoch，供原评估入口使用。
- 原 `saved_models/DREAMT/model.pth`（主工作流中位于运行目录的 `workspace/` 下）仍保存实际停止时的最终权重。主流程后续评估默认使用它，不自动将两项最佳模型之一认定为共同最优。

例如，训练结束后使用整体最佳权重执行原始采样（把路径替换成实际运行目录，输出目录须尚不存在）：

```bash
python scripts/evaluate_original_sampler.py \
  --training-args-json /path/to/run/training/best_joint_manifest.json \
  --output-dir /path/to/run/generation_best_joint
```

## 实验输出目录

在同一个 `dataset_config.json` 中设置 `output_root`，指定实验输出的父目录；项目默认值为 `"runs"`。例如（仅展示相关字段，保留其余数据配置）：

```json
{
  "data_dir": "/path/to/DREAMT_PIBI_SE_new",
  "output_root": "/scratch/your_username/dataset-generation-runs"
}
```

每次运行在该目录下创建 `dreamt_<UTC时间戳>/`，保存 `status.json`、数据预检以及训练、评估和生成结果。`--check-only` 也使用此位置。`output_root` 的相对路径以所选 JSON 文件所在目录为基准，支持 `~`；应填写非空字符串。旧配置未填写该字段时，仍使用项目内的 `runs/`。

直接执行 `sbatch submit.sh` 即会读取该配置。命令行 `--output-dir /path/to/new_run` 优先级更高，它指定最终的单次运行目录，不再追加时间戳；相对路径仍以启动命令的工作目录为基准，已有目录会报错以避免覆盖。Slurm 自身的 `slurm-<jobid>.out` 日志及单独运行的 smoke 脚本仍遵循各自的输出设置。

## CUDA 执行范围

生产计算入口强制检查 CUDA；没有可用 GPU、安装了 CPU 版 Torch 或设备初始化失败，会直接报错，不会回退 CPU。模型参数、buffer、训练随机数、条件 mask、时间步、前向/反向、损失、去噪重建、采样及评估张量运算在同一 CUDA 设备上。运行报告记录实际设备、GPU 名称和 Torch / CUDA / cuDNN 版本。

主机负责文件读写、WatchSleepNet 数据预处理、标签和索引元数据、原 seed42 模型初始化、NumPy 扩散表初始化及最终序列化；模型和扩散表在计算前整体转入 GPU。原 Adam 控制计数器允许留在主机，以保留原优化器配置。

`--check-only` 和旧短跑脚本的 `--prepare-only` 只验证数据，不运行模型，因此无需 GPU。CUDA 随机流不承诺与历史 CPU 随机流逐位一致。

## 与 WatchSleepNet 的一致范围

原始读取的 flatten → 倍率 → 整数 stride → 完整 30 秒切段 → 四类标签映射，以及信号零标记、tuple / collate 约定，都与提供的 `SSDataset` 对齐。原文件与冻结快照 SHA256 一致。原训练 engine 在调用模型前没有额外输入归一化或增强；网络内部 BN 属于原分类模型。

生成任务保留以下明确适配，因此整条分类和生成训练链路并不相同：

| 项目 | 原 WatchSleepNet | 本生成任务 |
|---|---|---|
| 模型输入 | 整夜 epoch 序列，可 padding | 同一人的四个 epoch，3000 点，不跨人或 padding |
| 长度上限 | 1100 个 epoch | 默认保留整夜；max_epochs=1100 可复现上限 |
| 标签条件 | 四类分类目标 | 四类 one-hot 条件，另加 AHI/BMI/AGE/GENDER |
| 数值标准化 | loader/engine 不做输入 z-score | 仅训练集拟合 IBI 与连续个人条件的 z-score |
| 分组与窗口 | 原任务的受试者 split/k-fold | 固定按人划分，120 秒窗口、30 秒步长（75% 重叠） |
| 无效片段 | 原 loader 保留并取段首标签 | 过滤非有限/负值、无效或段内不一致标签及已知时间缺口 |

原提取脚本的片段首尾、异常位置及峰提取失败产生的零值原样保留，不插值或裁剪；统计和原 MSE 包含这些存储零标记。没有时间索引的四字段 NPZ 只能依据存储顺序，无法恢复预处理已删除的原始时间间隔。

## 验证与限制

可复现检查命令：

```bash
python -B -m unittest discover -s tests -v
python -B scripts/verify_upstream_boundary.py --output runs/source_boundary.json
python -B scripts/audit_watchsleep_chain.py --output-dir runs/chain_audit
```

原数据链路回归检查覆盖 25 Hz 与 100 Hz 输入的 epoch 信号和标签、四段拼接的九个条件通道及反归一化。新增测试覆盖均值/总体方差、短 batch 加权平均、最后一轮验证、固定窗口、随机状态恢复、模型副本和原始 200 步采样一致性。缺少 CUDA 时，真实 GPU 测试会显式跳过，不能视为通过。

12 份 WaveStitch 上游快照和 5 个导入核心文件逐字未改；移除允许的数据、10 项设备适配、6 项 epoch 验证/日志改动及 1 项早停分支后，训练循环与上游相同。原始采样执行片段逐字未改。检查器可重新生成详细报告；本机历史运行日志和机器路径未纳入公共仓库。

目前尚未完成 Fir GPU 实测，也未用真实 DREAMT 数据验证生成质量。人工短记录和接口测试不能证明收敛，跳过的 GPU 测试不能视为通过。请在已分配的 GPU 计算节点上执行上述 `check_cuda.py` 和 `smoke_cuda.py` 后再进行正式训练。
