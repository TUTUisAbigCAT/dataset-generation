# DREAMT 条件 IBI 生成（CUDA 工作流）

基于固定版本 WaveStitch，适配 WatchSleepNet 的 DREAMT IBI 数据接口。每个输入窗口为 `[B, 750 × K, 10]`，其中 `K=window_epochs` 是连续 30 秒睡眠 epoch 的数量，默认 4（25 Hz、120 秒、3000 点）；生成目标为 IBI。九个条件通道依次为 W / Light / REM / Deep、AHI、BMI、AGE、gender_F、gender_M。

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

当前 `submit.sh` 使用用户指定的 `def-plago` 账户、4 CPU、1 GPU、10 小时时限，加载 StdEnv/2023、gcc/12.3、opencv/4.9.0 和 python/3.11，激活 `./.venv` 后运行主入口。依赖需提前安装，提交脚本中的 pip 行保持注释。

提交命令运行完整的 **训练 → 去噪评估 → 原始 200 步生成**，默认最多 50 个训练 epoch、batch size 8、4 个主机线程。各次运行单独保存源码、配置、checkpoint 和日志。

在**已经分配 GPU 的计算节点**上验证环境和完整流程：

```bash
python scripts/check_cuda.py
python scripts/smoke_cuda.py --threads 4
```

第二条命令自动创建四个人的人工短记录，运行 1 个 epoch、去噪评估和完整采样，并保存 `cuda_smoke_report.json`。它不修改真实数据，也不代表收敛或生成质量已合格。

## 数据目录与训练

`dataset_config.json` 当前保留用户指定的集群数据路径；本地格式检查可通过 `--data-dir example_data` 使用随附样例。其中四个人的短记录是**人工格式样例**，仅用于检查安装和入口，不是 DREAMT 观测数据。真实数据需用户自行准备，仓库不包含真实受试者数据或训练 checkpoint。

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
- `splits.json` 可选；未提供时按固定种子 42 按受试者划分，目标比例 75% / 12.5% / 其余，至少需要三个人，且训练集和验证集各有至少一个配置长度的合法窗口。已有 split 文件会严格校验。
- 100 Hz 输入可设置 `downsample_rate=4`；其他单位通过 `multiplier` 明确换算。

```bash
python scripts/run_dreamt.py --check-only
python scripts/run_dreamt.py --epochs 50 --batch-size 8 --threads 4
```

也可通过 `--data-dir /path/to/DREAMT_PIBI_SE_new` 覆盖 NPZ 目录，或把私人路径写进已忽略的 `dataset_config.local.json` 并传入 `--config dataset_config.local.json`。Windows 使用 `run.ps1` 时传入相同参数即可。

默认每次训练创建独立的 `runs/dreamt_<UTC时间戳>`，保存源码、配置、按人划分、训练集统计、checkpoint 和日志。每个 epoch 结束执行生成验证，训练结束后再执行去噪评估及原始 200 步采样。`--skip-generation` 只跳过最后的独立采样任务，每轮 validation 仍执行。旧模型评估绑定原训练 manifest，不跟随后来修改的数据目录。

## 配置窗口包含的睡眠 epoch 数

在 `dataset_config.json` 顶层设置：

```json
"window_epochs": 4
```

| window_epochs | 时间长度 | 窗口点数 / 向量维度 |
|---|---|---|
| 1 | 30 秒 | 750 |
| 2 | 60 秒 | 1500 |
| 4（默认） | 120 秒 | 3000 |
| 8 | 240 秒 | 6000 |

该值必须是正整数，省略时兼容旧配置使用 4。每个睡眠 epoch 固定为 30 秒 × 25 Hz = 750 点。主入口自动令 `window_size = s4_lmax = 750 × window_epochs`，数据加载、条件标签数、验证向量维度、去噪评估和生成使用同一长度；不需要改 `submit.sh` 或 Python 命令。短于该长度的记录无法提供窗口，若一个 split 没有合法窗口会明确报错。

`window_epochs` 与训练次数 `--epochs`、单条记录截断选项 `max_epochs` 含义不同，也不会改变 W/Light/REM/Deep 四个标签类别或 10 个输入通道。新运行将窗口配置写入源码快照、冻结数据身份和 manifest；评估 checkpoint 按其原训练长度运行，不受后来修改全局 JSON 的影响。旧身份未记录该字段时按 4 处理。

## 每轮 validation、对角高斯 KL 和平均 loss

每个 epoch 完成全部 optimizer 更新后（包括最后一轮），用当前权重在验证集条件下从噪声生成配置长度的 IBI。执行与训练相同的完整扩散日程，默认 200 步；直接复用经过哈希检查的原始采样片段。验证窗口独立生成，推理 stride 为完整的 `L = 750 × window_epochs` 点，不将不同片段或受试者互相拼接；训练 stride 固定为 750 点（30 秒），K=4 时相邻窗口重叠 90 秒，K=1 时不重叠。

在 `dataset_config.json` 中设置：

```json
"validation": {
  "max_windows": null,
  "batch_size": 8,
  "seed": 20261020,
  "kl_variance_floor_ms2": 1.0
}
```

默认使用按受试者隔离的 val split 中全部互不重叠的配置长度合法窗口（`max_windows: null`）。不足一个窗口的尾段和含无效睡眠标签的窗口不参与；候选搜索对齐 30 秒边界。只有显式设置整数 `max_windows` 才执行固定种子的子集验证。所选窗口及数量记录在 `training/validation/protocol.json`，不读取 test split，不重新拟合标准化参数。训练期间窗口、顺序、采样种子不变；修改 batch_size 可能改变采样随机流，跨实验比较应保持整个 validation 配置一致。

唯一的生成验证指标是 `gaussian_kl_real_to_generated`：**真实集到生成集的对角高斯近似 KL**，使用自然对数，单位为 nats，越小越好。每个完整窗口作为一个 L 维样本，全部窗口形成 `X_real, X_gen: [N, L]`。只统计 IBI，反归一化到毫秒后以 float64 沿 `dim=0` 在全部窗口上估计各坐标的均值和总体方差（`correction=0`）：

```text
mu_real[j] = mean_n(X_real[n, j])
v_real[j]  = mean_n((X_real[n, j] - mu_real[j])²)
mu_gen、v_gen 同理

vr[j] = max(v_real[j], kl_variance_floor_ms2)
vg[j] = max(v_gen[j], kl_variance_floor_ms2)

KL(real || generated)
  = 0.5 * sum_j(log(vg[j] / vr[j])
               + (vr[j] + (mu_real[j] - mu_gen[j])²) / vg[j] - 1)
```

先合并所有 batch 的坐标矩，再计算一次 KL，不平均逐 batch 或逐窗口的 KL。对全部 L 个坐标求和，**不除以 L**；改变 `window_epochs` 后，不直接横比原始 KL 或沿用非零阈值。两侧方差采用相同的固定下界，默认 `1.0 ms²`（对应 1 ms 标准差）；配置必须为正有限数，下界及方向写入验证协议并随运行冻结。下界只用于拟合高斯的方差，不裁剪 IBI 波形，不改变训练标准化；原存储零值（包括缺失标记）仍计入统计。

这只是匹配一阶矩和对角二阶矩的高斯近似，忽略跨时间坐标的相关性及非高斯结构，不能单独证明整夜连续性或条件生成质量。只有一个验证窗口时，跨窗口总体方差为 0，计算会使用下界；这种小样本结果不能评估总体分布质量。协议版本为 `ibi_epoch_diagonal_gaussian_kl_v3`，与旧 L2 / 逐窗口误差协议区分。

生成和坐标矩合并按 batch 在 CUDA 上执行，不保留跨 batch 的生成波形，运行统计张量规模为 O(L)。条件、参考数据和协议元数据存放在主机；退出验证时恢复随机数状态，不改变训练模型的模式、梯度、buffer 或后续训练随机流。KL 不增加采样次数，每轮仍需 `ceil(N/validation.batch_size) × 200` 次采样模型调用和一个独立模型副本。全量采样仍是主要计算工作，修改指标不会跳过它。

训练 `train_mean_loss = Σ(batch_MSE × batch样本数) / Σ(batch样本数)`，最后一个短 batch 按实际样本数加权，输出普通数值。`training_report.json` 中的 `mean_loss` 为实际训练过程的同口径平均；反向传播仍使用原始 MSE，没有将训练 loss 改为 KL。

`training/epoch_metrics.csv` 仅保留以下字段：

| 字段 | 含义 |
|---|---|
| `epoch` | 已完成的训练轮次 |
| `train_mean_loss` | 当轮按样本数加权的平均训练 MSE |
| `gaussian_kl_real_to_generated` | 当轮全验证集对角高斯 KL，nats |
| `smoothed_gaussian_kl_real_to_generated` | 最近完整滑动窗口的平均 KL；预热期为空 |
| `validation_seconds` | 当轮验证耗时，包含采样、KL 和最佳权重保存 |
| `smoothing_ready` | 是否已达到滑动平均所需轮数 |
| `metric_improvement` | 平滑 KL 是否达到本轮改善条件 |
| `early_stopping_bad_epochs` | 连续没有达到改善条件的轮数 |
| `stopped_early` | 是否在本轮触发早停 |

旧整体／逐窗口均值方差误差、方差比、向量 L2、`joint_improvement` 及其相关计算已从每轮验证中移除。每轮 `training/validation/epoch_0001/metrics.json` 等文件保留 KL、采样信息、早停状态；不再输出逐窗口误差。所选窗口元数据只在协议中保存。独立去噪评估、最终生成报告保持各自的输出用途。历史运行的 CSV 和冻结源码不改写，新配置对新启动的运行生效。

## 基于平滑 KL 的早停与 checkpoint

`dataset_config.json` 默认启用：

```json
"early_stopping": {
  "enabled": true,
  "patience": 10,
  "smoothing_window": 5,
  "min_delta_kl": 0.0
}
```

监控 `gaussian_kl_real_to_generated` 最近 5 个训练 epoch 的简单滑动平均（SMA）。前 4 轮预热，等待次数为 0，不生成最佳 KL checkpoint；第 5 轮建立平滑参考值并首次保存。以后仅当 `当前平滑 KL < 参考值 - min_delta_kl` 才更新参考值、清零等待计数并保存当前权重；否则等待次数加 1。连续 10 轮没有改善时，在当轮验证结束后停止，默认最早第 15 轮。相等不算改善，小幅下降可以累计超过固定参考值上的阈值。

`min_delta_kl` 必须为有限非负数，单位为 nats，默认 0 表示任何严格下降。旧配置中的 `min_delta_mean_ms` 和 `min_delta_variance_ms2` 仅在值为 0 时兼容并从冻结配置中移除；非零旧阈值会报错，需改用 KL 阈值。这里判断的是历史相对改善，没有绝对质量达标线。

主入口和 `submit.sh` 默认最多 `--epochs 50`，可以从命令行覆盖。`enabled: false` 只关闭提前停止，仍更新平滑指标、等待次数及最佳 KL 权重；未满滑动窗口的短训仅保存最终权重。

- `training/best_kl_model.pth` 与早停计数清零同步更新，保存触发平滑 KL 改善的当轮模型。只平滑指标，不平均权重、不做模型 EMA。
- 训练结束时，`training/best_kl_manifest.json` 记录所选 epoch、原始和平滑 KL、滑动区间、权重哈希以及验证协议，供原评估入口使用。新运行不再创建 `best_mean`、`best_variance`、`best_joint` checkpoint。
- 最终 `saved_models/DREAMT/model.pth`（主工作流位于运行目录 `workspace/` 下）仍保存实际停止时的权重。主流程后续评估默认使用最终模型，不自动切换为最佳 KL 模型。
- `training/training_report.json` 的 `epochs` 是实际完成轮数，`requested_epochs` 是上限；`stop_reason` 区分 `early_stopping` 和 `max_epochs`。

训练结束后使用最佳 KL 权重执行原始采样：

```bash
python scripts/evaluate_original_sampler.py \
  --training-args-json /path/to/run/training/best_kl_manifest.json \
  --output-dir /path/to/run/generation_best_kl
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
| 模型输入 | 整夜 epoch 序列，可 padding | 同一人的 K 个连续 epoch，750 × K 点，不跨人或 padding |
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
