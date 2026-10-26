# DREAMT 条件 IBI 生成（CUDA 工作流）

基于固定版本 WaveStitch，适配 WatchSleepNet 的 DREAMT IBI 数据接口。每个输入窗口为 `[B, 750 × K, 10]`，其中 `K=window_epochs` 是连续 30 秒睡眠 epoch 的数量，默认 4（25 Hz、120 秒、3000 点）；生成目标为 IBI。九个条件通道依次为 W / Light / REM / Deep、AHI、BMI、AGE、gender_F、gender_M。

网络结构、优化器、epsilon MSE 损失和训练更新公式沿用固定上游版本；本地 S4 慢速 Cauchy 回退计算补齐遗漏的共轭项。新训练通过独立的 `diffusion_schedule.py` 将200步日程的终点累计 alpha 调整到约 `1e-8`，使训练末态更接近采样使用的标准高斯起点；训练与推理使用同一记录日程。采样运行时另修正一处后验噪声系数：使用后验方差的平方根作为标准差；固定上游源码及 `training_utils.py` 不改写。在 `de30ac2` 的数据与 CUDA 设备适配基础上，增加每轮生成验证、loss 数值平均及早停；本地改动纳入源码检查器的逐项白名单。上游来源和许可证说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

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

提交命令运行完整的 **训练 → 去噪评估 → 修正后验噪声系数的 200 步生成**，默认最多 50 个训练 epoch、batch size 8、4 个主机线程。各次运行单独保存源码、配置、checkpoint 和日志。

在**已经分配 GPU 的计算节点**上验证环境和完整流程：

```bash
python scripts/check_cuda.py
python scripts/smoke_cuda.py --threads 4
```

第一条命令同时在实际 CUDA 设备上检查 S4 的 complex64 / complex128 Cauchy 数值和梯度，以及 S4 核与独立稠密递推的结果。第二条命令自动创建四个人的人工短记录，运行 1 个 epoch、去噪评估和完整采样，并保存 `cuda_smoke_report.json`。它不修改真实数据，也不代表收敛或生成质量已合格。

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

默认每次训练创建独立的 `runs/dreamt_<UTC时间戳>`，保存源码、配置、按人划分、训练集统计、checkpoint 和日志。每个 epoch 结束执行生成验证，训练结束后再执行去噪评估及修正后的 200 步采样。`--skip-generation` 只跳过最后的独立采样任务，每轮 validation 仍执行。旧模型评估的数据、长度、日程和模型参数绑定原训练 manifest，不跟随后来修改的全局配置；使用当前评估入口时，采样器采用后验方差平方根修正。

## 扩散日程与历史 checkpoint

在 `dataset_config.json` 顶层设置：

```json
"diffusion_schedule": "linear_power",
"terminal_alpha_bar": 1e-8
```

`run_dreamt.py` 和 `run_simulation.py` 的新运行省略这两项时也会显式记录上述默认值。`linear_power` 从原始200步线性 beta（`0.0001 → 0.02`）计算 `alpha = 1 - beta`，再使用 `c = log(target) / sum(log(alpha))`、`alpha_new = alpha ** c` 和 `beta_new = 1 - alpha_new`。累计 alpha 的终点约为 `1e-8`，对应信号系数约 `1e-4`；原线性日程的终点约为 `0.13218`。这是接近纯噪声的有限端点，不是精确零信噪比，也不单独证明生成质量提高。

保留200步、原始整数 timestep 条件、S4网络、epsilon MSE、Adam及反向均值公式。只接受 float32 下严格处于 `(0,1)` 的 beta、alpha 和累计 alpha，以免零端点导致现有 epsilon 反向式或引导式除零。可以显式选择 `"diffusion_schedule": "linear"` 运行原日程；此模式忽略终点目标，但仍记录配置。

新运行的冻结 JSON、`training/run_manifest.json`、最佳 checkpoint 对应的 manifest 以及验证协议记录同一日程。`diffusion_identity` 包含日程名、参数、有效终点、完整 float32 beta 数组及 beta/alpha/累计 alpha 的 SHA256。两个评估入口只按训练 manifest 重建，并核对身份；不能给现有 checkpoint 单独切换采样日程。历史 manifest 缺少新字段时始终恢复原 `linear`，不使用当前 JSON 的新默认值。非线性日程缺少身份记录或记录不一致会报错。不同日程应从新模型开始独立实验，不能把旧模型换日程后的结果当成正常评估。

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

该值必须是正整数，省略时兼容旧配置使用 4。每个睡眠 epoch 固定为 30 秒 × 25 Hz = 750 点。主入口自动令 `window_size = s4_lmax = 750 × window_epochs`，数据加载、条件标签数、验证窗口长度、去噪评估和生成使用同一长度；不需要改 `submit.sh` 或 Python 命令。短于该长度的记录无法提供窗口，若一个 split 没有合法窗口会明确报错。

`window_epochs` 与训练次数 `--epochs`、单条记录截断选项 `max_epochs` 含义不同，也不会改变 W/Light/REM/Deep 四个标签类别或 10 个输入通道。新运行将窗口配置写入源码快照、冻结数据身份和 manifest；评估 checkpoint 按其原训练长度运行，不受后来修改全局 JSON 的影响。旧身份未记录该字段时按 4 处理。

## 每轮 validation、逐窗口高斯 KL 和平均 loss

每个 epoch 完成全部 optimizer 更新后（包括最后一轮），用当前权重在验证集条件下从噪声生成配置长度的 IBI。执行与训练相同的完整扩散日程，默认 200 步；采样片段由固定上游源码提取，经哈希检查后仅施加下述后验噪声系数修正。验证窗口独立生成，推理 stride 为完整的 `L = 750 × window_epochs` 点，不将不同片段或受试者互相拼接；训练 stride 固定为 750 点（30 秒），K=4 时相邻窗口重叠 90 秒，K=1 时不重叠。

对于 `step > 0`，上游片段将 `posterior_variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)` 直接乘标准高斯噪声；当前运行时改为 `sqrt(posterior_variance) * noise`，因此注入噪声的方差等于后验方差。`step == 0` 仍不加噪。反向均值、随机数调用及顺序、条件与引导系数均不改动；日程由外部适配器按训练记录提供，每轮验证与最终采样一致。每轮验证与 `scripts/evaluate_original_sampler.py` 使用相同的平方根修正；该脚本保留旧名称以兼容现有入口。

`extract_original_pipeline()` 保留原始片段供审计，`extract_corrected_pipeline()` 生成只含单处修正的运行片段。报告分别记录上游快照逐字未改（`upstream_snapshot_byte_identical: true`）与运行采样片段已修正（`sampler_source_byte_identical: false`、`posterior_variance_sqrt_corrected: true`）；详细来源记录 `posterior_noise_scale: "sqrt_posterior_variance"`。历史日志、checkpoint 和冻结源码不改写。修正前后采样结果应在相同 checkpoint、数据与种子下对照，不能将公式修正本身当成生成质量已经改善的证据。

### S4 后端选择与历史 checkpoint

本地 `TSImputers/S4Model.py` 的 `cauchy_slow` 在求和前分别拼接 `v`、`w` 的共轭，修正回退路径仅计算一半共轭对的问题。该 PyTorch 实现始终可用，CPU 输入始终使用它。CUDA 输入只有在 dtype、形状和梯度要求均兼容时才使用可选 CUDA 扩展；否则使用已安装的 PyKeOps，或在同一 CUDA 设备上执行 PyTorch 回退。分派器检查输入设备、复数精度和可广播维度；扩展输入先解除共轭视图并转为连续内存，PyKeOps 按实际设备选择后端，不隐式搬到 CPU 或降低精度。

网络层数、维度和权重键不变，但回退算子和后端选择发生变化；报告记录 `imported_core_byte_identical: false`、`s4_fallback_conjugate_corrected: true`、`s4_device_dtype_dispatch_corrected: true`。源码检查只允许三块精确替换：PyKeOps 输入和设备、始终可用的共轭回退及分派器、S4 调用位置。这些修正不能直接证明生成质量改善。

生产评估入口要求训练 manifest 中的 `source_sha256["TSImputers/S4Model.py"]` 与当前实现完全匹配；缺少记录或不一致时停止。历史 checkpoint 应在其原冻结工作区中复现，不能仅凭权重形状一致就载入修正后的 S4 算子。每轮验证复制同一次训练的模型参数并使用相同源码与后端。

独立后端检查可按实际设备运行：

```bash
python scripts/check_s4_backends.py --device cpu
python scripts/check_s4_backends.py --device cuda
python scripts/check_s4_backends.py --device both
```

默认 `--device cpu`；此数值诊断不改变生产入口的 CUDA 要求。显式请求 `cuda` 或 `both` 而没有可用 GPU 时检查失败，不以 CPU 代替或记为跳过成功。检查覆盖两种复数精度、梯度和独立稠密 S4 递推；可选扩展只有在实际安装且设备可用时才验证。CPU 检查通过不能作为 CUDA 路径已验证的证据。

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

唯一的生成验证指标是 `gaussian_kl_real_to_generated`：**每个真实窗口与对应生成窗口的一维高斯近似 KL，再对全部窗口取算术平均**，使用自然对数，单位为 nats，越小越好。全部窗口形成 `X_real, X_gen: [N, L]`；同索引的两个窗口配对，生成窗口使用该真实窗口的条件。只统计 IBI，反归一化到毫秒后以 float64 沿每个窗口内部的时间点轴 `dim=1` 分别计算均值和总体方差（`correction=0`）：

```text
mu_real[n] = mean_j(X_real[n, j])
v_real[n]  = mean_j((X_real[n, j] - mu_real[n])²)
mu_gen、v_gen 同理

vr[n] = max(v_real[n], kl_variance_floor_ms2)
vg[n] = max(v_gen[n], kl_variance_floor_ms2)

KL_window[n](real || generated)
  = 0.5 * (log(vg[n] / vr[n])
           + (vr[n] + (mu_real[n] - mu_gen[n])²) / vg[n] - 1)

gaussian_kl_real_to_generated = sum_n(KL_window[n]) / N
```

每个窗口分别形成一个真实高斯和一个生成高斯，计算的是 `KL(real || generated)`，不是把全部窗口混合后求一次 KL，也不对 L 个时间坐标的 KL 求和。每个窗口等权，**不是每个受试者等权**；提供更多合法验证窗口的受试者会获得更大权重。跨 batch 累加 KL 总和与窗口数，最后除以总窗口数，最后一个短 batch 不会被额外放大权重。

两侧方差采用相同的固定下界，默认 `1.0 ms²`（对应 1 ms 标准差）；配置必须为正有限数，下界及方向写入验证协议并随运行冻结。下界只用于拟合高斯的方差，不裁剪 IBI 波形，不改变训练标准化；原存储零值（包括缺失标记）仍计入统计。常量窗口的方差为零，计算时使用下界。

该指标只比较每个窗口内的幅值均值和方差，**对窗口内部的时间排序不敏感**：任意打乱一个窗口的采样点，均值、方差和对应 KL 都不变。它忽略自相关、频谱、非高斯结构和接缝连续性，不是 L 维序列联合分布的 KL，也不是整夜生成质量或条件利用能力的充分证明。只有一个验证窗口时也能用窗口内 L 个点计算该指标，但一个配对样本不足以评价生成分布或泛化能力；同条件生成本身的随机性也会影响配对得分。

协议版本为 `ibi_epoch_window_gaussian_kl_v4`。旧 `ibi_epoch_diagonal_gaussian_kl_v3` 是沿 `dim=0` 跨窗口统计每个坐标，再对 L 个坐标的 KL 求和；现在改为沿 `dim=1` 统计每个独立窗口并平均逐窗 KL。**旧值与新值不能直接比较，旧的非零 KL 阈值也不能直接沿用。** 改变 `window_epochs` 同样会改变窗口内部统计和样本选择；比较实验仍需固定窗口长度、数据及验证配置。

生成、逐窗口统计和 KL 累加按 batch 在 CUDA 上执行，不保留跨 batch 的生成波形或逐窗统计结果。条件、参考数据和协议元数据存放在主机；退出验证时恢复随机数状态，不改变训练模型的模式、梯度、buffer 或后续训练随机流。KL 不增加采样次数，每轮仍需 `ceil(N/validation.batch_size) × 200` 次采样模型调用和一个独立模型副本。全量采样仍是主要计算工作，修改指标不会跳过它。

训练 `train_mean_loss = Σ(batch_MSE × batch样本数) / Σ(batch样本数)`，最后一个短 batch 按实际样本数加权，输出普通数值。`training_report.json` 中的 `mean_loss` 为实际训练过程的同口径平均；反向传播仍使用原始 MSE，没有将训练 loss 改为 KL。

`training/epoch_metrics.csv` 仅保留以下字段：

| 字段 | 含义 |
|---|---|
| `epoch` | 已完成的训练轮次 |
| `train_mean_loss` | 当轮按样本数加权的平均训练 MSE |
| `gaussian_kl_real_to_generated` | 当轮全验证集的配对窗口高斯 KL 算术平均，nats |
| `smoothed_gaussian_kl_real_to_generated` | 最近完整滑动窗口的平均 KL；预热期为空 |
| `validation_seconds` | 当轮验证耗时，包含采样、KL 和最佳权重保存 |
| `smoothing_ready` | 是否已达到滑动平均所需轮数 |
| `metric_improvement` | 平滑 KL 是否达到本轮改善条件 |
| `early_stopping_bad_epochs` | 连续没有达到改善条件的轮数 |
| `stopped_early` | 是否在本轮触发早停 |

旧整体／逐窗口均值方差误差、方差比、向量 L2、`joint_improvement` 及其相关计算已从每轮验证中移除。每轮 `training/validation/epoch_0001/metrics.json` 等文件保留平均 KL、采样信息、早停状态；逐窗口 KL 计算后汇总，不另输出逐窗列表。所选窗口元数据只在协议中保存。独立去噪评估、最终生成报告保持各自的输出用途。历史运行的 CSV 和冻结源码不改写，新配置对新启动的运行生效；读取相同名字的 KL 列时应同时检查协议版本。

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

训练结束后使用最佳 KL 权重执行修正后的采样：

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

原数据链路回归检查覆盖 25 Hz 与 100 Hz 输入的 epoch 信号和标签、四段拼接的九个条件通道及反归一化。新增测试覆盖旧线性日程逐bit兼容、新日程端点与身份核对、逐窗口均值/总体方差及配对 KL、短 batch 按窗口数汇总、最后一轮验证、固定窗口、随机状态恢复、模型副本，以及仅包含后验方差平方根修正的完整 200 步采样。缺少 CUDA 时，真实 GPU 测试会显式跳过，不能视为通过。

12 份 WaveStitch 上游快照逐字未改；5 个导入核心文件中，4 个逐字未改，`TSImputers/S4Model.py` 仅允许上述共轭回退与设备/精度分派的三块精确替换。移除允许的数据适配、外部日程导入/配置、设备适配、epoch 验证/日志和早停分支后，训练循环与上游相同。源码检查器还独立比较原始与运行时采样片段，仅允许给完整后验方差项加一次 `torch.sqrt`，其余代码和随机数调用必须一致；日程数组由明确记录的外部适配器提供。该检查不执行采样。检查器可重新生成详细报告；本机历史运行日志和机器路径未纳入公共仓库。

目前尚未完成 Fir GPU 实测，也未用真实 DREAMT 数据验证生成质量。人工短记录和接口测试不能证明收敛，跳过的 GPU 测试不能视为通过。请在已分配的 GPU 计算节点上执行上述 `check_cuda.py` 和 `smoke_cuda.py` 后再进行正式训练。
