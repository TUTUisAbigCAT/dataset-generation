# IBI 缺失零值训练对照

## 实验问题

检验“原始 NPZ 中的缺失标记 0 被作为真实 IBI 参与标准化和训练”对生成分布的影响。原始 NPZ、睡眠标签、时间轴和固定 S4 网络均不改写。这项实验不修复非零 IBI 中的漏峰、重复检测，也不证明屏蔽零值后生成信号已经生理合理。

本地 S092 审计得到 667500 个点，其中 38226 个零（5.7267%）。890 个完整 30 秒片段都含零；步长 30 秒时，887 个两分钟窗口和 883 个四分钟窗口全部含零。因此不能简单删除所有含零窗口。该统计只代表此份 S092，不代表全部训练集。

## 两组如何比较

| 设置 | preserve | masked |
|---|---|---|
| 训练 scaler | 包含存储零值 | 仅训练集中结构有效 epoch 的正值 |
| 原始零值输入 | 标准化为 `-mean/std` | 标准化空间 0，占位而非真实 0 秒 |
| epsilon MSE | 所有位置 | 只监督原始正值位置 |
| 窗口时间轴与条件 | 保持原顺序 | 保持原顺序 |
| 配对实验验证 | 真实正值位置 | 完全相同的真实正值位置 |

令 `m = (raw_ibi > 0)`，在标准化前生成并随 batch 传递；有效值恰好标准化为 0 时，`m` 仍为真。

`loss = sum(m * (predicted_epsilon - epsilon)^2) / sum(m)`。

实现先选择有效位置再做差，避免无效位置的数值污染损失。epoch loss 按全部有效点数加权，不按 batch 均值直接平均。既有负值、非有限值、未知标签及时间断裂仍按原结构规则排除；masked 窗口至少需要两个观测点。没有对高正值自动截断，也没有对缺口插值。

缺失输入的占位值仍被正常加噪；mask 仅作为损失旁路，不新增网络输入通道。这是观测点监督对照，不是“缺失位置完全不影响网络”。缺失区最终生成结果没有对应真实监督，不能用本实验断言其正确性。掩码去噪学习的相关方法见 [MissDiff 原论文](https://arxiv.org/abs/2307.00467)；这里不将表格数据上的理论或结果直接视作 S092 的保证。

## 集群执行

在已更新代码的 `dataset-generation` 项目根目录使用现有虚拟环境。数据和元数据路径仍读取原 `dataset_config.json`。预检不启动模型：

```bash
python scripts/run_missing_value_ablation.py --config dataset_config.json --check-only
mkdir -p /home/yxtao/links/scratch/dataset-generation/logs
sbatch submit_missing_value_ablation.sh
```

提交脚本使用 `def-plago`、4 CPU、1 GPU、10 小时时限和 `./.venv`，按顺序独立训练两组，每组 10 个 epoch。实验脚本关闭早停以固定训练轮数，每轮选至多 64 个相同的验证窗口，仍使用完整训练扩散日程生成。若存在少于两个正值的训练窗口，masked 组会排除它们，因此更新次数可能不同；`summary.json` 和 `comparison.csv` 会记录窗口数及更新次数，不能将固定 epoch 宣称为严格等计算预算。默认网络为原 S4；窗口长度沿用用户 JSON 的 `window_epochs`。普通 `submit.sh` 和原 JSON 不被改动。

在已分配 GPU 的节点也可以直接运行：

```bash
python scripts/run_missing_value_ablation.py --config dataset_config.json --epochs 10 --batch-size 8 --threads 4 --validation-max-windows 64
```

`--validation-max-windows 0` 使用所有合格验证窗口；`--output-dir /absolute/new/path` 指定一个尚不存在的实验目录。两组训练重新初始化，不续接旧 checkpoint。只做 masked 一组时可运行 `python scripts/run_dreamt.py --config dataset_config.json --missing-value-policy masked --epochs 10 --batch-size 8 --threads 4`。

## 输出与判断

输出目录位于 JSON 的 `output_root` 下，名称为 `missing_value_ablation_<UTC时间>`：

- `summary.json`：共享数据/划分/日程/验证窗口核验、两组 scaler、执行状态。
- `config_preserve.json` / `config_masked.json`：冻结的实验配置。
- `preserve/training/epoch_metrics.csv` / `masked/training/epoch_metrics.csv`：逐 epoch 同口径验证 KL。
- `comparison.csv`：两组首轮、末轮、最低 KL 与对应轮数、实际更新次数，以及末轮全部生成位置中非正值的比例。
- `preserve.log` / `masked.log`：独立运行日志；各组子目录继续保存源码、训练 manifest、权重和去噪诊断。

两组 KL 都在每个真实窗口的正值位置拟合一维高斯；生成信号在相同位置计算，负生成值不会被删掉，任意非有限生成值都会报错。该 KL 只比较幅值的均值与方差，不评价节律/时序或峰检测准确性。不要将它与历史含零验证 KL 直接比较，也不要直接用两组不同监督范围的 train loss 判定哪组更好。

每轮 `validation/epoch_NNNN/metrics.json` 的 `diagnostics` 另记录全部生成位置的非正值数量和比例，包含未参与 KL 的位置。这项诊断不影响早停，也不以删除负生成值来改善指标。

推理必须复用选中训练运行的冻结配置（包括 `missing_value_policy` 和 scaler）。仅有 `.pth` 不能推断此策略；把旧模型直接换成新 scaler 不构成有效对照。

## 本地验证范围

实现检查包括原始 mask、训练集统计隔离、无效点 loss 梯度、相同参考位置的 KL、权重/配置身份恢复、源码白名单和真实 S4 小规模前后向。生产训练和生成入口继续强制 CUDA。CPU 单元测试通过不表示 GPU 工作流或 DREAMT 训练质量已验证；实际对照结果以集群完成后的两个日志为准。

## 固定验证波形诊断与 30 轮等步数对照

`submit_missing_value_ablation_extended.sh` 将下列步骤串联在一个 GPU 作业中。脚本已填写这次参考运行的绝对路径 `missing_value_ablation_20261102T055527_103986Z`，原始 NPZ 和旧运行目录只读；新结果写到 `output_root` 下独立的 `missing_value_ablation_extended_<UTC时间>_<jobid>` 目录。

1. `scripts/export_ablation_validation.py` 读取旧实验两组**最终第 10 轮**权重，在原来的 64 个验证窗口上重新采样一次，保存完整波形、逐窗口统计和与旧最终 KL 的差值。它不选 `best_kl_model.pth`：旧 preserve 的 best 来自第 7 轮，直接和 masked 第 10 轮相比会混淆训练预算。旧权重、冻结配置、训练数据、scaler、验证窗口/掩码、采样日程及网络/采样源码均先核对；核对失败会阻止后续长训。
2. 对两组导出的同一批窗口画图，默认固定窗口编号 0、31、63。每张图上方是原始 IBI 与两组生成 IBI，下方是共享睡眠阶段条件；原始零值会显示出来，生成负值不裁剪。另有全部窗口均值、方差的散点图，以及 KL 两项贡献图。
3. 两个模型各**从头训练 30 轮**，关闭早停；沿用每臂旧配置并核对训练窗口、batch size、学习率、模型、日程、种子相关设置及数据身份。当前 51,266 个训练窗口、batch size 8，对应每轮 6,409 次更新，每组共 **192,270 次更新**。预检不等量或结束时未完成预算会报错，不会把不等步数结果当作完成的对照。
4. 每轮保存同一批 64 个验证窗口的波形与逐窗口统计，完成后绘制第 30 轮两组对比。

旧 `.pth` 仅有模型参数/缓冲区，没有 Adam 状态、RNG 和恢复训练所需状态，因此这里**不是从第 10 轮继续训练 20 轮**。新运行的第 10 轮验证结果可与旧实验核对；CUDA 硬件或软件差异可能影响逐位复现，脚本会记录重导出的 KL 差异，不伪造相同结果。

```bash
mkdir -p /home/yxtao/links/scratch/dataset-generation/logs
sbatch submit_missing_value_ablation_extended.sh
```

若只想先补导出已有权重的波形，在 GPU 节点执行：

```bash
python scripts/export_ablation_validation.py \
  --reference-run /home/yxtao/links/scratch/dataset-generation/runs/missing_value_ablation_20261102T055527_103986Z \
  --output-dir /home/yxtao/links/scratch/dataset-generation/runs/epoch10_validation_export \
  --threads 4
```

输出目录必须尚不存在。该诊断恢复记录中的训练路径，数据搬迁时会明确失败，而不是用全局 JSON 的新数据悄悄替代。普通训练也可在 JSON 中设置 `"validation": {"save_artifacts": true}`（与既有验证选项合并）来逐轮导出。默认关闭，既有训练入口行为不变。

新作业输出结构：

```text
missing_value_ablation_extended_<time>_<jobid>/
  epoch10_diagnostics/summary.json
  epoch10_diagnostics/{preserve,masked}/validation/epoch_0010/
  epoch10_plots/
  training_30epochs/summary.json
  training_30epochs/comparison.csv
  training_30epochs/{preserve,masked}/training/epoch_metrics.csv
  training_30epochs/{preserve,masked}/training/validation/epoch_NNNN/
  epoch30_plots/
```

每个 `epoch_NNNN` 保存：

| 文件 | 内容 |
|---|---|
| `waveforms.npz` | `generated_ibi_seconds`、`reference_ibi_seconds`，形状 `[64,3000]`；对应 `scoring_mask`、原始正值掩码、窗口编号、被试、起点、阶段和标准化条件。均为普通数组，`allow_pickle=False` 可读。 |
| `per_window_metrics.csv` | 每窗口真实/生成均值（ms）、未加 floor 的总体方差（ms²）、有符号差值和绝对差值、观测点数、单窗 KL、`kl_mean_term`、`kl_variance_term`、全部位置中的非正生成点数。 |
| `artifacts.json` | 单位、形状、计算口径、关联协议及 NPZ/CSV SHA256。 |
| `metrics.json` | 既有聚合 KL、非正值诊断和新产物索引。 |

令 `vr=max(reference_variance_ms2, floor)`、`vg=max(generated_variance_ms2, floor)`：

```text
kl_mean_term     = (reference_mean_ms - generated_mean_ms)^2 / (2 * vg)
kl_variance_term = 0.5 * (vr / vg - 1 - log(vr / vg))
window_kl       = kl_mean_term + kl_variance_term
```

两项都基于**同一个真实窗口的观测位置**。均值项也受生成方差分母影响，因此须同时看原始均值差和方差，不能把分解当作两个独立因果来源。`scoring_mask` 不输入采样器；保存操作复用当轮已生成信号，不新增采样、不更改模型更新、不改变训练随机状态。统计和模型运算在 CUDA 上，落盘和画图使用主机。

`epoch_metrics.csv` 继续保持现有简洁列；详细统计放在单独的逐窗口 CSV 中。best 选择继续采用 5 轮平均 KL，本次没有修改选择规则。生成是独立条件窗口，不是逐点重建或整晚 overlap 续接；S092 仍属于测试集，本实验不使用它挑选训练轮次。
