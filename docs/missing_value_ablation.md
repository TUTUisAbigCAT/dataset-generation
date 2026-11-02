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
