# 分开运行的 IBI 预处理实验

新增两项独立实验，各自与已完成的 **preserve + z-score** 第 30 轮结果比较。两个作业分别从头训练 30 个 epoch，不串联、不恢复旧权重，也不重跑旧 preserve/masked 两组。原始 NPZ、睡眠标签和时间顺序保持不变。

| 设置 | 历史基线 | A：`zero_fill_one` | B：`minmax` |
|---|---|---|---|
| 原始零值输入 | 保留 0 秒 | 读取时替换为 1.0 秒 | 保留 0 秒 |
| IBI 标准化 | 训练集 z-score | 填充后的训练集 z-score | 固定 `[0,2]` 秒映射到 `[-1,1]` |
| epsilon MSE | 所有位置 | 所有位置，包含填充位置 | 所有位置，包含原始零值 |
| 验证参考位置 | 原始正值位置 | 同一批原始正值位置 | 同一批原始正值位置 |

A 只改变缺失零值的处理，并重新拟合相应的 IBI signal scaler；B 只改变 IBI 的标准化方式。两组均不使用 masked loss，不新增 mask 通道，不改变网络、损失公式、扩散日程或采样公式。元数据条件的标准化仍沿用原规则。

## A：原始零值填为 1 秒

在标准化前，将原始 IBI 的 `0` 替换为 **1.0 秒（1000 ms）**。非零值不被修改、截断或平滑。填充后的训练信号用于拟合 z-score 的均值和标准差，验证集不参与拟合。

填充位置正常加噪，并参与普通全点 epsilon MSE。这与此前 `masked` 实验不同：填入的 1 秒在训练中确实受到监督，而不是仅作为无损失的占位输入。

## B：固定 min-max 标准化

保留原始零值，使用固定范围 `[0,2]` 秒：

```text
x0_normalized = 2 * (IBI_seconds - 0) / (2 - 0) - 1
IBI_seconds  = (x0_normalized + 1) * (2 - 0) / 2 + 0
```

因此 0 秒映射为 −1，1 秒为 0，2 秒为 +1。**这只作用于 IBI 信号列**；睡眠阶段 one-hot 和元数据条件列不映射到该范围。这个范围不是逐窗口拟合，也不由验证数据估计。超出固定范围的输入报错，**不裁剪到边界**；原有非有限值、负值及结构无效数据仍遵守原加载规则。

`[-1,1]` 约束的是有效的干净输入 **x0**。扩散中的 Gaussian noise、加噪状态 **xt** 和模型生成结果都不保证落在这个范围内，也不会被裁剪。反标准化后仍可能出现负生成值，必须如实记录。

固定范围映射不保证标准化后方差为 1。即使扩散日程不变，信号标准差与相对噪声大小也会改变，因此有效信噪比会变化；不同标准化实验的训练 loss 不能直接横向比较高低。

## 与历史基线对齐

统一入口是 `scripts/run_preprocessing_experiment.py`。它读取参考目录中 preserve 的冻结配置与历史记录，不从当前全局 `dataset_config.json` 重新决定数据。

- 核对训练/验证划分、原始 NPZ 哈希、训练窗口、元数据条件 scaler、窗口长度、扩散日程、模型及训练参数。
- 仅允许当前实验对应的输入策略和 IBI signal scaler 改变；不得同时把零填充与 min-max 组合进同一实验。
- 每轮使用原来的同一批 64 个验证窗口，关闭早停，保存生成波形与逐窗口统计。
- 从头初始化模型和优化器，完成全部 30 轮及对应更新数。

历史源码与运行身份检查保持严格。缺少证据、源码或输入不匹配会中止；旧版本复刻使用已有冻结源码，不通过放松哈希检查绕过失败。旧权重不能直接配新 scaler 代替重新训练。

验证真实信号始终来自原始 NPZ：原始 `0` 保留为缺失标记，**A 中填入的 1 秒不作为验证真值**。KL 在每个窗口的原始正值位置比较真实与生成的均值、方差。相同评分位置的负生成值不被丢弃，全部生成位置的非正值比例另行记录；生成波形不做裁剪、平滑或归零。

## 分别提交两个 Slurm 作业

在项目根目录执行。Slurm 打开日志早于脚本运行，因此先建日志目录：

```bash
cd /project/def-plago/yxtao/dataset-generation
mkdir -p /home/yxtao/links/scratch/dataset-generation/logs
sbatch submit_zero_fill_one.sh
sbatch submit_minmax.sh
```

这会提交两个独立作业，不是在一个作业里顺序训练两组。两份脚本均固定参考路径：

```text
/home/yxtao/links/scratch/dataset-generation/runs/missing_value_ablation_extended_20261102T170345Z_1221493/training_30epochs
```

每个作业使用 `def-plago`、4 CPU、1 GPU、10 小时；加载 `StdEnv/2023 gcc/12.3`、`opencv/4.9.0`、`python/3.11`，激活项目 `./.venv`。脚本通过 `SLURM_SUBMIT_DIR` 定位项目，应从上述项目根目录提交。

A 输出到 `runs/zero_fill_one_<UTC时间>_<job_id>`，B 输出到 `runs/minmax_<UTC时间>_<job_id>`。两者都在旧参考目录之外，不覆盖任何历史文件。

后续推理必须将各自新输出的 `config.json` 传给推理入口，并使用该实验对应的 checkpoint。统一训练脚本从旧 preserve 配置生成两份完全分离的新配置；不能继续使用旧全局 `dataset_config.json`，也不能将旧 checkpoint 与新 scaler 混用。

只检查配置与数据身份，不训练模型：

```bash
python scripts/run_preprocessing_experiment.py \
  --experiment zero_fill_one \
  --reference-run /home/yxtao/links/scratch/dataset-generation/runs/missing_value_ablation_extended_20261102T170345Z_1221493/training_30epochs \
  --output-dir /home/yxtao/links/scratch/dataset-generation/runs/zero_fill_one_check \
  --epochs 30 --batch-size 8 --threads 4 --check-only
```

将 `--experiment` 改成 `minmax` 并使用另一个新输出目录，可单独检查 B。输出目录必须尚不存在，且不能位于参考目录内；预检与正式训练使用不同目录。`--output-dir` 可省略，由脚本生成新目录。

## 输出与比较

```text
<experiment output>/
  config.json
  preflight/preflight.json
  run/workspace/
  run/training/
  summary.json
  comparison.csv
  training.log
```

`comparison.csv` 汇总旧 preserve、旧 masked 与本次新实验的**最终 epoch KL、非正值比例和训练预算**。逐窗口真实/生成均值、方差及完整波形保存在 `run/training/validation/epoch_NNNN/` 每轮产物中，不在这张简表里。

比较同一预算、同一验证窗口的原始均值差、方差差、KL 与非正值比例。每轮保存的是当轮生成波形，不是多个随机采样结果的平均。KL 只描述窗口幅值矩，不验证节律或生理质量；“负值减少”不能单独说明信号已经合理。

正式训练继续要求 CUDA。本地 CPU 测试只验证加载、身份门禁、指标与调度；模拟子进程测试不执行正式模型训练，不代表这两项实验已在 GPU 完成。
