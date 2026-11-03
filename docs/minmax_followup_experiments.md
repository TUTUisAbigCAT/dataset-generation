# min-max 基线上的两项独立实验

公共基线是已完成的 `minmax_20261103T003905Z_1223635`：原始 IBI 零值保留，干净 IBI 的固定物理范围 `[0,2]` 秒映射为 `[-1,1]`，普通全点 epsilon MSE。新建两个独立作业，各自从头训练 30 个 epoch，不恢复旧权重，不重跑公共基线或此前 preserve/masked 两组。

| 设置 | 公共基线 | A：`minmax01` | B：`minmax_dropout` |
|---|---|---|---|
| 原始零值 | 保留 | 保留 | 保留 |
| 干净 IBI 标准化范围 | `[-1,1]` | `[0,1]` | `[-1,1]` |
| 训练输入置零 | 无 | 无 | 加噪后逐点概率 0.1 |
| 验证/推理输入置零 | 无 | 无 | 关闭 |
| 损失 | 全点 epsilon MSE | 同基线 | 同基线，含置零点 |

两项变化不组合到同一个模型中。数据划分、原始 NPZ、窗口、睡眠阶段及元数据条件、模型、扩散日程、训练预算和验证协议保持一致。新配置分别从公共基线冻结配置生成，不读取全局 JSON 来替换历史设置。

## A：仅改为 `[0,1]`

配置使用 `signal_normalization` 中的 `output_range: [0,1]`，固定物理边界仍为 `min_seconds: 0`、`max_seconds: 2`：

```text
x0_normalized = IBI_seconds / 2
IBI_seconds  = 2 * x0_normalized
```

0、1、2 秒分别映射为 0、0.5、1。只改变 IBI 信号列，条件列不变；输入超出物理边界时仍报错，不裁剪。`[0,1]` 只描述有效的干净输入 x0，不限制 Gaussian noise、加噪状态 xt 或生成结果。

与 `[-1,1]` 相比，这同时改变干净信号的中心与尺度。保持扩散日程不变不等于保持有效信噪比不变，因此不能仅用训练 loss 降低来断言生成更好。

## B：仅增加训练输入 dropout

保持原始零值和 `[-1,1]` 标准化，新增：

```json
"training_input_dropout": {"probability": 0.1, "seed": 42}
```

先按原扩散公式得到 xt，再对 **xt 的 IBI 通道**独立逐点抽取 Bernoulli 掩码，将约 10% 的位置置为标准化空间 0。剩余位置不乘 `1/(1-p)`；条件通道、时间步、epsilon 目标和普通全点 MSE 不变。置零的训练点仍参与损失，验证和推理不应用这个操作。

这里的标准化 0 对应干净信号坐标中的 1 秒，但操作作用于加噪状态，不是在原始 NPZ 中把缺失零值填成 1 秒，也不是此前的 masked loss。它不改变模型内部 dropout 层。

置零会移除对应位置的独立 epsilon 信息，而这些位置仍要预测原 epsilon。因此全点 MSE 可能出现约 `p=0.1` 的不可约贡献；具体观测值还取决于其余位置误差和有限样本。不能把它的训练 loss 与无 dropout 组直接比较后判定质量退化，应查看同口径验证结果。

## 身份核对与验证

入口 `scripts/run_minmax_experiment.py` 检查旧运行是否完整，核对冻结源码、最终 checkpoint 哈希、原始数据和划分、训练窗口、条件 scaler、扩散日程、训练参数及更新预算。训练源码只允许本次明确授权的输入 dropout 接入；不通过宽泛忽略源码差异绕过核验。

两组继续关闭早停，每轮保存同一批 64 个验证窗口。真实参考始终使用原始正值位置；原始零值作为缺失保留，不当作生理零值评分。生成负值不裁剪：评分位置内参与 KL，所有位置的非正值比例另外记录。best 权重仍按现有 SMA5 KL 规则选择，最终比较使用同为第 30 轮的权重。

## 分别提交

```bash
cd /project/def-plago/yxtao/dataset-generation
mkdir -p /home/yxtao/links/scratch/dataset-generation/logs
sbatch submit_minmax01.sh
sbatch submit_minmax_dropout.sh
```

两条命令提交两个独立作业。每个作业使用 `def-plago`、4 CPU、1 GPU、10 小时；加载 `StdEnv/2023 gcc/12.3`、`opencv/4.9.0`、`python/3.11`，激活项目 `./.venv`。脚本从 `SLURM_SUBMIT_DIR` 进入项目，须在上述项目根目录提交。

参考目录已写入脚本：

```text
/home/yxtao/links/scratch/dataset-generation/runs/minmax_20261103T003905Z_1223635
```

输出分别为新的 `runs/minmax01_<UTC时间>_<job_id>` 和 `runs/minmax_dropout_<UTC时间>_<job_id>`，不覆盖旧目录。

只进行配置/数据检查、不训练模型：

```bash
python scripts/run_minmax_experiment.py \
  --experiment minmax01 \
  --reference-run /home/yxtao/links/scratch/dataset-generation/runs/minmax_20261103T003905Z_1223635 \
  --output-dir /home/yxtao/links/scratch/dataset-generation/runs/minmax01_check \
  --epochs 30 --batch-size 8 --threads 4 --check-only
```

将实验名改为 `minmax_dropout` 并使用另一个新目录，可预检 B。输出目录必须尚不存在且位于参考目录之外；预检与正式训练使用不同目录。

## 输出与使用

输出包含 `config.json`、`preflight/preflight.json`、`run/workspace/`、`run/training/`、`summary.json`、`comparison.csv` 和 `training.log`。`comparison.csv` 只比较旧 min-max 基线与当前新实验的最终第 30 轮指标和预算；逐窗口均值、方差、KL及波形保存在每轮 validation 产物中。

后续推理使用该实验对应的 checkpoint 与新输出的 `config.json`，不能混用旧全局配置或另一个标准化范围。dropout 配置保留作训练来源记录，推理不置零输入。

正式训练和生成仍要求 CUDA。本地人工数据与模拟子进程测试只验证实现和工作流，不表示这两项实验已完成真实训练。验证 KL 衡量每窗口幅值矩，不单独证明节律、整夜连续性或生理质量。
