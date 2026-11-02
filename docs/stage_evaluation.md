# S092 分睡眠阶段的条件生成评估

该评估使用已完成 30 轮等更新对照中的 `preserve` 和 `masked` **最终权重**，比较相同 S092 睡眠阶段与人员条件下的独立窗口生成。它不训练、不选择 best 权重，也不做整晚前缀续接。两臂必须具有相同完成轮数和更新次数；默认要求最终第 30 轮。

S092 属于原划分的测试集。这次明确读取它进行测试分析，不能再把本次结果当作未查看过的测试数据，也不应据此挑选训练轮次或反复调参后宣称独立测试提升。训练/验证划分不被修改，S092 不参与重新拟合 scaler。

## 条件与数据身份

- 从两臂 `training/run_manifest.json` 恢复最终 checkpoint、模型、扩散日程、训练数据身份和各自的 scaler，并核对文件哈希。不会根据当前全局 `dataset_config.json` 替换旧训练设置，也不会回退到 `best_kl_model.pth`。
- 两臂使用同一个 S092 文件、相同窗口起点、四类阶段 `W / Light / REM / Deep`，以及相同的 `AHI / BMI / AGE / GENDER` 条件。原始 N1/N2 合并为 Light。
- 两臂仍保留各自训练时的 IBI scaler。把某一模型改用另一臂的 scaler 会改变推理含义，不属于这个对照。
- 送入采样器的 IBI 输入通道全部置为标准化空间 0，作为未知输入；不传入真实 IBI 或观测 mask 作为已知信号。真实值仅用于生成后的评分。零输入并不表示要生成 0 秒 IBI。
- 采样使用完整记录的 200 步日程，在 CUDA 上运行；没有 CPU 模型回退。文件读写和绘图在主机上执行。

## 窗口和重复采样

保持训练窗口长度 `L = window_epochs × 750`，25 Hz；当前模型为 4 个 30 秒 epoch，即 120 秒、3000 点。候选起点为 `0, L, 2L, ...`，候选窗口互不重叠，最后不足 L 的尾部不补齐。

候选窗口中的任一 epoch 若结构无效、真实 IBI 含负值或非有限值、正值不足 2 点，或阶段条件不一致，都会被排除；已知窗口内部时间断裂也会排除。缺失标记 0 不会单独使整个窗口失效。筛选不按 IBI 高低或模型生成质量决定。

前置条件读取沿用 `prepare_night` 的严格检查：若完整记录存在未知/不一致的阶段标签或已知时间断裂，会先报错，不会悄悄压缩时间轴后继续选择窗口。

用固定 `--selection-seed 42` 打乱候选，按 W、Light、REM、Deep 轮询选择。`--windows-per-stage 8` 是各阶段目标窗口数：同一个混合阶段窗口可以贡献给多个阶段，但真实窗口只选择一次；不优先选择纯阶段窗口。部分阶段可能超过目标，或因数据不足未达目标，结果中会保留实际覆盖数量；不存在的阶段不伪造样本或 0 分。

默认使用 `--repeats 8 --seed 20261102`。两臂共享同一组选定窗口和对应的采样种子；每次重复都从新的噪声生成独立窗口，不把上一次结果当作前缀或观测。该重复次数衡量采样波动，不是 8 次独立模型训练。

## 评分口径

每个生成窗口按原始 30 秒标签拆分。逐段使用 `finite(reference) & (reference > 0)` 的原始观测位置，同时选择真实与生成 IBI，计算：

- 均值（ms）、总体方差（ms²，`correction=0`）及两者差值。
- 一维高斯 `KL(real || generated)` 及均值项、方差项；方差 floor 为 1 ms²，和当前验证口径一致。
- 所有生成位置中的非正值比例；分母是完整 750 点，不只统计观测位置，也不删除负生成值来改善结果。

同阶段内按每个独立的 30 秒 epoch 等权聚合，不按有效点数加权。同一真实 epoch 不因混合窗口计入多个阶段。跨重复给出均值和总体标准差；该标准差不是置信区间。不存在的阶段输出不可计算状态而非零误差。

这里先按 30 秒拆分再计算 KL，与训练验证时整段 120 秒拟合的 KL 不是同一数值口径，不能直接拿两者的绝对值判断改善幅度。

这些幅值统计不能验证峰检测、节律或时序是否正确，也不是逐点重建指标。当前 S092 的非零 IBI 仍存在提取算法的疑问，因此贴近该参考分布不等于已证明生理真实性。

## 集群运行

在更新后的项目根目录，先确保 Slurm 日志父目录存在：

```bash
mkdir -p /home/yxtao/links/scratch/dataset-generation/logs
sbatch submit_stage_evaluation.sh
```

脚本已填写这次 30 轮 A/B 目录，申请 `def-plago`、4 CPU、1 GPU、10 小时，沿用 `StdEnv/2023`、`gcc/12.3`、`opencv/4.9.0`、`python/3.11` 和 `./.venv`。输出目录带 UTC 时间及作业号，旧数据、权重和实验目录均不改写；任一步失败会停止后续绘图。

只核对身份和窗口，不执行模型采样：

```bash
python scripts/evaluate_stage_generation.py \
  --ablation-run /home/yxtao/links/scratch/dataset-generation/runs/missing_value_ablation_extended_20261102T170345Z_1221493/training_30epochs \
  --subject S092 --expected-epochs 30 --repeats 8 --seed 20261102 \
  --selection-seed 42 --windows-per-stage 8 --batch-size 8 --threads 4 \
  --output-dir /home/yxtao/links/scratch/dataset-generation/runs/S092_stage_check \
  --check-only
```

`--output-dir` 必须是尚不存在的新目录。移除 `--check-only` 后需要实际 CUDA GPU。成功生成后，提交脚本调用 `scripts/plot_stage_evaluation.py`，直接使用保存的波形与统计，不重新采样。

结果结构如下：

```text
S092_stage_eval_<UTC时间>_<jobid>/
  summary.json
  protocol.json
  reference.npz
  stage_summary.csv
  preserve/draw_000/ ... draw_007/
  masked/draw_000/ ... draw_007/
    waveforms.npz
    per_epoch_metrics.csv
    per_stage_metrics.csv
    metrics.json
  plots/stage_metrics.png
  plots/stage_w_example.png
  plots/stage_light_example.png
  plots/stage_rem_example.png
  plots/stage_deep_example.png
```

`reference.npz` 保存共享真实 IBI、评分 mask、原条件、阶段和窗口起点。每个 `draw_NNN` 的 NPZ 保存该臂该次生成的未裁剪 IBI 与种子；逐 epoch/阶段 CSV 保留该次统计，根目录 `stage_summary.csv` 保存跨种子的 mean/std。缺失阶段没有可用波形时，对应示例图会明确标注 NA，不绘制虚构波形。

`summary.json` 和 `protocol.json` 记录完成状态、数据/模型/窗口选择身份、final checkpoint 哈希及产物哈希。请先核对这些记录、实际阶段覆盖和评分口径，再比较数值或图形。本地 CPU fixture / mock 测试只检查实现与流程，不能代替真实集群 CUDA 运行。
