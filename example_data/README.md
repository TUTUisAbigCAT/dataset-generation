# 人工接口样例（非真实 DREAMT 数据）

本目录由 scripts/create_example_data.py 现场生成：4 位虚构受试者，每人 8 个 30 秒 epoch，25 Hz、4 分钟、IBI 单位为秒。波形仅用于文件接口和程序执行测试，不模拟真实生理分布。

每个 NPZ 仅含 data [6000,1]、stages [6000]、fs、ahi 四字段；原始阶段编码为 W=0、N1=1、N2=2、N3=3、REM=4。CSV 提供 SID/AHI/BMI/AGE/GENDER。局部 0 是刻意加入的缺测标记，不是真实生理 IBI。

未附带 splits.json，用于演示适配器的固定种子自动受试者划分。

训练或生成结果只能说明接口可执行，不能用于真实 DREAMT 效果或生成质量结论。
