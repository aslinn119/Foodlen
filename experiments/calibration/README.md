# 创新二 营养预测区间校准

状态更新：2026年11月28日纳入刘杰正式交付，基础实验已完成。本页集中导航，实际源码和结果保存在[营养极限挑战正式模块](../nutrition_challenge/README.md)，避免平行维护两套校准代码。

| 内容 | 实现和结果 |
|---|---|
| 基础Split Conformal | [conformal.py](../nutrition_challenge/challenge_external/conformal.py)，绝对残差的有限样本次序统计量 |
| 独立角色划分与冻结协议 | [主实验协议](../nutrition_challenge/challenge_audit/omnifood/experiment_protocol.json)、[主报告](../nutrition_challenge/challenge_external/实验报告.md) |
| 主实验阈值 | [calibration.json](../nutrition_challenge/challenge_external/results/calibration.json)，5668校准场景、逐字段90%目标 |
| 覆盖率、区间宽度和误差 | [metrics.csv](../nutrition_challenge/challenge_external/results/metrics.csv)，2368评估场景、三模型五字段 |
| 欠覆盖案例 | [完整记录](../nutrition_challenge/challenge_external/results/undercoverage_cases.jsonl) |
| 真实图片区间接口 | [runtime.py](../nutrition_challenge/challenge_external/runtime.py)、[接口样例](../nutrition_challenge/challenge_external/results/interface_example.json) |
| 餐盘域适配与按预测量缩放区间 | [NutritionVerse报告](../nutrition_challenge/challenge_external/nutritionverse_extension/餐盘优化实验报告.md) |
| 单对象域补充评价 | [MetaFood结果](../nutrition_challenge/challenge_external/metafood_extension/metrics.json) |
| 交付复核及环境限制 | [复核记录](../nutrition_challenge/integration_checks.json) |

各模型、数据域及校准版本分别使用，不把挑战历史CLIP阈值直接用于Task2 normalized B。外部校准数据未用于已登记基线训练/选型；采集批次独立性和交换性未被证实，只报告实测覆盖。90%目标没有在所有字段达标，不等于单图或五项联合保证。负责人按开题为刘杰。
