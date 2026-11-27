# Task2 营养预测

先看[本次修订成果说明](成果说明_2026-11-23.txt)，再看[模块使用说明](nutrition/使用说明.txt)与[README](nutrition/README.md)。

- 主成果：CNN+GRU生成模型A、CLIP+LoRA结构化预测模型B。
- B-gen仅保留历史生成式对照；均值和ResNet18为基础数值基线。
- 当前食材F1及数值误差优先引用[11月23日保存预测复核](nutrition/results_repaired_20261123/summary.json)。旧报告/旧结果保留作追溯，不能优先于修订说明。
- 最新模型依赖见[MODELS.json](nutrition/MODELS.json)。新B与历史B的权重及接口不同；照模块说明整体使用。
- 本轮是保存预测复算，未重跑全量推理；文本指标仍需按说明区分历史成绩与新复核。
- 完整图片仍缺，食材筛选阈值调整不等于创新二的保形区间校准。
