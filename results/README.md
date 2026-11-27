# 正式结果入口

结果文件保留在各模块原内部路径，只在此集中索引，避免多份总表不同步。

| 内容 | 结果与说明 |
|---|---|
| Task1四模型统一测试 | [4938张总表](../evaluation/task1_stage2/outputs/v1_v4_test/v1_v2_reevaluation.csv)；原V2与重训V1，不含词表修复V2 |
| 新版V2 | [新旧比较](../task1/v2_vocab_repaired/comparison.csv)、[训练结果](../task1/v2_vocab_repaired/训练结果.md)、[完整结果](../task1/v2_vocab_repaired/final_results.json) |
| Task2修订 | [六份保存预测复核汇总](../task2/nutrition/results_repaired_20261123/summary.json)、[成果说明](../task2/成果说明_2026-11-23.txt) |
| Judge与注意力 | [阶段交付](../evaluation/task1_stage2/最终交付/交接说明.md)、[报告](../evaluation/task1_stage2/最终交付/FoodLens阶段2报告_李馨瑶.docx) |
| 创新一 | [重排序报告](../experiments/reranking/REPORT.md)、[用户确认实体指标](../experiments/reranking/results/entity_metrics_human.json) |
| SCST正式预算实验 | [报告](../experiments/scst/REPORT.md)、[比较表](../experiments/scst/results/comparison.csv)、[训练曲线](../experiments/scst/results/training_and_results.png) |
| SCST早期试跑 | [历史33步报告](../experiments/scst_pilot/REPORT.md)，不作正式实验结果 |
| 创新二 | [尚待完成](../experiments/calibration/README.md) |

引用时带上模型版本、样本集合、指标尺度和评价实现。历史测试集此前已被查看；新一轮整理不改变该事实。创新一标注是助手辅助、用户确认，不能称独立人工盲标；SCST已完成单种子固定预算实验；本次结果不能推出RL普遍无效。
