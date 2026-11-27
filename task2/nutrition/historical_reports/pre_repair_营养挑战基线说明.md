# 营养挑战基线包说明

本文件对应任务书 11 月 9 日的营养挑战基线交付。提交包中包含三类可复用基线：

| 基线 | 权重或来源 | 输出能力 |
|---|---|---|
| 训练集均值 | `eval_baseline.py` 与训练集统计 | 五个营养数值字段，不依赖模型权重 |
| ResNet18 回归 | `checkpoints/best_model.pt` | 五个营养数值字段 |
| CLIP+LoRA 多任务 | `checkpoints_lora_mt/best_model.pt` | 食材、分项重量和五个营养数值字段 |

挑战统一入口为 `predict_challenge_baseline(image_path, model_id)`，支持 `mean`、`resnet18` 和 `clip`，均返回相同的五项数值字段。历史结果见下列文件。

- `checkpoints/baseline_vs_resnet18.csv`
- `final_results/test_metrics.json`
- `final_results/metrics_summary.csv`
- `final_results/modelB_test_preds.jsonl`

B-gen 是生成式扩展基线，结果见 `final_results/bgen_metrics.json` 和 `final_results/modelBgen_test_preds.jsonl`。

