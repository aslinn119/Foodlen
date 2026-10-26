# FoodLens Task 1 李馨瑶

本目录只包含 Task 1（FoodLens 食物图像描述）所需的数据、V2 注意力模型、实验代码和交付材料。

## 任务与模型

V2 使用 ImageNet 预训练 ResNet18 提取 7×7 空间特征，在每个 GRU 解码步通过 Bahdanau 加性注意力得到上下文向量，并支持 greedy/beam 解码与词级注意力热图。V1 代码来自组长提供的源码包，数据加载、图像变换和 BLEU 计算保持复用。

## 数据准备与核验

原始清洗数据压缩包位于 `D:\foodlens_data\qingxishuju\Food500Cap_ready.zip`。运行 `prepare_data.py` 将其解压到本目录下的 `Food500Cap_ready`，并从训练 caption 按 min_count=5 重建词表。ZIP 实际缺少 README 中声明的 `vocab.json`；重建后词表大小为 2,681，特殊 token id 为 PAD=0、BOS=1、EOS=2、UNK=3。

`audit_ready_data.py` 检查三份标注中的图像路径、重复路径、空 caption 与词表规模。审计结果保存在 `checks/ready_data_audit.json`。本次核验通过：train 17,784、val 1,967、test 4,938，共 24,689 张图像。

## 环境与运行

从工作区 `A3卷积网络` 运行：

```powershell
$env:TORCH_HOME = "$PWD\FoodLens_Task1_李馨瑶\.torch_cache"
\.venv\Scripts\python.exe FoodLens_Task1_李馨瑶\v2_attention\test_model.py
\.venv\Scripts\python.exe FoodLens_Task1_李馨瑶\audit_ready_data.py
\.venv\Scripts\python.exe FoodLens_Task1_李馨瑶\v2_attention\train_v2.py --root FoodLens_Task1_李馨瑶\Food500Cap_ready --epochs 20 --workers 4
```

V2 的配置和逐轮训练记录在 `v2_attention/runs/v2/config.json`、`history.json`，验证集 BLEU-4 最佳权重为 `best_v2.pt`。训练结束后，对完整 test 集做统一 BLEU 评测：

```powershell
\.venv\Scripts\python.exe FoodLens_Task1_李馨瑶\v2_attention\evaluate_task1.py --root FoodLens_Task1_李馨瑶\Food500Cap_ready --checkpoint FoodLens_Task1_李馨瑶\v2_attention\runs\v2\best_v2.pt --decode greedy --output FoodLens_Task1_李馨瑶\results\v2_greedy_test.json
\.venv\Scripts\python.exe FoodLens_Task1_李馨瑶\v2_attention\evaluate_task1.py --root FoodLens_Task1_李馨瑶\Food500Cap_ready --checkpoint FoodLens_Task1_李馨瑶\v2_attention\runs\v2\best_v2.pt --decode beam --beam-size 3 --output FoodLens_Task1_李馨瑶\results\v2_beam_test.json
```

`v2_attention/train_v2.py --variant v1_ablation` 可从头训练同配置 V1 结构作为无注意力消融。原始 V1 测试指标来自组长提供的结果记录；提供的源码 ZIP 未包含对应 checkpoint，因此该历史结果与当前重训消融应在报告中明确区分。

本次完整 test 结果（BLEU-1/2/3/4；ROUGE-L；CIDEr-D）：重训 V1 无注意力为 31.56/15.57/8.55/4.88；V2 greedy 为 32.33/16.54/9.26/5.29；V2 beam-3 为 31.74/16.46/9.39/5.40。相应 ROUGE-L 为 0.2616/0.2695/0.2709，CIDEr-D 为 0.3374/0.3713/0.4070。组长提供的历史 V1 BLEU-1/2/3/4 为 30.88/15.26/8.26/4.57，但旧 checkpoint 未提供。与重训 V1 消融相比，V2 greedy 的 BLEU-4 高 0.41 分；beam-3 比 greedy 高 0.11 分。完整记录在 `results/task1_results_summary.json`。

`compute_saved_metrics.py` 对已保存的 full-test captions 计算 ROUGE-L 与 CIDEr-D。ROUGE-L 使用 LCS、β=1.2；CIDEr-D 使用 1–4 gram TF-IDF、σ=6 和标准 ×10 标度。每张图只有一条参考描述，解释和比较时应注明单参考设置。

## 报告与结果

`task1_李馨瑶.docx` 为可单独交给组长统稿的 Word 章节，含模型方法、超参数、完整测试对比表、训练曲线、BLEU 对比图、三组预测/参考文本及注意力热图。图表生成脚本为 `build_report_figures.py`，报告生成脚本为 `create_task1_report.py`。
