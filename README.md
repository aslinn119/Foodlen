# FoodLens —— 食物图像描述生成与营养估计

《神经网络与深度学习》课程设计项目。输入食物 RGB 图像，完成两个任务：

- **Task 1 图像描述生成**：为食物图像生成自然语言描述（image captioning）
- **Task 2 结构化营养预测**：预测菜品质量与营养成分（热量 / 蛋白质 / 脂肪 / 碳水化合物）
- **极限挑战（选做）**：单图像营养估计，普通 RGB 输入 → 总质量/总热量/宏量营养素，冻结评测协议

## 小组成员

| 成员 | 分工 |
| --- | --- |
| 刘九尔（组长） | 数据集构建与清洗、GPU 训练环境、V1（CNN+GRU）、V3（Transformer，待开展）、极限挑战集成与冻结协议、报告统稿 |
| 刘杰 | Task2 结构化预测全流程（均值基线 / ResNet18 / 生成式 JSON 基线 / CLIP+LoRA 多任务）、极限挑战冻结评测与结题数据 |
| 李馨瑶 | V2（CNN+GRU+注意力）与消融实验、演示系统、报告图表 |

## 目录结构

```
v1_cnn_gru/                    # 刘九尔：Task1 V1 基线（CNN+GRU）
FoodLens_V1_CNN_GRU/           # V1 整理版（供复现）
Task1_李馨瑶_组长整合包/         # 李馨瑶：V2 注意力模型 + 消融 + 结果图表
FoodLens_Task2_刘杰_提交包/      # 刘杰：Task2 全部代码、结果与报告
A1_mlp_numpy_提交包/             # 课程作业一：纯 NumPy MLP
任务书/                         # 三人的任务书
FoodLens_开题报告_v3.docx       # 开题报告（最新版）
报告.pdf / 报告骨架.md           # 中期/结题报告材料
```

## 关键结果（截至 2026-10-24）

- **V1**（CNN+GRU）：test BLEU-1 30.88 / BLEU-4 4.57
- **V2**（+注意力）：BLEU-4 5.29；无注意力消融 4.88（注意力带来一致提升）
- **Task2**（CLIP+LoRA 多任务，最佳）：平均 MAE 25.86；均值基线 65.51、ResNet18 回归 27.24、生成式基线 44.28（14.2% 解析失败）

## 数据集与权重说明（重要）

- **数据集不进仓库**：Food500Cap（ISIA Food-500 + 标注）约 3 GB，请从 ISIA 官方渠道下载；课程原始 `images.zip` / `annotations.zip` 由课程提供。
- **模型权重不进仓库**：所有 `*.pt` / `*.bin` 已被 `.gitignore` 排除（GitHub 单文件限 100 MB）。CLIP 预训练权重从 HuggingFace 下载，其余权重用各目录下的 `train*.py` 复现。
- 数据放置路径请参照各子目录 README 中的配置。

## 复现环境

- Python 3.10+，PyTorch（GPU 版 CUDA 12.x），torchvision，transformers，Pillow，numpy，tqdm
- 各子目录有独立 `requirements.txt` 与 `README.md`，进入对应目录按其说明运行
