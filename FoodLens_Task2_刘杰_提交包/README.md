# FoodLens Task 2 提交包（刘杰）

任务二：食物图像 → 结构化营养 JSON 生成。包含代码、训练好的模型权重、两份报告与全部实验结果。
**包内目录结构与原工作区一致，代码以自身所在目录为根查找模型与数据，解压到任何路径均可直接运行。**

## 包内容

```
（包根）
  README.md                 # 本文件
  requirements.txt          # 依赖清单

  # —— 代码（训练 / 评测 / 推理 / 出图）——
  data.py                   # 数据集与 DataLoader
  model.py                  # ResNet18 多输出回归
  train.py                  # ResNet18 训练 + 评估
  eval_baseline.py          # 均值基线 vs ResNet18 对比
  model_caption.py          # 模型 A：ResNet18+GRU 生成式 JSON 基线
  train_caption.py          # 模型 A 训练（支持 --time-budget 断点续训）
  model_lora.py             # 模型 B：CLIP ViT-B/32 + LoRA 多任务
  train_lora_mt.py          # 模型 B 训练
  metrics.py                # 统一评测：BLEU-1/4、ROUGE-L、CIDEr-D、F1、MSE/MAE/RMSE（自实现）
  eval_models.py            # 两模型统一评测入口
  json_parse.py             # JSON 文本解析
  stats.py                  # 数据统计（4 张图）
  make_figures.py / make_arch_figures.py / make_export_figs.py / make_resnet_arch.py
                            # 报告图表生成
  error_analysis_final.py   # 错误分析
  predict.py                # 交付接口 predict_nutrition(image_path) -> dict
  demo_task2.py             # 命令行 demo：python demo_task2.py <图片路径>

  # —— 模型权重（均已训练至最优，可直接加载）——
  checkpoints/best_model.pt              # ResNet18 回归最优权重
  checkpoints/baseline_vs_resnet18.csv   # 均值基线对比结果
  checkpoints_caption/best_model.pt      # 模型 A 最优权重
  checkpoints_lora_mt/best_model.pt      # 模型 B 最优权重（含食材词表与判定阈值）
  clip_model/                            # CLIP ViT-B/32 本地权重（离线可加载）
  ingredients.json                       # 151 种训练集食材词表

  # —— 标注与样例 ——
  annotations/              # {train,val,test}.jsonl（教学云平台 Nutrition5k 学生版）
  demo_images/              # 3 张测试样例图，可直接跑 demo

  # —— 报告 ——
  reports/FoodLens任务二报告_刘杰.docx   # 完整个人报告（含环境与复现说明）
  reports/task2_刘杰.docx                # 组报告节选（第 4 节 + 5.2 节）

  # —— 实验结果（复现产出，供查阅）——
  final_results/            # 两模型 val/test 全部指标与预测明细
  stats_result/             # 数据统计图
  report_figures_final/     # 报告图表（架构、训练曲线等）
  export_figs/              # 组报告用图（散点、demo、结构拼图）
  error_analysis_final/     # Top10 误差样本拼图与明细
```

## 环境复现

- Windows 11，GPU：RTX 4060 Laptop 8GB（CPU 亦可运行，仅速度较慢），Python 3.10 虚拟环境
- 安装依赖：`pip install -r requirements.txt`（torch 2.5.1+cu121；若官方源无 cu121 轮子，用 `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`）
- 完整图片数据（约 1.3 GB）来自教学云平台，解压为 `images/{train,val,test}/dish_*.png` 放在包根即可复现训练与评测

## 快速开始

```bash
python demo_task2.py demo_images/dish_1556575327.png   # 单张图片推理（开箱即用）
python eval_models.py                                   # 复现两模型统一评测
```

## 核心结果（test 集）

| 指标 | 模型 A（CNN+GRU） | 模型 B（CLIP+LoRA） |
|---|---|---|
| BLEU-4 | 0.6516 | 0.5427 |
| ROUGE-L | 0.8051 | 0.7579 |
| CIDEr-D | 8.48 | 7.43 |
| 食材 micro-F1 | 0.172 | 0.472 |
| 平均 MSE | 9074.8 | 3219.9 |
| 平均 MAE | 44.28 | 25.86（均值基线 65.51，ResNet18 27.24） |
