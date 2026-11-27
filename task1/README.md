# Task1 图像描述

| 目录 | 定位 |
|---|---|
| [v1_original](v1_original/README.md) | 组长原始V1，ResNet18+GRU；与下列重训消融分开 |
| [v2_original_and_ablation](v2_original_and_ablation/README.md) | 原V2注意力、无注意力重训V1及原报告，保留其内部依赖 |
| [v2_vocab_repaired](v2_vocab_repaired/README.md) | 新词表修复V2，25轮、第24轮最佳；作为新版接入候选 |
| [v3](v3/README.md) | V3 Transformer与layer4实验，保留正式及历史实验记录 |
| [v4](v4/v4/README.md) | ViT-B/32+Transformer；骨干冻结 |

四模型统一测试表见[评价总入口](../evaluation/README.md)。表中V1是无注意力重训消融、V2是旧注意力版本，不能标成组长原V1或词表修复V2。权重未复制，详见[外置清单](../assets/外置文件清单.csv)。
