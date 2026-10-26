# V1：ResNet18 全局特征 + GRU 解码器（CNN Global + GRU 基线）

对应开题报告模型路线 **V1：CNN Global + GRU**，Show and Tell 风格的编码器—解码器基线。

## 文件

| 文件 | 作用 |
|---|---|
| `data.py` | Food500Cap_ready 数据集加载、vocab 读取、训练/验证图像变换、变长 caption 填充 collate |
| `model.py` | `EncoderCNN`（ResNet18 去分类层 → 512 维全局特征 → 投影到嵌入空间）、`DecoderGRU`（图像特征初始化隐状态，自回归生成）、贪心解码 + 束搜索 |
| `bleu.py` | 自包含语料级 BLEU-1~4（训练监控用，正式指标后续换 pycocoevalcap） |
| `train.py` | 训练 / 验证 / 早停 / 断点保存 + `--smoke` 小样本过拟合测试 |

## 环境要求

Python 3.10+，安装 PyTorch（GPU 版）与 torchvision：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 使用

```bash
# 第一步：管线自检（64 张图过拟合 300 步，应输出 PASS）
python train.py --smoke

# 正式训练（默认 20 epoch，val BLEU-4 早停，耐心 4）
python train.py

# 常用调参
python train.py --epochs 30 --batch-size 64 --hidden-size 512 --lr-dec 1e-3
```

数据路径默认指向 `C:\Users\刘九尔\Desktop\神经网络与深度学习\Food500Cap_ready`，
换位置时用 `--root` 指定。

## 输出

- `runs/v1/best_v1.pt`：验证集 BLEU-4 最优模型（含 vocab 与配置，可直接加载推理）；
- `runs/v1/history.json`：每 epoch 的 loss / BLEU / 耗时记录。

## 说明

- 编码器使用 ImageNet 预训练 ResNet18，`enc_finetune=True` 全程微调（学习率 1e-4，比解码器 1e-3 小一个量级）；
- 训练用 Teacher Forcing + pack_padded_sequence，交叉熵损失忽略 `<PAD>`，梯度裁剪 5.0；
- 验证监控用贪心解码 + 自包含 BLEU（单参考、语料级），与最终 pycocoevalcap 指标会有系统偏差，仅用于模型选择和曲线记录；
- 束搜索已实现（`model.beam_decode`），V1 默认贪心，解码策略对比在实验阶段统一做。
