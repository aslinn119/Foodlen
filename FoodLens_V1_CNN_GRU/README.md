# FoodLens V1：ResNet18 + GRU

适配当前 Food500Cap_ready 数据格式：`annotations/{train,val,test}.jsonl`，每条记录包含 `file_name`、`cat`、`caption`；图片位于对应相对路径；词表位于 `vocab.json`。

## 1. 安装依赖（PowerShell）

在解压后的项目根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 PyTorch/torchvision 安装失败，请根据电脑是否有 NVIDIA GPU，使用 PyTorch 官网安装器生成对应命令。

## 2. 先试跑 1 个 epoch

```powershell
python src/train.py --data-root "C:\Users\刘九尔\Desktop\神经网络与深度学习\Food500Cap_ready" --epochs 1 --batch-size 8 --num-workers 0
```

如果成功，再开始正式训练：

```powershell
python src/train.py --data-root "C:\Users\刘九尔\Desktop\神经网络与深度学习\Food500Cap_ready" --epochs 15 --batch-size 32 --num-workers 0
```

显存不足时将 batch size 改成 16、8 或 4。Windows 初次运行建议 `--num-workers 0`。默认尝试下载 ImageNet 预训练权重；首次可能需要联网。如果无法下载，可加 `--no-pretrained`，但通常会影响效果。

## 3. 生成测试集预测

```powershell
python src/predict.py --data-root "C:\Users\刘九尔\Desktop\神经网络与深度学习\Food500Cap_ready" --checkpoint outputs_v1/best_model.pt --split test --output outputs_v1/test_predictions.jsonl
```

## 4. 计算基础 BLEU-1 到 BLEU-4

```powershell
python src/evaluate_bleu.py --predictions outputs_v1/test_predictions.jsonl
```

## 5. 训练产物

- `outputs_v1/best_model.pt`：验证集 loss 最优模型
- `outputs_v1/last_model.pt`：最后一个 epoch 模型
- `outputs_v1/history.json`：训练历史
- `outputs_v1/test_predictions.jsonl`：测试预测（运行预测脚本后生成）

## 模型说明

V1 使用 ResNet18 提取全局图像特征，GRU 逐词生成描述。它是后续注意力模型、Transformer 和 ViT 的对照基线，不保证第一轮训练就达到最终高分。正式提交前，应统一评估分词规则，并实现至少三种要求的 caption 指标。
