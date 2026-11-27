# FoodLens V3 — V1 对齐版（2026-10-28）

本版直接复用你上传的 V1 `data.py`：
- Tokenizer: `caption.lower().split()`；`max_len=30`；特殊符号 `<PAD>/<BOS>/<EOS>/<UNK>` 分别为 0/1/2/3。
- 训练图像: `Resize(256) -> RandomCrop(224) -> RandomHorizontalFlip() -> ToTensor() -> ImageNet Normalize`。
- 验证图像: `Resize(256) -> CenterCrop(224) -> ToTensor() -> ImageNet Normalize`。
- 直接从**可信的** V1 checkpoint `best_v1.pt` 中读取词表，不使用早期 75 token 词表，也不重新生成词表。
- V3 架构：ResNet18 `layer4` 的 7×7 网格（49 token）+ Transformer Encoder（视觉自注意力）+ Transformer Decoder（因果自注意力、跨注意力）。

## 一、文件夹说明
`v3/v1_data.py` 为你上传的 V1 原版数据模块拷贝；其余 `model.py`, `data.py`, `train.py`, `infer.py` 为 V3 专用代码。

## 二、Windows 操作（在 v3 文件夹内打开 PowerShell）
在命令行先 `cd` 进入解压后的 `FoodLens_V3_V1Aligned\v3` 文件夹。请将下面路径改成你自己的真实路径。

快速小样本检查（不下载预训练模型，网络不通也可试运行）：

```powershell
python train.py --data-root "D:\Food500Cap_ready" --vocab "D:\FoodCaption_v1\runs\v1\best_v1.pt" --run-dir "outputs\smoke" --epochs 2 --train-limit 8 --val-limit 8 --batch-size 2 --no-pretrained --freeze-cnn
```

有预训练 ResNet18 权重时，进一步小样本训练（建议使用 GPU）：

```powershell
python train.py --data-root "D:\Food500Cap_ready" --vocab "D:\FoodCaption_v1\runs\v1\best_v1.pt" --run-dir "outputs\overfit" --epochs 40 --train-limit 16 --val-limit 16 --batch-size 4 --freeze-cnn
```

正式训练（整个 train 17784 张、val 1967 张；保持数据隔离）：

```powershell
python train.py --data-root "D:\Food500Cap_ready" --vocab "D:\FoodCaption_v1\runs\v1\best_v1.pt" --run-dir "outputs\formal" --epochs 20 --train-limit 17784 --val-limit 1967 --batch-size 16
```

产生**验证集**预测（给李馨瑶；`--limit 0` 代表全量）：

```powershell
python infer.py --checkpoint "outputs\formal\best_v3.pt" --data-root "D:\Food500Cap_ready" --split val --limit 0 --output "outputs\v3_val_predictions.jsonl"
```

输出 JSONL 包含 `file_name`、`caption`（预测）与 `reference`（原始参考）。

## 三、重要提示
1. `--data-root` 必须是包含 `annotations/train.jsonl`、`val.jsonl` 和对应图像的真实数据集目录。以上 `D:\Food500Cap_ready` 只是示例路径，不一定存在。
2. `--vocab` 可以指向原 V1 `vocab.json` 或可信 V1 权重 `best_v1.pt`。如权重词表字段不是 `vocab`，脚本会直接报出键名，便于排查。
3. 不要用 test 集调参；验证集用于选模。此版最佳模型按 val loss 选取，正式与 V1/V2 公平评测应另用统一指标脚本。
4. 小样本测试只是流程自检，不代表模型性能；是否真正过拟合须观察训练 loss 持续降低，`--no-pretrained` 仅为离线 smoke 备选。
5. 文件尚未在你的真实数据上训练。第一次试跑请把完整终端输出发回来。

## 断点续训版（2026-10-28）

- `--train-limit`、`--val-limit` 默认是 `None`，即读取全部训练/验证样本。之前的初版默认仅各读取 16 张，正式训练必须使用此修订版！
- 每轮结束自动保存 `last_v3.pt`（含模型、优化器、历史、已完成 epoch），验证最优权重另存 `best_v3.pt`。
- `--resume outputs/full_v3/last_v3.pt` 恢复。`--epochs` 是**总轮数**，不是额外轮数。
- 续训要保持数据位置、batch size、限制数量、冻结策略、词表及模型参数不变。
- 若在首轮保存 `last_v3.pt` 前中断，无法恢复。已存在的旧版 `best_v3.pt` 不含优化器参数，不能精确断点续训。
