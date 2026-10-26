"""
train_lora_mt.py —— 训练模型 B：CLIP + LoRA 参数高效微调（多任务）

任务头与损失：
  1) 营养回归（5 维）：对 (x-mean)/std 归一化后做加权 MSE
  2) 食材多标签（151 类）：BCEWithLogits，pos_weight 缓解类别不平衡
  3) 分项重量（151 类）：仅对出现的食材做 MSE（重量/100 归一化）

训练后在 val 上：
  - 选定食材分类阈值（最大化 micro-F1），随最优模型一并保存
选型指标：val 平均 MAE（与既有实验口径一致）。
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model_lora import (CLIPLoRAMultiTask, build_ingredient_vocab, count_lora_params,
                        NUM_KEYS, WEIGHT_NORM, DATA_ROOT, CLIP_PATH)

CKPT_DIR = os.path.join(DATA_ROOT, "checkpoints_lora_mt")
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {DEVICE}")

BATCH_SIZE = 16
NUM_EPOCHS = 25
LR = 5e-4
WEIGHT_DECAY = 1e-4
LAST_N = 4
RANK = 16
ALPHA = 16.0

LOSS_W_NUMERIC = 1.0
LOSS_W_INGREDIENT = 1.0
LOSS_W_ITEMWT = 0.5

# 数值字段加权（蛋白/脂肪最难，权重更高，与 v2 实验一致）
NUM_LOSS_WEIGHTS = torch.tensor([1.0, 1.5, 2.0, 2.0, 1.5], dtype=torch.float32)

CLIPT_TF_TRAIN = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4815, 0.4578, 0.4082],
                         std=[0.2686, 0.2613, 0.2758]),
])
CLIPT_TF_EVAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4815, 0.4578, 0.4082],
                         std=[0.2686, 0.2613, 0.2758]),
])


def load_records(split):
    with open(os.path.join(DATA_ROOT, "annotations", f"{split}.jsonl"), "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def compute_train_stats(records):
    vals = np.array([[r["target"][k] for k in NUM_KEYS] for r in records], dtype=np.float64)
    return vals.mean(axis=0), vals.std(axis=0)


class MultiTaskDataset(Dataset):
    def __init__(self, split, vocab, transform):
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}
        self.transform = transform
        self.records = load_records(split)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(os.path.join(DATA_ROOT, r["image"])).convert("RGB")
        img = self.transform(img)
        numeric = torch.tensor([r["target"][k] for k in NUM_KEYS], dtype=torch.float32)
        C = len(self.vocab)
        label = torch.zeros(C, dtype=torch.float32)
        wt = torch.zeros(C, dtype=torch.float32)
        for it in r["target"]["items"]:
            i = self.index.get(it["name"])
            if i is not None:
                label[i] = 1.0
                wt[i] = it["weight_g"] / WEIGHT_NORM
        return img, numeric, label, wt


@torch.no_grad()
def evaluate(model, loader, mean_np, std_np):
    """返回 numeric preds/targets、ingredient probs/labels、item-weight preds（反归一化后）"""
    model.eval()
    mean = torch.tensor(mean_np, dtype=torch.float32)
    std = torch.tensor(std_np, dtype=torch.float32)
    P, T, GP, GT, WP = [], [], [], [], []
    for imgs, numeric, label, wt in loader:
        imgs = imgs.to(DEVICE)
        out = model(imgs)
        P.append(out["numeric"].cpu() * std + mean)
        T.append(numeric)
        GP.append(torch.sigmoid(out["ingredient"]).cpu())
        GT.append(label)
        WP.append(out["item_weight"].cpu())
    return (np.concatenate(P), np.concatenate(T),
            np.concatenate(GP), np.concatenate(GT),
            np.concatenate(WP))


def numeric_mae(preds, targets):
    return np.abs(preds - targets).mean(axis=0)


def best_threshold(probs, labels, grid=None):
    """在 val 上网格搜索 sigmoid 阈值，最大化 micro-F1。"""
    if grid is None:
        grid = np.arange(0.05, 0.65, 0.025)
    best_t, best_f1 = 0.3, -1.0
    for t in grid:
        pred = probs >= t
        tp = float((pred & labels).sum())
        fp = float((pred & ~labels).sum())
        fn = float((~pred & labels).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def micro_f1(probs, labels, t):
    pred = probs >= t
    tp = float((pred & labels).sum())
    fp = float((pred & ~labels).sum())
    fn = float((~pred & labels).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-budget", type=float, default=None,
                        help="本次调用最多训练秒数，超时保存断点退出（下次自动续训）")
    args = parser.parse_args()
    t_start = time.time()

    vocab = build_ingredient_vocab()
    C = len(vocab)
    print(f"食材类别数: {C}")

    train_records = load_records("train")
    mean_np, std_np = compute_train_stats(train_records)
    mean = torch.tensor(mean_np, dtype=torch.float32, device=DEVICE)
    std = torch.tensor(std_np, dtype=torch.float32, device=DEVICE)
    print("train 均值:", np.round(mean_np, 1))
    print("train 标准差:", np.round(std_np, 1))

    # 多标签 BCE 的 pos_weight（按训练集正负比例，截断防止爆炸）
    pos = np.zeros(C)
    for r in train_records:
        for it in r["target"]["items"]:
            i = vocab.index(it["name"]) if it["name"] in vocab else -1
            if i >= 0:
                pos[i] += 1
    neg = len(train_records) - pos
    pos_weight_np = np.clip(neg / np.maximum(pos, 1), 1.0, 20.0)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=DEVICE)

    train_ds = MultiTaskDataset("train", vocab, CLIPT_TF_TRAIN)
    val_ds = MultiTaskDataset("val", vocab, CLIPT_TF_EVAL)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CLIPLoRAMultiTask(num_targets=5, num_ingredients=C,
                              last_n=LAST_N, rank=RANK, alpha=ALPHA).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数 {total/1e6:.2f}M，可训练 {trainable/1e6:.3f}M "
          f"({trainable/total*100:.2f}%)，其中 LoRA {count_lora_params(model)/1e6:.3f}M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    mse = nn.MSELoss(reduction="none")
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    w_num = NUM_LOSS_WEIGHTS.to(DEVICE)

    best_val_mae = float("inf")
    best_t, best_f1 = 0.3, 0.0
    history = {"train_loss": [], "val_mae": [], "val_f1": []}
    start_epoch = 1
    latest = os.path.join(CKPT_DIR, "latest.pt")
    if os.path.exists(latest):  # 断点续训
        ck = torch.load(latest, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = ck["epoch"] + 1
        best_val_mae = ck["best_val_mae"]
        best_t, best_f1 = ck["best_t"], ck["best_f1"]
        history = ck["history"]
        print(f"从断点续训：epoch {start_epoch}/{NUM_EPOCHS}，best_val_MAE={best_val_mae:.2f}")

    done = start_epoch > NUM_EPOCHS
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        if args.time_budget and (time.time() - t_start) > args.time_budget:
            print(f"[time-budget] 已用 {time.time()-t_start:.0f}s，保存断点退出")
            break
        model.train()
        total_loss = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for imgs, numeric, label, wt in pbar:
            imgs = imgs.to(DEVICE)
            numeric = numeric.to(DEVICE)
            label = label.to(DEVICE)
            wt = wt.to(DEVICE)

            out = model(imgs)
            loss_num = (mse(out["numeric"], (numeric - mean) / std) * w_num).mean()
            loss_ing = bce(out["ingredient"], label)
            mask = label > 0.5
            loss_wt = (mse(out["item_weight"][mask], wt[mask])).mean() if mask.any() \
                else torch.tensor(0.0, device=DEVICE)
            loss = LOSS_W_NUMERIC * loss_num + LOSS_W_INGREDIENT * loss_ing + \
                LOSS_W_ITEMWT * loss_wt

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        avg_loss = total_loss / len(train_loader.dataset)

        P, T, GP, GT, WP = evaluate(model, val_loader, mean_np, std_np)
        mae = numeric_mae(P, T)
        t, f1 = best_threshold(GP, GT.astype(bool))
        history["train_loss"].append(float(avg_loss))
        history["val_mae"].append(float(mae.mean()))
        history["val_f1"].append(float(f1))
        print(f"Epoch {epoch}: loss={avg_loss:.4f}  val_MAE={mae.mean():.2f}  "
              f"val_microF1={f1:.3f} (t={t:.2f})  ({time.time()-t0:.0f}s)")

        if mae.mean() < best_val_mae:
            best_val_mae = mae.mean()
            best_t, best_f1 = t, f1
            torch.save({"model_state": model.state_dict(),
                        "mean": mean_np.tolist(), "std": std_np.tolist(),
                        "threshold": best_t, "val_mae": float(mae.mean()),
                        "val_f1": float(f1),
                        "lora_last_n": LAST_N, "rank": RANK, "alpha": ALPHA,
                        "num_ingredients": C},
                       os.path.join(CKPT_DIR, "best_model.pt"))
            print(f"  -> 保存最优模型 (val_MAE={best_val_mae:.2f}, F1={f1:.3f})")

        torch.save({"model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch, "best_val_mae": best_val_mae,
                    "best_t": best_t, "best_f1": best_f1, "history": history},
                   latest)
        with open(os.path.join(CKPT_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
    else:
        done = True

    if done:
        os.remove(latest) if os.path.exists(latest) else None
        print(f"训练完成：最优 val_MAE={best_val_mae:.2f}，阈值 t={best_t:.3f} (F1={best_f1:.3f})")
        print("DONE")
    else:
        print(f"进度 {history and len(history['train_loss'])}/{NUM_EPOCHS} epochs，稍后继续")


if __name__ == "__main__":
    main()
