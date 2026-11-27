"""
train_lora_ablation.py —— P1③ 消融对照训练：CLIP 冻结头 vs CLIP+LoRA（同框架、同数据、同超参）

仅改变"视觉编码器是否注入 LoRA"这一个因素：
  --rank 16  -> CLIP 最后 last_n 层注意力注入 LoRA（同模型 B 官方配置）
  --rank 0   -> 冻结 CLIP，无任何适配器，仅训练三个任务头（纯冻结基线）

两种配置共享：数据划分、增强、损失权重、优化器、epoch 数与选型口径（val 平均 MAE）。
支持 --time-budget 断点续训（每次调用限时运行，退出后再次调用自动续训）。
每个 epoch 记录墙钟时间，训练结束后输出总耗时供消融表引用。
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {DEVICE}")

BATCH_SIZE = 16
NUM_EPOCHS = 25
LR = 5e-4
WEIGHT_DECAY = 1e-4

LOSS_W_NUMERIC = 1.0
LOSS_W_INGREDIENT = 1.0
LOSS_W_ITEMWT = 0.5

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0,
                        help="LoRA rank；0 = 不注入 LoRA（纯冻结基线），16 = 同模型 B 官方配置")
    parser.add_argument("--last-n", type=int, default=4, help="注入 LoRA 的层数（rank>0 时生效）")
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="输出目录，默认 checkpoints_ablation_frozen(rank=0) / checkpoints_ablation_lora(rank>0)")
    parser.add_argument("--time-budget", type=float, default=None,
                        help="本次调用最多训练秒数，超时保存断点退出（下次自动续训）")
    args = parser.parse_args()
    t_start = time.time()

    tag = "frozen" if args.rank <= 0 else f"lora_r{args.rank}"
    ckpt_dir = args.ckpt_dir or os.path.join(DATA_ROOT, f"checkpoints_ablation_{tag}")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vocab = build_ingredient_vocab()
    C = len(vocab)
    print(f"食材类别数: {C}，配置: rank={args.rank} last_n={args.last_n} seed={args.seed}")

    train_records = load_records("train")
    mean_np, std_np = compute_train_stats(train_records)
    mean = torch.tensor(mean_np, dtype=torch.float32, device=DEVICE)
    std = torch.tensor(std_np, dtype=torch.float32, device=DEVICE)

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
    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CLIPLoRAMultiTask(num_targets=5, num_ingredients=C,
                              last_n=args.last_n, rank=args.rank, alpha=args.alpha).to(DEVICE)
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
    history = {"train_loss": [], "val_mae": [], "val_f1": [], "epoch_sec": []}
    start_epoch = 1
    train_sec_total = 0.0
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest):  # 断点续训
        ck = torch.load(latest, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = ck["epoch"] + 1
        best_val_mae = ck["best_val_mae"]
        best_t, best_f1 = ck["best_t"], ck["best_f1"]
        history = ck["history"]
        train_sec_total = ck.get("train_sec_total", 0.0)
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
        ep_sec = time.time() - t0
        train_sec_total += ep_sec

        P, T, GP, GT, WP = evaluate(model, val_loader, mean_np, std_np)
        mae = numeric_mae(P, T)
        t, f1 = best_threshold(GP, GT.astype(bool))
        history["train_loss"].append(float(avg_loss))
        history["val_mae"].append(float(mae.mean()))
        history["val_f1"].append(float(f1))
        history["epoch_sec"].append(float(ep_sec))
        print(f"Epoch {epoch}: loss={avg_loss:.4f}  val_MAE={mae.mean():.2f}  "
              f"val_microF1={f1:.3f} (t={t:.2f})  ({ep_sec:.0f}s)")

        if mae.mean() < best_val_mae:
            best_val_mae = mae.mean()
            best_t, best_f1 = t, f1
            torch.save({"model_state": model.state_dict(),
                        "mean": mean_np.tolist(), "std": std_np.tolist(),
                        "threshold": best_t, "val_mae": float(mae.mean()),
                        "val_f1": float(f1),
                        "lora_last_n": args.last_n, "rank": args.rank, "alpha": args.alpha,
                        "num_ingredients": C},
                       os.path.join(ckpt_dir, "best_model.pt"))
            print(f"  -> 保存最优模型 (val_MAE={best_val_mae:.2f}, F1={f1:.3f})")

        torch.save({"model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch, "best_val_mae": best_val_mae,
                    "best_t": best_t, "best_f1": best_f1, "history": history,
                    "train_sec_total": train_sec_total},
                   latest)
        with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
    else:
        done = True

    if done:
        os.remove(latest) if os.path.exists(latest) else None
        summary = {"tag": tag, "rank": args.rank, "last_n": args.last_n,
                   "seed": args.seed, "epochs": NUM_EPOCHS,
                   "best_val_mae": float(best_val_mae), "best_val_f1": float(best_f1),
                   "threshold": float(best_t), "train_sec_total": float(train_sec_total),
                   "trainable_M": trainable / 1e6, "lora_M": count_lora_params(model) / 1e6}
        with open(os.path.join(ckpt_dir, "ablation_result.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"训练完成：最优 val_MAE={best_val_mae:.2f}，阈值 t={best_t:.3f} (F1={best_f1:.3f})")
        print(f"总训练耗时 {train_sec_total/60:.1f} min（纯训练+验证，不含数据加载）")
        print("DONE")
    else:
        print(f"进度 {history and len(history['train_loss'])}/{NUM_EPOCHS} epochs，稍后继续")


if __name__ == "__main__":
    main()
