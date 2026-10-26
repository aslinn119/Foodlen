"""
train_caption.py —— 训练模型 A：CNN(ResNet18) + GRU 营养 JSON 描述生成模型

训练目标：给定图像，自回归生成 target_text（营养 JSON 字符串）。
损失：token 级交叉熵（忽略 <pad>）。
选型：val loss 最低；训练曲线与最终 BLEU/ROUGE-L/CIDEr-D 由 eval_models.py 统一评测。
"""

import os
import sys
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

from model_caption import (build_vocab_from_train, CaptionModel,
                         PAD, MAX_LEN, DATA_ROOT)

CKPT_DIR = os.path.join(DATA_ROOT, "checkpoints_caption")
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {DEVICE}")

BATCH_SIZE = 32
NUM_EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

# ---------- 数据增强（与回归模型一致，保持风格统一） ----------
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CaptionDataset(Dataset):
    def __init__(self, split, vocab, transform, max_len=MAX_LEN):
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len
        with open(os.path.join(DATA_ROOT, "annotations", f"{split}.jsonl"), "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(os.path.join(DATA_ROOT, r["image"])).convert("RGB")
        img = self.transform(img)
        ids = self.vocab.encode(r["target_text"])[: self.max_len]
        return img, torch.tensor(ids, dtype=torch.long)


def collate(batch):
    imgs, seqs = zip(*batch)
    T = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return torch.stack(imgs), padded


@torch.no_grad()
def eval_loss(model, loader, criterion):
    model.eval()
    total, ntok = 0.0, 0
    for imgs, captions in loader:
        imgs = imgs.to(DEVICE)
        captions = captions.to(DEVICE)
        logits = model(imgs, captions)
        loss = criterion(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))
        total += loss.item() * (captions[:, 1:] != PAD).sum().item()
        ntok += (captions[:, 1:] != PAD).sum().item()
    return total / max(ntok, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-budget", type=float, default=None,
                        help="本次调用最多训练秒数，超时保存断点退出（下次自动续训）")
    args = parser.parse_args()
    t_start = time.time()

    vocab = build_vocab_from_train()
    print(f"词表大小: {len(vocab)}")
    with open(os.path.join(CKPT_DIR, "vocab_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab_size": len(vocab)}, f)

    train_ds = CaptionDataset("train", vocab, train_tf)
    val_ds = CaptionDataset("val", vocab, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=collate)

    model = CaptionModel(len(vocab)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    start_epoch = 1
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}
    latest = os.path.join(CKPT_DIR, "latest.pt")
    if os.path.exists(latest):  # 断点续训
        ck = torch.load(latest, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        history = ck["history"]
        print(f"从断点续训：epoch {start_epoch}/{NUM_EPOCHS}，best_val={best_val:.4f}")

    done = start_epoch > NUM_EPOCHS
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        if args.time_budget and (time.time() - t_start) > args.time_budget:
            print(f"[time-budget] 已用 {time.time()-t_start:.0f}s，保存断点退出")
            break
        model.train()
        total, ntok = 0.0, 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for imgs, captions in pbar:
            imgs = imgs.to(DEVICE)
            captions = captions.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs, captions)
            loss = criterion(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            n = (captions[:, 1:] != PAD).sum().item()
            total += loss.item() * n
            ntok += n
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        scheduler.step()

        train_loss = total / max(ntok, 1)
        val_loss = eval_loss(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"({time.time()-t0:.0f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(),
                        "vocab_size": len(vocab)},
                       os.path.join(CKPT_DIR, "best_model.pt"))

        torch.save({"model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch, "best_val": best_val, "history": history},
                   latest)
        with open(os.path.join(CKPT_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
    else:
        done = True

    if done:
        os.remove(latest) if os.path.exists(latest) else None
        print(f"训练完成，最优 val_loss={best_val:.4f}，模型保存在 {CKPT_DIR}")
        print("DONE")
    else:
        print(f"进度 {history and len(history['train_loss'])}/{NUM_EPOCHS} epochs，稍后继续")


if __name__ == "__main__":
    main()
