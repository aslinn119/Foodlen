"""
train_bgen.py —— 训练 B-gen 的字符级 GRU 解码器（CLIP+LoRA 编码器冻结且特征已缓存）

训练设定：
  - teacher forcing，交叉熵（label smoothing 0.1），仅对非填充位置计损失
  - AdamW + cosine，梯度裁剪
  - 早停：以 val 集非填充位置 token 准确率为选型指标
  - 断点续训：latest.pt；完成删除断点并输出 DONE（与项目内其他训练脚本同约定）
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:  # 训练环境没有 tqdm 时仍可正常训练
    def tqdm(iterable, **kwargs):
        return iterable

from model_bgen import BGenModel, build_char_vocab, PAD, DATA_ROOT
from data_bgen import BGenSeqDataset, collate

CKPT_DIR = os.path.join(DATA_ROOT, "checkpoints_bgen")
os.makedirs(CKPT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {DEVICE}")

BATCH_SIZE = 128
NUM_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10


@torch.no_grad()
def val_token_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    for feats, ids, lengths in loader:
        feats, ids = feats.to(DEVICE), ids.to(DEVICE)
        logits = model(feats, ids[:, :-1])
        tgt = ids[:, 1:]
        mask = tgt != PAD
        pred = logits.argmax(dim=-1)
        correct += ((pred == tgt) & mask).sum().item()
        total += mask.sum().item()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-budget", type=float, default=None,
                        help="本次调用最多训练秒数，超时保存断点退出（下次自动续训）")
    args = parser.parse_args()
    t_start = time.time()

    char2idx, idx2char = build_char_vocab()
    V = len(idx2char)
    print(f"字符词表: {V}")

    train_ds = BGenSeqDataset("train")
    val_ds = BGenSeqDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            collate_fn=collate, num_workers=0)

    model = BGenModel(num_ingredients=151)
    model.set_vocab_size(V)
    model.to(DEVICE)
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"解码器可训练参数: {n_train/1e6:.3f}M（CLIP+LoRA 编码器冻结）")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    best_acc, history = -1.0, {"train_loss": [], "val_acc": []}
    start_epoch = 1
    latest = os.path.join(CKPT_DIR, "latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch, best_acc, history = ck["epoch"] + 1, ck["best_acc"], ck["history"]
        print(f"从断点续训：epoch {start_epoch}/{NUM_EPOCHS}，best_val_acc={best_acc:.4f}")

    done = start_epoch > NUM_EPOCHS
    bad_epochs = 0
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        if args.time_budget and (time.time() - t_start) > args.time_budget:
            print(f"[time-budget] 已用 {time.time()-t_start:.0f}s，保存断点退出")
            break
        model.train()
        # scheduled sampling 概率：前 10 轮 0 -> 0.30（缓解曝光偏差，保证开环生成能收尾）
        ss_prob = min(0.30, max(0.0, (epoch - 10) / 20.0) * 0.30)
        total_loss, n_tok = 0.0, 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for feats, ids, lengths in pbar:
            feats, ids = feats.to(DEVICE), ids.to(DEVICE)
            logits = model(feats, ids[:, :-1], ss_prob=ss_prob)
            tgt = ids[:, 1:]
            loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            n = (tgt != PAD).sum().item()
            total_loss += loss.item() * n
            n_tok += n
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        acc = val_token_acc(model, val_loader)
        avg_loss = total_loss / max(n_tok, 1)
        history["train_loss"].append(float(avg_loss))
        history["val_acc"].append(float(acc))
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}  val_tok_acc={acc:.4f}  ({time.time()-t0:.0f}s)")

        if acc > best_acc:
            best_acc, bad_epochs = acc, 0
            torch.save({"model_state": model.state_dict(),
                        "vocab_size": V, "emb_dim": model.emb_dim, "hidden": model.hidden,
                        "best_val_acc": float(acc)},
                       os.path.join(CKPT_DIR, "best_decoder.pt"))
            print(f"  -> 保存最优解码器 (val_tok_acc={best_acc:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"早停：连续 {PATIENCE} 轮无提升")
                done = True
                break

        torch.save({"model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch, "best_acc": best_acc, "history": history}, latest)
        with open(os.path.join(CKPT_DIR, "history_decoder.json"), "w") as f:
            json.dump(history, f, indent=2)
    else:
        done = True

    if done:
        if os.path.exists(latest):
            os.remove(latest)
        print(f"训练完成：最优 val token acc = {best_acc:.4f}")
        print("DONE")
    else:
        print(f"进度 {len(history['train_loss'])}/{NUM_EPOCHS} epochs，稍后继续")


if __name__ == "__main__":
    main()
