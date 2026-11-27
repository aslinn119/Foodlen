# -*- coding: utf-8 -*-
"""V1 训练脚本：ResNet18 + GRU 图像描述基线。

用法：
    python train.py                # 正式训练（默认配置）
    python train.py --smoke        # 小样本过拟合测试：64 张图、300 步，验证管线正确性
    python train.py --epochs 30 --batch-size 64

训练日志与最佳模型保存在 ./runs/v1/ 下。
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data import FoodCapDataset, load_vocab, train_transforms, eval_transforms, CollateFn
from model import CaptionModel
from bleu import corpus_bleu

# ---------- 默认配置 ----------
ROOT = r"C:\Users\刘九尔\Desktop\神经网络与深度学习\Food500Cap_ready"
RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "v1")
CKPT = os.path.join(RUN_DIR, "best_v1.pt")

p = argparse.ArgumentParser()
p.add_argument("--root", default=ROOT)
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--embed-dim", type=int, default=256)
p.add_argument("--hidden-size", type=int, default=512)
p.add_argument("--lr-dec", type=float, default=1e-3)
p.add_argument("--lr-enc", type=float, default=1e-4)
p.add_argument("--max-len", type=int, default=30)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--val-num", type=int, default=500, help="每次验证用多少张 val 图")
p.add_argument("--patience", type=int, default=4)
p.add_argument("--smoke", action="store_true", help="小样本过拟合测试模式")
p.add_argument("--smoke-steps", type=int, default=300)
p.add_argument("--smoke-size", type=int, default=64)
args = p.parse_args()


def evaluate(model, loader, vocab, device, max_decode=300):
    """贪心解码 + 语料级 BLEU（仅监控用）。"""
    model.eval()
    hyps, refs = [], []
    n = 0
    inv = {i: w for w, i in vocab.items()}
    with torch.no_grad():
        for imgs, _, _, raw in loader:
            imgs = imgs.to(device)
            for words in model.greedy_decode(imgs, vocab, max_len=args.max_len):
                hyps.append(words)
            refs += [r.lower().split() for r in raw]
            n += imgs.size(0)
            if n >= max_decode:
                break
    b = corpus_bleu(hyps[:max_decode], refs[:max_decode])
    model.train()
    return b  # [bleu1, bleu2, bleu3, bleu4]


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)

    vocab = load_vocab(os.path.join(args.root, "vocab.json"))
    pad_id = vocab["<PAD>"]
    print(f"词表大小: {len(vocab)}", flush=True)

    if args.smoke:
        # ---- 小样本过拟合测试：64 张图反复训练，要求 loss 显著下降、能背出标注 ----
        ds = FoodCapDataset(args.root, "train", vocab, max_len=args.max_len,
                            transform=train_transforms())
        ds = Subset(ds, list(range(args.smoke_size)))
        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0,
                            collate_fn=CollateFn(pad_id))
        model = CaptionModel(len(vocab), args.embed_dim, args.hidden_size,
                             pad_id, enc_finetune=True).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr_dec)
        crit = nn.CrossEntropyLoss(ignore_index=pad_id)
        model.train()
        losses, t0 = [], time.time()
        for step in range(args.smoke_steps):
            imgs, caps, lens, _ = next(iter(loader))
            imgs, caps = imgs.to(device), caps.to(device)
            logits = model(imgs, caps, lens)
            loss = crit(logits.reshape(-1, logits.size(-1)), caps[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
            if (step + 1) % 50 == 0:
                print(f"step {step+1}/{args.smoke_steps}, loss {sum(losses[-50:])/50:.4f}", flush=True)
        model.eval()
        imgs, _, _, raw = next(iter(DataLoader(ds, batch_size=4, num_workers=0,
                                               collate_fn=CollateFn(pad_id))))
        hyps = model.greedy_decode(imgs.to(device), vocab, max_len=args.max_len)
        print("\n===== 过拟合测试样例（应接近背出训练标注）=====", flush=True)
        for i in range(2):
            print(f"参考: {' '.join(raw[i].lower().split())}")
            print(f"生成: {' '.join(hyps[i])}\n", flush=True)
        tail = losses[-min(50, len(losses)):]
        ok = len(tail) > 0 and (sum(tail) / len(tail)) < losses[0] * 0.4
        print(f"初始 loss {losses[0]:.4f} -> 末期 {sum(tail)/len(tail):.4f}", flush=True)
        print("SMOKE TEST:", "PASS ✔" if ok else "FAIL ✘（管线可跑但拟合不足，检查数据/模型）", flush=True)
        return

    # ---- 正式训练 ----
    train_ds = FoodCapDataset(args.root, "train", vocab, max_len=args.max_len,
                              transform=train_transforms())
    val_ds = FoodCapDataset(args.root, "val", vocab, max_len=args.max_len,
                            transform=eval_transforms())
    print(f"train {len(train_ds)} 张, val {len(val_ds)} 张", flush=True)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          collate_fn=CollateFn(pad_id), persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        collate_fn=CollateFn(pad_id), persistent_workers=args.workers > 0)

    model = CaptionModel(len(vocab), args.embed_dim, args.hidden_size,
                         pad_id, enc_finetune=True).to(device)
    opt = torch.optim.Adam([
        {"params": model.decoder.parameters(), "lr": args.lr_dec},
        {"params": model.encoder.parameters(), "lr": args.lr_enc},
    ])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    crit = nn.CrossEntropyLoss(ignore_index=pad_id)

    best_bleu4, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for imgs, caps, lens, _ in train_ld:
            imgs, caps = imgs.to(device), caps.to(device)
            logits = model(imgs, caps, lens)
            loss = crit(logits.reshape(-1, logits.size(-1)), caps[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        bleus = evaluate(model, val_ld, vocab, device, max_decode=args.val_num)
        sched.step(bleus[3])
        entry = {"epoch": epoch, "train_loss": tot / nb, "val_bleu1": bleus[0],
                 "val_bleu4": bleus[3], "minutes": (time.time() - t0) / 60}
        history.append(entry)
        print(f"epoch {epoch:02d} | loss {tot/nb:.4f} | val BLEU-1 {bleus[0]:.2f} "
              f"BLEU-4 {bleus[3]:.2f} | {entry['minutes']:.1f} min", flush=True)
        if bleus[3] > best_bleu4:
            best_bleu4, bad_epochs = bleus[3], 0
            torch.save({"model": model.state_dict(), "vocab": vocab, "args": vars(args),
                        "epoch": epoch, "val_bleu4": bleus[3]}, CKPT)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"早停：val BLEU-4 {best_bleu4:.2f} 未提升超过 {args.patience} 轮", flush=True)
                break
    json.dump(history, open(os.path.join(RUN_DIR, "history.json"), "w"), indent=1)
    print(f"最佳模型: {CKPT} (val BLEU-4 {best_bleu4:.2f})", flush=True)


if __name__ == "__main__":
    main()
