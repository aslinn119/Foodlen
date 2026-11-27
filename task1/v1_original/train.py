# -*- coding: utf-8 -*-
"""V1 训练脚本：ResNet18 + GRU 图像描述基线。

用法：
    python train.py                # 正式训练（默认配置）
    python train.py --smoke        # 小样本过拟合测试：64 张图、300 步，验证管线正确性
    python train.py --epochs 30 --batch-size 64

训练日志与最佳模型保存在 ./runs/v1/ 下。
"""
import argparse  # 命令行参数解析
import json      # 保存训练历史到 history.json
import math      # （保留导入）
import os        # 路径操作
import time      # 计时

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset  # Subset 用于 smoke 模式取子集

from data import FoodCapDataset, load_vocab, train_transforms, eval_transforms, CollateFn
from model import CaptionModel
from bleu import corpus_bleu

# ---------- 默认配置 ----------
ROOT = r"C:\Users\刘九尔\Desktop\神经网络与深度学习\Food500Cap_ready"  # 数据集根目录
# 运行输出目录（本脚本所在目录/runs/v1）
RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "v1")
CKPT = os.path.join(RUN_DIR, "best_v1.pt")  # 最佳模型保存路径

# ---------- 命令行参数 ----------
p = argparse.ArgumentParser()
p.add_argument("--root", default=ROOT)                      # 数据集路径
p.add_argument("--epochs", type=int, default=20)            # 最大训练轮数
p.add_argument("--batch-size", type=int, default=64)        # 批大小
p.add_argument("--embed-dim", type=int, default=256)        # 词嵌入维度
p.add_argument("--hidden-size", type=int, default=512)      # GRU 隐状态维度
p.add_argument("--lr-dec", type=float, default=1e-3)        # 解码器学习率
p.add_argument("--lr-enc", type=float, default=1e-4)        # 编码器学习率（预训练主干用小学习率）
p.add_argument("--max-len", type=int, default=30)           # caption 最大长度
p.add_argument("--workers", type=int, default=2)            # DataLoader 加载进程数
p.add_argument("--val-num", type=int, default=500, help="每次验证用多少张 val 图")  # 验证抽样数量
p.add_argument("--patience", type=int, default=4)           # 早停耐心：BLEU-4 连续几轮不提升就停
p.add_argument("--smoke", action="store_true", help="小样本过拟合测试模式")  # 加 --smoke 即开启
p.add_argument("--smoke-steps", type=int, default=300)      # smoke 模式训练步数
p.add_argument("--smoke-size", type=int, default=64)        # smoke 模式用多少张图
args = p.parse_args()  # 解析参数到 args


def evaluate(model, loader, vocab, device, max_decode=300):
    """贪心解码 + 语料级 BLEU（仅监控用）。"""
    model.eval()           # 切到评估模式（关闭 dropout、BN 用滑动统计）
    hyps, refs = [], []    # hyps 存模型生成的词序列，refs 存参考句词序列
    n = 0
    inv = {i: w for w, i in vocab.items()}  # id->词 反向表（本函数里未实际用到，可忽略）
    with torch.no_grad():  # 不算梯度
        for imgs, _, _, raw in loader:  # raw 是原始 caption 文本
            imgs = imgs.to(device)
            for words in model.greedy_decode(imgs, vocab, max_len=args.max_len):
                hyps.append(words)  # 收集生成结果（已是词列表）
            refs += [r.lower().split() for r in raw]  # 参考句转小写分词
            n += imgs.size(0)
            if n >= max_decode:
                break  # 评够 max_decode 张就停（省时间）
    b = corpus_bleu(hyps[:max_decode], refs[:max_decode])  # 计算 BLEU-1~4
    model.train()          # 切回训练模式
    return b  # [bleu1, bleu2, bleu3, bleu4]


def main():
    os.makedirs(RUN_DIR, exist_ok=True)  # 创建输出目录（已存在不报错）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有显卡用显卡
    print(f"设备: {device}", flush=True)

    vocab = load_vocab(os.path.join(args.root, "vocab.json"))  # 加载词表
    pad_id = vocab["<PAD>"]  # PAD 的 id（损失函数里要忽略它）
    print(f"词表大小: {len(vocab)}", flush=True)

    if args.smoke:
        # ---- 小样本过拟合测试：64 张图反复训练，要求 loss 显著下降、能背出标注 ----
        # 目的：快速验证 数据加载->模型->损失->反向传播 整条管线没有 bug
        ds = FoodCapDataset(args.root, "train", vocab, max_len=args.max_len,
                            transform=train_transforms())
        ds = Subset(ds, list(range(args.smoke_size)))  # 只取前 64 张
        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0,
                            collate_fn=CollateFn(pad_id))
        model = CaptionModel(len(vocab), args.embed_dim, args.hidden_size,
                             pad_id, enc_finetune=True).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr_dec)  # 所有参数一个学习率
        crit = nn.CrossEntropyLoss(ignore_index=pad_id)  # 交叉熵，忽略 PAD 位置
        model.train()
        losses, t0 = [], time.time()
        for step in range(args.smoke_steps):
            imgs, caps, lens, _ = next(iter(loader))  # 每步重新随机取一个 batch
            imgs, caps = imgs.to(device), caps.to(device)
            logits = model(imgs, caps, lens)  # [B, L-1, V]
            # 目标是把 caps[:, 1:]（去掉 BOS）预测出来；reshape 成 (N, V) 与 (N,) 算交叉熵
            loss = crit(logits.reshape(-1, logits.size(-1)), caps[:, 1:].reshape(-1))
            opt.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪防爆
            opt.step()       # 更新参数
            losses.append(loss.item())
            if (step + 1) % 50 == 0:
                print(f"step {step+1}/{args.smoke_steps}, loss {sum(losses[-50:])/50:.4f}", flush=True)
        # 训练完抽 4 张图看生成效果（应该接近背出训练标注）
        model.eval()
        imgs, _, _, raw = next(iter(DataLoader(ds, batch_size=4, num_workers=0,
                                               collate_fn=CollateFn(pad_id))))
        hyps = model.greedy_decode(imgs.to(device), vocab, max_len=args.max_len)
        print("\n===== 过拟合测试样例（应接近背出训练标注）=====", flush=True)
        for i in range(2):
            print(f"参考: {' '.join(raw[i].lower().split())}")
            print(f"生成: {' '.join(hyps[i])}\n", flush=True)
        # 判定：末期平均 loss 是否降到初始的 40% 以下
        tail = losses[-min(50, len(losses)):]
        ok = len(tail) > 0 and (sum(tail) / len(tail)) < losses[0] * 0.4
        print(f"初始 loss {losses[0]:.4f} -> 末期 {sum(tail)/len(tail):.4f}", flush=True)
        print("SMOKE TEST:", "PASS ✔" if ok else "FAIL ✘（管线可跑但拟合不足，检查数据/模型）", flush=True)
        return  # smoke 模式到此结束，不进正式训练

    # ---- 正式训练 ----
    train_ds = FoodCapDataset(args.root, "train", vocab, max_len=args.max_len,
                              transform=train_transforms())  # 训练集（带增强）
    val_ds = FoodCapDataset(args.root, "val", vocab, max_len=args.max_len,
                            transform=eval_transforms())     # 验证集（不带增强）
    print(f"train {len(train_ds)} 张, val {len(val_ds)} 张", flush=True)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          collate_fn=CollateFn(pad_id), persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        collate_fn=CollateFn(pad_id), persistent_workers=args.workers > 0)
    # pin_memory：加速 CPU->GPU 传输；persistent_workers：worker 进程常驻省重启开销

    model = CaptionModel(len(vocab), args.embed_dim, args.hidden_size,
                         pad_id, enc_finetune=True).to(device)
    # 参数分组：解码器用大学习率，预训练编码器用小学习率（防止破坏已学特征）
    opt = torch.optim.Adam([
        {"params": model.decoder.parameters(), "lr": args.lr_dec},
        {"params": model.encoder.parameters(), "lr": args.lr_enc},
    ])
    # 指标停滞时自动把学习率减半：mode="max" 监控越大越好的指标（BLEU-4），2 轮不动就降
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    crit = nn.CrossEntropyLoss(ignore_index=pad_id)  # 损失函数，忽略 PAD

    best_bleu4, bad_epochs, history = -1.0, 0, []  # 最佳分数、连续未提升轮数、历史记录
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0  # 计时、loss 总和、batch 数
        for imgs, caps, lens, _ in train_ld:
            imgs, caps = imgs.to(device), caps.to(device)
            logits = model(imgs, caps, lens)  # 前向
            # 目标序列是 caps[:, 1:]（每个位置预测下一个词），与 logits 错位对齐
            loss = crit(logits.reshape(-1, logits.size(-1)), caps[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪
            opt.step()
            tot += loss.item()
            nb += 1
        bleus = evaluate(model, val_ld, vocab, device, max_decode=args.val_num)  # 验证
        sched.step(bleus[3])  # 用 BLEU-4 驱动学习率调度
        entry = {"epoch": epoch, "train_loss": tot / nb, "val_bleu1": bleus[0],
                 "val_bleu4": bleus[3], "minutes": (time.time() - t0) / 60}
        history.append(entry)  # 记录本轮数据
        print(f"epoch {epoch:02d} | loss {tot/nb:.4f} | val BLEU-1 {bleus[0]:.2f} "
              f"BLEU-4 {bleus[3]:.2f} | {entry['minutes']:.1f} min", flush=True)
        if bleus[3] > best_bleu4:
            best_bleu4, bad_epochs = bleus[3], 0  # 刷新最佳分数，清零计数
            # 保存最佳模型（含权重、词表、参数、轮数、分数）
            torch.save({"model": model.state_dict(), "vocab": vocab, "args": vars(args),
                        "epoch": epoch, "val_bleu4": bleus[3]}, CKPT)
        else:
            bad_epochs += 1  # 未提升，计数加一
            if bad_epochs >= args.patience:
                print(f"早停：val BLEU-4 {best_bleu4:.2f} 未提升超过 {args.patience} 轮", flush=True)
                break  # 触发早停
    # 训练结束：把每轮历史写入 history.json
    json.dump(history, open(os.path.join(RUN_DIR, "history.json"), "w"), indent=1)
    print(f"最佳模型: {CKPT} (val BLEU-4 {best_bleu4:.2f})", flush=True)


if __name__ == "__main__":
    main()  # 直接运行本脚本时执行 main()
