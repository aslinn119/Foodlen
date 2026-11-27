# -*- coding: utf-8 -*-
"""在 test 集上评测 best_v1.pt：BLEU-1~4（greedy）+ 可选 beam 对比 + 样例输出。

用法：
  python eval_test.py                  # greedy 跑完整 test 集
  python eval_test.py --beam-num 200   # 额外在前 200 张上跑 beam search 对比
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from data import FoodCapDataset, eval_transforms, CollateFn
from model import CaptionModel
from bleu import corpus_bleu

# ---------- 命令行参数 ----------
p = argparse.ArgumentParser()
p.add_argument("--root", default=r"C:\Users\刘九尔\Desktop\神经网络与深度学习\Food500Cap_ready")  # 数据集根目录
p.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "runs", "v1", "best_v1.pt"))  # 模型权重路径
p.add_argument("--split", default="test")            # 评测哪个划分：train/val/test
p.add_argument("--limit", type=int, default=0, help="只评前 N 张，0=全部")
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--workers", type=int, default=2)     # DataLoader 进程数
p.add_argument("--beam-num", type=int, default=0, help="在前 N 张上额外做 beam 对比")  # 0=不做
p.add_argument("--beam-size", type=int, default=3)   # 束宽
p.add_argument("--show", type=int, default=5, help="打印多少条生成样例")
def main():
    args = p.parse_args()  # 在 main 里解析，避免导入本模块时误触发

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载 checkpoint；map_location 保证无显卡机器也能读 GPU 保存的模型
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]              # 训练时保存的词表
    ck_args = ckpt.get("args", {})     # 训练时的超参数（用于保持结构一致）
    max_len = ck_args.get("max_len", 30)
    print(f"checkpoint: {args.ckpt}")
    print(f"epoch {ckpt.get('epoch')}, val BLEU-4 {ckpt.get('val_bleu4')}, 词表 {len(vocab)}")

    pad_id = vocab["<PAD>"]
    # 构建评测数据集（用确定性预处理）
    ds = FoodCapDataset(args.root, args.split, vocab, max_len=max_len,
                        transform=eval_transforms())
    if args.limit > 0:
        ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))  # 只取前 N 张
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    collate_fn=CollateFn(pad_id))

    # 用 checkpoint 里的超参数重建模型结构，再载入权重
    model = CaptionModel(len(vocab), ck_args.get("embed_dim", 256),
                         ck_args.get("hidden_size", 512), pad_id).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()  # 评估模式

    # ---- greedy 解码整个数据集 ----
    hyps, refs = [], []
    with torch.no_grad():
        for imgs, _, _, raw in ld:
            hyps += model.greedy_decode(imgs.to(device), vocab, max_len=max_len)
            refs += [r.lower().split() for r in raw]  # 参考句转小写分词

    b = corpus_bleu(hyps, refs)  # 语料级 BLEU-1~4
    print(f"\n===== {args.split} 集 {len(hyps)} 张 (greedy) =====")
    for i, name in enumerate(["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"], 1):
        print(f"{name}: {b[i-1]:.2f}")

    # 把评测结果存成 JSON，放 checkpoint 同目录
    out_dir = os.path.dirname(args.ckpt)
    result = {"split": args.split, "n": len(hyps), "decode": "greedy",
              "bleu1": b[0], "bleu2": b[1], "bleu3": b[2], "bleu4": b[3],
              "ckpt": args.ckpt, "epoch": ckpt.get("epoch")}
    json.dump(result, open(os.path.join(out_dir, f"test_result_{args.split}_greedy.json"), "w"),
              indent=1)

    # ---- 打印若干条生成样例，并把全部样例写入 txt ----
    print("\n===== 生成样例 =====")
    for i in range(min(args.show, len(hyps))):
        print(f"[{i}] 参考: {' '.join(refs[i])}")
        print(f"    生成: {' '.join(hyps[i])}\n")
    with open(os.path.join(out_dir, f"samples_{args.split}_greedy.txt"), "w", encoding="utf-8") as f:
        for i in range(len(hyps)):
            f.write(f"[{i}] 参考: {' '.join(refs[i])}\n    生成: {' '.join(hyps[i])}\n\n")

    # ---- 可选：beam search 对比（逐张解码，较慢，所以只在前 N 张上做） ----
    if args.beam_num > 0:
        n = min(args.beam_num, len(ds))
        hyps_b, refs_b = [], []
        with torch.no_grad():
            for i in range(n):
                img, _, _, raw = ds[i]  # 逐张取（beam_decode 只支持 batch=1）
                hyps_b.append(model.beam_decode(img.unsqueeze(0).to(device), vocab,
                                                beam_size=args.beam_size, max_len=max_len))
                refs_b.append(raw.lower().split())
        bb = corpus_bleu(hyps_b, refs_b)       # beam 的 BLEU
        gb = corpus_bleu(hyps[:n], refs_b)     # 同样这 n 张上 greedy 的 BLEU（公平对比）
        print(f"===== 前 {n} 张 beam({args.beam_size}) 对比 =====")
        print("greedy:", [f"{x:.2f}" for x in gb])
        print("beam  :", [f"{x:.2f}" for x in bb])
        json.dump({"n": n, "beam_size": args.beam_size,
                   "greedy": gb, "beam": bb},
                  open(os.path.join(out_dir, "beam_compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
