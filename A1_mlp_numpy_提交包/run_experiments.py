# -*- coding: utf-8 -*-
"""实验脚本：基准训练 + 验证集调参 + 附加题对比实验。
用法: python run_experiments.py all      # 跑全部实验
      python run_experiments.py baseline # 只跑某一个
结果保存到 results/results.jsonl，每个实验一行。
"""
import json
import os
import time

import numpy as np

from mlp import (TwoLayerNet, load_mnist, train,
                 SGD, AdaGrad, RMSProp, Adam, Adamax)

RESULTS = "results/results.jsonl"
os.makedirs("results", exist_ok=True)

EPOCHS = 10          # 常规实验轮数
HIDDEN = 128
SEED = 0


def run(name, opt, Xtr, ytr, Xva, yva, Xte, yte, epochs=EPOCHS,
        batch_size=128, loss_name="ce", lam=0.0, seed=SEED,
        subset=None, iter_log_every=100):
    if subset:
        Xtr, ytr = Xtr[:subset], ytr[:subset]
    net = TwoLayerNet(784, HIDDEN, 10, seed=seed)
    t0 = time.time()
    hist = train(net, opt, Xtr, ytr, Xva, yva, epochs=epochs,
                 batch_size=batch_size, loss_name=loss_name, lam=lam,
                 seed=seed, iter_log_every=iter_log_every, verbose=False)
    wall = time.time() - t0
    rec = {
        "name": name, "loss": loss_name, "lam": lam,
        "batch_size": batch_size, "epochs": epochs,
        "subset": subset or len(Xtr),
        "val_acc": net.accuracy(Xva, yva),
        "test_acc": net.accuracy(Xte, yte),
        "best_val_acc": max(hist["val_acc"]),
        "final_val_loss": hist["val_loss"][-1],
        "wall_time": round(wall, 1),
        "history": {k: (v if k in ("iter", "iter_loss") else
                        [round(x, 5) for x in v]) for k, v in hist.items()},
    }
    print(f"[{name:28s}] best_val {rec['best_val_acc']:.4f} | "
          f"test {rec['test_acc']:.4f} | {wall:.1f}s")
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def main(which="all"):
    Xtr, ytr, Xva, yva, Xte, yte = load_mnist()
    print(f"data: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}\n")

    if which in ("all", "tune_lr"):
        print("=== 验证集调参：学习率扫描 (SGD-momentum, 3 epochs) ===")
        for lr in [0.001, 0.01, 0.05, 0.1, 0.5]:
            run(f"tune_lr={lr}", SGD(lr=lr, momentum=0.9), Xtr, ytr, Xva, yva, Xte, yte,
                epochs=3, batch_size=128)

    if which in ("all", "baseline"):
        print("\n=== 基准模型 (Adam, CE, L2=1e-3, 10 epochs) ===")
        run("baseline_adam", Adam(lr=1e-3), Xtr, ytr, Xva, yva, Xte, yte, lam=1e-3)

    if which in ("all", "loss"):
        print("\n=== 附加题1a: 损失函数对比 (Adam, L2=1e-3) ===")
        for loss_name in ["ce", "mse"]:
            run(f"loss_{loss_name}", Adam(lr=1e-3), Xtr, ytr, Xva, yva, Xte, yte,
                lam=1e-3, loss_name=loss_name)

    if which in ("all", "reg"):
        print("\n=== 附加题1b: L2 正则化强度对比 (Adam, CE) ===")
        for lam in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
            run(f"reg_lam={lam:g}", Adam(lr=1e-3), Xtr, ytr, Xva, yva, Xte, yte,
                lam=lam)

    if which in ("all", "opt"):
        print("\n=== 附加题2: 优化算法对比 (CE, L2=1e-3) ===")
        # batch GD: 全量数据一个 batch
        run("opt_batch_gd", SGD(lr=0.5), Xtr, ytr, Xva, yva, Xte, yte, batch_size=50000)
        # online GD: batch=1，用子集保证时间可控
        run("opt_online_gd", SGD(lr=0.005), Xtr, ytr, Xva, yva, Xte, yte,
            batch_size=1, subset=10000)
        # mini-batch SGD
        run("opt_minibatch_sgd", SGD(lr=0.05), Xtr, ytr, Xva, yva, Xte, yte)
        # Momentum / Nesterov
        run("opt_momentum", SGD(lr=0.05, momentum=0.9), Xtr, ytr, Xva, yva, Xte, yte)
        run("opt_nesterov", SGD(lr=0.05, momentum=0.9, nesterov=True),
            Xtr, ytr, Xva, yva, Xte, yte)
        # 自适应学习率
        run("opt_adagrad", AdaGrad(lr=0.05), Xtr, ytr, Xva, yva, Xte, yte)
        run("opt_rmsprop", RMSProp(lr=1e-3), Xtr, ytr, Xva, yva, Xte, yte)
        run("opt_adam", Adam(lr=1e-3), Xtr, ytr, Xva, yva, Xte, yte)
        run("opt_adamax", Adamax(lr=2e-3), Xtr, ytr, Xva, yva, Xte, yte)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
