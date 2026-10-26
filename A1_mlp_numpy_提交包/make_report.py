# -*- coding: utf-8 -*-
"""读取 results/results.jsonl，生成对比图表到 results/figs/"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()

import matplotlib.pyplot as plt
import numpy as np

recs = [json.loads(l) for l in open("results/results.jsonl", encoding="utf-8")]
FIG = "results/figs"
os.makedirs(FIG, exist_ok=True)

by = {r["name"]: r for r in recs}


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("saved", name)


# 1) 学习率扫描
tune = [(float(n.split("=")[1]), r) for n, r in by.items() if n.startswith("tune_lr")]
tune.sort()
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([t[0] for t in tune], [t[1]["best_val_acc"] for t in tune], "o-")
ax.set_xscale("log")
ax.set_xlabel("学习率")
ax.set_ylabel("验证集准确率")
ax.set_title("验证集学习率扫描（SGD-momentum, 3 epochs）")
ax.grid(alpha=0.3)
save(fig, "fig1_lr_sweep")

# 2) 基准训练曲线
b = by["baseline_adam"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
ep = range(1, len(b["history"]["train_loss"]) + 1)
a1.plot(ep, b["history"]["train_loss"], label="train")
a1.plot(ep, b["history"]["val_loss"], label="val")
a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.legend(); a1.grid(alpha=0.3)
a1.set_title("训练/验证损失（CE + L2 1e-3, Adam）")
a2.plot(ep, b["history"]["val_acc"], "g-o")
a2.set_xlabel("epoch"); a2.set_ylabel("验证准确率"); a2.grid(alpha=0.3)
a2.set_title("验证准确率")
fig.tight_layout()
save(fig, "fig2_baseline_curves")

# 3) 损失函数对比
fig, ax = plt.subplots(figsize=(6, 4))
for n, lab in [("loss_ce", "交叉熵 CE"), ("loss_mse", "均方误差 MSE")]:
    r = by[n]
    ax.plot(range(1, len(r["history"]["val_acc"]) + 1), r["history"]["val_acc"], "-o", label=lab)
ax.set_xlabel("epoch"); ax.set_ylabel("验证准确率"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("不同损失函数对比（Adam, L2=1e-3）")
save(fig, "fig3_loss_compare")

# 4) L2 正则化对比
fig, ax = plt.subplots(figsize=(6, 4))
for lam in ["0", "0.0001", "0.001", "0.01", "0.1"]:
    r = by[f"reg_lam={lam}"]
    ax.plot(range(1, len(r["history"]["val_acc"]) + 1), r["history"]["val_acc"],
            "-o", label=f"λ={lam}")
ax.set_xlabel("epoch"); ax.set_ylabel("验证准确率"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("L2 正则化强度对比（Adam, CE）")
save(fig, "fig4_reg_compare")

# 5) 优化器对比：曲线 + 最终条形图
opt_names = ["opt_batch_gd", "opt_online_gd", "opt_minibatch_sgd", "opt_momentum",
             "opt_nesterov", "opt_adagrad", "opt_rmsprop", "opt_adam", "opt_adamax"]
labels = ["batch GD", "online GD", "mini-batch SGD", "Momentum", "Nesterov",
          "AdaGrad", "RMSProp", "Adam", "Adamax"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
for n, lab in zip(opt_names, labels):
    r = by[n]
    a1.plot(range(1, len(r["history"]["val_acc"]) + 1), r["history"]["val_acc"], label=lab)
a1.set_xlabel("epoch"); a1.set_ylabel("验证准确率"); a1.legend(fontsize=8)
a1.grid(alpha=0.3); a1.set_title("各优化器验证准确率曲线")
x = np.arange(len(opt_names))
a2.bar(x, [by[n]["test_acc"] for n in opt_names])
a2.set_xticks(x); a2.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
a2.set_ylabel("测试准确率"); a2.set_ylim(0.6, 1.0); a2.grid(alpha=0.3, axis="y")
a2.set_title("各优化器测试准确率（10 epochs）")
fig.tight_layout()
save(fig, "fig5_optimizer_compare")
