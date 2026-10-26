# -*- coding: utf-8 -*-
"""画 ResNet18 多输出回归结构示意图，并与 CLIP+LoRA 结构图拼成组报告「图6」。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "export_figs")
os.makedirs(OUT, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#EEF3F6", ec="#37474F", fs=11, bold=False):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, text=None, fs=9):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#37474F", lw=1.4))
    if text:
        ax.text((x1 + x2) / 2 + 0.6, (y1 + y2) / 2, text, fontsize=fs, va="center")


# ---------- 左：ResNet18 回归 ----------
fig, ax = plt.subplots(figsize=(4.6, 6.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 30); ax.axis("off")
box(ax, 2, 26.5, 6, 2.6, "输入图像 224×224\n（ImageNet 标准化）", fs=10)
arrow(ax, 5, 26.5, 5, 24.2)
box(ax, 1.6, 18.6, 6.8, 5.6, "ResNet18 骨干\n（ImageNet-1K 预训练）\n\nconv1 → layer1 → layer2\n→ layer3 → layer4", fs=10)
arrow(ax, 5, 18.6, 5, 16.2)
box(ax, 2, 13.4, 6, 2.8, "全局平均池化\n512 维特征", fs=10)
arrow(ax, 5, 13.4, 5, 11.2)
box(ax, 2, 8.6, 6, 2.6, "fc：512 → 5\n（替换原分类头）", fc="#FFF3E0", fs=10)
arrow(ax, 5, 8.6, 5, 6.4, "反标准化\n(×train 标准差 + 均值)", fs=9)
box(ax, 1.2, 3.2, 7.6, 3.0, "总重量 / 热量 / 蛋白质\n脂肪 / 碳水化合物", fc="#E8F5E9", fs=10)
ax.set_title("ResNet18 多输出回归", fontsize=12, pad=8)
fig.tight_layout()
p_left = os.path.join(OUT, "_arch_resnet18.png")
fig.savefig(p_left, dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------- 与 CLIP+LoRA 图拼接 ----------
p_right = os.path.join(ROOT, "report_figures_final", "arch_modelB.png")
im_l = Image.open(p_left).convert("RGB")
im_r = Image.open(p_right).convert("RGB")
H = 1200
im_l = im_l.resize((int(im_l.width * H / im_l.height), H))
im_r = im_r.resize((int(im_r.width * H / im_r.height), H))
W = im_l.width + im_r.width + 40
canvas = Image.new("RGB", (W, H), "white")
canvas.paste(im_l, (0, 0))
canvas.paste(im_r, (im_l.width + 40, 0))
p_out = os.path.join(OUT, "fig6_arch.png")
canvas.save(p_out)
print("saved", p_out, canvas.size)
os.remove(p_left)
