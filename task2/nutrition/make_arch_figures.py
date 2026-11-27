"""make_arch_figures.py —— 绘制模型 A / 模型 B 结构示意图"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_figures_final")
os.makedirs(OUT, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#EEF3F6", ec="#37474F", fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                 fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, text=None, fs=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, color="#37474F", lw=1.4))
    if text:
        ax.text((x1 + x2) / 2 + 0.012, (y1 + y2) / 2, text, fontsize=fs, va="center")


# ---------- 模型 A ----------
fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.02, 0.62, 0.15, 0.28, "菜品图像\n224×224", fc="#D8E0E3", bold=True)
box(ax, 0.24, 0.62, 0.20, 0.28, "ResNet18\n(ImageNet 预训练)\n整体表示 512 维", bold=True)
box(ax, 0.49, 0.62, 0.16, 0.28, "线性映射\nh0 = W·f", fc="#FFF3E0")
box(ax, 0.71, 0.62, 0.26, 0.28, "GRU 解码器\n嵌入256 → 隐状态512\n自回归逐 token 生成", bold=True)
box(ax, 0.71, 0.12, 0.26, 0.28, "JSON 文本输出\n+ 稳健解析\n(严格→修复→正则兜底)", fc="#E8F5E9")
box(ax, 0.24, 0.12, 0.41, 0.28, "target_text 监督学习\ntoken 级交叉熵损失（teacher forcing）", fc="#FFF3E0")
arrow(ax, 0.17, 0.76, 0.24, 0.76)
arrow(ax, 0.44, 0.76, 0.49, 0.76)
arrow(ax, 0.65, 0.76, 0.71, 0.76)
arrow(ax, 0.84, 0.62, 0.84, 0.40)
arrow(ax, 0.65, 0.26, 0.71, 0.26)
ax.set_title("模型 A：CNN 整体表示 + GRU 解码（任务一图像描述生成结构）", fontsize=13)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "arch_modelA.png"), dpi=150); plt.close()

# ---------- 模型 B ----------
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.02, 0.66, 0.13, 0.24, "菜品图像\n224×224", fc="#D8E0E3", bold=True)
box(ax, 0.20, 0.60, 0.22, 0.36,
    "CLIP ViT-B/32 视觉编码器\n88.3M 参数（冻结）\n最后 4 层注意力 q/k/v/out\n注入 LoRA 低秩矩阵 rank=16\nLoRA 0.39M + 任务头 0.43M\n可训练共 0.82M（0.93%）",
    fc="#E3F2FD", bold=True, fs=10)
box(ax, 0.47, 0.66, 0.13, 0.24, "[CLS] 池化\n768 维", fc="#FFF3E0")
box(ax, 0.65, 0.78, 0.32, 0.16, "营养回归头：5 个数值字段\n(加权 MSE, 归一化空间)", fs=9.5)
box(ax, 0.65, 0.50, 0.32, 0.16, "食材多标签头：151 类 0/1\n(BCE, pos_weight)", fs=9.5)
box(ax, 0.65, 0.22, 0.32, 0.16, "分项重量头：151 类分量回归\n(仅对出现食材计 MSE)", fs=9.5)
box(ax, 0.24, 0.06, 0.32, 0.16, "sigmoid 阈值 0.625（val 上网格搜索）\n分项重量等比缩放到总重", fc="#E8F5E9", fs=9.5)
box(ax, 0.02, 0.06, 0.17, 0.16, "完整营养 JSON\n(items + 5 字段)", fc="#D8E0E3", bold=True, fs=10)
arrow(ax, 0.15, 0.78, 0.20, 0.78)
arrow(ax, 0.42, 0.78, 0.47, 0.78)
arrow(ax, 0.60, 0.78, 0.65, 0.86)
arrow(ax, 0.60, 0.78, 0.65, 0.58)
arrow(ax, 0.60, 0.74, 0.65, 0.30)
arrow(ax, 0.65, 0.14, 0.56, 0.14)
arrow(ax, 0.24, 0.14, 0.19, 0.14)
ax.set_title("模型 B：CLIP 多模态预训练模型 + LoRA 参数高效微调（多任务）", fontsize=13)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "arch_modelB.png"), dpi=150); plt.close()
print("arch figures saved")
