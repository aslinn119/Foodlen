"""Build publication-ready Task 1 architecture, training, and score figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


BASE = Path(__file__).resolve().parent
OUT = BASE / "report_figures" / "task1"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def architecture():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set(xlim=(0, 12), ylim=(0, 4.2)); ax.axis("off")
    boxes = [
        (0.2, 1.65, 1.55, 0.85, "Input image\n224 × 224 × 3"),
        (2.15, 1.65, 1.8, 0.85, "ResNet18\nconv feature map"),
        (4.35, 1.65, 1.65, 0.85, "49 spatial regions\n7 × 7 × 512"),
        (6.7, 1.65, 1.75, 0.85, "Bahdanau\nattention"),
        (9.05, 1.65, 1.45, 0.85, "GRU\nword decoder"),
        (10.95, 1.65, 0.85, 0.85, "Next\nword"),
    ]
    for x, y, w, h, label in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                               facecolor="#EAF2F8", edgecolor="#345B74", linewidth=1.4)
        ax.add_patch(patch); ax.text(x+w/2, y+h/2, label, ha="center", va="center", color="#152733")
    for x1, x2 in [(1.75,2.15),(3.95,4.35),(6.0,6.7),(8.45,9.05),(10.5,10.95)]:
        ax.add_patch(FancyArrowPatch((x1,2.075),(x2,2.075),arrowstyle="-|>",mutation_scale=14,color="#345B74"))
    ax.text(7.55, 3.28, "spatial features", ha="center", color="#345B74")
    ax.add_patch(FancyArrowPatch((5.15,2.55),(7.15,2.55),connectionstyle="arc3,rad=0.28",arrowstyle="-|>",mutation_scale=14,color="#568B65"))
    ax.text(8.0, 0.86, "previous hidden state", ha="center", color="#8A5A2B")
    ax.add_patch(FancyArrowPatch((8.0,1.0),(7.9,1.65),connectionstyle="arc3,rad=-0.1",arrowstyle="-|>",mutation_scale=14,color="#8A5A2B"))
    ax.text(9.75, 0.48, "word embedding + attended context", ha="center", color="#345B74")
    ax.add_patch(FancyArrowPatch((9.75,0.62),(9.75,1.65),arrowstyle="-|>",mutation_scale=14,color="#345B74"))
    ax.text(6, 3.95, "V2 CNN GRU with spatial attention", ha="center", va="center", fontsize=15, weight="bold", color="black")
    fig.tight_layout(); fig.savefig(OUT/"v2_architecture.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    entries = []
    for label, path, color in [
        ("V1 ablation", BASE/"v2_attention"/"runs"/"v1_ablation"/"history.json", "#8A9A5B"),
        ("V2 attention", BASE/"v2_attention"/"runs"/"v2"/"history.json", "#345B74"),
    ]:
        if path.exists():
            rows=json.loads(path.read_text(encoding="utf-8")); entries.append((label,rows,color))
            ep=[r["epoch"] for r in rows]
            axes[0].plot(ep,[r["train_loss"] for r in rows],marker="o",label=label,color=color)
            axes[1].plot(ep,[r["val_bleu4"] for r in rows],marker="o",label=label,color=color)
    axes[0].set(title="Training loss",xlabel="Epoch",ylabel="Cross-entropy")
    axes[1].set(title="Validation BLEU-4",xlabel="Epoch",ylabel="BLEU-4")
    for ax in axes: ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT/"training_curves.png",dpi=180,bbox_inches="tight"); plt.close(fig)


def scores():
    sources = [("V2 greedy", BASE/"results"/"v2_greedy_test.json", "#345B74"),
               ("V2 beam", BASE/"results"/"v2_beam_test.json", "#568B65"),
               ("V1 ablation", BASE/"results"/"v1_ablation_test.json", "#8A9A5B")]
    legacy=json.loads((BASE/"baseline_reference"/"group_v1_reported_metrics.json").read_text(encoding="utf-8"))
    available=[("V1 group baseline (reported)",legacy["bleu1_to_4"],"#9C6B4E")]
    for label,path,color in sources:
        if path.exists():
            data=json.loads(path.read_text(encoding="utf-8")); available.append((label,data["bleu1_to_4"],color))
    if not available:return
    fig,ax=plt.subplots(figsize=(8.5,4))
    xs=list(range(1,5)); width=.75/len(available)
    for i,(label,values,color) in enumerate(available):
        pos=[x-.375+width/2+i*width for x in xs]
        ax.bar(pos,values,width,label=label,color=color)
    ax.set(xticks=xs,xlabel="Metric",ylabel="BLEU score",title="Task 1 test-set BLEU comparison")
    ax.set_xticklabels([f"BLEU-{x}" for x in xs]);ax.grid(axis="y",alpha=.25);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(OUT/"test_bleu_comparison.png",dpi=180,bbox_inches="tight");plt.close(fig)


if __name__ == "__main__":
    architecture();curves();scores()
    print(f"Figures written to {OUT}")
