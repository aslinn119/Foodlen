"""
make_figures.py —— 生成报告所需的全部图表（report_figures_final/）

图 1  数据分布与食材统计沿用 stats_result/（阶段 1 产物）
图 2  模型 A 训练曲线（train/val loss）
图 3  模型 B 训练曲线（loss / val MAE / val micro-F1）
图 4  模型 A vs 模型 B 文本类指标对比（BLEU / ROUGE-L / CIDEr-D）
图 5  模型 A vs 模型 B F1 对比（micro / macro / sample）
图 6  test 集预测 vs 真值散点图（5 个字段 × 2 模型）
图 7  test 集各字段 MAE/RMSE 对比柱状图
图 8  模型 A 生成示例（生成文本 vs 标注文本）
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DATA_ROOT, "report_figures_final")
os.makedirs(OUT, exist_ok=True)

NUM_KEYS = ["total_weight_g", "total_calories_kcal", "protein_g", "fat_g", "carbohydrate_g"]
CN_KEYS = {"total_weight_g": "总重量 (g)", "total_calories_kcal": "总热量 (kcal)",
           "protein_g": "蛋白质 (g)", "fat_g": "脂肪 (g)", "carbohydrate_g": "碳水 (g)"}

with open(os.path.join(DATA_ROOT, "final_results", "test_metrics.json"), encoding="utf-8") as f:
    TEST = json.load(f)
with open(os.path.join(DATA_ROOT, "final_results", "val_metrics.json"), encoding="utf-8") as f:
    VAL = json.load(f)
with open(os.path.join(DATA_ROOT, "annotations", "test.jsonl"), encoding="utf-8") as f:
    TEST_RECS = [json.loads(l) for l in f]


def load_history(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------- 图 2/3：训练曲线 ----------
hA = load_history(os.path.join(DATA_ROOT, "checkpoints_caption", "history.json"))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(hA["train_loss"], label="train loss", color="steelblue")
axes[0].plot(hA["val_loss"], label="val loss", color="darkorange")
axes[0].set_title("模型 A（CNN+GRU）训练曲线")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("token CE loss"); axes[0].legend()
hB = load_history(os.path.join(DATA_ROOT, "checkpoints_lora_mt", "history.json"))
ax2 = axes[1].twinx()
axes[1].plot(hB["train_loss"], label="train loss", color="steelblue")
ax2.plot(hB["val_mae"], label="val MAE", color="darkorange")
ax2.plot(np.array(hB["val_f1"]) * 30, label="val micro-F1 (×30)", color="seagreen")
axes[1].set_title("模型 B（CLIP+LoRA 多任务）训练曲线")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss"); ax2.set_ylabel("val MAE")
l1, lab1 = axes[1].get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
axes[1].legend(l1 + l2, lab1 + lab2, loc="upper right")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "training_curves.png"), dpi=150); plt.close()
print("-> training_curves.png")

# ---------- 图 4：文本类指标对比 ----------
labels = ["BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr-D"]
va = [VAL["modelA_cnn_gru"][k] for k in ["bleu_1", "bleu_4", "rouge_l", "cider_d"]]
vb = [VAL["modelB_clip_lora"][k] for k in ["bleu_1", "bleu_4", "rouge_l", "cider_d"]]
ta = [TEST["modelA_cnn_gru"][k] for k in ["bleu_1", "bleu_4", "rouge_l", "cider_d"]]
tb = [TEST["modelB_clip_lora"][k] for k in ["bleu_1", "bleu_4", "rouge_l", "cider_d"]]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
x = np.arange(len(labels)); w = 0.35
for ax, A, B, name in [(axes[0], va, vb, "val"), (axes[1], ta, tb, "test")]:
    ax.bar(x - w / 2, A, w, label="模型A CNN+GRU", color="steelblue")
    ax.bar(x + w / 2, B, w, label="模型B CLIP+LoRA", color="darkorange")
    for i, (a, b) in enumerate(zip(A, B)):
        ax.text(i - w / 2, a + 0.15, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.15, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_title(f"{name} 集")
    ax.set_ylim(0, max(max(A), max(B)) * 1.25); ax.legend()
plt.suptitle("文本类指标对比（JSON 序列化文本）", y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "text_metrics_compare.png"), dpi=150); plt.close()
print("-> text_metrics_compare.png")

# ---------- 图 5：F1 对比 ----------
f1_labels = ["micro-F1", "macro-F1", "sample-F1"]
va = [VAL["modelA_cnn_gru"][k] for k in ["micro_f1", "macro_f1", "sample_f1"]]
vb = [VAL["modelB_clip_lora"][k] for k in ["micro_f1", "macro_f1", "sample_f1"]]
ta = [TEST["modelA_cnn_gru"][k] for k in ["micro_f1", "macro_f1", "sample_f1"]]
tb = [TEST["modelB_clip_lora"][k] for k in ["micro_f1", "macro_f1", "sample_f1"]]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
x = np.arange(3); w = 0.35
for ax, A, B, name in [(axes[0], va, vb, "val"), (axes[1], ta, tb, "test")]:
    ax.bar(x - w / 2, A, w, label="模型A CNN+GRU", color="steelblue")
    ax.bar(x + w / 2, B, w, label="模型B CLIP+LoRA", color="darkorange")
    for i, (a, b) in enumerate(zip(A, B)):
        ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(f1_labels); ax.set_title(f"{name} 集")
    ax.set_ylim(0, 0.65); ax.legend()
plt.suptitle("食材识别 F1 对比（items 名称集合匹配）", y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "f1_compare.png"), dpi=150); plt.close()
print("-> f1_compare.png")

# ---------- 图 6：预测 vs 真值散点 ----------
preds = {}
for tag, fn in [("A", "modelA_test_preds.jsonl"), ("B", "modelB_test_preds.jsonl")]:
    with open(os.path.join(DATA_ROOT, "final_results", fn), encoding="utf-8") as f:
        preds[tag] = [json.loads(l)["parsed"] for l in f]
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for j, k in enumerate(NUM_KEYS):
    truth = np.array([r["target"][k] for r in TEST_RECS])
    for i, tag in enumerate(["A", "B"]):
        p = np.array([d[k] for d in preds[tag]])
        mae = np.abs(p - truth).mean()
        ax = axes[i][j]
        ax.scatter(truth, p, s=6, alpha=0.4, color="steelblue" if tag == "A" else "darkorange")
        lim = max(truth.max(), p.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_title(f"模型{tag}  {CN_KEYS[k]}  MAE={mae:.1f}", fontsize=10)
        if i == 1:
            ax.set_xlabel("真值")
        if j == 0:
            ax.set_ylabel("预测")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "pred_vs_truth.png"), dpi=140); plt.close()
print("-> pred_vs_truth.png")

# ---------- 图 7：MAE/RMSE 柱状对比 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
x = np.arange(5); w = 0.35
maeA = [TEST["modelA_cnn_gru"]["numeric"][k]["mae"] for k in NUM_KEYS]
maeB = [TEST["modelB_clip_lora"]["numeric"][k]["mae"] for k in NUM_KEYS]
rmseA = [TEST["modelA_cnn_gru"]["numeric"][k]["rmse"] for k in NUM_KEYS]
rmseB = [TEST["modelB_clip_lora"]["numeric"][k]["rmse"] for k in NUM_KEYS]
axes[0].bar(x - w / 2, maeA, w, label="模型A CNN+GRU", color="steelblue")
axes[0].bar(x + w / 2, maeB, w, label="模型B CLIP+LoRA", color="darkorange")
axes[0].set_xticks(x)
axes[0].set_xticklabels([CN_KEYS[k] for k in NUM_KEYS], fontsize=8, rotation=12)
axes[0].set_title("test 集各字段 MAE"); axes[0].legend()
axes[1].bar(x - w / 2, rmseA, w, label="模型A CNN+GRU", color="steelblue")
axes[1].bar(x + w / 2, rmseB, w, label="模型B CLIP+LoRA", color="darkorange")
axes[1].set_xticks(x)
axes[1].set_xticklabels([CN_KEYS[k] for k in NUM_KEYS], fontsize=8, rotation=12)
axes[1].set_title("test 集各字段 RMSE"); axes[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "mae_rmse_compare.png"), dpi=150); plt.close()
print("-> mae_rmse_compare.png")

# ---------- 图 8：模型 A 生成示例 ----------
with open(os.path.join(DATA_ROOT, "final_results", "modelA_test_preds.jsonl"), encoding="utf-8") as f:
    rows = [json.loads(l) for l in f]
rows.sort(key=lambda r: len(set(r["parsed"]["items"][0]["name"] if r["parsed"]["items"] else "")
                            & set()), reverse=True)  # 占位，后面按解析码排序
ok = [r for r in rows if r["parse_code"] == 1][:2]
bad = [r for r in rows if r["parse_code"] != 1][:2]
rec_map = {r["dish_id"]: r for r in TEST_RECS}
fig, axes = plt.subplots(2, 2, figsize=(15, 6))
for ax, r, name in [(axes[0][0], ok[0], "示例1（严格解析成功）"),
                    (axes[0][1], ok[1], "示例2（严格解析成功）"),
                    (axes[1][0], bad[0], "示例3（修复/兜底解析）"),
                    (axes[1][1], bad[1], "示例4（修复/兜底解析）")]:
    ref = rec_map[r["dish_id"]]["target_text"]
    text = r["text"] if len(r["text"]) <= 300 else r["text"][:300] + "…"
    rj = json.dumps(r["parsed"], ensure_ascii=False)
    rj = rj if len(rj) <= 300 else rj[:300] + "…"
    ax.axis("off")
    ax.set_title(name, fontsize=11)
    ax.text(0, 1, "[GEN]\n" + text + "\n\n[PARSED]\n" + rj + "\n\n[REF]\n" + ref[:300],
            fontsize=6.5, va="top", family="monospace")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "caption_examples.png"), dpi=140); plt.close()
print("-> caption_examples.png")

print("\n全部图表已保存到:", OUT)
