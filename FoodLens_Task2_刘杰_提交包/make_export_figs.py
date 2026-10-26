# -*- coding: utf-8 -*-
"""为组报告导出刘杰章节所需图表：图9 散点图 + 图6 demo 展示图，并输出表4数据。"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "export_figs")
os.makedirs(OUT, exist_ok=True)

FIELDS = ["total_weight_g", "total_calories_kcal", "protein_g", "fat_g", "carbohydrate_g"]
CN = {
    "total_weight_g": "总重量 (g)",
    "total_calories_kcal": "热量 (kcal)",
    "protein_g": "蛋白质 (g)",
    "fat_g": "脂肪 (g)",
    "carbohydrate_g": "碳水 (g)",
}

# ---------- 真值 ----------
test_records = [json.loads(l) for l in open(os.path.join(ROOT, "annotations", "test.jsonl"), encoding="utf-8")]
truth = {r["dish_id"]: r for r in test_records}
true_by_field = {f: np.array([r["target"][f] for r in test_records]) for f in FIELDS}
test_mean = {f: float(true_by_field[f].mean()) for f in FIELDS}

# ---------- 模型B (CLIP+LoRA) 预测 ----------
preds_b = {}
for line in open(os.path.join(ROOT, "final_results", "modelB_test_preds.jsonl"), encoding="utf-8"):
    d = json.loads(line)
    preds_b[d["dish_id"]] = d["parsed"]
pred_by_field_b = {f: np.array([preds_b[r["dish_id"]][f] for r in test_records]) for f in FIELDS}

# ---------- 模型 ResNet18 预测 ----------
r18 = np.load(os.path.join(ROOT, "checkpoints", "test_results.npz"))
pred_by_field_r18 = {f: r18["preds"][:, i] for i, f in enumerate(FIELDS)}

# ---------- 图9：热量 预测 vs 真值 ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, name, pred in [
    (axes[0], "ResNet18 回归", pred_by_field_r18["total_calories_kcal"]),
    (axes[1], "CLIP+LoRA 多任务（本文）", pred_by_field_b["total_calories_kcal"]),
]:
    y = true_by_field["total_calories_kcal"]
    lim = [0, max(y.max(), pred.max()) * 1.05]
    ax.scatter(y, pred, s=12, alpha=0.45, edgecolors="none")
    ax.plot(lim, lim, "r--", lw=1, label="理想预测")
    mae = float(np.mean(np.abs(y - pred)))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("真值 (kcal)")
    ax.set_ylabel("预测 (kcal)")
    ax.set_title(f"{name}  MAE={mae:.1f}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
fig.suptitle("图9  test 集热量预测 vs 真值", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "fig9_calories_scatter.png"), dpi=200)
plt.close(fig)
print("saved fig9")

# ---------- 图6：demo 展示（3 样例：图 + 预测 JSON） ----------
demo_ids = ["dish_1556575327", "dish_1557862345"]
# 再挑一个食材数较多、误差中等的样例（排除已选）
cals = {r["dish_id"]: abs(pred_by_field_b["total_calories_kcal"][i] - true_by_field["total_calories_kcal"][i])
        for i, r in enumerate(test_records)}
cand = [r for r in test_records
        if 3 <= len(json.loads(r["target_text"])["items"]) <= 5 and r["dish_id"] not in demo_ids]
cand.sort(key=lambda r: abs(cals[r["dish_id"]] - 30))  # 误差≈30 的多样食材样例
demo_ids.append(cand[0]["dish_id"])

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
for ax, did in zip(axes, demo_ids):
    img = Image.open(os.path.join(ROOT, "images", "test", did + ".png"))
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(did.replace("dish_", "样例 "), fontsize=10)
    p = preds_b[did]
    lines = [f"{it['name']} {it['weight_g']:.0f}g" for it in p["items"]]
    txt = (f"预测 食材: {'、'.join(lines)}\n"
           f"总重 {p['total_weight_g']:.0f}g | 热量 {p['total_calories_kcal']:.0f} kcal\n"
           f"蛋白质 {p['protein_g']:.1f}g | 脂肪 {p['fat_g']:.1f}g | 碳水 {p['carbohydrate_g']:.1f}g")
    t = json.loads(truth[did]["target_text"])
    t_lines = [f"{it['name']} {it['weight_g']:.0f}g" for it in t["items"]]
    ttxt = f"真值 食材: {'、'.join(t_lines)} | 总重 {t['total_weight_g']:.0f}g | 热量 {t['total_calories_kcal']:.0f} kcal"
    ax.text(0.5, -0.32, txt + "\n" + ttxt, transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
fig.suptitle("图6  predict_nutrition 端到端输出示例（每组上为预测、下为真值）", fontsize=12)
fig.tight_layout(rect=[0, 0.10, 1, 0.94])
fig.savefig(os.path.join(OUT, "fig6_demo.png"), dpi=200)
plt.close(fig)
print("saved fig6, demo_ids =", demo_ids)

# ---------- 表4 数据 ----------
r18_mae = {f: float(np.mean(np.abs(pred_by_field_r18[f] - true_by_field[f]))) for f in FIELDS}
r18_rmse = {f: float(np.sqrt(np.mean((pred_by_field_r18[f] - true_by_field[f]) ** 2))) for f in FIELDS}
b_mae = {f: float(np.mean(np.abs(pred_by_field_b[f] - true_by_field[f]))) for f in FIELDS}
b_rmse = {f: float(np.sqrt(np.mean((pred_by_field_b[f] - true_by_field[f]) ** 2))) for f in FIELDS}
base_mae = {"total_weight_g": 120.68, "total_calories_kcal": 167.59, "protein_g": 14.96,
            "fat_g": 10.57, "carbohydrate_g": 13.73}
base_rmse = {"total_weight_g": 152.11, "total_calories_kcal": 212.1, "protein_g": 18.97,
             "fat_g": 14.15, "carbohydrate_g": 17.48}

table4 = []
for f in FIELDS:
    m = test_mean[f]
    table4.append({
        "field": f, "cn": CN[f], "test_mean": round(m, 2),
        "base_mae": base_mae[f], "base_rmse": base_rmse[f], "base_rel": round(base_mae[f] / m * 100, 1),
        "r18_mae": round(r18_mae[f], 2), "r18_rmse": round(r18_rmse[f], 2), "r18_rel": round(r18_mae[f] / m * 100, 1),
        "b_mae": round(b_mae[f], 2), "b_rmse": round(b_rmse[f], 2), "b_rel": round(b_mae[f] / m * 100, 1),
    })
json.dump({"test_mean": test_mean, "rows": table4}, open(os.path.join(OUT, "table4_data.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
for row in table4:
    print(row)
