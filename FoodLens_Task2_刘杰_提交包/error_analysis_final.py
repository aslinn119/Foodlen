"""
error_analysis_final.py —— 模型 B 在 test 集上误差最大的 10 个样本分析

输出：
  error_analysis_final/top10_worst.csv     误差明细表
  error_analysis_final/top10_contact.png   10 张菜品的拼图（供人工看图）
"""

import os
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DATA_ROOT, "error_analysis_final")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(DATA_ROOT, "annotations", "test.jsonl"), encoding="utf-8") as f:
    recs = [json.loads(l) for l in f]
with open(os.path.join(DATA_ROOT, "final_results", "modelB_test_preds.jsonl"), encoding="utf-8") as f:
    preds = [json.loads(l)["parsed"] for l in f]

rows = []
for r, p in zip(recs, preds):
    err_cal = abs(p["total_calories_kcal"] - r["target"]["total_calories_kcal"])
    rel = err_cal / max(r["target"]["total_calories_kcal"], 1)
    rows.append({
        "dish_id": r["dish_id"],
        "n_items": len(r["target"]["items"]),
        "true_cal": r["target"]["total_calories_kcal"],
        "pred_cal": round(p["total_calories_kcal"], 1),
        "abs_err": round(err_cal, 1),
        "rel_err": round(rel, 3),
        "mass_warning": r["metadata"].get("mass_warning", False),
        "coverage": round(r["metadata"].get("visible_item_mass_coverage", 1.0), 3),
    })
rows.sort(key=lambda x: -x["abs_err"])
top10 = rows[:10]

with open(os.path.join(OUT, "top10_worst.csv"), "w", encoding="utf-8-sig") as f:
    f.write("rank,dish_id,n_items,true_cal,pred_cal,abs_err,rel_err,mass_warning,coverage\n")
    for i, r in enumerate(top10, 1):
        f.write(f'{i},{r["dish_id"]},{r["n_items"]},{r["true_cal"]},{r["pred_cal"]},'
                f'{r["abs_err"]},{r["rel_err"]},{r["mass_warning"]},{r["coverage"]}\n')
print("已保存 top10_worst.csv")

# 整体统计：误差与食材数量/coverage 的关系
alln = np.array([r["n_items"] for r in rows])
alle = np.array([r["abs_err"] for r in rows])
allc = np.array([r["coverage"] for r in rows])
print(f"\ntest 集: n={len(rows)}, 平均绝对误差={alle.mean():.1f} kcal")
for lo, hi in [(1, 2), (3, 5), (6, 8), (9, 16)]:
    m = (alln >= lo) & (alln <= hi)
    print(f"  食材数 {lo}-{hi}: {m.sum():3d} 样本, 平均误差 {alle[m].mean():7.1f} kcal")
low = allc < 0.9
print(f"  coverage<0.9: {low.sum()} 样本, 平均误差 {alle[low].mean():7.1f} kcal（其余 {alle[~low].mean():.1f}）")
big = alle > 200
print(f"  误差>200kcal: {big.sum()} 样本, 其中食材数>=6 的占 {(alln[big]>=6).sum()}/{big.sum()}")

# 拼图
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for i, r in enumerate(top10):
    ax = axes[i // 5][i % 5]
    img = Image.open(os.path.join(DATA_ROOT, f'images/test/{r["dish_id"]}.png')).convert("RGB")
    ax.imshow(img)
    ax.set_title(f'#{i+1} {r["dish_id"][-6:]}\n'
                 f'{r["n_items"]}种食材 真{r["true_cal"]:.0f}/预{r["pred_cal"]:.0f} kcal\n'
                 f'cov={r["coverage"]:.2f}', fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "top10_contact.png"), dpi=120)
print("已保存 top10_contact.png")
