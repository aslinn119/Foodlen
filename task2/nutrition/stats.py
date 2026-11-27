import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DATA_ROOT, "stats_result")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 读取 train.jsonl ----------
records = []
with open(os.path.join(DATA_ROOT, "annotations", "train.jsonl"), "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"train 样本数: {len(records)}")

# 提取 5 个营养值
targets = {
    "total_weight_g": [],
    "total_calories_kcal": [],
    "protein_g": [],
    "fat_g": [],
    "carbohydrate_g": [],
}
for r in records:
    for k in targets:
        targets[k].append(r["target"][k])

df = pd.DataFrame(targets)
print("\n营养值统计:")
print(df.describe())

# ---------- 图1: 5个营养值分布直方图 ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, k in enumerate(targets):
    axes[i].hist(df[k], bins=50, color="steelblue", edgecolor="black")
    axes[i].set_title(f"{k}\nmean={df[k].mean():.1f}, std={df[k].std():.1f}")
    axes[i].set_xlabel(k)
    axes[i].set_ylabel("Count")
axes[-1].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "nutrition_distribution.png"), dpi=150)
print(f"\n-> 已保存 nutrition_distribution.png")

# ---------- 图2: 食材频率 Top 20 ----------
from collections import Counter
item_counter = Counter()
for r in records:
    for it in r["target"]["items"]:
        item_counter[it["name"]] += 1

top20 = item_counter.most_common(20)
names = [x[0] for x in top20][::-1]
counts = [x[1] for x in top20][::-1]

plt.figure(figsize=(10, 8))
plt.barh(names, counts, color="coral", edgecolor="black")
plt.title("Top 20 Ingredients (train)")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ingredient_freq_top20.png"), dpi=150)
print("-> 已保存 ingredient_freq_top20.png")

# ---------- 图3: 每图食材数量分布 ----------
item_nums = [len(r["target"]["items"]) for r in records]
plt.figure(figsize=(8, 5))
plt.hist(item_nums, bins=range(1, max(item_nums)+2), color="mediumseagreen", edgecolor="black", align="left")
plt.title(f"Ingredients per Dish (train)\nmean={np.mean(item_nums):.2f}, max={max(item_nums)}")
plt.xlabel("Number of ingredients")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "items_per_dish.png"), dpi=150)
print("-> 已保存 items_per_dish.png")

# ---------- 图4: 缺失值/异常值检查 ----------
print("\n===== 缺失值/异常值检查 =====")
print("NaN 数量:")
print(df.isna().sum())

warning_count = sum(1 for r in records if r["metadata"].get("mass_warning", False))
print(f"\nmass_warning=true 的样本数: {warning_count}")

coverage = [r["metadata"].get("visible_item_mass_coverage", 1.0) for r in records]
plt.figure(figsize=(8, 5))
plt.hist(coverage, bins=50, color="orchid", edgecolor="black")
plt.title("visible_item_mass_coverage (train)")
plt.xlabel("coverage")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mass_coverage.png"), dpi=150)
print("-> 已保存 mass_coverage.png")

# ---------- 保存统计表 ----------
df.to_csv(os.path.join(OUT_DIR, "nutrition_stats.csv"), index=False)
print(f"\n所有结果已保存到: {OUT_DIR}")