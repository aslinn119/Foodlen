import os
import json
import numpy as np
import pandas as pd

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(DATA_ROOT, "checkpoints")

TARGET_KEYS = [
    "total_weight_g",
    "total_calories_kcal",
    "protein_g",
    "fat_g",
    "carbohydrate_g",
]


def load_targets(split):
    path = os.path.join(DATA_ROOT, "annotations", f"{split}.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return np.array([[r["target"][k] for k in TARGET_KEYS] for r in records])


# ---------- 1. train 集均值（基线预测值） ----------
train_targets = load_targets("train")
train_mean = train_targets.mean(axis=0)
train_std = train_targets.std(axis=0)

print("===== train 集统计 =====")
for k, m, s in zip(TARGET_KEYS, train_mean, train_std):
    print(f"  {k:25s} mean={m:8.2f}  std={s:8.2f}")


# ---------- 2. test 集均值基线 ----------
test_targets = load_targets("test")
baseline_preds = np.tile(train_mean, (len(test_targets), 1))

baseline_mae = np.abs(baseline_preds - test_targets).mean(axis=0)
baseline_rmse = np.sqrt(((baseline_preds - test_targets) ** 2).mean(axis=0))
baseline_rel = baseline_mae / train_mean  # 相对误差

print("\n===== 均值基线（test 集）=====")
for k, mae, rmse, rel in zip(TARGET_KEYS, baseline_mae, baseline_rmse, baseline_rel):
    print(f"  {k:25s} MAE={mae:8.2f}  RMSE={rmse:8.2f}  RelMAE={rel:.3f}")


# ---------- 3. ResNet18 模型结果 ----------
resnet_path = os.path.join(CKPT_DIR, "test_results.npz")
if os.path.exists(resnet_path):
    data = np.load(resnet_path)
    resnet_mae = data["mae"]
    resnet_rmse = data["rmse"]
    resnet_rel = resnet_mae / train_mean

    print("\n===== ResNet18 回归（test 集）=====")
    for k, mae, rmse, rel in zip(TARGET_KEYS, resnet_mae, resnet_rmse, resnet_rel):
        print(f"  {k:25s} MAE={mae:8.2f}  RMSE={rmse:8.2f}  RelMAE={rel:.3f}")

    # ---------- 4. 对比表 ----------
    print("\n===== 对比表（test 集）=====")
    rows = []
    for i, k in enumerate(TARGET_KEYS):
        rows.append({
            "target": k,
            "baseline_MAE": round(baseline_mae[i], 2),
            "resnet18_MAE": round(resnet_mae[i], 2),
            "improvement_%": round((baseline_mae[i] - resnet_mae[i]) / baseline_mae[i] * 100, 1),
            "baseline_RMSE": round(baseline_rmse[i], 2),
            "resnet18_RMSE": round(resnet_rmse[i], 2),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    out_csv = os.path.join(CKPT_DIR, "baseline_vs_resnet18.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n-> 对比表已保存: {out_csv}")

else:
    print(f"\n[警告] 未找到 {resnet_path}，请先跑 train.py")