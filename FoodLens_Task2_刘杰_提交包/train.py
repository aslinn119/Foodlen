import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data import get_dataloaders, TARGET_KEYS
from model import NutritionRegressor

# ---------------- 配置 ----------------
DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(DATA_ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")


# ---------------- 计算 train 集均值和标准差（用于归一化） ----------------
def compute_stats():
    records = []
    with open(os.path.join(DATA_ROOT, "annotations", "train.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    vals = np.array([[r["target"][k] for k in TARGET_KEYS] for r in records])
    return vals.mean(axis=0), vals.std(axis=0)


MEAN, STD = compute_stats()
MEAN = torch.tensor(MEAN, dtype=torch.float32, device=DEVICE)
STD = torch.tensor(STD, dtype=torch.float32, device=DEVICE)
print("均值:", MEAN.cpu().numpy())
print("标准差:", STD.cpu().numpy())


# ---------------- 评估函数 ----------------
@torch.no_grad()
def evaluate(model, loader, mean, std):
    model.eval()
    all_preds, all_targets = [], []
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE)
        preds_norm = model(imgs)
        # 反归一化
        preds = preds_norm * std + mean
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds, axis=0)   # [N, 5]
    targets = np.concatenate(all_targets, axis=0)

    mae = np.abs(preds - targets).mean(axis=0)
    rmse = np.sqrt(((preds - targets) ** 2).mean(axis=0))
    return mae, rmse, preds, targets


# ---------------- 均值基线 ----------------
def mean_baseline(loader):
    all_targets = []
    for _, targets in loader:
        all_targets.append(targets.numpy())
    targets = np.concatenate(all_targets, axis=0)
    preds = np.tile(MEAN.cpu().numpy(), (len(targets), 1))
    mae = np.abs(preds - targets).mean(axis=0)
    rmse = np.sqrt(((preds - targets) ** 2).mean(axis=0))
    return mae, rmse


# ---------------- 训练 ----------------
def main():
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=0)

    # === 基线：均值预测 ===
    print("\n===== 均值基线（val）=====")
    mae, rmse = mean_baseline(val_loader)
    for k, m, r in zip(TARGET_KEYS, mae, rmse):
        print(f"  {k:25s} MAE={m:8.2f}  RMSE={r:8.2f}")

    # === 模型 ===
    model = NutritionRegressor(pretrained=True).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    history = {"train_loss": [], "val_mae": [], "val_rmse": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for imgs, targets in pbar:
            imgs = imgs.to(DEVICE)
            targets_norm = (targets.to(DEVICE) - MEAN) / STD

            optimizer.zero_grad()
            preds_norm = model(imgs)
            loss = criterion(preds_norm, targets_norm)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader.dataset)

        # 验证
        mae, rmse, _, _ = evaluate(model, val_loader, MEAN, STD)
        avg_mae = mae.mean()
        avg_rmse = rmse.mean()

        history["train_loss"].append(avg_loss)
        history["val_mae"].append(avg_mae)
        history["val_rmse"].append(avg_rmse)

        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_MAE={avg_mae:.2f}, val_RMSE={avg_rmse:.2f}")

        # 保存最优
        if avg_mae < best_val_mae:
            best_val_mae = avg_mae
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_model.pt"))
            print(f"  -> 保存最优模型 (val_MAE={best_val_mae:.2f})")

    # === 测试集评估 ===
    print("\n===== 测试集评估（最优模型）=====")
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_model.pt")))
    mae, rmse, preds, targets = evaluate(model, test_loader, MEAN, STD)
    for k, m, r in zip(TARGET_KEYS, mae, rmse):
        print(f"  {k:25s} MAE={m:8.2f}  RMSE={r:8.2f}")

    # 保存结果
    np.savez(os.path.join(CKPT_DIR, "test_results.npz"),
             preds=preds, targets=targets, mae=mae, rmse=rmse)

    with open(os.path.join(CKPT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n结果已保存到: {CKPT_DIR}")


if __name__ == "__main__":
    main()