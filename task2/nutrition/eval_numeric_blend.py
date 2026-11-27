"""Validation-only evaluation of a numeric blend of Model B and B-gen."""
import json
import random
from pathlib import Path

import numpy as np

from metrics import NUM_KEYS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "final_results"


def read_by_dish(path, field):
    values = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            dish = row["dish_id"]
            if dish in values:
                raise ValueError(f"Duplicate dish_id in {path}: {dish}")
            values[dish] = [row[field][key] for key in NUM_KEYS]
    return values


def errors(pred, target):
    delta = pred - target
    return {"mean_mae": float(np.abs(delta).mean()),
            "mean_mse": float(np.square(delta).mean()),
            "field_mae": dict(zip(NUM_KEYS, np.abs(delta).mean(axis=0).tolist()))}


def main():
    target = read_by_dish(ROOT / "annotations/val.jsonl", "target")
    base = read_by_dish(OUT / "modelB_val_preds.jsonl", "parsed")
    gen = read_by_dish(OUT / "modelBgen_val_preds.jsonl", "parsed")
    if target.keys() != base.keys() or target.keys() != gen.keys():
        raise ValueError("Validation dish IDs differ across inputs")
    ids = sorted(target)
    random.Random(7).shuffle(ids)
    y, b, g = (np.array([data[dish] for dish in ids]) for data in (target, base, gen))
    grid = [i / 20 for i in range(21)]
    candidates = {w: b + w * (g - b) for w in grid}
    pooled = np.empty_like(y)
    folds = []
    for fold in range(5):
        held = np.arange(len(ids)) % 5 == fold
        weight = min(grid, key=lambda w: np.abs(candidates[w][~held] - y[~held]).mean())
        pooled[held] = candidates[weight][held]
        folds.append({"n": int(held.sum()), "bgen_weight": weight,
                      "b": errors(b[held], y[held]), "blend": errors(pooled[held], y[held])})
    summary = {"split": "val", "n": len(ids), "seed": 7, "folds": folds,
               "b": errors(b, y), "bgen": errors(g, y),
               "cross_validation_blend": errors(pooled, y),
               "fixed_weight_0.25_full_validation": errors(candidates[0.25], y)}
    path = OUT / "numeric_blend_validation.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Model B MAE {summary['b']['mean_mae']:.4f}")
    print(f"Five-fold held-out MAE {summary['cross_validation_blend']['mean_mae']:.4f}")


if __name__ == "__main__":
    main()
