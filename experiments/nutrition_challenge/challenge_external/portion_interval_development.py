"""Grouped development comparison of absolute and portion-scaled intervals."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from challenge_external.metafood_experiment import KEYS, read_split

OUT = Path(__file__).resolve().parent / "portion_interval_development"
METHODS = ("absolute", "portion_scaled")


def fit_head(x, y, test):
    center = x.mean(axis=0)
    centered = x - center
    gram = centered @ centered.T
    penalty = np.trace(gram) / len(x)
    dual = np.linalg.solve(gram + penalty * np.eye(len(x)), y - y.mean(axis=0))
    return np.maximum(0, (test - center) @ centered.T @ dual + y.mean(axis=0))


def group_folds(rows):
    groups = {}
    for i, row in enumerate(rows):
        category = row["object_id"].split("/")[0].casefold()
        groups.setdefault(category, []).append(i)
    def order(group):
        return -len(groups[group]), hashlib.sha256(group.encode()).hexdigest()
    folds = [[] for _ in range(5)]
    for group in sorted(groups, key=order):
        min(folds, key=len).extend(groups[group])
    return [np.array(f) for f in folds]


def main():
    OUT.mkdir(exist_ok=True)
    data = [read_split(role) for role in ("train", "validation")]
    rows = data[0][0] + data[1][0]
    x = np.concatenate([s[1] for s in data])
    y = np.concatenate([s[3] for s in data])
    folds = group_folds(rows)
    records = []
    for fold, test in enumerate(folds):
        cal = folds[(fold + 1) % 5]
        train = np.setdiff1d(np.arange(len(x)), np.r_[cal, test])
        norm = y[train].std(axis=0)
        typical = np.maximum(y[train].mean(axis=0), 1e-8)
        points = fit_head(x[train], y[train], x[np.r_[cal, test]])
        for method in METHODS:
            # A positive floor limits how aggressively the radius shrinks near zero.
            scale = np.sqrt(0.25 + points / typical) if method == "portion_scaled" else np.ones_like(points)
            residual = np.abs(points[:len(cal)] - y[cal]) / scale[:len(cal)]
            rank = math.ceil((len(cal) + 1) * 0.9)
            q = np.sort(residual, axis=0)[rank - 1]
            radius = q * scale[len(cal):]
            pred = points[len(cal):]
            low, high = np.maximum(0, pred-radius), pred+radius
            for j, i in enumerate(test):
                records.append(dict(id=rows[i]["sample_id"],fold=fold,method=method,truth=y[i].tolist(),
                                    low=low[j].tolist(),high=high[j].tolist(),train_std=norm.tolist()))
    (OUT/"heldout.json").write_text(json.dumps(records, allow_nan=False))
    table = []
    for method in METHODS:
        sample = [r for r in records if r["method"]==method]
        assert len({r["id"] for r in sample}) == len(sample) == 250
        low, high, truth, std = [np.array([r[k] for r in sample]) for k in ("low","high","truth","train_std")]
        width = high-low
        hit = (low<=truth)&(truth<=high)
        score = width+20*(np.maximum(low-truth,0)+np.maximum(truth-high,0))
        for j, field in enumerate(KEYS):
            table.append(dict(method=method,field=field,hits=int(hit[:,j].sum()),coverage=hit[:,j].mean(),width=width[:,j].mean(),normalized_interval_score=(score[:,j]/std[:,j]).mean()))
    (OUT/"metrics.json").write_text(json.dumps(table,indent=2))
    print(json.dumps(table))
if __name__=="__main__":
    main()
