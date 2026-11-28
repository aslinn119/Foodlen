"""Evaluate one fixed RGB-prediction blend on saved category-held-out predictions."""
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "final_development_review"
FIELDS = ("total_weight_g", "total_calories_kcal", "protein_g", "fat_g", "carbohydrate_g")


def write(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def main():
    OUT.mkdir(exist_ok=True)
    write("protocol.json", {
        "inputs": "Prior five-fold category-held-out linear and square-root kernel predictions on 250 development objects",
        "gate": "For each field: use kernel below fitting-fold target median, linear above fitting-fold 80th percentile, interpolate linearly between, gated by the linear RGB prediction.",
        "criteria": "For each field require overall MAE <= unweighted kernel overall MAE AND high-group MAE <= linear high-group MAE. All five must pass to justify pursuing the blend.",
        "stopping": "One fixed rule, no retries on these folds, no final test or calibration access. Development success would still need independent calibration and evaluation."
    })
    rows = json.loads((ROOT/"point_head_development/predictions.json").read_text())
    weighted = {r["id"]:r for r in json.loads((ROOT/"high_value_development/heldout.json").read_text())}
    assert len(rows)==len(weighted)==len({r["id"] for r in rows})==250
    assert len({r["category"] for r in rows})==91
    assert all(len({r["fold"] for r in rows if r["category"]==g})==1 for g in {r["category"] for r in rows})
    truth = np.array([r["truth"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    linear = np.array([r["base"] for r in rows])
    kernel = np.array([r["tuned"] for r in rows])
    tail_head = np.array([weighted[r["id"]]["new"] for r in rows])
    blended = np.empty_like(kernel)
    high = np.zeros_like(kernel,dtype=bool)
    for k in range(5):
        mask = fold==k
        median,upper = np.quantile(truth[~mask],[.5,.8],axis=0)
        np.testing.assert_allclose(upper, rows[np.flatnonzero(mask)[0]]["high_cut"])
        gate = np.clip((linear[mask]-median)/np.maximum(upper-median,1e-8),0,1)
        blended[mask] = (1-gate)*kernel[mask]+gate*linear[mask]
        high[mask] = truth[mask]>=upper
    table=[]
    for i,field in enumerate(FIELDS):
        record={"field":field,"high_n":int(high[:,i].sum())}
        for name,p in [("linear",linear),("kernel",kernel),("weighted",tail_head),("blend",blended)]:
            err=np.abs(p[:,i]-truth[:,i])
            record[name]={"mae":err.mean(),"high_mae":err[high[:,i]].mean()}
        record["pass"]=bool(record["blend"]["mae"]<=record["kernel"]["mae"] and record["blend"]["high_mae"]<=record["linear"]["high_mae"])
        table.append(record)
    write("metrics.json",table)
    write("blend_predictions.json",[{"id":r["id"],"fold":r["fold"],"point":p.tolist()} for r,p in zip(rows,blended)])
    for r in table:
        print(r["field"],r["blend"],r["pass"])

if __name__=="__main__":
    main()
