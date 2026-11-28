"""Verify the shipped tables and all three live baseline outputs from bundled files."""
import json
import math
from pathlib import Path
import numpy as np
from challenge_external.display_session import IntervalSession
from challenge_external.runtime import KEYS, MODELS, RESULTS, sha


def read_rows(name):
    return [json.loads(line) for line in (RESULTS/name).read_text(encoding="utf8").splitlines()]


def verify():
    cal=read_rows("calibration_predictions.jsonl")
    ev=read_rows("evaluation_predictions.jsonl")
    splits=[{r["scene_id"] for r in records} for records in (cal,ev)]
    assert [len(s) for s in splits]==[len(cal),len(ev)]==[5668,2368] and splits[0].isdisjoint(splits[1])
    artifact=json.loads((RESULTS/"calibration.json").read_text())
    assert sha(RESULTS/"calibration_predictions.jsonl")==artifact["calibration_predictions_sha256"]
    metrics=json.loads((RESULTS/"metrics.json").read_text())
    expected={(r["model"],r["field"]):r for r in metrics}
    assert len(expected)==len(metrics)==15
    failures={}
    for model in MODELS:
        for i,field in enumerate(KEYS):
            scores=sorted(abs(r["target"][field]-r["points"][model][i]) for r in cal)
            q=scores[math.ceil(.9*(len(cal)+1))-1]
            assert q==artifact["models"][model]["q"][i]
            points=np.array([r["points"][model][i] for r in ev])
            truth=np.array([r["target"][field] for r in ev])
            low=np.maximum(0,points-q);high=points+q
            hit=(low<=truth)&(truth<=high)
            err=points-truth
            actual={"MAE":np.abs(err).mean(),"MSE":np.square(err).mean(),"RMSE":np.sqrt(np.square(err).mean()),
                    "coverage":hit.mean(),"mean_width":(high-low).mean(),"median_width":np.median(high-low)}
            for key,val in actual.items():
                assert math.isclose(val,expected[model,field][key],rel_tol=1e-11,abs_tol=1e-10),(model,field,key)
            for j in np.flatnonzero(~hit):
                failures.setdefault((model,ev[j]["scene_id"]),[]).append(field)
    misses=read_rows("undercoverage_cases.jsonl")
    assert len(misses)==len(failures) and failures=={(r["model"],r["scene_id"]):r["failed_fields"] for r in misses}
    live=IntervalSession().predict_all(RESULTS/"interface_sample.jpg")
    example=json.loads((RESULTS/"interface_example.json").read_text())["outputs"]
    for model in MODELS:
        for key in ("model_version","target_coverage","calibration_version","coverage_scope"):
            assert live[model][key]==example[model][key]
        for field in KEYS:
            f=live[model]["fields"][field];ref=example[model]["fields"][field]
            assert f["unit"]==ref["unit"] and f["lower"]<=f["point"]<=f["upper"]
            for key in ("point","lower","upper"):
                assert math.isclose(f[key],ref[key],rel_tol=1e-4,abs_tol=.101)
    print(f"PASS: 15 thresholds and metric rows, {len(misses)} failure records, 3 live RGB outputs.")

if __name__=="__main__":
    verify()
