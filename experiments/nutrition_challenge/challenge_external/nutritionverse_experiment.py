"""Register, develop, calibrate and evaluate the dish-disjoint NutritionVerse experiment."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from statistics import NormalDist

import numpy as np

from challenge_external.conformal import bounds, fit_quantiles
from challenge_external.nutritionverse_model import (KEYS, OUT, ROOT, extract, fit, points,
                                                    read, scale, sha, validate_hashes)

DATA = ROOT / "challenge_external/data_supplement_v1/nutritionverse"
ROLES = ("train", "validation", "calibration_reserved", "test_reserved")
WEIGHTS = [1, 2, 1, 1, 1]
METHODS = ("absolute", "portion_scaled")
MODELS = ("mean_train", "resnet18", "clip_lora", "adapted")
BASE_DEPS = ROOT / "challenge_external/results/calibration.json"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def save(name, obj):
    with NamedTemporaryFile(dir=OUT, mode="w", encoding="utf8", delete=False) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        temp = Path(f.name)
    temp.replace(OUT/name)


def register():
    OUT.mkdir(exist_ok=True)
    if (OUT/"protocol.json").exists():
        raise FileExistsError("Protocol already exists")
    source = read(DATA/"protocol.json")
    audit = read(DATA/"audit.json")
    review = read(DATA/"release_review.json")
    assert review["status"] == "import_verified" and review["model_runs"] == 0
    assert review["audit_sha256"] == sha(DATA/"audit.json") and not audit["exact_duplicates"]
    assert sha(DATA/"download_manifest.jsonl") == source["manifest_sha256"]
    assert all(sha(DATA/(r+".jsonl")) == audit["manifest_hashes"][r] for r in ROLES)
    deps = read(BASE_DEPS)["inference_dependencies"].copy()
    for filename in ("nutritionverse_model.py", "nutritionverse_experiment.py"):
        rel = "challenge_external/"+filename
        deps[rel] = sha(ROOT/rel)
    validate_hashes(deps)
    configs = [dict(view=v, ridge=r, transform=t) for v,r,t in itertools.product(
        ("full","plate","plate_size"), (.1,1.), ("sqrt","identity"))]
    save("protocol.json", {
        "version":"foodlens-nv-dish-rgb-v1", "domain":"NutritionVerse-Real v2; one registered RGB per dish",
        "roles":{r:{"n":source["role_counts"][r],"sha256":audit["manifest_hashes"][r]} for r in ROLES},
        "dependencies":deps, "point_candidates":configs, "model_order":MODELS, "field_order":KEYS,
        "baseline":"training mean of 68 dishes; original frozen Task2 ResNet18 and CLIP+LoRA",
        "selection":"Lowest validation weighted mean MAE normalized by training SD. Heat gets weight 2, other fields 1; ties use grid order. Head is trained on 68 only.",
        "interval_selection":"Compare absolute and portion-scaled radii on validation, using provisional quantiles from five-fold out-of-fold training residuals and train mean as typical. Minimize normalized 90% interval score with same field weights; ties prefer absolute.",
        "calibration":"After freeze, recompute exact split-conformal q on 60 reserved dishes only, order 55 of 60. Baselines use absolute scores. Clip lower bounds to 0.",
        "evaluation":"One evaluation on 67 reserved dishes. All four methods and all fields. MAE, MSE, empirical coverage, width, interval score and all misses. Paired dish bootstrap (5000, seed 20261125) for adapted minus mean and CLIP MAE. Descriptive Wilson 95% coverage intervals.",
        "independence":"Dish IDs disjoint. Cooking batches unknown. Do not generalize to OmniFood8K. Test never used for any decision.",
        "roles_source_sha256":sha(DATA/"protocol.json"),"image_input":"RGB pixels only; full frame, automatic plate crop and optionally simple color/portion descriptors."})


def batch(role, protocol):
    spec=protocol["roles"][role]
    path=DATA/(role+".jsonl")
    assert sha(path)==spec["sha256"]
    rr=rows(path)
    assert len(rr)==spec["n"] and len({r["object_id"] for r in rr})==len(rr)
    paths=[DATA/r["image"] for r in rr]
    for p,r in zip(paths,rr):
        assert sha(p)==r["image_sha256"] and r["role"]==role
    blocks, raw=extract(paths)
    y=np.array([[r["target"][k] for k in KEYS] for r in rr],dtype=float)
    return [r["sample_id"] for r in rr],blocks,raw,y


def weighted(values, sd):
    return float(np.average(np.asarray(values).mean(0)/sd,weights=WEIGHTS))


def quantiles(p,y,s):
    residuals=np.abs(p-y)/s
    return fit_quantiles(np.zeros_like(p),residuals,.9)


def fit_stage(protocol):
    validate_hashes(protocol["dependencies"])
    ids,x,_,y=batch("train",protocol)
    _,vx,raw,vy=batch("validation",protocol)
    sd=np.maximum(y.std(0),1e-8)
    typical=np.maximum(y.mean(0),1e-8)
    scores=[]
    for cfg in protocol["point_candidates"]:
        candidate=fit(x,y,cfg)
        pred=points(vx,candidate,cfg)
        scores.append({"config":cfg,"score":weighted(abs(pred-vy),sd),"mae":abs(pred-vy).mean(0).tolist()})
    selected=min(scores,key=lambda v:v["score"])["config"]
    head=fit(x,y,selected)
    vp=points(vx,head,selected)
    ordered=sorted(range(len(y)),key=lambda i:hashlib.sha256(("nv-oof-v1:"+ids[i]).encode()).digest())
    oof=np.empty_like(y)
    for val in np.array_split(ordered,5):
        train=np.setdiff1d(np.arange(len(y)),val)
        tx=tuple(a[train] for a in x)
        oof[val]=points(tuple(a[val] for a in x),fit(tx,y[train],selected),selected)
    iscores={}
    for m in METHODS:
        q,_=quantiles(oof,y,scale(oof,typical,m))
        lo,hi=bounds(vp,q*scale(vp,typical,m))
        iscores[m]=weighted(hi-lo+20*(np.maximum(lo-vy,0)+np.maximum(vy-hi,0)),sd)
    method=min(iscores,key=iscores.get)
    with NamedTemporaryFile(dir=OUT,delete=False) as f:
        np.savez(f,**head)
        tmp=Path(f.name)
    tmp.replace(OUT/"head.npz")
    deps=protocol["dependencies"].copy()
    for name in ("protocol.json","head.npz"):
        deps[(OUT/name).relative_to(ROOT).as_posix()]=sha(OUT/name)
    val_models={"mean_train":np.broadcast_to(y.mean(0),vy.shape),**raw,"adapted":vp}
    save("freeze.json",{"selected":selected,"scale":method,"dependencies":deps,
         "typical":typical.tolist(),"train_sd":sd.tolist(),"point_scores":scores,"interval_scores":iscores,
         "validation_mae":{m:np.abs(p-vy).mean(0).tolist() for m,p in val_models.items()}})
    print("development",selected,method,{m:round(np.abs(p-vy).mean(0)[1],2) for m,p in val_models.items()},flush=True)


def model_predictions(role, protocol, freeze):
    ids,x,raw,y=batch(role,protocol)
    with np.load(OUT/"head.npz") as h:
        adapted=points(x,h,freeze["selected"])
    forecasts={"mean_train":np.broadcast_to(freeze["typical"],y.shape),**raw,"adapted":adapted}
    return ids,forecasts,y


def score_scale(m, p, freeze):
    return scale(p,freeze["typical"],freeze["scale"] if m=="adapted" else "absolute")


def calibrate(protocol):
    freeze=read(OUT/"freeze.json")
    validate_hashes(freeze["dependencies"])
    ids,models,y=model_predictions("calibration_reserved",protocol,freeze)
    qs={}
    for m,p in models.items():
        q,rank=quantiles(p,y,score_scale(m,p,freeze))
        qs[m]=q.tolist()
    assert rank==55 and len(y)==60
    save("calibration_predictions.json",{"ids":ids,"targets":y.tolist(),"points":{m:p.tolist() for m,p in models.items()}})
    save("calibration.json",{"q":qs,"rank":rank,"n":len(y),"target_coverage":.9,
         "freeze_sha256":sha(OUT/"freeze.json"),"calibration_predictions_sha256":sha(OUT/"calibration_predictions.json")})


def evaluate(protocol):
    freeze,cal=read(OUT/"freeze.json"),read(OUT/"calibration.json")
    validate_hashes(freeze["dependencies"])
    assert cal["freeze_sha256"]==sha(OUT/"freeze.json")
    assert cal["calibration_predictions_sha256"]==sha(OUT/"calibration_predictions.json")
    ids,models,y=model_predictions("test_reserved",protocol,freeze)
    save("evaluation_predictions.json",{"ids":ids,"targets":y.tolist(),"points":{m:p.tolist() for m,p in models.items()},
                                         "calibration_sha256":sha(OUT/"calibration.json")})


def summarize():
    ev=read(OUT/"evaluation_predictions.json")
    cal,freeze=read(OUT/"calibration.json"),read(OUT/"freeze.json")
    assert ev["calibration_sha256"]==sha(OUT/"calibration.json")
    y=np.array(ev["targets"]); n=len(y)
    indices=np.random.default_rng(20261125).integers(0,n,size=(5000,n))
    results=[]; misses=[]
    for model, values in ev["points"].items():
        p=np.array(values)
        lo,hi=bounds(p,np.array(cal["q"][model])*score_scale(model,p,freeze))
        hit=(lo<=y)&(y<=hi)
        for j,k in enumerate(KEYS):
            err=p[:,j]-y[:,j]
            h=hit[:,j].mean()
            z=NormalDist().inv_cdf(.975)
            center=(h+z*z/(2*n))/(1+z*z/n)
            delta=z*np.sqrt(h*(1-h)/n+z*z/(4*n*n))/(1+z*z/n)
            w=hi[:,j]-lo[:,j]
            results.append({"model":model,"field":k,"mae":abs(err).mean(),"mse":np.square(err).mean(),
                "hits":int(hit[:,j].sum()),"n":n,"coverage":h,"wilson95":[center-delta,center+delta],
                "mean_width":w.mean(),"interval_score":(w+20*(np.maximum(lo[:,j]-y[:,j],0)+np.maximum(y[:,j]-hi[:,j],0))).mean()})
            for i in np.flatnonzero(~hit[:,j]):
                misses.append({"model":model,"dish":ev["ids"][i],"field":k,"point":p[i,j],"truth":y[i,j],
                               "lower":lo[i,j],"upper":hi[i,j]})
    diffs={}
    for m in ("mean_train","clip_lora"):
        paired=abs(np.array(ev["points"]["adapted"])-y)-abs(np.array(ev["points"][m])-y)
        diffs[m]={"difference":paired.mean(0).tolist(),"bootstrap95":np.quantile(paired[indices].mean(1),[.025,.975],axis=0).T.tolist()}
    save("metrics.json",results)
    save("undercoverage.json",misses)
    save("paired_mae.json",diffs)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("register","fit","calibrate","evaluate","summarize"))
    stage=parser.parse_args().stage
    if stage=="register":
        register()
    elif stage=="summarize":
        summarize()
    else:
        marker={"fit":"freeze.json","calibrate":"calibration.json","evaluate":"evaluation_predictions.json"}[stage]
        if (OUT/marker).exists():
            raise FileExistsError(f"Preserving frozen {marker}")
        protocol=read(OUT/"protocol.json")
        {"fit":fit_stage,"calibrate":calibrate,"evaluate":evaluate}[stage](protocol)
