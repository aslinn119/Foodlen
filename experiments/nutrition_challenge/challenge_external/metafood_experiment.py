"""MetaFood single-object extension. Run from the project root with -m."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from challenge_external.conformal import bounds, fit_quantiles
from challenge_external.runtime import DEVICE, KEYS, ROOT, PointPredictor, sha

OUT = ROOT / "challenge_external/metafood_extension"
DATA = ROOT / "challenge_external/data_supplement_v1/metafood"

def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")

@lru_cache(maxsize=1)
def encoder():
    return PointPredictor()

@torch.inference_mode()
def features(paths):
    model = encoder()
    vectors, points = [], []
    for start in range(0, len(paths), 16):
        batch = []
        for path in paths[start:start+16]:
            with Image.open(path) as im:
                batch.append(model.tf(im.convert("RGB")))
        x = torch.stack(batch).to(DEVICE)
        feat = model.clip.clip(pixel_values=model.cnorm(x)).pooler_output
        raw = (model.clip.norm_head(feat).cpu()*model.cs+model.cm).numpy()
        vectors.extend(feat.cpu().numpy())
        points.extend([[round(max(0,float(v)),1) for v in row] for row in raw])
    return np.asarray(vectors,dtype=float),np.asarray(points)

def read_split(role):
    rows = [json.loads(l) for l in (DATA/(role+".jsonl")).read_text(encoding="utf8").splitlines()]
    x,p = features([DATA/r["image"] for r in rows])
    return rows,x,p,np.array([[r["target"][k] for k in KEYS] for r in rows])

def ridge(x,h):
    return np.maximum(0,(x-h["center"])@h["coef"]+h["intercept"])

def predict_metafood_intervals(image_path):
    artifact=json.loads((OUT/"calibration.json").read_text())
    for rel,digest in artifact["dependencies"].items():
        if sha(ROOT/rel)!=digest:
            raise ValueError("Calibration dependency changed: "+rel)
    x,base=features([Path(image_path)])
    with np.load(OUT/"head.npz") as head:
        point=(ridge(x,head) if artifact["selected"]=="ridge" else base)[0]
    lo,hi=bounds(point,artifact["q"][artifact["selected"]])
    return {"model_version":artifact["selected"]+"-"+artifact["dependencies"]["challenge_external/metafood_extension/head.npz"][:12],
            "calibration_version":artifact["version"],"target_coverage":0.9,"coverage_scope":"per_field_marginal_metafood_objects",
            "fields":{k:{"point":point[i].item(),"lower":lo[i].item(),"upper":hi[i].item(),"unit":"kcal" if i==1 else "g"} for i,k in enumerate(KEYS)}}

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/"protocol.json").exists():
        raise RuntimeError("Existing experiment. Preserve registered split and results.")
    roles=["train","validation","calibration_reserved","test_reserved"]
    manifests=[ [json.loads(l) for l in (DATA/(r+".jsonl")).read_text().splitlines()] for r in roles ]
    for key in ["object_id","image_sha256"]:
        values=[v[key] for rows in manifests for v in rows]
        assert len(set(values))==len(values)==495
    assert list(map(len,manifests))==[200,50,150,95]
    deps=json.loads((ROOT/"challenge_external/results/calibration.json").read_text())["inference_dependencies"]
    assert all(sha(ROOT/f)==v for f,v in deps.items())
    deps["challenge_external/metafood_experiment.py"]=sha(__file__)
    save("protocol.json",{"domain":"MetaFood3D single object; batch independence unknown","roles":{role:sha(DATA/(role+".jsonl")) for role in roles},
         "models":"Frozen historical CLIP vs centered ridge on pooled RGB features; lambda=mean diagonal of training Gram matrix; no tuning grid",
         "selection":"lower mean validation MAE / training target SD, ties prefer CLIP; freeze before calibration; evaluate both for paired comparison",
         "coverage":"90% per field; absolute residuals; nonnegative intervals","dependencies":deps.copy()})
    _,x,_,y=read_split("train")
    center=x.mean(0); xc=x-center; gram=xc@xc.T
    coef=xc.T@np.linalg.solve(gram+np.eye(len(x))*np.trace(gram)/len(x),y-y.mean(0))
    head=dict(center=center,coef=coef,intercept=y.mean(0))
    np.savez(OUT/"head.npz",**head)
    deps["challenge_external/metafood_extension/head.npz"]=sha(OUT/"head.npz")
    _,x,b,t=read_split("validation")
    scores={m:float((np.abs(p-t).mean(0)/y.std(0)).mean()) for m,p in {"clip":b,"ridge":ridge(x,head)}.items()}
    selected=min(scores,key=scores.get)
    save("freeze.json",{"validation_scores":scores,"selected":selected,"dependencies":deps})
    rows,x,b,t=read_split("calibration_reserved")
    qs={};records={}
    for m,p in {"clip":b,"ridge":ridge(x,head)}.items():
        q,rank=fit_quantiles(p,t,0.9);qs[m]=q.tolist();records[m]=p.tolist()
    save("calibration_predictions.json",{"ids":[r["sample_id"] for r in rows],"targets":t.tolist(),"points":records})
    save("calibration.json",{"q":qs,"rank":rank,"n":len(t),"dependencies":deps,"selected":selected,"target_coverage":0.9,
         "version":"metafood-abs90-"+sha(OUT/"calibration_predictions.json")[:12]})
    rows,x,b,t=read_split("test_reserved")
    metrics=[];miss=[];points={}
    for m,p in {"clip":b,"ridge":ridge(x,head)}.items():
        points[m]=p.tolist();lo,hi=bounds(p,qs[m]);hit=(lo<=t)&(t<=hi)
        for i,k in enumerate(KEYS):
            err=p[:,i]-t[:,i]
            metrics.append(dict(model=m,field=k,mae=float(np.abs(err).mean()),mse=float((err**2).mean()),hits=int(hit[:,i].sum()),n=len(t),coverage=float(hit[:,i].mean()),mean_width=float((hi-lo)[:,i].mean())))
            for j in np.flatnonzero(~hit[:,i]):
                miss.append(dict(model=m,id=rows[j]["sample_id"],field=k,point=p[j,i],lower=lo[j,i],upper=hi[j,i],target=t[j,i]))
    save("evaluation_predictions.json",{"ids":[r["sample_id"] for r in rows],"targets":t.tolist(),"points":points})
    save("metrics.json",metrics);save("undercoverage.json",miss)
    demo= predict_metafood_intervals(DATA/rows[0]["image"])
    for i,k in enumerate(KEYS):
        assert np.isclose(demo["fields"][k]["point"],points[selected][0][i],atol=.11,rtol=.0001)
    save("interface_example.json",demo)
    print(selected,scores,metrics,flush=True)

if __name__=="__main__":
    main()
