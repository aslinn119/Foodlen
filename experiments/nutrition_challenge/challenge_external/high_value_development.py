"""Tail-aware kernel ridge experiment. Uses only MetaFood train and validation."""
import argparse
import json
from pathlib import Path

import numpy as np
from challenge_external.point_candidate import infer, features, OUT as BASE
from challenge_external.metafood_experiment import read_split, KEYS, ROOT, sha
from challenge_external.portion_interval_development import group_folds

OUT = Path(__file__).resolve().parent / "high_value_development"
WEIGHTS = (1, 2, 4)


def fit(x, y, weights):
    norms = (x*x).sum(1)
    distance = np.maximum(norms[:,None]+norms-2*x@x.T, 0)
    bandwidth = np.median(distance[np.triu_indices(len(x),1)])
    kernel = np.exp(-distance/bandwidth)
    high = y >= np.quantile(y,.8,axis=0)
    w = np.where(high, weights, 1)
    target = np.sqrt(y)
    avg = np.average(target,axis=0,weights=w)
    coef = np.column_stack([
        np.linalg.solve(kernel+np.diag(.3/w[:,i]),target[:,i]-avg[i])
        for i in range(y.shape[1])
    ])
    return dict(train=x, bandwidth=bandwidth, avg=avg, coef=coef)


def select(rows, x, y):
    folds = group_folds(rows)
    predictions = np.empty((len(WEIGHTS),*y.shape))
    high = np.empty_like(y,dtype=bool)
    for val in folds:
        tr = np.setdiff1d(np.arange(len(x)),val)
        high[val] = y[val]>=np.quantile(y[tr],.8,axis=0)
        for j,w in enumerate(WEIGHTS):
            predictions[j,val]=infer(x[val],fit(x[tr],y[tr],w))
    error=np.abs(predictions-y)
    overall=error.mean(axis=1)
    tail=(error*high).sum(axis=1)/high.sum(0)
    # Pick by inner tail MAE subject to at most 5% inner overall MAE cost.
    index=np.where(overall<=1.05*overall[0],tail,np.inf).argmin(axis=0)
    return np.array(WEIGHTS)[index]


def predict_image(path):
    identity=json.loads((OUT/"version.json").read_text())
    for rel,digest in identity.items():
        if sha(ROOT/rel)!=digest:
            raise ValueError("Model dependency has changed: "+rel)
    x,_=features([Path(path)])
    with np.load(BASE/"candidate.npz") as base, np.load(OUT/"head.npz") as params:
        p=infer(x,dict(train=base["train"],**dict(params)))[0]
    return {"status":"development_only_uncalibrated","point":dict(zip(KEYS,p.tolist()))}


def experiment():
    OUT.mkdir(exist_ok=True)
    data=[read_split(role) for role in ("train","validation")]
    rows=data[0][0]+data[1][0]
    x=np.concatenate([s[1] for s in data]);y=np.concatenate([s[3] for s in data])
    old={r["id"]:r for r in json.loads((BASE/"predictions.json").read_text())}
    records=[]
    for fold,test in enumerate(group_folds(rows)):
        tr=np.setdiff1d(np.arange(len(x)),test)
        w=select([rows[i] for i in tr],x[tr],y[tr])
        pred=infer(x[test],fit(x[tr],y[tr],w))
        base=infer(x[test],fit(x[tr],y[tr],1))
        cut=np.quantile(y[tr],.8,axis=0)
        for j,i in enumerate(test):
            r=old[rows[i]["sample_id"]]
            assert r["fold"]==fold
            np.testing.assert_allclose(r["tuned"],base[j],atol=1e-5)
            np.testing.assert_allclose(r["high_cut"],cut)
            records.append(dict(id=r["id"],fold=fold,weights=w.tolist(),target=y[i].tolist(),
                                high=(y[i]>=cut).tolist(),previous=base[j].tolist(),new=pred[j].tolist()))
    assert len({r["id"] for r in records})==len(records)==250
    (OUT/"heldout.json").write_text(json.dumps(records,allow_nan=False))
    weights=select(rows,x,y)
    model=fit(x,y,weights)
    with np.load(BASE/"candidate.npz") as base:
        np.testing.assert_allclose(model.pop("train"),base["train"],atol=1e-7)
    np.savez_compressed(OUT/"head.npz",**model)
    identity=json.loads((BASE/"candidate_version.json").read_text())
    for p in [Path(__file__),OUT/"head.npz"]:
        identity[p.resolve().relative_to(ROOT).as_posix()]=sha(p)
    (OUT/"version.json").write_text(json.dumps(identity,indent=2))
    print("Full-data development weights",weights)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--image")
    args=parser.parse_args()
    if args.image:
        print(json.dumps(predict_image(args.image)))
    else:
        experiment()
