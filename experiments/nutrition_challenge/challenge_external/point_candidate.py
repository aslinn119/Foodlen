"""MetaFood development-only point predictor. Usage: python -m challenge_external.point_candidate IMAGE"""
import argparse
import json
from pathlib import Path
import numpy as np
from challenge_external.metafood_experiment import KEYS, features, read_split
from challenge_external.runtime import ROOT, sha
from challenge_external.point_head_development import OUT, candidate


def infer(x, head):
    z=head["train"]
    dist=np.maximum((x*x).sum(1)[:,None]+(z*z).sum(1)-2*x@z.T,0)
    return np.maximum(0,head["avg"]+np.exp(-dist/head["bandwidth"])@head["coef"])**2


def export():
    data=[read_split(r) for r in ("train","validation")]
    x=np.concatenate([s[1] for s in data]);y=np.concatenate([s[3] for s in data])
    dist=np.maximum((x*x).sum(1)[:,None]+(x*x).sum(1)-2*x@x.T,0)
    bandwidth=np.median(dist[np.triu_indices(len(x),1)])
    avg=np.sqrt(y).mean(0)
    coef=np.linalg.solve(np.exp(-dist/bandwidth)+.3*np.eye(len(x)),np.sqrt(y)-avg)
    file=OUT/"candidate.npz"
    np.savez_compressed(file,train=x,bandwidth=bandwidth,avg=avg,coef=coef)
    with np.load(file) as head:
        np.testing.assert_allclose(infer(x,head),candidate(x,y,x,"kernel_sqrt"),atol=1e-8)
    frozen=json.loads((ROOT/"challenge_external/metafood_extension/freeze.json").read_text())["dependencies"]
    paths=[rel for rel in frozen if not rel.endswith("/head.npz")]
    paths.extend(["challenge_external/point_candidate.py",file.relative_to(ROOT).as_posix()])
    (OUT/"candidate_version.json").write_text(json.dumps({p:sha(ROOT/p) for p in paths},indent=2))


def predict_candidate(image):
    versions=json.loads((OUT/"candidate_version.json").read_text())
    for path,value in versions.items():
        if sha(ROOT/path)!=value:
            raise ValueError("Changed development model dependency: "+path)
    x,_=features([Path(image)])
    with np.load(OUT/"candidate.npz") as head:
        point=infer(x,head)[0]
    return {"model":"metafood_kernel_sqrt","status":"development_only_uncalibrated","point":dict(zip(KEYS,point.tolist()))}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("image")
    print(json.dumps(predict_candidate(parser.parse_args().image),indent=2))
