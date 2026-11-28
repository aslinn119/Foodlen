"""Point prediction development on existing train/validation images only."""
from pathlib import Path
import json
import numpy as np
from challenge_external.metafood_experiment import KEYS, read_split
from challenge_external.portion_interval_development import fit_head, group_folds

OUT = Path(__file__).resolve().parent / "point_head_development"
MODES = ("linear", "kernel", "kernel_sqrt")


def candidate(x, y, query, mode):
    if mode == "linear":
        return fit_head(x, y, query)
    # A radial kernel captures local similarities between frozen image embeddings.
    def distance(a, b):
        return np.maximum((a*a).sum(1)[:, None] + (b*b).sum(1) - 2*a@b.T, 0)
    distances = distance(x, x)
    bandwidth = max(np.median(distances[np.triu_indices(len(x), 1)]), 1e-8)
    gram = np.exp(-distances / bandwidth)
    cross = np.exp(-distance(query, x) / bandwidth)
    target = np.sqrt(y) if mode == "kernel_sqrt" else y
    avg = target.mean(0)
    coef = np.linalg.solve(gram + 0.3*np.eye(len(x)), target-avg)
    result = np.maximum(0, avg + cross@coef)
    return result**2 if mode == "kernel_sqrt" else result


def main():
    OUT.mkdir(exist_ok=True)
    data = [read_split(role) for role in ("train","validation")]
    rows = data[0][0] + data[1][0]
    x = np.concatenate([v[1] for v in data])
    y = np.concatenate([v[3] for v in data])
    folds = group_folds(rows)
    saved = []
    for k, test in enumerate(folds):
        train = np.concatenate([indices for j, indices in enumerate(folds) if j != k])
        inner = group_folds([rows[i] for i in train])
        # Choose one method for all fields. Compare pooled normalized absolute error.
        scores = {}
        for mode in MODES:
            errors = []
            for val in inner:
                fit = np.setdiff1d(np.arange(len(train)), val)
                u, v = train[fit], train[val]
                p = candidate(x[u], y[u], x[v], mode)
                errors.extend(abs(p-y[v])/np.maximum(y[u].std(0),1e-8))
            scores[mode] = np.mean(errors)
        selected = min(scores, key=scores.get)
        base = candidate(x[train],y[train],x[test],"linear")
        tuned = candidate(x[train],y[train],x[test],selected)
        cut = np.quantile(y[train],.8,axis=0)
        for j,i in enumerate(test):
            saved.append(dict(id=rows[i]["sample_id"], category=rows[i]["object_id"].split("/")[0].casefold(),
                              fold=k, selected=selected, inner_scores=scores, truth=y[i].tolist(),
                              high_cut=cut.tolist(), base=base[j].tolist(), tuned=tuned[j].tolist()))
    assert len({r["id"] for r in saved})==len(saved)==250
    assert all(len({r["fold"] for r in saved if r["category"]==g})==1 for g in {r["category"] for r in saved})
    (OUT/"predictions.json").write_text(json.dumps(saved,allow_nan=False))
    truth,cut,base,tuned=[np.array([r[f] for r in saved]) for f in ("truth","high_cut","base","tuned")]
    summary=[]
    for j,field in enumerate(KEYS):
        high=truth[:,j]>=cut[:,j]
        record=dict(field=field,high_count=int(high.sum()))
        for name,p in [("base",base),("tuned",tuned)]:
            err=p[:,j]-truth[:,j]
            record[name]=dict(mae=np.abs(err).mean(),mse=(err**2).mean(),high_mae=np.abs(err[high]).mean())
        summary.append(record)
    (OUT/"metrics.json").write_text(json.dumps(summary,indent=2))
    print([next(r["selected"] for r in saved if r["fold"]==k) for k in range(5)],summary)
if __name__=="__main__":
    main()
