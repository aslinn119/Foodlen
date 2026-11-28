"""Independent saved-result verification plus live image-interface smoke test."""
import json,math,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from challenge_external.runtime import KEYS,MODELS,RESULTS,predict_intervals,sha

def rows(name): return [json.loads(l) for l in (RESULTS/name).read_text(encoding='utf-8').splitlines()]
cal=rows('calibration_predictions.jsonl');ev=rows('evaluation_predictions.jsonl')
artifact=json.loads((RESULTS/'calibration.json').read_text())
metrics=json.loads((RESULTS/'metrics.json').read_text())
miss=rows('undercoverage_cases.jsonl')
assert len(cal)==5668 and len(ev)==2368
assert len({r['scene_id'] for r in cal})==len(cal)
assert len({r['scene_id'] for r in ev})==len(ev)
assert not {r['scene_id'] for r in cal}&{r['scene_id'] for r in ev}
assert sha(RESULTS/'calibration_predictions.jsonl')==artifact['calibration_predictions_sha256']
expected_misses={}
for model in MODELS:
    for i,k in enumerate(KEYS):
        residuals=sorted(abs(r['target'][k]-r['points'][model][i]) for r in cal)
        rank=math.ceil((len(residuals)+1)*.9)
        q=residuals[rank-1]
        assert q==artifact['models'][model]['q'][i]
        errors=[];widths=[];hits=0
        for r in ev:
            p=r['points'][model][i];y=r['target'][k]
            lo=max(0,p-q);hi=p+q;errors.append(p-y);widths.append(hi-lo)
            if lo<=y<=hi: hits+=1
            else: expected_misses.setdefault((model,r['scene_id']),[]).append(k)
        m=next(x for x in metrics if x['model']==model and x['field']==k)
        recomputed={'MAE':float(np.mean(np.abs(errors))),'MSE':float(np.mean(np.square(errors))),
                    'RMSE':float(np.sqrt(np.mean(np.square(errors)))),'coverage':hits/len(ev),
                    'mean_width':float(np.mean(widths)),'median_width':float(np.median(widths))}
        for field,value in recomputed.items(): assert math.isclose(m[field],value,rel_tol=1e-12,abs_tol=1e-10),(model,k,field)
assert {(r['model'],r['scene_id']):r['failed_fields'] for r in miss}==expected_misses
assert len(miss)==len(expected_misses)
# Fixed first evaluation scene; selection does not depend on success or error.
sample=ev[0];path=RESULTS/'interface_sample.jpg'
with zipfile.ZipFile(ROOT/'0-OminiFood8k.zip') as z:path.write_bytes(z.read(sample['image_member']))
outputs={}
for model in MODELS:
    out=predict_intervals(path,model);outputs[model]=out
    for i,k in enumerate(KEYS):
        f=out['fields'][k]
        assert math.isclose(f['point'],sample['points'][model][i],abs_tol=.101,rel_tol=1e-4)
        assert 0<=f['lower']<=f['point']<=f['upper']
        q=artifact['models'][model]['q'][i]
        assert math.isclose(f['upper']-f['point'],q,rel_tol=1e-12,abs_tol=1e-10)
    assert out['target_coverage']==.9 and out['calibration_version']==artifact['version']
(RESULTS/'interface_example.json').write_text(json.dumps({'scene_id':sample['scene_id'],'selection':'first evaluation record, not selected by prediction quality','outputs':outputs},ensure_ascii=False,indent=2),encoding='utf-8')
(RESULTS/'verification.json').write_text(json.dumps({'passed':True,'calibration_n':len(cal),'evaluation_n':len(ev),'metrics_rows_verified':len(metrics),'undercoverage_model_scene_rows_verified':len(miss),'live_interface_models':MODELS,'single_vs_batch_tolerance':{'atol':.101,'rtol':1e-4},'note':'No new model choices or threshold fitting on evaluation data.'},indent=2),encoding='utf-8')
print('Independent verification and all three live interfaces PASSED',flush=True)
