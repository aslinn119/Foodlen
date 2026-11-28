"""Frozen external baseline experiment: calibration first, evaluation last."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from challenge_external.runtime import KEYS,MODELS,RESULTS,PointPredictor,sha,DEVICE
from challenge_external.conformal import fit_quantiles,bounds
AUDIT=ROOT/'challenge_audit/omnifood'

def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    RESULTS.mkdir(exist_ok=True)
    if (RESULTS/'calibration.json').exists() or (RESULTS/'evaluation_predictions.jsonl').exists():
        raise RuntimeError('Existing experiment artifacts: do not overwrite a frozen evaluation')
    identity=json.loads((AUDIT/'baseline_identity.json').read_text())
    for rel,info in identity['files'].items():
        assert sha(ROOT/rel)==info['sha256'], 'Changed baseline '+rel
    protocol=json.loads((AUDIT/'experiment_protocol.json').read_text())
    for role,m in protocol['manifests'].items():
        assert sha(AUDIT/f'{role}_manifest.jsonl')==m['sha256']
    targets=np.array([[json.loads(l)['target'][k] for k in KEYS] for l in (ROOT/'annotations/train.jsonl').read_text(encoding='utf-8').splitlines()])
    dump(RESULTS/'frozen_stats.json',{'mean':targets.mean(0).tolist(),'std':targets.std(0).tolist(),'source_sha256':identity['files']['annotations/train.jsonl']['sha256']})
    dependencies=['model.py','model_lora.py','checkpoints/best_model.pt','checkpoints_lora_mt/best_model.pt',
                  'challenge_external/runtime.py','challenge_external/conformal.py','challenge_external/results/frozen_stats.json']
    dependencies += [str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'clip_model').rglob('*') if p.is_file() and p.suffix in ['.json','.bin','.safetensors']]
    hashes={p:sha(ROOT/p) for p in dependencies}
    dump(RESULTS/'run_freeze.json',{'protocol':protocol,'dependencies':hashes,'runner_sha256':sha(__file__),'device':DEVICE,'torch':torch.__version__,'batch_size':32,'training':False})
    predictor=PointPredictor();start=time.time();all_metrics=[]
    artifact=None
    with zipfile.ZipFile(ROOT/'0-OminiFood8k.zip') as z:
        for role in ['calibration','evaluation']:
            records=[json.loads(l) for l in (AUDIT/f'{role}_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
            arrays={m:[] for m in MODELS}
            with (RESULTS/f'{role}_predictions.jsonl').open('w',encoding='utf-8') as out:
                for begin in range(0,len(records),32):
                    batch=records[begin:begin+32];images=[]
                    for r in batch:
                        with Image.open(io.BytesIO(z.read(r['image_member']))) as im: images.append(predictor.tf(im.convert('RGB')))
                    preds=predictor.predict_tensors(torch.stack(images))
                    for m in MODELS: arrays[m].append(preds[m])
                    for j,r in enumerate(batch):
                        out.write(json.dumps({'scene_id':r['scene_id'],'image_member':r['image_member'],'target':r['target'],'quality_flags':r['data_quality_flags'],'points':{m:preds[m][j].tolist() for m in MODELS}},allow_nan=False)+'\n')
                    if begin%512==0: print(role,min(begin+32,len(records)), '/',len(records),'elapsed_sec',round(time.time()-start),flush=True)
            arrays={m:np.concatenate(v) for m,v in arrays.items()}
            y=np.array([[r['target'][k] for k in KEYS] for r in records])
            if role=='calibration':
                artifact={'version':'omnifood-abs90-v1-'+sha(RESULTS/'calibration_predictions.jsonl')[:12],
                          'target_coverage':.9,'coverage_scope':'per_field_marginal','n':len(y),
                          'calibration_manifest_sha256':sha(AUDIT/'calibration_manifest.jsonl'),
                          'calibration_predictions_sha256':sha(RESULTS/'calibration_predictions.jsonl'),
                          'inference_dependencies':hashes,'models':{}}
                for m,p in arrays.items():
                    q,rank=fit_quantiles(p,y,.9)
                    artifact['models'][m]={'q':q.tolist(),'rank':rank,'model_version':hashes['challenge_external/results/frozen_stats.json'][:16] if m=='mean' else hashes['checkpoints/best_model.pt' if m=='resnet18' else 'checkpoints_lora_mt/best_model.pt'][:16]}
                dump(RESULTS/'calibration.json',artifact)
                print('Calibration frozen before evaluation',artifact['version'],flush=True)
            else:
                misses=[];summary={}
                for m,p in arrays.items():
                    low,high=bounds(p,artifact['models'][m]['q']);covered=(y>=low)&(y<=high);width=high-low
                    summary[m]={'all_five_covered_fraction':float(covered.all(1).mean()),'any_field_missed_scenes':int((~covered.all(1)).sum())}
                    for i,k in enumerate(KEYS):
                        error=p[:,i]-y[:,i]
                        all_metrics.append({'model':m,'field':k,'unit':'kcal' if i==1 else 'g','n':len(y),'MAE':float(np.abs(error).mean()),'MSE':float(np.square(error).mean()),'RMSE':float(np.sqrt(np.square(error).mean())),
                                            'target_coverage':.9,'coverage':float(covered[:,i].mean()),'covered_count':int(covered[:,i].sum()),'miss_count':int((~covered[:,i]).sum()),'mean_width':float(width[:,i].mean()),'median_width':float(np.median(width[:,i])),'q':artifact['models'][m]['q'][i]})
                    for j,r in enumerate(records):
                        if not covered[j].all():
                            misses.append({'model':m,'scene_id':r['scene_id'],'image_member':r['image_member'],'quality_flags':r['data_quality_flags'],
                                           'failed_fields':[k for i,k in enumerate(KEYS) if not covered[j,i]],'point':p[j].tolist(),'target':y[j].tolist(),'lower':low[j].tolist(),'upper':high[j].tolist()})
                dump(RESULTS/'metrics.json',all_metrics);dump(RESULTS/'coverage_summary.json',summary)
                with (RESULTS/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
                    writer=csv.DictWriter(f,fieldnames=list(all_metrics[0]));writer.writeheader();writer.writerows(all_metrics)
                with (RESULTS/'undercoverage_cases.jsonl').open('w',encoding='utf-8') as f:
                    for r in misses:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    dump(RESULTS/'completion.json',{'finished':True,'seconds':time.time()-start,'calibration_sha256':sha(RESULTS/'calibration.json'),'evaluation_predictions_sha256':sha(RESULTS/'evaluation_predictions.jsonl'),'training':False})
    print('FINISHED',json.dumps(all_metrics),flush=True)

if __name__=='__main__': main()
