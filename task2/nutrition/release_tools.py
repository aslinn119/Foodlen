"""Release checks and saved-prediction scoring. Uses only Python standard library."""
import json, math, argparse, hashlib, collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
KEYS=['total_weight_g','total_calories_kcal','protein_g','fat_g','carbohydrate_g']
def read_rows(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def index_rows(rows):
    result={}
    for row in rows:
        key=row['dish_id']
        if key in result: raise ValueError('Duplicate dish_id: '+key)
        result[key]=row
    return result
def align_predictions(preds,refs):
    pi=index_rows(preds);ri=index_rows(refs)
    if set(pi)!=set(ri): raise ValueError('ID mismatch: missing=%s extra=%s'%(sorted(set(ri)-set(pi)),sorted(set(pi)-set(ri))))
    return [pi[r['dish_id']] for r in refs]
def check_images(root,rows):
    missing=[];unreadable=[];hashes=collections.defaultdict(list)
    try: from PIL import Image
    except ImportError: Image=None
    for row in rows:
        p=root/row['image']
        if not p.is_file():missing.append(row['image']);continue
        try:
            hashes[hashlib.sha256(p.read_bytes()).hexdigest()].append(row['dish_id'])
            if Image:
                with Image.open(p) as im:im.verify()
        except Exception as e:unreadable.append({'image':row['image'],'error':str(e)})
    return {'n':len(rows),'missing':missing,'unreadable':unreadable,'image_decode_checked':Image is not None,'identical_file_groups':[v for v in hashes.values() if len(v)>1]}
def score(preds,refs,labels):
    preds=align_predictions(preds,refs);classes=collections.defaultdict(lambda:[0,0,0]);tp=fp=fn=0
    errors={k:[] for k in KEYS}
    for row,ref in zip(preds,refs):
        p=row['parsed'];r=ref['target'];ps={i['name'] for i in p['items']};rs={i['name'] for i in r['items']}
        tp+=len(ps&rs);fp+=len(ps-rs);fn+=len(rs-ps)
        for name in ps|rs:
            v=classes[name];v[0]+=name in ps&rs;v[1]+=name in ps-rs;v[2]+=name in rs-ps
        for k in KEYS:
            if not math.isfinite(float(p[k])) or p[k]<0:raise ValueError('Invalid numeric field '+row['dish_id']+':'+k)
            errors[k].append(p[k]-r[k])
    f=lambda v:2*v[0]/max(2*v[0]+v[1]+v[2],1)
    return {'micro_p':tp/max(tp+fp,1),'micro_r':tp/max(tp+fn,1),'micro_f1':2*tp/max(2*tp+fp+fn,1),'macro_f1':sum(f(classes[c]) for c in labels)/len(labels),'macro_class_scope':'fixed_ingredient_vocabulary','macro_class_count':len(labels),'out_of_vocabulary_names':sorted(set(classes)-set(labels)),'numeric':{k:{'mae':sum(map(abs,e))/len(e),'mse':sum(v*v for v in e)/len(e),'rmse':math.sqrt(sum(v*v for v in e)/len(e))} for k,e in errors.items()}}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['rescore','data-check']);a=ap.parse_args()
    out=ROOT/'results_repaired_20261123';out.mkdir(exist_ok=True)
    refs={s:read_rows(ROOT/f'annotations/{s}.jsonl') for s in ['train','val','test']}
    if a.action=='data-check':
        report={s:check_images(ROOT,rows) for s,rows in refs.items()}
        for rows in refs.values():index_rows(rows)
        report['split_intersections']={x+'_'+y:sorted(set(index_rows(refs[x]))&set(index_rows(refs[y]))) for x,y in [('train','val'),('train','test'),('val','test')]}
        (out/'data_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Data check saved; missing images:',sum(len(report[s]['missing']) for s in refs));return
    labels=json.loads((ROOT/'ingredients.json').read_text(encoding='utf-8'))
    summary=[]
    for s in ['val','test']:
        for model in ['a','b','bgen']:
            source=ROOT/f'final_results_v3_20261119/{model}_{s}_predictions.jsonl';preds=read_rows(source)
            old=json.loads((source.parent/f'{model}_{s}_metrics.json').read_text(encoding='utf-8'));new=score(preds,refs[s],labels)
            old['previous_macro_f1']=old['macro_f1']
            for k,v in new.items():
                if k=='numeric':old[k].update(v)
                else:old[k]=v
            if 'before_weight_projection' in old:
                raw=[dict(p,parsed=p['raw_parsed']) for p in preds];old['before_weight_projection'].update(score(raw,refs[s],labels))
            old['revision']={'source':'saved_predictions_rescored_not_new_inference','prediction_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'text_metrics':'inherited unchanged; not recomputed in this release','model_role':'historical_optional_comparison' if model=='bgen' else 'main_model'}
            (out/f'{model}_{s}_metrics.json').write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
            summary.append({'model':model,'split':s,'n':len(preds),'micro_f1':new['micro_f1'],'macro_f1_fixed151':new['macro_f1'],'weight_mae_g':new['numeric']['total_weight_g']['mae'],'calories_mae_kcal':new['numeric']['total_calories_kcal']['mae']})
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
