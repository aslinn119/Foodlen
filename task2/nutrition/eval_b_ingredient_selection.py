from pathlib import Path
import json,random
import numpy as np,torch
from torch.utils.data import DataLoader
from predict import _get, DEVICE, select_b_ingredients
from eval_models import ImgDataset, CLIPT_TF_EVAL
from metrics import ingredient_f1
from model_lora import WEIGHT_NORM
from postprocess_bgen import ingredient_weight_errors
rs=[json.loads(l) for l in Path('annotations/val.jsonl').read_text(encoding='utf8').splitlines()]
s=_get('b')
probs=[]; weights=[]
with torch.no_grad():
 for x in DataLoader(ImgDataset(rs,CLIPT_TF_EVAL),batch_size=16):
  o=s['model'](x.to(DEVICE)); probs.extend(o['ingredient'].sigmoid().cpu().numpy()); weights.extend(o['item_weight'].cpu().numpy()*WEIGHT_NORM)
refs=[[i['name'] for i in r['target']['items']] for r in rs]
def gen(t,mode):
 result=[]
 for p,w in zip(probs,weights):
  idx=select_b_ingredients(p,t,score_order=mode=='score')
  if mode=='fill' and not len(idx) and p.max()>=t-.1: idx=np.array([p.argmax()])
  result.append([s['vocab'][i] for i in idx] if w[idx].clip(0).sum()>0 else [])
 return result
modes=['base','score','fill']
grid=[(t/1000,m) for t in range(500,951,25) for m in modes]
pred={c:gen(*c) for c in grid}
base=gen(s['threshold'],'base')
ix=sorted(range(len(rs)),key=lambda i:rs[i]['dish_id']); random.Random(7).shuffle(ix)
def measure(p, ds): return ingredient_f1([p[i] for i in ds],[refs[i] for i in ds])
pooled=base.copy()
folds=[]
for f in range(5):
 test=ix[f::5]; train=[i for k,i in enumerate(ix) if k%5!=f]
 c=max(grid,key=lambda c:measure(pred[c],train)['micro_f1'])
 for i in test: pooled[i]=pred[c][i]
 folds.append({'threshold':c[0],'selection':c[1],'base':measure(base,test)['micro_f1'],'candidate':measure(pooled,test)['micro_f1']})
result={'split':'val','n':len(rs),'fold_seed':7,'checkpoint_threshold':s['threshold'],'grid':[{'threshold':c[0],'selection':c[1],**measure(v,ix)} for c,v in pred.items()],'folds':folds,'checkpoint':measure(base,ix),'held_out':measure(pooled,ix),'overflow_at_checkpoint':int(sum((p>=s['threshold']).sum()>16 for p in probs))}
saved = [json.loads(line) for line in Path('final_results/modelB_val_preds.jsonl').read_text(encoding='utf-8').splitlines()]
if [row['dish_id'] for row in saved] != [row['dish_id'] for row in rs]:
    raise ValueError('Saved B predictions and validation labels do not align')
if base != [[item['name'] for item in row['parsed']['items']] for row in saved]:
    raise ValueError('Current B inference differs from saved validation ingredients')
ranked = []
for prob, weight, row in zip(probs, weights, saved):
    idx = select_b_ingredients(prob, s['threshold'], score_order=True)
    masses = weight[idx].clip(0).tolist()
    total = sum(masses)
    scale = row['parsed']['total_weight_g'] / total if total else 0
    items = [{'name': s['vocab'][i], 'weight_g': round(w * scale, 1)}
             for i, w in zip(idx, masses)] if total else []
    ranked.append({'items': items})
ranked_names = [[item['name'] for item in row['items']] for row in ranked]
result['rank_overflow'] = {
    'threshold': s['threshold'],
    'baseline_weight': ingredient_weight_errors([row['parsed'] for row in saved], rs),
    'ranked_weight': ingredient_weight_errors(ranked, rs),
    'ranked_metrics': measure(ranked_names, ix),
    'baseline_empty_count': sum(not row for row in base),
    'ranked_empty_count': sum(not row for row in ranked_names),
    'fold_micro_f1': [{'baseline': measure(base, ix[f::5])['micro_f1'],
                       'ranked': measure(ranked_names, ix[f::5])['micro_f1']}
                      for f in range(5)],
}
Path('final_results/b_ingredient_selection_validation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print([(f['threshold'],f['selection'],f['base'],f['candidate']) for f in folds]); print(result['checkpoint'],result['held_out'],result['overflow_at_checkpoint'])

