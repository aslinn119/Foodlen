"""Recompute scores with pycocoevalcap implementations, using explicit whitespace tokenization."""
import argparse,json,hashlib,importlib.metadata
from pathlib import Path
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from common import ROOT,read,dump
from pipeline import select

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'results');a=p.parse_args()
 rows=read(a.results/'reranked.jsonl');manifest=json.loads((ROOT/'results/manifest.json').read_text(encoding='utf-8'))
 if [r['image_id'] for r in rows]!=manifest['tune_ids']+manifest['audit_ids']:raise ValueError('Expected all 200 frozen validation IDs in order')
 audit=rows[100:];gts={i:[r['reference'].lower()] for i,r in enumerate(audit)}
 output={'n':100,'split':'frozen audit validation','tokenization':'lowercase, whitespace; no PTB tokenizer or Java; COCO scorer implementations, not complete COCO evaluation pipeline','pycocoevalcap':importlib.metadata.version('pycocoevalcap'),'methods':{}}
 per={}
 for method in ['greedy','beam','beam_clip']:
  caps=[r['greedy'] if method=='greedy' else r['candidates'][0]['caption'] if method=='beam' else r['caption'] for r in audit]
  res={i:[s.lower()] for i,s in enumerate(caps)}
  b,_=Bleu(4).compute_score(gts,res,verbose=0);rouge,_=Rouge().compute_score(gts,res);cider,scores=Cider().compute_score(gts,res)
  output['methods'][method]={'bleu_1_to_4_percent':[float(x)*100 for x in b],'rouge_l':float(rouge),'cider_d':float(cider)}
  per[method]=np.asarray(scores);output['methods'][method]['avg_words']=float(np.mean([len(s.split()) for s in caps]));output['methods'][method]['unk_image_rate']=float(np.mean(['<UNK>' in s for s in caps]));output['methods'][method]['repeated_trigram_image_rate']=float(np.mean([len(list(zip(t,t[1:],t[2:])))>len(set(zip(t,t[1:],t[2:]))) for t in [s.split() for s in caps]]))
 rng=np.random.default_rng(20261121);samples=rng.integers(0,100,(5000,100));output['paired_bootstrap']={}
 for baseline in ['greedy','beam']:
  delta=per['beam_clip']-per[baseline];boots=delta[samples].mean(1)
  output['paired_bootstrap']['beam_clip_minus_'+baseline]={'cider_d_delta':float(delta.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist(),'replicates':5000,'note':'paired image resampling of per-image scores with document frequencies held fixed; exploratory'}
 dump(a.results/'coco_metrics.json',output)
 dump(a.results/'coco_per_image.json',[{'image_id':r['image_id'],**{m:float(v[i]) for m,v in per.items()}} for i,r in enumerate(audit)])
 print(json.dumps(output,indent=2))
if __name__=='__main__':main()
