"""Meaningful decoder, overlap-matcher and frozen-selection checks."""
import json
import torch
from common import ROOT,read,beams,greedy
from pipeline import select
from entities import extract
class Toy:
 def decode(self,mem,tokens):
  out=torch.full((len(tokens),tokens.shape[1],5),-10.);out[:,-1,2]=5.;out[:,-1,3]=4.;return out
c={'bos_id':1,'eos_id':2,'pad_id':0,'max_len':5};mem=torch.zeros(1,1,1)
assert greedy(Toy(),mem,c)[0]==[1,2]
assert beams(Toy(),mem,c,2)[0][0]==[1,2]
assert all(2 not in ids[1:-1] for ids,_ in beams(Toy(),mem,c,2))
assert extract('rice noodles with chicken and spring onions')==['meat','noodle','scallion']
assert extract('orange salmon pie')==['fish']
assert extract('applesauce cake with whipped cream')==['apple','cream']
rows=read(ROOT/'results/reranked.jsonl');manifest=json.loads((ROOT/'results/manifest.json').read_text(encoding='utf-8'))
assert [r['image_id'] for r in rows]==manifest['tune_ids']+manifest['audit_ids']
assert len(rows)==200 and len({r['image_id'] for r in rows})==200
for r in rows:
 assert len(r['candidates'])==4 and len({tuple(c['token_ids']) for c in r['candidates']})==4
 assert select(r,0)['selected_index']==0 and select(r,1)['selected_index']==r['selected_index']
 for candidate in r['candidates']:
  assert candidate['length']==len(candidate['token_ids'])-1
  assert abs(candidate['sum_logprob']-candidate['generation_score']*candidate['length'])<1e-5
print('PASS: EOS termination, ingredient overlap/color matching, frozen IDs, 200x4 distinct candidates, score normalization and alpha endpoints')
