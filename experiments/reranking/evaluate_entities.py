import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'review'))
from visual_labels_source import LABELS
from common import ROOT,read,dump,lines
from entities import canonical_labels,extract
import argparse
parser=argparse.ArgumentParser(description='Score assistant draft annotations, or a fully signed human review CSV.')
parser.add_argument('--human-csv',type=Path)
args=parser.parse_args();tag='_human' if args.human_csv else ''
human=None
if args.human_csv:
 with args.human_csv.open(encoding='utf-8-sig',newline='') as f:human=list(csv.DictReader(f))
 if not any(r.get('human_eligible')=='1' for r in human):raise ValueError('At least one eligible image is required')
 if len(human)!=100 or any(r['review_status']!='verified' or not r['reviewer'].strip() or not r['reviewed_at'].strip() or r['human_eligible'] not in ['0','1'] for r in human):raise ValueError('Require all 100 rows signed, dated, review_status=verified, and human_eligible=0 or 1')
confirmed_draft=bool(human) and all(r.get('review_method')=='user_confirmed_assistant_draft_via_chat' for r in human)
annotation_source='user confirmation of assistant draft via chat' if confirmed_draft else 'human review' if human else 'assistant visual inspection, not human ground truth'
rows=read(ROOT/'results/reranked.jsonl')[100:]
labels=LABELS.strip().splitlines();assert len(labels)==100,len(labels)
annotations=[];scores=[]
for i,(r,label) in enumerate(zip(rows,labels)):
 confirmed,uncertain,note=[x.strip() for x in label.split('|')]
 if human:
  h=human[i]
  if h['image_id']!=r['image_id'] or int(h['index'])!=i:raise ValueError('Human review IDs/order mismatch')
  confirmed=h['human_visible'].replace(';',' ');uncertain=h['human_uncertain'].replace(';',' ');note=('Draft confirmed by ' if confirmed_draft else 'Human verified by ')+h['reviewer']+' at '+h['reviewed_at']
 visible=canonical_labels(confirmed.split());unknown=canonical_labels(uncertain.split())-visible
 # General herbs/leafy greens do not establish their exact botanical species.
 # Meat species collapse to meat; visibly unresolved possible meat remains neutral.
 ann={'index':i,'image_id':r['image_id'],'visible':sorted(visible),'uncertain':sorted(unknown),'note':note,'eligible':(human[i]['human_eligible']=='1') if human else i!=33,'annotation_source':annotation_source,'independent_blind_annotation':False if confirmed_draft else None,'human_verified':bool(human)}
 annotations.append(ann)
 for method,caption in [('greedy',r['greedy']),('beam',r['candidates'][0]['caption']),('beam_clip',r['caption'])]:
  predicted=set(extract(caption));tp=predicted&visible;neutral=predicted&unknown;unsupported=predicted-visible-unknown;missing=visible-predicted
  scores.append({'index':i,'image_id':r['image_id'],'method':method,'caption':caption,'predicted':sorted(predicted),'matched':sorted(tp),'unresolved':sorted(neutral),'unsupported':sorted(unsupported),'omitted':sorted(missing),'eligible':ann['eligible']})
lines(ROOT/('review/human_annotations.jsonl' if human else 'review/assistant_annotations.jsonl'),annotations);lines(ROOT/f'results/entity_per_image{tag}.jsonl',scores)
if not human and not (ROOT/'review/human_review.csv').exists():
 with (ROOT/'review/human_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['index','image_id','assistant_visible','assistant_uncertain','assistant_note','human_visible','human_uncertain','human_eligible','reviewer','reviewed_at','review_status'])
  for a in annotations:w.writerow([a['index'],a['image_id'],';'.join(a['visible']),';'.join(a['uncertain']),a['note'],'','','','','','pending_human_review'])
# Closed lexicon, image-entity set metrics. Zero-claim captions remain in recall and omission counts.
output={'annotation_source':annotation_source,'independent_blind_annotation':False if confirmed_draft else None,'human_verified_images':100 if human else 0,'reviewed_images':100,'eligible_images':sum(a['eligible'] for a in annotations),'excluded_index':[a['index'] for a in annotations if not a['eligible']],'excluded_reason':'see signed human annotations' if human else 'polymer-clay cake model; retained in text evaluation','protocol':'visible evidence only; uncertain claims excluded from assessable precision denominator; unsupported is not proof of factual absence; repeated mentions count once per image','methods':{}}
for method in ['greedy','beam','beam_clip']:
 rr=[s for s in scores if s['method']==method and s['eligible']];tp=sum(len(s['matched']) for s in rr);fp=sum(len(s['unsupported']) for s in rr);fn=sum(len(s['omitted']) for s in rr);neutral=sum(len(s['unresolved']) for s in rr)
 output['methods'][method]={'tp':tp,'unsupported':fp,'omitted':fn,'unresolved':neutral,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'unsupported_rate':fp/(tp+fp) if tp+fp else None,'omission_rate':fn/(tp+fn) if tp+fn else None,'unsupported_image_rate':sum(bool(s['unsupported']) for s in rr)/len(rr),'zero_entity_caption_count':sum(not s['predicted'] for s in rr)}
# Sensitivity: keep the post-hoc flagged model image, and vary treatment of unresolved claims.
output['sensitivity_all_100']={}
for method in output['methods']:
 rr=[s for s in scores if s['method']==method];tp=sum(len(s['matched']) for s in rr);fp=sum(len(s['unsupported']) for s in rr);fn=sum(len(s['omitted']) for s in rr)
 output['sensitivity_all_100'][method]={'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'unsupported_rate':fp/(tp+fp) if tp+fp else None}
 v=output['methods'][method];n=v['tp']+v['unsupported']+v['unresolved'];v['unsupported_rate_unresolved_bounds']=[v['unsupported']/n,(v['unsupported']+v['unresolved'])/n]
import numpy as np
methods={m:[s for s in scores if s['method']==m and s['eligible']] for m in output['methods']}
rng=np.random.default_rng(20261121);n=output['eligible_images'];samples=rng.integers(0,n,(5000,n));output['paired_bootstrap']={}
def boot(m):
 tp=np.array([len(s['matched']) for s in methods[m]]);fp=np.array([len(s['unsupported']) for s in methods[m]]);fn=np.array([len(s['omitted']) for s in methods[m]])
 t=tp[samples].sum(1);f=fp[samples].sum(1);n=fn[samples].sum(1)
 return t/(t+f),t/(t+n),f/(t+f)
clip=boot('beam_clip')
for base in ['greedy','beam']:
 b=boot(base);output['paired_bootstrap']['beam_clip_minus_'+base]={k:{'delta':output['methods']['beam_clip'][k]-output['methods'][base][k],'ci95':np.quantile(x-y,[.025,.975]).tolist()} for k,x,y in zip(['precision','recall','unsupported_rate'],clip,b)}
dump(ROOT/f'results/entity_metrics{tag}.json',output)
print(json.dumps(output,indent=2))
