import json, re
from pathlib import Path
import torch
from PIL import Image
from experiment import read,dump,lines,metrics

def run(a):
    from transformers import CLIPModel,CLIPProcessor
    cache=Path(__file__).parent/'model_cache'
    model=CLIPModel.from_pretrained(a.clip_model,cache_dir=str(cache)).eval()
    processor=CLIPProcessor.from_pretrained(a.clip_model,cache_dir=str(cache))
    rows=read(a.out/'candidates.jsonl')
    if len(rows)!=200:raise ValueError('Require complete fixed validation set')
    for i,r in enumerate(rows):
        with Image.open(a.data_root/r['file_name']) as im:
            inputs=processor(text=[x['caption'] for x in r['candidates']],images=im.convert('RGB'),return_tensors='pt',padding=True,truncation=True)
        with torch.no_grad():
            output=model(**inputs)
            sims=(output.image_embeds @ output.text_embeds.T)[0].tolist()
        for c,score in zip(r['candidates'],sims):c['clip_cosine']=score
        if (i+1)%20==0:print('CLIP',i+1,flush=True)
    lines(a.out/'clip_candidates.jsonl',rows)
    def select(r,alpha):
        # Normalize within each image; alpha=0 exactly recovers beam ordering.
        def norm(key):
            vals=[c[key] for c in r['candidates']];lo,hi=min(vals),max(vals)
            return [(v-lo)/(hi-lo) if hi>lo else 0. for v in vals]
        gs,cs=norm('generation_score'),norm('clip_cosine')
        j=max(range(len(gs)),key=lambda i:(1-alpha)*gs[i]+alpha*cs[i])
        return {**r,'caption':r['candidates'][j]['caption'],'selected_index':j}
    grid=[0,.25,.5,.75,1]
    scores=[{'alpha':alpha,**metrics([select(r,alpha) for r in rows[:100]])} for alpha in grid]
    best=max(scores,key=lambda r:r['cider_d'])['alpha']
    selected=[select(r,best) for r in rows];lines(a.out/'reranked.jsonl',selected)
    dump(a.out/'rerank_metrics.json',{'clip_model':a.clip_model,'selection':'max CIDEr-D on first 100 validation IDs only; ties favor lower alpha','alpha':best,'tune_grid':scores,'audit_n':100,'audit':{'greedy':metrics(rows[100:],'greedy'),'beam':metrics([{**r,'caption':r['candidates'][0]['caption']} for r in rows[100:]]),'beam_clip':metrics(selected[100:])}})
    # Empty annotation cells require human review of the image, not reference-only claims.
    import csv
    with (a.out/'ingredient_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['image_id','reference','greedy','beam','beam_clip','visible_ingredients','greedy_hallucinations','greedy_omissions','beam_hallucinations','beam_omissions','clip_hallucinations','clip_omissions','review_status'])
        for r in selected[100:]:w.writerow([r['image_id'],r['reference'],r['greedy'],r['candidates'][0]['caption'],r['caption']]+['']*7+['pending_image_review'])
