"""Generate or rerank innovation-one captions; no training functionality."""
import argparse,hashlib,json
from pathlib import Path
import torch
from PIL import Image
from common import ROOT,DEFAULT_CP,read,lines,dump,load,tensor,greedy,beams,text,metrics

def select(row,alpha):
    def norm(key):
        vals=[c[key] for c in row['candidates']];lo,hi=min(vals),max(vals)
        return [(v-lo)/(hi-lo) if hi>lo else 0. for v in vals]
    gs,cs=norm('generation_score'),norm('clip_cosine')
    j=max(range(len(gs)),key=lambda i:(1-alpha)*gs[i]+alpha*cs[i])
    return {**row,'caption':row['candidates'][j]['caption'],'selected_index':j}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['generate','clip','rerank'])
    p.add_argument('--out',type=Path,default=ROOT/'reproduction');p.add_argument('--checkpoint',type=Path,default=DEFAULT_CP)
    p.add_argument('--data-root',type=Path,default=ROOT/'assets/images');p.add_argument('--clip-model',default=str(ROOT/'weights/clip'))
    p.add_argument('--subset',type=Path,default=ROOT/'results/val_subset.jsonl');p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    if a.mode=='generate':
        target=a.out/'candidates.jsonl'
        if target.exists():raise FileExistsError('Use a new output directory to avoid mixing configurations.')
        device='cuda' if torch.cuda.is_available() else 'cpu';m,s=load(a.checkpoint,device);c=s['config'];v=s['vocab']
        rows=read(a.subset);rows=rows[:a.limit] if a.limit else rows;results=[]
        with target.open('w',encoding='utf-8') as f,torch.no_grad():
            for r in rows:
                mem=m.encode_image(tensor(r,a.data_root,device));g,gs=greedy(m,mem,c)
                candidates=[{'token_ids':seq,'caption':text(seq,v),'sum_logprob':score,'length':len(seq)-1,'generation_score':score/(len(seq)-1)} for seq,score in beams(m,mem,c,4)]
                item={'image_id':r['file_name'],'file_name':r['file_name'],'reference':r['caption'],'greedy':text(g,v),'greedy_score':gs/(len(g)-1),'candidates':candidates}
                f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush();results.append(item)
                if len(results)%10==0:print('Generated',len(results),flush=True)
        dump(a.out/'generation_provenance.json',{'checkpoint_sha256':hashlib.file_digest(a.checkpoint.open('rb'),'sha256').hexdigest(),'n':len(rows),'device':device,'torch':torch.__version__})
    elif a.mode=='clip':
        from transformers import CLIPModel,CLIPProcessor
        model=CLIPModel.from_pretrained(a.clip_model,local_files_only=True).eval();processor=CLIPProcessor.from_pretrained(a.clip_model,local_files_only=True)
        rows=read(a.out/'candidates.jsonl')
        for i,r in enumerate(rows):
            with Image.open(a.data_root/r['file_name']) as im:inputs=processor(text=[c['caption'] for c in r['candidates']],images=im.convert('RGB'),return_tensors='pt',padding=True,truncation=True)
            with torch.no_grad():result=model(**inputs);sims=(result.image_embeds @ result.text_embeds.T)[0].tolist()
            for c,sim in zip(r['candidates'],sims):c['clip_cosine']=sim
            if (i+1)%20==0:print('CLIP',i+1,flush=True)
        lines(a.out/'clip_candidates.jsonl',rows)
    else:
        rows=read(a.out/'clip_candidates.jsonl');manifest=json.loads((ROOT/'results/manifest.json').read_text(encoding='utf-8'))
        if [r['image_id'] for r in rows]!=manifest['tune_ids']+manifest['audit_ids']:raise ValueError('Expected the frozen 200 validation IDs in their original order')
        tunes=[{'alpha':x,**metrics([select(r,x) for r in rows[:100]])} for x in manifest['alpha_grid']]
        alpha=max(tunes,key=lambda x:x['cider_d'])['alpha'];lines(a.out/'reranked.jsonl',[select(r,alpha) for r in rows])
        dump(a.out/'selection.json',{'alpha':alpha,'grid':tunes,'criterion':'local CIDEr-D, 100 tuning validation images only'})
if __name__=='__main__':main()
