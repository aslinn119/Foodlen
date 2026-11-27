import argparse, hashlib, json, random, sys, time
from pathlib import Path
import torch
from PIL import Image
from model import ViTCaptioner
from v1_data import eval_transforms
from caption_metrics import compute_caption_metrics, rouge_l
ROOT=Path(__file__).resolve().parent
DEFAULT_CP=ROOT/'weights/v4_full.pt'

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def lines(path,rows):
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def load(cp,device):
    s=torch.load(cp,map_location='cpu',weights_only=False); c=s['config']
    kwargs={k:c[k] for k in ['pad_id','max_len','d_model','nhead','num_decoder_layers','dim_feedforward','dropout']}
    m=ViTCaptioner(len(s['vocab']),**kwargs,pretrained=False)
    if 'full_model_state' in s:m.load_state_dict(s['full_model_state'],strict=True)
    else:raise ValueError('Use the bundled full-model checkpoint; original decoder-only checkpoint requires the original ImageNet backbone.')
    m.load_trainable_state_dict(s['trainable_state']); return m.to(device).eval(),s

def text(ids,vocab):
    inv={v:k for k,v in vocab.items()}; out=[]
    for i in ids[1:]:
        if i==vocab['<EOS>']:break
        if i!=vocab['<PAD>']:out.append(inv[i])
    return ' '.join(out)

def distribution(m,mem,tokens,c):
    logits=m.decode(mem,tokens)[:,-1,:].clone()
    logits[:,c['pad_id']]=-torch.inf; logits[:,c['bos_id']]=-torch.inf
    return logits.log_softmax(-1)

@torch.no_grad()
def beams(m,mem,c,width=4):
    active=[([c['bos_id']],0.)]; finished=[]
    for _ in range(c['max_len']-1):
        if not active:break
        tokens=torch.tensor([x[0] for x in active],device=mem.device)
        probs=distribution(m,mem.expand(len(active),-1,-1),tokens,c)
        values,indices=probs.topk(width,dim=-1); pool=[]
        for j,(seq,score) in enumerate(active):
            for v,i in zip(values[j].tolist(),indices[j].tolist()):
                item=(seq+[i],score+v)
                if i==c['eos_id']:finished.append(item)
                else:pool.append(item)
        active=sorted(pool,key=lambda x:x[1],reverse=True)[:width]
    pool=finished+active
    # Average log probability including EOS; identical score used in reranking.
    return sorted(pool,key=lambda x:x[1]/(len(x[0])-1),reverse=True)[:width]

def tensor(row,data,device):
    with Image.open(data/row['file_name']) as im:return eval_transforms()(im.convert('RGB')).unsqueeze(0).to(device)

def metrics(rows,key='caption'):
    return compute_caption_metrics([r[key].lower().split() for r in rows],[r['reference'].lower().split() for r in rows])

@torch.no_grad()
def greedy(m,mem,c):
    ids=[c['bos_id']]; score=0.
    for _ in range(c['max_len']-1):
        lp=distribution(m,mem,torch.tensor([ids],device=mem.device),c)[0]
        i=int(lp.argmax());score+=float(lp[i]);ids.append(i)
        if i==c['eos_id']:break
    return ids,score
