import argparse, hashlib, json, random, sys, time
from pathlib import Path
import torch
from PIL import Image
from model import ViTCaptioner
from v1_data import eval_transforms
from caption_metrics import compute_caption_metrics, rouge_l
ROOT=Path(__file__).resolve().parent
BASE=Path.home()/'Desktop/神经网络与深度学习'
DEFAULT_CP=BASE/'FoodLens_V4_First_Version/FoodLens_V4/v4/outputs/full_v4/best_v4.pt'

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def lines(path,rows):
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def load(cp,device):
    s=torch.load(cp,map_location='cpu',weights_only=False); c=s['config']
    m=ViTCaptioner(len(s['vocab']),**{k:c[k] for k in ['pad_id','max_len','d_model','nhead','num_decoder_layers','dim_feedforward','dropout','pretrained']})
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

def rollout(m,mem,c,sample=False):
    ids=[c['bos_id']]; logs=[]
    for _ in range(c['max_len']-1):
        lp=distribution(m,mem,torch.tensor([ids],device=mem.device),c)[0]
        token=int(torch.distributions.Categorical(logits=lp).sample()) if sample else int(lp.argmax())
        logs.append(lp[token]); ids.append(token)
        if token==c['eos_id']:break
    return ids,torch.stack(logs).sum()

def tensor(row,data,device):
    with Image.open(data/row['file_name']) as im:return eval_transforms()(im.convert('RGB')).unsqueeze(0).to(device)

def metrics(rows,key='caption'):
    return compute_caption_metrics([r[key].lower().split() for r in rows],[r['reference'].lower().split() for r in rows])

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','candidates','rl','clip'])
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CP);p.add_argument('--data-root',type=Path,default=BASE/'Food500Cap_ready')
    p.add_argument('--out',type=Path,default=ROOT/'results');p.add_argument('--seed',type=int,default=20261121)
    p.add_argument('--steps',type=int,default=32);p.add_argument('--resume',type=Path);p.add_argument('--clip-model',default='openai/clip-vit-base-patch32')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);random.seed(a.seed);torch.manual_seed(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.mode=='prepare':
        allrows={s:read(a.data_root/'annotations'/f'{s}.jsonl') for s in ['train','val']}
        sets={}
        for s,n in [('train',64),('val',200)]:
            rows=sorted(allrows[s],key=lambda r:r['file_name']);random.Random(a.seed).shuffle(rows);rows=rows[:n]
            if len({r['file_name'] for r in rows})!=n:raise ValueError('Duplicate/missing IDs')
            for r in rows:
                with Image.open(a.data_root/r['file_name']) as im:im.verify()
            lines(a.out/f'{s}_subset.jsonl',rows);sets[s]=[r['file_name'] for r in rows]
        assert not set(sets['train'])&set(sets['val'])
        dump(a.out/'manifest.json',{'seed':a.seed,'checkpoint':str(a.checkpoint),'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'subsets':sets,'tune_ids':sets['val'][:100],'audit_ids':sets['val'][100:],'torch':torch.__version__,'device':str(device),'reward':'ROUGE-L beta=1.2, whitespace lower; not CIDEr','rl_lr':1e-6,'beam_width':4,'beam_score':'sum_logprob / generated_length including EOS','alpha_grid':[0,0.25,0.5,0.75,1]})
        print('Prepared fixed 64 train / 200 validation images',flush=True);return
    if a.mode=='clip':
        from rerank import run
        run(a);return
    m,s=load(a.checkpoint,device);c=s['config'];v=s['vocab'];rows=read(a.out/'val_subset.jsonl')
    if a.mode=='candidates':
        out=a.out/'candidates.jsonl';done=read(out) if out.exists() else [];seen={r['image_id'] for r in done}
        with out.open('a',encoding='utf-8') as f:
            for row in rows:
                if row['file_name'] in seen:continue
                with torch.no_grad():
                    mem=m.encode_image(tensor(row,a.data_root,device));g,gl=rollout(m,mem,c)
                    candidates=[{'token_ids':seq,'caption':text(seq,v),'sum_logprob':score,'length':len(seq)-1,'generation_score':score/(len(seq)-1)} for seq,score in beams(m,mem,c)]
                r={'image_id':row['file_name'],'file_name':row['file_name'],'reference':row['caption'],'greedy':text(g,v),'greedy_score':float(gl)/(len(g)-1),'candidates':candidates}
                f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();done.append(r)
                if len(done)%10==0:print('Candidates',len(done),flush=True)
        dump(a.out/'decode_metrics.json',{'n':len(done),'greedy':metrics(done,'greedy'),'beam':metrics([{**r,'caption':r['candidates'][0]['caption']} for r in done])});return
    # Eval mode disables dropout for both policies, while autograd remains enabled for sampled policy.
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-6)
    start=0
    if a.resume:
        rs=torch.load(a.resume,map_location=device,weights_only=False);m.load_trainable_state_dict(rs['trainable_state']);opt.load_state_dict(rs['optimizer']);torch.set_rng_state(rs['rng_cpu'].cpu());random.setstate(rs['rng_python']);start=rs['step']
        if torch.cuda.is_available() and rs.get('rng_cuda') is not None:torch.cuda.set_rng_state_all(rs['rng_cuda'])
    train=read(a.out/'train_subset.jsonl'); history=read(a.out/'rl_history.jsonl') if a.resume else []
    def evaluate():
        predictions=[]
        with torch.no_grad():
            for r in rows:
                ids,_=rollout(m,m.encode_image(tensor(r,a.data_root,device)),c)
                predictions.append({'file_name':r['file_name'],'reference':r['caption'],'caption':text(ids,v)})
        return predictions
    if not a.resume:
        before=evaluate();lines(a.out/'ce_control.jsonl',before);dump(a.out/'ce_metrics.json',metrics(before))
    for step in range(start,a.steps):
        r=train[step%len(train)];opt.zero_grad(set_to_none=True)
        mem=m.encode_image(tensor(r,a.data_root,device))
        with torch.no_grad():g,_=rollout(m,mem.detach(),c)
        sampled,logprob=rollout(m,mem,c,True);gt=text(g,v);st=text(sampled,v);ref=r['caption'].lower().split()
        rg=rouge_l(gt.split(),ref);rs=rouge_l(st.split(),ref);adv=rs-rg
        loss=-adv*logprob;loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        old=m.output_projection.weight.detach().clone();opt.step();delta=float((old-m.output_projection.weight).abs().max())
        record={'step':step+1,'image_id':r['file_name'],'sampled':st,'greedy':gt,'reference':r['caption'],'sample_reward':rs,'greedy_reward':rg,'advantage':adv,'loss':float(loss),'grad_norm':float(norm),'parameter_max_delta':delta}
        history.append(record);lines(a.out/'rl_history.jsonl',history)
        state={**s,'trainable_state':m.trainable_state_dict(),'optimizer':opt.state_dict(),'step':step+1,'rng_cpu':torch.get_rng_state(),'rng_python':random.getstate(),'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(state,a.out/'rl_last.pt');print('RL',step+1,'reward',rs,'grad',float(norm),flush=True)
    # Restore saved tensors and verify strict equivalence before final evaluation.
    saved=torch.load(a.out/'rl_last.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(t.cpu(),saved['trainable_state'][k]) for k,t in m.trainable_state_dict().items())
    m.load_trainable_state_dict(saved['trainable_state']);opt.load_state_dict(saved['optimizer'])
    after=evaluate();lines(a.out/'rl_after.jsonl',after);dump(a.out/'rl_metrics.json',{'n':len(after),'metrics':metrics(after),'steps':a.steps,'checkpoint_reload_equal':True,'nonzero_updates':sum(r['parameter_max_delta']>0 for r in history),'mean_sample_reward':sum(r['sample_reward'] for r in history)/len(history),'mean_greedy_reward':sum(r['greedy_reward'] for r in history)/len(history)})
if __name__=='__main__':main()
