"""Fixed-budget SCST and continued-CE control, with frozen-backbone feature caching."""
import argparse, collections, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from model import ViTCaptioner
from v1_data import eval_transforms

HERE=Path(__file__).resolve().parent
def read(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def lines(p,rows): Path(p).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def grams(t):
    return [collections.Counter(tuple(t[i:i+n]) for i in range(len(t)-n+1)) for n in range(1,5)]

class FixedCider:
    """CIDEr-D x10; document frequencies fixed on training references, never on a minibatch."""
    def __init__(self,refs):
        self.df=collections.Counter();self.logn=math.log(len(refs))
        for r in refs:
            for g in grams(r.lower().split()): self.df.update(g.keys())
    def vec(self,t):
        return [{g:count*(self.logn-math.log(max(1,self.df[g]))) for g,count in c.items()} for c in grams(t)]
    def score(self,h,r):
        ht=h.lower().split();rt=r.lower().split();hv=self.vec(ht);rv=self.vec(rt)
        penalty=math.exp(-((max(0,len(ht)-1)-max(0,len(rt)-1))**2)/72.)
        scores=[]
        for h,r in zip(hv,rv):
            a=math.sqrt(sum(x*x for x in h.values()));b=math.sqrt(sum(x*x for x in r.values()))
            dot=sum(min(x,r.get(g,0))*r.get(g,0) for g,x in h.items())
            scores.append(dot/(a*b) if a*b else 0.)
        return 10*sum(scores)/4*penalty

def captions(ids,v):
    inv={i:t for t,i in v.items()};out=[]
    for row in ids.tolist():
        words=[]
        for i in row[1:]:
            if i==v['<EOS>']:break
            if i!=v['<PAD>']:words.append(inv[i])
        out.append(' '.join(words))
    return out

@torch.no_grad()
def rollout(m,mem,c,sample=False):
    b=mem.shape[0];ids=torch.full((b,1),c['bos_id'],device=mem.device,dtype=torch.long)
    done=torch.zeros(b,device=mem.device,dtype=torch.bool)
    for _ in range(c['max_len']-1):
        logits=m.decode(mem,ids)[:,-1,:].clone();logits[:,c['pad_id']]=-torch.inf;logits[:,c['bos_id']]=-torch.inf
        tok=torch.multinomial(logits.softmax(-1),1).squeeze(1) if sample else logits.argmax(-1)
        tok=tok.masked_fill(done,c['pad_id']);ids=torch.cat([ids,tok[:,None]],1)
        done|=tok.eq(c['eos_id'])
        if done.all():break
    return ids

def sequence_logprobs(m,mem,ids,c):
    logits=m.decode(mem,ids[:,:-1]).clone();logits[:,:,c['pad_id']]=-torch.inf;logits[:,:,c['bos_id']]=-torch.inf
    chosen=logits.log_softmax(-1).gather(-1,ids[:,1:,None]).squeeze(-1)
    return chosen.masked_fill(ids[:,1:].eq(c['pad_id']),0).sum(1)

def score_rows(rows):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    g={i:[r['reference'].lower()] for i,r in enumerate(rows)};h={i:[r['caption'].lower()] for i,r in enumerate(rows)}
    b,_=Bleu(4).compute_score(g,h,verbose=0);r,_=Rouge().compute_score(g,h);c,per=Cider().compute_score(g,h)
    ts=[x['caption'].split() for x in rows]
    return {'n':len(rows),'bleu_1_to_4_percent':[float(x)*100 for x in b],'rouge_l':float(r),'cider_d':float(c),
      'unk_image_rate':sum('<UNK>' in t for t in ts)/len(ts),
      'repeated_trigram_image_rate':sum(len(list(zip(t,t[1:],t[2:])))>len(set(zip(t,t[1:],t[2:]))) for t in ts)/len(ts),
      'mean_words':float(np.mean([len(t) for t in ts]))},[float(x) for x in per]

def save_state(path,m,opt,epoch,step,base):
    tmp=path.with_suffix('.tmp')
    torch.save({'trainable_state':m.trainable_state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'step':step,
      'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_python':random.getstate(),
      'config':base['config'],'vocab':base['vocab']},tmp)
    os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--backbone',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True,help='Frozen 100-image visual annotations from Innovation1')
    p.add_argument('--out',type=Path,default=HERE/'results');p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--train-size',type=int,default=2048);p.add_argument('--val-size',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=20261127);p.add_argument('--lr',type=float,default=1e-6)
    p.add_argument('--mode',choices=['all','tests'],default='all');a=p.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('Formal run requires CUDA; refusing silent CPU fallback')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    a.out.mkdir(parents=True,exist_ok=True);cache=a.out/'local_cache';cache.mkdir(exist_ok=True)
    base=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=base['config'];v=base['vocab']
    kw={k:c[k] for k in ['pad_id','max_len','d_model','nhead','num_decoder_layers','dim_feedforward','dropout']}
    m=ViTCaptioner(len(v),**kw,pretrained=False)
    backbone=torch.load(a.backbone,map_location='cpu',weights_only=True)
    m.vit.load_state_dict({k:t for k,t in backbone.items() if not k.startswith('heads.')},strict=True)
    m.load_trainable_state_dict(base['trainable_state']);m=m.cuda().eval()
    allrows={s:read(a.data_root/'annotations'/f'{s}.jsonl') for s in ['train','val','test']}
    for s,rr in allrows.items():
        ids=[r['file_name'] for r in rr]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate '+s)
    assert not(set(r['file_name'] for r in allrows['train']) & set(r['file_name'] for r in allrows['val']))
    ann=read(a.annotations);assert len(ann)==100
    audit_ids={r['image_id'] for r in ann};val_by={r['file_name']:r for r in allrows['val']}
    audit=[val_by[r['image_id']] for r in ann]
    train=sorted(allrows['train'],key=lambda r:r['file_name']);random.Random(a.seed).shuffle(train);train=train[:a.train_size]
    val=sorted([r for r in allrows['val'] if r['file_name'] not in audit_ids],key=lambda r:r['file_name'])
    random.Random(a.seed+1).shuffle(val);val=val[:a.val_size]
    sets={'train':train,'selection_val':val,'entity_audit':audit,'test':allrows['test']}
    protocol={'seed':a.seed,'train_n':len(train),'selection_val_n':len(val),'entity_audit_n':100,'test_n':len(allrows['test']),
      'epochs':a.epochs,'batch_size':a.batch_size,'updates_per_arm':a.epochs*math.ceil(len(train)/a.batch_size),'lr':a.lr,
      'reward':'CIDEr-D x10; fixed DF from all training references only; lowercase whitespace',
      'selection':'maximum selection_val CIDEr-D, ties select earliest epoch; no audit/test model selection',
      'ce_control':'original frozen CE checkpoint plus continued CE arm with identical examples, batches and Adam learning rate',
      'dropout':'eval mode in both training arms, to align sampled and scored SCST policies',
      'precision':'float32 raw frozen-ViT features and decoder; TF32 disabled; no AMP',
      'checkpoint_sha256':sha(a.checkpoint),'backbone_sha256':sha(a.backbone),'annotations_sha256':sha(a.annotations),
      'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
      'test_history':'Test set has previously been inspected by historical project work; this is retrospective evaluation, not a pristine benchmark.',
      'entity_annotation_source':'assistant-assisted labels confirmed by user in prior chat; not independent blind human ground truth',
      'single_seed':True,'created_before_training':True}
    protocol_path=a.out/'protocol.json'
    if protocol_path.exists():
        old=json.loads(protocol_path.read_text(encoding='utf-8'))
        if old!=protocol:raise ValueError('Existing output protocol differs; use a fresh output directory')
    else:dump(protocol_path,protocol)
    for k,rr in sets.items():lines(a.out/f'{k}_inputs.jsonl',rr)
    lines(a.out/'entity_annotations.jsonl',ann)
    # Tests check the reward against the established scorer and full-prefix gradient scoring.
    from pycocoevalcap.cider.cider import Cider
    refs=['a red apple on a plate','a bowl of rice and meat','two bananas on the table','a glass of milk']
    hyps=['a red apple','rice and meat','two bananas','milk on a table']
    rew=FixedCider(refs);official,_=Cider().compute_score({i:[r] for i,r in enumerate(refs)},{i:[h] for i,h in enumerate(hyps)})
    local=np.mean([rew.score(h,r) for h,r in zip(hyps,refs)])
    assert abs(local-official)<1e-10,(local,official)
    mem=torch.randn(2,50,c['d_model'],device='cuda')*.01
    ids=torch.tensor([[c['bos_id'],4,c['eos_id'],c['pad_id']],[c['bos_id'],5,6,c['eos_id']]],device='cuda')
    parallel=sequence_logprobs(m,mem,ids,c);sequential=torch.zeros(2,device='cuda')
    for j in range(1,ids.shape[1]):
        z=m.decode(mem,ids[:,:j])[:,-1,:].clone();z[:,c['pad_id']]=-torch.inf;z[:,c['bos_id']]=-torch.inf
        token_lp=z.log_softmax(-1).gather(1,ids[:,j,None]).squeeze(1).masked_fill(ids[:,j].eq(c['pad_id']),0)
        sequential+=token_lp
    err=float((parallel-sequential).abs().max());assert err<2e-5,err
    loss=-parallel.mean();loss.backward();assert torch.isfinite(m.output_projection.weight.grad).all();m.zero_grad(set_to_none=True)
    dump(a.out/'implementation_checks.json',{'fixed_df_reward_matches_pycocoevalcap':True,'reward_error':abs(float(local-official)),
      'teacher_forced_vs_sequential_max_error':err,'eos_scored_pad_excluded':True,'finite_policy_gradient':True})
    print('Implementation checks passed',flush=True)
    if a.mode=='tests':return
    reward=FixedCider([r['caption'] for r in allrows['train']])
    transform=eval_transforms();features={}
    image_hashes={};cross_duplicates=[]
    for split,rr in sets.items():
        # Detect duplicate bytes across train/evaluation domains without altering the official split.
        for r in rr:
            h=sha(a.data_root/r['file_name']);previous=image_hashes.get(h)
            if previous and (previous[0]=='train')!=(split=='train'):cross_duplicates.append([previous,[split,r['file_name']]])
            image_hashes.setdefault(h,[split,r['file_name']])
        fp=cache/f'{split}_features.pt'
        ids=[r['file_name'] for r in rr]
        if fp.exists():
            data=torch.load(fp,weights_only=False);assert data['ids']==ids;assert data['backbone_sha256']==protocol['backbone_sha256']
            features[split]=data['features'];continue
        chunks=[]
        for start in range(0,len(rr),32):
            ims=[]
            for r in rr[start:start+32]:
                with Image.open(a.data_root/r['file_name']) as im:ims.append(transform(im.convert('RGB')))
            with torch.no_grad():
                x=m.vit._process_input(torch.stack(ims).cuda());x=torch.cat([m.vit.class_token.expand(len(ims),-1,-1),x],1)
                chunks.append(m.vit.encoder(x).cpu())
            if start%512==0:print('Cache',split,start,'/',len(rr),flush=True)
        features[split]=torch.cat(chunks);torch.save({'ids':ids,'features':features[split],'backbone_sha256':protocol['backbone_sha256']},fp)
    dump(a.out/'data_audit.json',{'readable_images':sum(len(x) for x in sets.values()),'train_eval_identical_byte_pairs':cross_duplicates,
      'selection_audit_overlap':len({r['file_name'] for r in val}&audit_ids),'train_unique':len(train),'note':'byte equality, not perceptual near-duplicate detection'})
    if cross_duplicates:raise ValueError('Train/evaluation image duplicate bytes; resolve protocol before training')

    def evaluate(split,label):
        rr=sets[split];pred=[]
        for start in range(0,len(rr),16):
            with torch.no_grad():cap=captions(rollout(m,m.visual_projection(features[split][start:start+16].cuda()),c),v)
            pred.extend({'image_id':r['file_name'],'reference':r['caption'],'caption':h} for r,h in zip(rr[start:start+16],cap))
        met,per=score_rows(pred)
        for r,s in zip(pred,per):r['cider_d']=s
        lines(a.out/f'{label}_{split}_predictions.jsonl',pred);dump(a.out/f'{label}_{split}_metrics.json',met)
        print('Evaluate',label,split,met,flush=True);return met

    # Original CE is frozen; all final comparisons use identical evaluation code.
    if not (a.out/'ce_original_test_metrics.json').exists():
        for split in ['selection_val','entity_audit','test']:evaluate(split,'ce_original')
    for arm in ['ce_continued','scst']:
        m.load_trainable_state_dict(base['trainable_state']);m.eval()
        opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=a.lr)
        history_path=a.out/f'{arm}_history.jsonl';last=cache/f'{arm}_last.pt';bestfile=cache/f'{arm}_best.pt'
        history=read(history_path) if history_path.exists() else [];epoch_start=0;step=0
        summaries=json.loads((a.out/f'{arm}_epochs.json').read_text()) if (a.out/f'{arm}_epochs.json').exists() else []
        if last.exists():
            saved=torch.load(last,map_location='cpu',weights_only=False);m.load_trainable_state_dict(saved['trainable_state']);opt.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['rng_cpu']);torch.cuda.set_rng_state_all(saved['rng_cuda']);random.setstate(saved['rng_python'])
            epoch_start=saved['epoch'];step=saved['step']
            history=[r for r in history if r['step']<=step]
            # A partially written next epoch is not additional training evidence.
            if history_path.exists():
                import shutil
                shutil.copy2(history_path,cache/f'{arm}_history_before_resume.jsonl')
                lines(history_path,history)
            summaries=[r for r in summaries if r['epoch']<=epoch_start]
        else:torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        best=max([r['metrics']['cider_d'] for r in summaries],default=-1)
        for epoch in range(epoch_start,a.epochs):
            order=list(range(len(train)));random.Random(a.seed+epoch).shuffle(order);started=time.perf_counter()
            for start in range(0,len(order),a.batch_size):
                indices=order[start:start+a.batch_size];raw=features['train'][indices].cuda();rr=[train[i] for i in indices]
                opt.zero_grad(set_to_none=True);mem=m.visual_projection(raw)
                record={'epoch':epoch+1,'step':step+1,'image_ids':[r['file_name'] for r in rr]}
                if arm=='scst':
                    sampled=rollout(m,mem.detach(),c,True);greedy=rollout(m,mem.detach(),c)
                    st=captions(sampled,v);gt=captions(greedy,v)
                    rs=[reward.score(h,r['caption']) for h,r in zip(st,rr)];rg=[reward.score(h,r['caption']) for h,r in zip(gt,rr)]
                    adv=torch.tensor(np.array(rs)-np.array(rg),device='cuda',dtype=torch.float32)
                    lp=sequence_logprobs(m,mem,sampled,c);loss=-(adv*lp).mean()
                    record.update(sample_reward=float(np.mean(rs)),greedy_reward=float(np.mean(rg)),advantage=float(adv.mean()),
                      positive_advantage_count=int((adv>0).sum()),sampled_captions=st,greedy_captions=gt)
                else:
                    encoded=[]
                    for r in rr:encoded.append([v['<BOS>']]+[v.get(w,v['<UNK>']) for w in r['caption'].lower().split()][:c['max_len']-2]+[v['<EOS>']])
                    ids=torch.zeros(len(rr),max(map(len,encoded)),device='cuda',dtype=torch.long)
                    for i,x in enumerate(encoded):ids[i,:len(x)]=torch.tensor(x,device='cuda')
                    logits=m.decode(mem,ids[:,:-1]);loss=torch.nn.functional.cross_entropy(logits.reshape(-1,len(v)),ids[:,1:].reshape(-1),ignore_index=c['pad_id'])
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.)
                if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Nonfinite loss/gradient')
                old=m.output_projection.weight.detach().clone();opt.step();delta=float((m.output_projection.weight-old).abs().max())
                step+=1;record.update(loss=float(loss),grad_norm=float(norm),output_projection_max_delta=delta,elapsed_epoch_seconds=time.perf_counter()-started)
                history.append(record)
                with history_path.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                if step%16==0:print(arm,'epoch',epoch+1,'step',step,'loss',float(loss),'sec',round(time.perf_counter()-started,1),flush=True)
            met=evaluate('selection_val',f'{arm}_epoch{epoch+1}')
            summaries.append({'epoch':epoch+1,'step':step,'metrics':met,'epoch_seconds':time.perf_counter()-started})
            dump(a.out/f'{arm}_epochs.json',summaries)
            if met['cider_d']>best:
                best=met['cider_d'];save_state(bestfile,m,opt,epoch+1,step,base)
            save_state(last,m,opt,epoch+1,step,base)
            saved=torch.load(last,map_location='cpu',weights_only=False)
            assert all(torch.equal(t.cpu(),saved['trainable_state'][k]) for k,t in m.trainable_state_dict().items())
        saved=torch.load(bestfile,map_location='cpu',weights_only=False);m.load_trainable_state_dict(saved['trainable_state'])
        for split in ['entity_audit','test']:evaluate(split,arm)
        dump(a.out/f'{arm}_completion.json',{'steps':step,'best_epoch':saved['epoch'],'best_validation_cider_d':best,
          'nonzero_updates':sum(r['output_projection_max_delta']>0 for r in history),'finite_losses':all(math.isfinite(r['loss']) for r in history),
          'checkpoint_sha256':sha(bestfile),'checkpoint_reload_tensor_equal':True,'single_seed':True})
    print('ALL TRAINING AND INFERENCE COMPLETE',flush=True)

if __name__=='__main__':main()
