"""Train V2 or a fresh V1 ablation on the leader's identical prepared data.

V1 data transforms, collator and BLEU implementation are imported unchanged.
No raw-data fallback: inventing a split or vocabulary would break comparison.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'v1_cnn_gru'))
from data import FoodCapDataset, load_vocab, train_transforms, eval_transforms, CollateFn
from bleu import corpus_bleu
from model import CaptionModel
from model_v2 import CaptionModelV2


def check_ready(root):
    required=[root/'vocab.json']+[root/'annotations'/f'{s}.jsonl' for s in ['train','val','test']]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing prepared V1 data. Ask leader for the same split and vocabulary:\n'+'\n'.join(missing))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in required}


@torch.no_grad()
def evaluate(model, loader, vocab, device, max_len, limit):
    model.eval(); hyps=[]; refs=[]
    for images, _, _, raw in loader:
        hyps.extend(model.greedy_decode(images.to(device),vocab,max_len=max_len))
        refs.extend([r.lower().split() for r in raw])
        if len(hyps)>=limit:
            break
    return corpus_bleu(hyps[:limit],refs[:limit])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--variant',choices=['attention','v1_ablation'],default='attention')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--embed-dim',type=int,default=256)
    p.add_argument('--hidden-size',type=int,default=512)
    p.add_argument('--lr-enc',type=float,default=1e-4)
    p.add_argument('--lr-dec',type=float,default=1e-3)
    p.add_argument('--max-len',type=int,default=30)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--val-num',type=int,default=500)
    p.add_argument('--patience',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--smoke',action='store_true',help='20 real-data optimizer steps on one fixed batch')
    a=p.parse_args()
    hashes=check_ready(a.root)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab=load_vocab(a.root/'vocab.json'); pad=vocab['<PAD>']
    run=a.run_dir or Path(__file__).resolve().parent/'runs'/('v2' if a.variant=='attention' else 'v1_ablation')
    if a.smoke: run=run/'smoke'
    run.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(data_sha256=hashes,device=str(device),torch=torch.__version__,vocab_size=len(vocab),dropout=.5,gradient_clip=5.)
    (run/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    ds=FoodCapDataset(str(a.root),'train',vocab,max_len=a.max_len,transform=train_transforms())
    val=FoodCapDataset(str(a.root),'val',vocab,max_len=a.max_len,transform=eval_transforms())
    collate=CollateFn(pad)
    train=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,
                     collate_fn=collate,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    validation=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,
                          collate_fn=collate,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    model_class=CaptionModelV2 if a.variant=='attention' else CaptionModel
    model=model_class(len(vocab),a.embed_dim,a.hidden_size,pad,enc_finetune=True).to(device)
    optimizer=torch.optim.Adam([{'params':model.encoder.parameters(),'lr':a.lr_enc},
                                {'params':model.decoder.parameters(),'lr':a.lr_dec}])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2)
    criterion=nn.CrossEntropyLoss(ignore_index=pad)
    def update(batch):
        images,captions,lengths,_=batch; images=images.to(device);captions=captions.to(device)
        logits=model(images,captions,lengths)
        loss=criterion(logits.reshape(-1,len(vocab)),captions[:,1:].reshape(-1))
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        return loss.item()
    if a.smoke:
        batch=next(iter(train)); model.train(); losses=[]
        for step in range(20):
            losses.append(update(batch));print(f'real-data smoke {step+1}/20 loss={losses[-1]:.4f}',flush=True)
        passed=np.mean(losses[-5:])<np.mean(losses[:5])
        (run/'smoke.json').write_text(json.dumps(dict(losses=losses,passed=bool(passed)),indent=2),encoding='utf-8')
        if not passed: raise RuntimeError('Real-data smoke loss did not decrease')
        return
    best=-1.;bad=0;history=[]
    for epoch in range(1,a.epochs+1):
        model.train();start=time.time();total=0.;count=0
        for batch in train:
            total+=update(batch);count+=1
        bleus=evaluate(model,validation,vocab,device,a.max_len,a.val_num)
        scheduler.step(bleus[3])
        entry=dict(epoch=epoch,train_loss=total/count,val_bleu1=bleus[0],val_bleu4=bleus[3],val_bleu=bleus,minutes=(time.time()-start)/60)
        history.append(entry);(run/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps(entry),flush=True)
        if bleus[3]>best:
            best=bleus[3];bad=0
            checkpoint='best_v2.pt' if a.variant=='attention' else 'best_v1_ablation.pt'
            torch.save(dict(model=model.state_dict(),vocab=vocab,args=config,epoch=epoch,val_bleu4=best,
                            variant=a.variant,optimizer=optimizer.state_dict()),run/checkpoint)
        else:
            bad+=1
            if bad>=a.patience:break


if __name__=='__main__':main()
