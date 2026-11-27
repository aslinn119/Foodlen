"""V2 vocabulary repair experiment. Train-only vocabulary; full-val selection."""
import os,json,re,random,time,hashlib,math,argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18
from PIL import Image
from model_v2 import CaptionModelV2
ROOT=Path(__file__).resolve().parent
DATA=Path.home()/"Desktop/神经网络与深度学习/Food500Cap_ready"
SPECIAL=["<PAD>","<BOS>","<EOS>","<UNK>"]
def tokenize(text):return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*",text.lower())
def read(split):return [json.loads(l) for l in (DATA/f"annotations/{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
def dump(path,obj):
 path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Images(Dataset):
 def __init__(self,split,vocab):
  self.rows=read(split);self.vocab=vocab
  self.tf=T.Compose([T.Resize(256),T.RandomCrop(224) if split=="train" else T.CenterCrop(224)]+([T.RandomHorizontalFlip()] if split=="train" else [])+[T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(DATA/r["file_name"]) as im:x=self.tf(im.convert("RGB"))
  ids=[1]+[self.vocab.get(t,3) for t in tokenize(r["caption"])]+[2]
  return x,torch.tensor(ids),r["file_name"],r["caption"]
def collate(batch):
 imgs,ids,names,refs=zip(*batch)
 return torch.stack(imgs),nn.utils.rnn.pad_sequence(ids,batch_first=True,padding_value=0),[len(s) for s in ids],names,refs
@torch.no_grad()
def predict(model,loader,device,vocab):
 model.eval();rows=[]
 for imgs,_,_,names,refs in loader:
  words=model.greedy_decode(imgs.to(device),vocab,max_len=67)
  rows.extend(dict(file_name=n,caption=" ".join(w),reference=r) for n,w,r in zip(names,words,refs))
 return rows
def metrics(rows,normalized=True):
 from pycocoevalcap.bleu.bleu import Bleu
 from pycocoevalcap.rouge.rouge import Rouge
 from pycocoevalcap.cider.cider import Cider
 import contextlib,io
 tok=tokenize if normalized else lambda t:t.lower().split()
 gt={i:[" ".join(tok(r["reference"]))] for i,r in enumerate(rows)}
 hyp={i:[" ".join(tok(r["caption"]))] for i,r in enumerate(rows)}
 with contextlib.redirect_stdout(io.StringIO()):
  b,_=Bleu(4).compute_score(gt,hyp,verbose=0);rl,_=Rouge().compute_score(gt,hyp);c,_=Cider().compute_score(gt,hyp)
 return dict(n=len(rows),bleu=[100*x for x in b],rouge_l=rl,cider_d=c,unknown_count=sum("<unk>" in r["caption"].lower() for r in rows),repeated_trigram_count=sum(any(v>1 for v in Counter(tuple(w[i:i+3]) for i in range(len(w)-2)).values()) for w in [r["caption"].lower().split() for r in rows]))
def save_rows(p,rows):p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
def main():
 a=argparse.ArgumentParser();a.add_argument("--epochs",type=int,default=25);a.add_argument("--resume",action="store_true");args=a.parse_args()
 torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 counts=Counter(t for r in read("train") for t in tokenize(r["caption"]))
 vocab={w:i for i,w in enumerate(SPECIAL+sorted(counts))};dump(ROOT/"vocab.json",vocab)
 audit={s:dict(n=len(read(s)),oov_tokens=sum(t not in vocab for r in read(s) for t in tokenize(r["caption"])),oov_sentences=sum(any(t not in vocab for t in tokenize(r["caption"])) for r in read(s))) for s in ["train","val","test"]}
 dump(ROOT/"data_audit.json",audit)
 cfg=dict(seed=42,vocab_size=len(vocab),max_generation_tokens=67,tokenizer="lowercase regex words, internal apostrophe/hyphen retained",vocab_source="train only, min_frequency=1",architecture="unchanged ResNet18 spatial additive attention GRU",batch_size=32,lr_encoder=1e-4,lr_decoder=1e-3,epochs=args.epochs,patience=6,selection="full validation normalized BLEU4",device=str(device),torch=str(torch.__version__),data_root=str(DATA),sha256={s:sha(DATA/f"annotations/{s}.jsonl") for s in ["train","val","test"]})
 dump(ROOT/"config.json",cfg)
 loaders={s:DataLoader(Images(s,vocab),batch_size=32,shuffle=s=="train",num_workers=2,collate_fn=collate,pin_memory=device.type=="cuda",persistent_workers=True) for s in ["train","val","test"]}
 model=CaptionModelV2(len(vocab),pretrained=False).to(device)
 pre=Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
 net=resnet18(weights=None);net.load_state_dict(torch.load(pre,map_location="cpu",weights_only=True))
 model.encoder.backbone.load_state_dict(nn.Sequential(*list(net.children())[:-2]).state_dict(),strict=True)
 del net
 optimizer=torch.optim.Adam([dict(params=model.encoder.parameters(),lr=1e-4),dict(params=model.decoder.parameters(),lr=1e-3)])
 scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=2)
 criterion=nn.CrossEntropyLoss(ignore_index=0);best=-1;bad=0;history=[];start=1
 if args.resume:
  c=torch.load(ROOT/"last.pt",map_location=device,weights_only=False)
  assert c["vocab"]==vocab
  model.load_state_dict(c["model"]);optimizer.load_state_dict(c["optimizer"]);scheduler.load_state_dict(c["scheduler"])
  best=c["best"];bad=c["bad"];history=c["history"];start=c["epoch"]+1
  random.setstate(c["python_rng"]);np.random.set_state(c["numpy_rng"]);torch.set_rng_state(c["torch_rng"].cpu())
  if device.type=="cuda":torch.cuda.set_rng_state_all([x.cpu() for x in c["cuda_rng"]])
 print("TRAIN",cfg,flush=True)
 for epoch in range(start,args.epochs+1):
  before=time.time();model.train();total=0;nt=0
  for step,(imgs,caps,lengths,_,_) in enumerate(loaders["train"]):
   imgs=imgs.to(device);caps=caps.to(device);optimizer.zero_grad(set_to_none=True)
   logits=model(imgs,caps,lengths);loss=criterion(logits.reshape(-1,len(vocab)),caps[:,1:].reshape(-1))
   if not torch.isfinite(loss):raise ValueError("nonfinite training loss")
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
   n=int((caps[:,1:]!=0).sum());total+=loss.item()*n;nt+=n
   if step%100==0:print("STEP",epoch,step,float(loss),flush=True)
  rows=predict(model,loaders["val"],device,vocab);score=metrics(rows);scheduler.step(score["bleu"][3])
  history.append(dict(epoch=epoch,train_loss=total/nt,val=score,seconds=time.time()-before))
  dump(ROOT/"history.json",history);print("EPOCH",json.dumps(history[-1]),flush=True)
  if score["bleu"][3]>best:
   best=score["bleu"][3];bad=0
   torch.save(dict(model=model.state_dict(),vocab=vocab,epoch=epoch,config=cfg,val=score),ROOT/"best_v2.pt")
   save_rows(ROOT/"best_val_predictions.jsonl",rows)
  else:bad+=1
  checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best=best,bad=bad,history=history,vocab=vocab,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else [])
  torch.save(checkpoint,ROOT/"last.tmp");os.replace(ROOT/"last.tmp",ROOT/"last.pt")
  if bad>=6:break
 c=torch.load(ROOT/"best_v2.pt",map_location=device,weights_only=False);model.load_state_dict(c["model"])
 final={}
 for split in ["val","test"]:
  rows=predict(model,loaders[split],device,vocab);save_rows(ROOT/f"{split}_predictions.jsonl",rows)
  final[split]=dict(normalized=metrics(rows),legacy_whitespace=metrics(rows,False))
  source=ROOT.parent/"Step1_修复与复评_2026-11-17/task1"
  oldpath=source/"V2_val_fresh.jsonl" if split=="val" else source/"inputs/v2_greedy_test.json"
  if oldpath.suffix==".jsonl":old=[json.loads(l) for l in oldpath.read_text(encoding="utf-8").splitlines()]
  else:
   obj=json.loads(oldpath.read_text(encoding="utf-8"));old=obj if isinstance(obj,list) else obj["predictions"]
   old=[dict(r,caption=r.get("hypothesis",r.get("caption"))) for r in old]
  assert {r["file_name"] for r in old}=={r["file_name"] for r in rows}
  final[split]["old_v2_same_normalization"]=metrics(old)
  final[split]["old_v2_legacy"]=metrics(old,False)
 final["best_epoch"]=c["epoch"];final["weight_sha256"]=sha(ROOT/"best_v2.pt")
 dump(ROOT/"final_results.json",final)
 report=["# V2词表修复重训结果","",f"最佳轮次：{c['epoch']}；词表{len(vocab)}项，只从训练集构建；没有继承旧V2训练权重。视觉骨干采用原ImageNet预训练ResNet18。", "", "| 划分 | 模型 | 新分词BLEU-4 % | 原空格口径BLEU-4 % | 含UNK描述 |", "|---|---|---:|---:|---:|"]
 for s in ["val","test"]:
  for label,key,legacy in [("旧V2","old_v2_same_normalization","old_v2_legacy"),("新V2","normalized","legacy_whitespace")]:
   report.append(f"| {s} | {label} | {final[s][key]['bleu'][3]:.4f} | {final[s][legacy]['bleu'][3]:.4f} | {final[s][legacy]['unknown_count']} |")
 report += ["", "同一列内才是相同分词口径的对比。完整验证集选权重；测试只在训练结束后评价。单随机种子，不能声称统计显著。训练目标取消标点并延长句长，因此不是只改变词表大小的单变量实验。", "", "最终权重：best_v2.pt；推理见predict.py；完整指标见final_results.json；训练曲线数据见history.json。"]
 (ROOT/"训练结果.md").write_text("\n".join(report),encoding="utf-8")
 dump(ROOT/"status.json",dict(status="complete",best_epoch=c["epoch"],results="final_results.json"))
 print("COMPLETE",final,flush=True)
if __name__=="__main__":main()
