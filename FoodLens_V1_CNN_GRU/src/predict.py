import argparse, json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from model import ImageCaptioner

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", default="outputs_v1/best_model.pt")
    ap.add_argument("--split", choices=["val","test"], default="test")
    ap.add_argument("--output", default="outputs_v1/test_predictions.jsonl")
    ap.add_argument("--batch-size", type=int, default=32)
    a=ap.parse_args()
    root=Path(a.data_root)
    ck=torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    vocab=ck["vocab"]; inv={int(v):k for k,v in vocab.items()}
    pad,bos,eos=vocab["<PAD>"],vocab["<BOS>"],vocab["<EOS>"]
    cfg=ck.get("config",{})
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ImageCaptioner(len(vocab), pretrained=False)
    model.load_state_dict(ck["model_state_dict"]); model.to(device).eval()
    size=int(cfg.get("image_size",224)); max_len=int(cfg.get("max_len",40))
    tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    ann=root/"annotations"/f"{a.split}.jsonl"
    records=[json.loads(x) for x in ann.read_text(encoding="utf-8").splitlines() if x.strip()]
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as f:
        for start in tqdm(range(0,len(records),a.batch_size),desc="生成描述"):
            batch=records[start:start+a.batch_size]; ims=[]
            for r in batch:
                with Image.open(root/r["file_name"]) as im: ims.append(tf(im.convert("RGB")))
            ims=torch.stack(ims).to(device)
            with torch.no_grad():
                feat=model.encoder(ims)
                seq=model.decoder.generate(feat,bos,eos,max_len).cpu().tolist()
            for r,ids in zip(batch,seq):
                words=[]
                for i in ids:
                    i=int(i)
                    if i==eos: break
                    if i not in (pad,bos): words.append(inv.get(i,"<UNK>"))
                pred=" ".join(words)
                for mark in [".",",","!","?",":",";",")"]: pred=pred.replace(" "+mark,mark)
                pred=pred.replace("( ","(")
                f.write(json.dumps({"file_name":r["file_name"],"cat":r.get("cat",""),
                    "reference":r["caption"],"prediction":pred},ensure_ascii=False)+"\\n")
    print("预测结果已保存：",out.resolve())

if __name__=="__main__": main()
