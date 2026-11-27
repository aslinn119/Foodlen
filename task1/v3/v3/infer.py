import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data import FoodCaptionDataset, make_collate, build_transform, find_special
from model import FoodLensV3
from PIL import Image


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root')
    p.add_argument('--split',default='val')
    p.add_argument('--limit',type=int,default=16,help='0 for full split')
    p.add_argument('--image')
    p.add_argument('--output',default='outputs/v3_predictions.jsonl')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--max-len',type=int,default=30)
    args=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    vocab=cp['vocab']; inv={i:t for t,i in vocab.items()}
    model=FoodLensV3(**cp['model_config']); model.load_state_dict(cp['model_state'])
    model.to(device).eval()
    bos=find_special(vocab,'bos'); eos=find_special(vocab,'eos'); pad=find_special(vocab,'pad')
    def decode(seq):
        words=[]
        for ix in seq:
            ix=int(ix)
            if ix==eos: break
            if ix not in (bos,pad): words.append(inv.get(ix,'<unk>'))
        return ' '.join(words)
    if args.image:
        with Image.open(args.image) as im:
            img=build_transform()(im.convert('RGB')).unsqueeze(0).to(device)
        print(decode(model.greedy(img,bos,eos,args.max_len)[0].tolist()))
        return
    if not args.data_root: p.error('--data-root required unless --image is provided')
    ds=FoodCaptionDataset(args.data_root,args.split,vocab,args.max_len, args.limit or None)
    loader=DataLoader(ds,batch_size=args.batch_size,collate_fn=make_collate(pad))
    outfile=Path(args.output);outfile.parent.mkdir(parents=True,exist_ok=True)
    with outfile.open('w',encoding='utf-8') as f:
        for images, _, names, raw_captions in loader:
            seq=model.greedy(images.to(device),bos,eos,args.max_len).cpu().tolist()
            for name,s,ref in zip(names,seq,raw_captions):
                f.write(json.dumps({'file_name':name,'caption':decode(s),'reference':ref,'model_id':'v3'},ensure_ascii=False)+'\n')
    print('Wrote',outfile,len(ds),'predictions')

if __name__=='__main__': main()
