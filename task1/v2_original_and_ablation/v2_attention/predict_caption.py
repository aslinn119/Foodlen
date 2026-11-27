"""V2 inference API and word-aligned attention overlays on the actual crop."""
import argparse
import json
import sys
from pathlib import Path
import torch
from PIL import Image
from model_v2 import CaptionModelV2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'v1_cnn_gru'))
from data import eval_transforms


class CaptionPredictor:
    def __init__(self,checkpoint,device=None):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        ck=torch.load(checkpoint,map_location=self.device,weights_only=False)
        if ck.get('variant')!='attention':raise ValueError('Expected V2 attention checkpoint')
        self.vocab=ck['vocab'];args=ck['args'];self.max_len=args['max_len']
        self.model=CaptionModelV2(len(self.vocab),args['embed_dim'],args['hidden_size'],self.vocab['<PAD>'],pretrained=False).to(self.device)
        self.model.load_state_dict(ck['model']);self.model.eval()
        self.transform=eval_transforms()

    def caption(self,image_path,decode='greedy',beam_size=3):
        image=self.transform(Image.open(image_path).convert('RGB')).to(self.device)
        if decode=='greedy':words=self.model.greedy_decode(image[None],self.vocab,self.max_len)[0]
        elif decode=='beam':
            inv={i:w for w,i in self.vocab.items()}
            words=[inv[i] for i in self.model.beam_decode(image,self.vocab,beam_size,self.max_len)]
        else:raise ValueError(decode)
        return ' '.join(words)

    @torch.no_grad()
    def visualize(self,image_path,out_dir):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
        tensor=self.transform(Image.open(image_path).convert('RGB')).to(self.device)
        sequences,maps=self.model.greedy_decode(tensor[None],self.vocab,self.max_len,True)
        words=sequences[0];weights=maps[0]
        mean=torch.tensor([.485,.456,.406])[:,None,None]
        std=torch.tensor([.229,.224,.225])[:,None,None]
        crop=(tensor.cpu()*std+mean).clamp(0,1).permute(1,2,0).numpy()
        # Overlay on CenterCrop(224) itself; overlaying on raw image misaligns.
        count=min(8,len(words));cols=3;rows=(count+1+cols-1)//cols
        fig,axes=plt.subplots(rows,cols,figsize=(cols*3,rows*3),squeeze=False)
        for axis in axes.flat:axis.axis('off')
        axes.flat[0].imshow(crop);axes.flat[0].set_title('Actual model input crop')
        for i in range(count):
            heat=torch.nn.functional.interpolate(weights[i].reshape(1,1,7,7),size=(224,224),mode='bilinear',align_corners=False)[0,0].numpy()
            axis=axes.flat[i+1];axis.imshow(crop);axis.imshow(heat,cmap='jet',alpha=.45);axis.set_title(f'{i+1}: {words[i]}')
        fig.tight_layout();fig.savefig(out/'attention.png',dpi=180);plt.close(fig)
        np.savez_compressed(out/'attention_weights.npz',weights=weights.numpy(),tokens=np.array(words))
        (out/'caption.json').write_text(json.dumps(dict(image=str(image_path),caption=' '.join(words),tokens=words,coordinates='Resize256 CenterCrop224'),ensure_ascii=False,indent=2),encoding='utf-8')
        return ' '.join(words)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True)
    p.add_argument('--decode',choices=['greedy','beam'],default='greedy');p.add_argument('--attention-dir')
    a=p.parse_args();predictor=CaptionPredictor(a.checkpoint)
    if a.attention_dir:
        if a.decode!='greedy':p.error('Attention overlay is for greedy decoding only')
        print(predictor.visualize(a.image,a.attention_dir))
    else:print(predictor.caption(a.image,a.decode))


if __name__=='__main__':main()
