"""Single-image inference from a SCST/continued-CE checkpoint and frozen ViT weights."""
import argparse, json
from pathlib import Path
import torch
from PIL import Image
from model import ViTCaptioner
from v1_data import eval_transforms
from run import rollout, captions

def main():
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--backbone',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(4)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=state['config'];v=state['vocab']
    m=ViTCaptioner(len(v),**{k:c[k] for k in ['pad_id','max_len','d_model','nhead','num_decoder_layers','dim_feedforward','dropout']},pretrained=False)
    backbone=torch.load(a.backbone,map_location='cpu',weights_only=True)
    m.vit.load_state_dict({k:t for k,t in backbone.items() if not k.startswith('heads.')},strict=True)
    m.load_trainable_state_dict(state['trainable_state']);m=m.to(a.device).eval()
    with Image.open(a.image) as im:x=eval_transforms()(im.convert('RGB')).unsqueeze(0).to(a.device)
    with torch.no_grad():caption=captions(rollout(m,m.encode_image(x),c),v)[0]
    print(json.dumps({'caption':caption,'model_id':a.checkpoint.stem,'epoch':state.get('epoch'),'device':a.device},ensure_ascii=False))

if __name__=='__main__':main()
