"""Export the original decoder checkpoint plus frozen ImageNet backbone as a portable inference checkpoint."""
import argparse
from pathlib import Path
import torch
from model import ViTCaptioner
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--backbone',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
s=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=s['config']
kwargs={k:c[k] for k in ['pad_id','max_len','d_model','nhead','num_decoder_layers','dim_feedforward','dropout']}
m=ViTCaptioner(len(s['vocab']),**kwargs,pretrained=False)
vision=torch.load(a.backbone,map_location='cpu',weights_only=False)
m.vit.load_state_dict({k:v for k,v in vision.items() if not k.startswith('heads.')},strict=True)
m.load_trainable_state_dict(s['trainable_state'])
result={k:s[k] for k in ['vocab','config','trainable_state','epoch'] if k in s};result['full_model_state']=m.state_dict()
a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(result,a.output);print(a.output)
