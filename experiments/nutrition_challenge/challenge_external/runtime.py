"""Image-only prediction. No external labels or scene-ID lookup at inference."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from challenge_external.conformal import bounds

KEYS=['total_weight_g','total_calories_kcal','protein_g','fat_g','carbohydrate_g']
MODELS=['mean','resnet18','clip']
RESULTS=ROOT/'challenge_external'/'results'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

def sha(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

class PointPredictor:
    def __init__(self):
        stats=json.loads((RESULTS/'frozen_stats.json').read_text())
        self.mean=np.array(stats['mean']);self.std=np.array(stats['std'])
        from model import NutritionRegressor
        from model_lora import CLIPLoRAMultiTask
        self.resnet=NutritionRegressor(pretrained=False)
        self.resnet.load_state_dict(torch.load(ROOT/'checkpoints/best_model.pt',map_location='cpu',weights_only=True))
        ck=torch.load(ROOT/'checkpoints_lora_mt/best_model.pt',map_location='cpu',weights_only=False)
        self.clip=CLIPLoRAMultiTask(num_ingredients=ck['model_state']['ingredient_head.weight'].shape[0],last_n=ck['lora_last_n'],rank=ck['rank'],alpha=ck['alpha'])
        self.clip.load_state_dict(ck['model_state'])
        self.cm=torch.tensor(ck['mean'],dtype=torch.float32);self.cs=torch.tensor(ck['std'],dtype=torch.float32)
        self.resnet.to(DEVICE).eval();self.clip.to(DEVICE).eval()
        self.tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
        self.rnorm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
        self.cnorm=transforms.Normalize([.4815,.4578,.4082],[.2686,.2613,.2758])

    @torch.inference_mode()
    def predict_tensors(self,x):
        x=x.to(DEVICE)
        rp=self.resnet(self.rnorm(x)).cpu().numpy()*self.std+self.mean
        cp=(self.clip(self.cnorm(x))['numeric'].cpu()*self.cs+self.cm).numpy()
        cp=np.array([[round(max(float(v),0),1) for v in row] for row in cp])
        out={'mean':np.tile(self.mean,(len(x),1)),'resnet18':np.maximum(0,rp),'clip':cp}
        if any(not np.isfinite(p).all() for p in out.values()): raise ValueError('Nonfinite prediction')
        return out

_predictor=None

def predict_intervals(image_path, model_id='clip', calibration_path=None):
    """Return five marginal intervals. Target coverage fixed by calibration artifact."""
    global _predictor
    if model_id not in MODELS: raise ValueError('Unknown model')
    artifact=json.loads(Path(calibration_path or RESULTS/'calibration.json').read_text())
    for rel,digest in artifact['inference_dependencies'].items():
        if sha(ROOT/rel)!=digest: raise ValueError('Calibration incompatible with modified file: '+rel)
    if _predictor is None: _predictor=PointPredictor()
    with Image.open(image_path) as im: x=_predictor.tf(im.convert('RGB')).unsqueeze(0)
    p=_predictor.predict_tensors(x)[model_id][0]
    q=np.array(artifact['models'][model_id]['q'])
    lower,upper=bounds(p,q)
    return {'model_id':model_id,'model_version':artifact['models'][model_id]['model_version'],
            'calibration_version':artifact['version'],'target_coverage':artifact['target_coverage'],
            'coverage_scope':'per_field_marginal','fields':{k:{'point':float(p[i]),'lower':float(lower[i]),'upper':float(upper[i]),'unit':'kcal' if i==1 else 'g'} for i,k in enumerate(KEYS)},
            'limitations':'Calibrated on OmniFood8K official train; batch independence and coverage on other domains are not established.'}
