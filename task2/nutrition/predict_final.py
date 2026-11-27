"""Final v3 entry point. Old predict.py remains available to reproduce legacy models."""
import json
from pathlib import Path
import torch
from PIL import Image
from predict import _load_a,_load_b,_infer_b,RESNET_TF,DEVICE
from constrained_caption import generate_constrained,reconcile_weights
ROOT=Path(__file__).resolve().parent
_cache={}

@torch.no_grad()
def predict_nutrition_verbose(image_path,model_id='b'):
    if model_id not in ('a','b'):raise ValueError('v3 supports a or b; legacy bgen uses predict.py')
    if model_id not in _cache:
        _cache[model_id]=_load_a() if model_id=='a' else _load_b('checkpoints_lora_mt_normalized')
        _cache[model_id]['require_items']=True
    s=_cache[model_id]
    with Image.open(image_path) as image:
        image=image.convert('RGB')
        if model_id=='a':
            names=json.loads((ROOT/'ingredients.json').read_text())
            seq=generate_constrained(s['model'],RESNET_TF(image).unsqueeze(0).to(DEVICE),s['vocab'],names)[0]
            text=s['vocab'].decode(seq);raw=json.loads(text);obj=reconcile_weights(raw)
        else:
            obj,text,_=_infer_b(image,s);raw=None
    return {'model':model_id,'version':'20261119-v3','nutrition':obj,'text':text,'raw_generated':raw,
            'parse_code':1,'postprocess':'largest_remainder_weight_projection; B selects top1 if threshold set is empty',
            'warning':'Output validity/consistency is enforced; food identity and nutrition remain estimates.'}

def predict_nutrition(image_path,model_id='b'):
    return predict_nutrition_verbose(image_path,model_id)['nutrition']
