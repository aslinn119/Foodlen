"""New final protocol: constrained A, normalized-selection B, reference text scorers."""
from release_tools import align_predictions, check_images
from pathlib import Path
import os,json,argparse,hashlib,contextlib,io
import numpy as np
import torch
from torch.utils.data import DataLoader
from eval_models import ImgDataset,RESNET_TF,load_records,train_means,infer_model_b
import eval_models
from model_caption import build_vocab_from_train,CaptionModel
from constrained_caption import generate_constrained,reconcile_weights
from metrics import NUM_KEYS,nutrition_to_text,tokenize,ingredient_f1,numeric_metrics
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results_repaired_20261123';OUT.mkdir(exist_ok=True)

def reference_metrics(preds,refs):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    gt={i:[' '.join(tokenize(r['target_text']))] for i,r in enumerate(refs)}
    hyp={i:[' '.join(tokenize(nutrition_to_text(p)))] for i,p in enumerate(preds)}
    with contextlib.redirect_stdout(io.StringIO()):bs,_=Bleu(4).compute_score(gt,hyp)
    ro,_=Rouge().compute_score(gt,hyp);ci,_=Cider().compute_score(gt,hyp)
    m={f'bleu_{i+1}':float(x) for i,x in enumerate(bs)};m.update(rouge_l=float(ro),cider_d=float(ci),
        text_protocol='pycocoevalcap 1.2 scorers; fixed JSON lexical tokens, not PTB; one reference; CIDEr df=reference corpus')
    m.update(ingredient_f1([[i['name'] for i in p['items']] for p in preds],[[i['name'] for i in r['target']['items']] for r in refs]))
    m['numeric']=numeric_metrics(preds,[r['target'] for r in refs])
    from postprocess_bgen import ingredient_weight_errors
    m['ingredient_weight']=ingredient_weight_errors(preds,refs)
    _,std=train_means();m['numeric']['normalized_mae']=float(np.mean([m['numeric'][k]['mae']/max(std[i],1e-8) for i,k in enumerate(NUM_KEYS)]))
    m['health']={'n':len(preds),'duplicate_samples':sum(len(p['items'])!=len({i['name'] for i in p['items']}) for p in preds),
        'weight_diff_gt_0_11g':sum(abs(sum(i['weight_g'] for i in p['items'])-p['total_weight_g'])>.11 for p in preds)}
    return m

@torch.no_grad()
def a_predict(refs):
    v=build_vocab_from_train();ck=torch.load(ROOT/'checkpoints_caption/best_model.pt',map_location='cpu',weights_only=False)
    model=CaptionModel(ck['vocab_size'],pretrained=False).to(eval_models.DEVICE);model.load_state_dict(ck['model_state']);model.eval()
    names=json.loads((ROOT/'ingredients.json').read_text());records=[]
    for imgs in DataLoader(ImgDataset(refs,RESNET_TF),batch_size=32):
        seqs=generate_constrained(model,imgs.to(eval_models.DEVICE),v,names)
        for seq in seqs:
            text=v.decode(seq);raw=json.loads(text)
            records.append({'text':text,'raw_parsed':raw,'parsed':reconcile_weights(raw),'parse_code':1,'generation':'schema_constrained','postprocess':'weight_projection_to_predicted_total'})
        print('A generated',len(records),flush=True)
    return records

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['val','test'],required=True);ap.add_argument('--model',choices=['a','b','bgen'],required=True);args=ap.parse_args()
    refs=load_records(args.split)
    if args.model != 'bgen':
        report = check_images(ROOT, refs)
        if report['missing'] or report['unreadable']:
            raise RuntimeError('Image preflight failed: ' + json.dumps(report, ensure_ascii=False))
    if args.split=='test' and not (ROOT/'final_results_v3_20261119/freeze.json').exists():raise RuntimeError('Freeze protocol before test')
    if args.model=='a':preds=a_predict(refs)
    elif args.model=='b':
        eval_models.CKPT_B=str(ROOT/'checkpoints_lora_mt_normalized')
        preds=[{'parsed':p,'generation':'multitask','parse_code':1} for p in infer_model_b(refs,*train_means(),require_items=True)]
    else:
        preds=[json.loads(l) for l in (ROOT/f'final_results_v3_20261119/bgen_{args.split}_predictions.jsonl').read_text().splitlines()]
    if args.model == 'bgen':
        preds = align_predictions(preds, refs)
    else:
        if len(preds) != len(refs): raise ValueError('Prediction/reference count mismatch')
        for p,r in zip(preds,refs): p['dish_id']=r['dish_id']
    m=reference_metrics([p['parsed'] for p in preds],refs)
    m['provenance'] = 'fresh_image_inference' if args.model != 'bgen' else 'historical_saved_predictions_rescored'
    if args.model=='a':m['before_weight_projection']=reference_metrics([p['raw_parsed'] for p in preds],refs)
    if args.model != 'bgen':
        (OUT/f'{args.model}_{args.split}_predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in preds),encoding='utf-8')
    (OUT/f'{args.model}_{args.split}_metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    print(json.dumps(m,indent=2))
if __name__=='__main__':main()
