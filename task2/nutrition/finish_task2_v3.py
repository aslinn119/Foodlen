"""Wait for the authorized 25-epoch run, freeze after validation, then evaluate/package."""
from pathlib import Path
import subprocess,sys,json,time,hashlib
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'final_results_v3_20261119'
def run(*args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    last=-1
    while True:
        try:n=len(json.loads((ROOT/'checkpoints_lora_mt_normalized/history.json').read_text())['train_loss'])
        except (OSError,ValueError):n=0
        if n!=last:print('TRAINING_EPOCHS',n,flush=True);last=n
        if n==25 and not (ROOT/'checkpoints_lora_mt_normalized/latest.pt').exists():break
        time.sleep(5)
    print('VALIDATING_NORMALIZED_B',flush=True)
    run('eval_task2_final.py','--split','val','--model','b')
    sources=['model_caption.py','constrained_caption.py','predict.py','predict_final.py','eval_models.py','eval_task2_final.py','metrics.py',
             'checkpoints_caption/best_model.pt','checkpoints_lora_mt_normalized/best_model.pt','checkpoints_lora_mt_normalized/history.json','ingredients.json']
    for m in ['a','b','bgen']:assert (OUT/f'{m}_val_metrics.json').exists()
    freeze={'version':'20261119-v3','files':{n:sha(ROOT/n) for n in sources},'selection':'25 epochs; minimum validation mean MAE/train standard deviation; no test selection',
            'A_acceptance':'fixed schema correctness; validation F1 approximately unchanged; no claim of recognition gain',
            'test_status':'previously inspected historical test; retrospective correction evaluation'}
    (OUT/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    print('FROZEN_BEFORE_TEST',flush=True)
    for model in ['a','b','bgen']:run('eval_task2_final.py','--split','test','--model',model)
    run('build_task2_v3.py')
    print('ALL_DONE',flush=True)
if __name__=='__main__':main()
