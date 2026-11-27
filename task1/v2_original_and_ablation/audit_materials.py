"""Audit available inputs and reproduce BLEU from supplied V1 text outputs."""
from pathlib import Path
import json
import re
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'v1_cnn_gru'))
from bleu import corpus_bleu

def main():
    checks=ROOT/'checks';checks.mkdir(exist_ok=True)
    # Do not fabricate ready-data manifests or infer a vocabulary from test text.
    ready=ROOT/'Food500Cap_ready'
    missing=[str(p.relative_to(ROOT)) for p in [ready/'vocab.json',*[ready/'annotations'/f'{s}.jsonl' for s in ['train','val','test']]] if not p.is_file()]
    source=ROOT/'baseline_reference'/'samples_test_greedy.txt'
    refs=[];hyps=[]
    if source.exists():
        reference=None
        for line in source.read_text(encoding='utf-8-sig').splitlines():
            match=re.match(r'^\[(\d+)\]\s*参考:\s*(.*)$',line)
            if match:
                reference=match.group(2).lower().split()
            elif line.strip().startswith('生成:'):
                if reference is None:raise ValueError('Unpaired generated caption')
                refs.append(reference);hyps.append(line.strip().split(':',1)[1].strip().split());reference=None
    result=dict(v1_source_present=(ROOT/'v1_cnn_gru/model.py').exists(),
                raw_dataset_root=r'D:\foodlens_data\Food500Cap',
                prepared_data_missing=missing,
                v1_checkpoint_present=any(ROOT.rglob('best_v1.pt')),
                v1_sample_pairs=len(refs),
                v1_bleu_recomputed_with_supplied_monitor=corpus_bleu(hyps,refs) if refs else None,
                limitations=['Baseline samples have no image identifiers; pairing with prepared data needs verification.',
                             'Source defaults are not evidence of the actual baseline run configuration.',
                             'V1 BLEU script is documented as a monitoring metric, not official COCO evaluation.'])
    (checks/'materials_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
