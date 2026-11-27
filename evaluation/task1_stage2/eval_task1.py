"""Recalculate caption scores from saved predictions against a fixed test split."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / 'metrics'))
from caption_metrics import compute_caption_metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(value):
    return value.lower().split()


def load_reference(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = {}
    for row in rows:
        name, caption = row.get('file_name'), row.get('caption')
        if not isinstance(name, str) or not name or not isinstance(caption, str) or not caption.strip():
            raise ValueError('Reference rows require non-empty file_name and caption')
        if name in result:
            raise ValueError(f'Duplicate reference image: {name}')
        result[name] = caption
    if not result:
        raise ValueError('Reference split is empty')
    return result


def load_predictions(path, refs):
    if path.suffix.lower() == '.jsonl':
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        obj = records
    else:
        obj = json.loads(path.read_text(encoding='utf-8'))
    records = obj if isinstance(obj, list) else obj.get('predictions')
    if not isinstance(records, list):
        raise ValueError(f'{path}: expected a JSON list or an object containing predictions')
    if isinstance(obj, dict) and obj.get('split', 'test') != 'test':
        raise ValueError(f'{path}: split is not test')
    by_name = {}
    for row in records:
        name = row.get('file_name')
        hypothesis = row.get('hypothesis', row.get('caption'))
        if not isinstance(name, str) or not name or not isinstance(hypothesis, str):
            raise ValueError(f'{path}: each prediction requires file_name and hypothesis strings')
        if name in by_name:
            raise ValueError(f'{path}: duplicate image {name}')
        if name in refs and 'reference' in row and words(row['reference']) != words(refs[name]):
            raise ValueError(f'{path}: reference mismatch for {name}')
        by_name[name] = hypothesis
    missing, extra = set(refs) - set(by_name), set(by_name) - set(refs)
    if missing or extra:
        raise ValueError(f'{path}: image set mismatch: missing={len(missing)}, extra={len(extra)}')
    order = list(refs)
    result = compute_caption_metrics([words(by_name[n]) for n in order], [words(refs[n]) for n in order])
    if isinstance(obj, dict) and 'bleu1_to_4' in obj:
        old = obj['bleu1_to_4'] + [obj.get('rouge_l'), obj.get('cider_d')]
        new = result['bleu1_to_4'] + [result['rouge_l'], result['cider_d']]
        for before, after in zip(old, new):
            if before is not None and abs(float(before) - after) > 1e-8:
                raise ValueError(f'{path}: recalculated score differs from stored score: {before} vs {after}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, default=HERE/'inputs'/'test_annotations.jsonl')
    p.add_argument('--split', choices=('train','val','test'), default='test')
    p.add_argument('--model', action='append', metavar='NAME=JSON', help='Repeat for each model; JSON has predictions with file_name and hypothesis')
    p.add_argument('--output-dir', type=Path, default=HERE/'outputs')
    args = p.parse_args()
    models = args.model or [f'{name}={HERE / "inputs" / file}' for name, file in [
        ('V1重训消融 greedy','v1_ablation_test.json'),
        ('V2注意力 greedy','v2_greedy_test.json'),
        ('V2注意力 beam-3','v2_beam_test.json')]]
    refs = load_reference(args.annotations)
    records = []
    for spec in models:
        if '=' not in spec:
            p.error('--model needs NAME=JSON')
        name, filename = spec.split('=', 1)
        if not name.strip() or not filename:
            p.error('--model needs a non-empty name and file')
        path = Path(filename)
        score = load_predictions(path, refs)
        records.append({'model':name, 'n':len(refs), **score, 'source':str(path), 'source_sha256':sha(path)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'reference_file':str(args.annotations), 'reference_sha256':sha(args.annotations),
              'split':args.split, 'reference_count':len(refs), 'matching':'all file_name and reference text checked',
              'tokenization':'lowercase + whitespace split; one reference per image',
              'bleu_scale':'0–100', 'rouge_l_scale':'0–1', 'cider_d_scale':'x10', 'results':records}
    (args.output_dir/'v1_v2_reevaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.output_dir/'v1_v2_reevaluation.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w = csv.writer(f); w.writerow(['模型','样本数','BLEU-1','BLEU-2','BLEU-3','BLEU-4','ROUGE-L','CIDEr-D'])
        for row in records:
            w.writerow([row['model'],row['n'],*[f'{v:.4f}' for v in row['bleu1_to_4']],f'{row["rouge_l"]:.4f}',f'{row["cider_d"]:.4f}'])
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
