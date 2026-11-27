"""Verify and summarize the completed DSH blind review."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent / 'outputs' / 'judge_four_model'
RESULTS = ROOT / 'deepseek_web_formal' / 'results'
DIMS = {
    'visual_consistency': '视觉一致性',
    'food_identification': '食物识别准确性',
    'detail_accuracy': '细节准确性',
    'language_quality': '语言质量',
}
CATEGORIES = ('explicit_conflicts', 'unverifiable_from_image', 'visible_but_omitted',
              'suspected_ingredient_hallucinations')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def index_jsonl(path):
    rows = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            row = json.loads(line)
            cid = row['candidate_id']
            if cid in rows:
                raise ValueError(f'{path.name}: duplicate {cid}')
            rows[cid] = row
    return rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    scores = index_jsonl(RESULTS / 'judge_scores.jsonl')
    samples = index_jsonl(ROOT / 'blind_candidates.jsonl')
    key_rows = read_json(ROOT / 'blind_key.json')['rows']
    key = {r['candidate_id']: r for r in key_rows}
    expected = {f'{i:02}{c}' for i in range(1, 31) for c in 'ABCD'}
    require(len(key_rows) == 120 and set(scores) == set(samples) == set(key) == expected,
            'Candidate IDs do not match the complete blind sample')
    require({p.stem for p in (RESULTS / 'raw').glob('*.txt')} == expected, 'Raw files do not match the sample')
    models = defaultdict(list)
    images_by_model = defaultdict(set)
    run_specs = set()
    for cid, row in sorted(scores.items()):
        sample = samples[cid]
        require(key[cid]['image_id'] == sample['image_id'], f'{cid}: image mismatch in key')
        require(all(row[k] == sample[k] for k in ('reference', 'candidate')), f'{cid}: text mismatch')
        require(Path(row['image_path']).name == 'image.jpg'
                and Path(row['image_path']).parent.name == cid[:2], f'{cid}: group image mismatch')
        require(all(type(row[k]) is int and 1 <= row[k] <= 5 for k in DIMS), f'{cid}: invalid scores')
        reasons = row['reasons']
        require(set(reasons) == set(DIMS) and all(isinstance(v, str) and v.strip() for v in reasons.values()),
                f'{cid}: invalid reasons')
        require(all(isinstance(row[k], list) and all(isinstance(v, str) for v in row[k]) for k in CATEGORIES),
                f'{cid}: invalid category arrays')
        fields = ('candidate_id', *DIMS, 'reasons', *CATEGORIES)
        raw = read_json(RESULTS / 'raw' / f'{cid}.txt')
        single = read_json(RESULTS / f'{cid}.json')
        require(all(row[k] == raw[k] == single[k] for k in fields), f'{cid}: raw or individual result mismatch')
        run_specs.add(tuple(row[k] for k in ('judge_provider', 'judge_model', 'reasoning_effort')))
        model = key[cid]['model']
        models[model].append(row)
        images_by_model[model].add(sample['image_id'])
    image_ids = {r['image_id'] for r in samples.values()}
    require(len(run_specs) == 1, 'Judge settings are inconsistent')
    require(len(image_ids) == 30 and len(models) == 4 and all(
        len(rows) == 30 and images_by_model[name] == image_ids for name, rows in models.items()),
        'Model coverage is incomplete')
    provider, judge, effort = next(iter(run_specs))
    counts = {k: Counter(row[k] for row in scores.values()) for k in DIMS}
    summary = [[name, len(rows), *(f'{mean(r[k] for r in rows):.4f}' for k in DIMS)]
               for name, rows in sorted(models.items())]
    headers = ['Caption 模型', '样本数', *DIMS.values()]
    with (ROOT / '正式盲评_四模型均分.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(summary)
    report = [
        '# FoodLens 正式视觉盲评结果', '',
        '对照盲评样本和密钥核验了 120 条候选。每个模型覆盖同一批 30 张图。逐条 JSON、汇总 JSONL 与原始回复的评分字段及理由一致。', '',
        f'评审模型为 DeepSeek V4.1 Flash 1017（`{judge}`），提供方为 OpenCode Go（`{provider}`），推理强度为 `{effort}`。'
        '按 DSH 执行记录，每条候选由一名独立子 Agent 读取图片并评分。', '',
        '| ' + ' | '.join(headers) + ' |',
        '|---|---:|---:|---:|---:|---:|',
        *['| ' + ' | '.join(map(str, row)) + ' |' for row in summary],
        '', '表中为各模型在 30 张图上的四项均分，量表为 1～5 分。', '',
        '| 维度 | 1 分 | 2 分 | 3 分 | 4 分 | 5 分 |', '|---|---:|---:|---:|---:|---:|',
        *['| ' + ' | '.join([label, *(str(counts[k][v]) for v in range(1, 6))]) + ' |'
          for k, label in DIMS.items()],
        '', '上述分数反映单一 Judge 对这批图片的评价。人工复核尚待填写，Task2 同图一致性分析仍需对应 JSON。', '',
    ]
    (ROOT / '正式盲评_汇总.md').write_text('\n'.join(report), encoding='utf-8')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
