"""Compare same-image Caption and Task2 ingredient mentions conservatively."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLIND = HERE.parent / 'outputs/judge_four_model'


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def tokens(text):
    return re.findall(r'[a-z]+', text.lower())


def main():
    vocab = json.loads((HERE / 'model_b/ingredients.json').read_text(encoding='utf-8'))
    lexicon = {}
    for name in vocab:
        phrase = tuple(tokens(name))
        if phrase:
            lexicon[phrase] = name
    for phrase, name in list(lexicon.items()):
        word = phrase[-1]
        if word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
            variant = word[:-3] + 'y' if word.endswith('ies') else word[:-1]
        else:
            variant = word[:-1] + 'ies' if word.endswith('y') else word + 's'
        lexicon.setdefault((*phrase[:-1], variant), name)
    longest = max(map(len, lexicon))
    predictions = read_lines(HERE / 'task2_model_b_same_images.jsonl')
    captions = read_lines(BLIND / 'blind_candidates.jsonl')
    by_image = {r['file_name']: r for r in predictions}
    keys = {r['candidate_id']: r['model'] for r in json.loads(
        (BLIND / 'blind_key.json').read_text(encoding='utf-8'))['rows']}
    if len(by_image) != len(predictions) or len(by_image) != 30 or set(by_image) != {r['image_id'] for r in captions}:
        raise ValueError('Expected exact one-to-one image coverage')
    output = []
    for row in captions:
        words = tokens(row['candidate'])
        mentioned = set()
        i = 0
        while i < len(words):
            for size in range(min(longest, len(words) - i), 0, -1):
                matched = lexicon.get(tuple(words[i:i + size]))
                if matched:
                    mentioned.add(matched)
                    i += size
                    break
            else:
                i += 1
        items = {item['name'] for item in by_image[row['image_id']]['nutrition']['items']}
        output.append({
            'candidate_id': row['candidate_id'], 'file_name': row['image_id'],
            'caption_model': keys[row['candidate_id']], 'caption': row['candidate'],
            'common': sorted(mentioned & items), 'caption_only': sorted(mentioned - items),
            'task2_only': sorted(items - mentioned),
        })
    (HERE / 'ingredient_comparison.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in output), encoding='utf-8')
    grouped = defaultdict(list)
    for row in output:
        grouped[row['caption_model']].append(row)
    with (HERE / 'ingredient_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['caption_model', 'images', 'at_least_one_shared', 'caption_has_vocab_match'])
        for model, rows in sorted(grouped.items()):
            writer.writerow([model, len(rows), sum(bool(r['common']) for r in rows),
                             sum(bool(r['common'] or r['caption_only']) for r in rows)])
    print(f'Aligned {len(by_image)} images and {len(output)} captions')


if __name__ == '__main__':
    main()
