"""Predict Task2 nutrition for the fixed Task1 visual blind review sample."""
import csv
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import CLIPVisionConfig, CLIPVisionModel

HERE = Path(__file__).resolve().parent
EXPORT = HERE / 'model_b'
IMAGE_ROOT = HERE.parent.parent / 'Food500Cap_ready'
MANIFEST = HERE.parent / 'outputs/judge_four_model/task2_30_image_manifest.csv'
sys.path.insert(0, str(EXPORT))
from predict import predict_nutrition


def main():
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        names = [row['file_name'] for row in csv.DictReader(f)]
    if len(names) != 30 or len(set(names)) != 30:
        raise ValueError('Expected 30 unique images')
    config = CLIPVisionConfig.from_dict(json.loads(
        (EXPORT / 'clip_config.json').read_text(encoding='utf-8'))['vision_config'])
    # The supplied B checkpoint stores the entire model state. Its strict load
    # replaces initialization values, including every frozen vision parameter.
    rows = []
    torch.set_num_threads(4)
    with patch.object(CLIPVisionModel, 'from_pretrained', side_effect=lambda _: CLIPVisionModel(config)):
        for name in names:
            nutrition = predict_nutrition(str(IMAGE_ROOT / name), 'b')
            if not isinstance(nutrition.get('items'), list):
                raise ValueError(f'Invalid items for {name}')
            rows.append({'file_name': name, 'model_id': 'b', 'nutrition': nutrition})
            print(f'{len(rows)}/30', flush=True)
    (HERE / 'task2_model_b_same_images.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')


if __name__ == '__main__':
    main()
