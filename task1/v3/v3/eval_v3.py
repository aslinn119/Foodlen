

import json
from pathlib import Path
from bleu import corpus_bleu

# 两个模型的预测文件
files = {
    "V3 Baseline": "outputs/full_v3/v3_val_predictions.jsonl",
    "V3.2 Layer4": "outputs/v3_layer4/v3_2_val_predictions.jsonl",
}


def load_predictions(path):
    records = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records[r["file_name"]] = r

    return records


# 加载两份预测结果
baseline = load_predictions(files["V3 Baseline"])
optimized = load_predictions(files["V3.2 Layer4"])

# 确保两个模型预测的是同一组图片
assert baseline.keys() == optimized.keys(), "两个模型的图片集合不一致！"

# 比较每张图片的预测描述
changed = 0

for name in baseline:
    if baseline[name]["caption"] != optimized[name]["caption"]:
        changed += 1

print(f"Total images: {len(baseline)}")
print(f"Changed captions: {changed}")
print(f"Change rate: {changed / len(baseline) * 100:.2f}%")
print()

# 分别计算 BLEU
for model_name, path in files.items():
    records = load_predictions(path)

    hyps = []
    refs = []

    for name in baseline:
        r = records[name]
        hyps.append(r["caption"].lower().split())
        refs.append(r["reference"].lower().split())

    scores = corpus_bleu(hyps, refs)

    print(f"===== {model_name} =====")
    print("Prediction file:", Path(path).resolve())

    for i, score in enumerate(scores, start=1):
        print(f"BLEU-{i}: {score:.4f}%")

    print()

# 显示前五个发生变化的预测样例
print("===== Changed examples =====")

count = 0

for name in baseline:
    old = baseline[name]["caption"]
    new = optimized[name]["caption"]

    if old != new:
        print("\nImage:", name)
        print("Reference:", baseline[name]["reference"])
        print("Baseline:", old)
        print("V3.2:", new)

        count += 1

        if count >= 5:
            break
