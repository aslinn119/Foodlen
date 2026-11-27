import json
from bleu import corpus_bleu

path = "outputs/full_v4/v4_test_predictions.jsonl"

hyps = []
refs = []

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        hyps.append(record["caption"].lower().split())
        refs.append(record["reference"].lower().split())

scores = corpus_bleu(hyps, refs)

print("Total samples:", len(hyps))

for i, score in enumerate(scores, start=1):
    print(f"BLEU-{i}: {score:.4f}%")
