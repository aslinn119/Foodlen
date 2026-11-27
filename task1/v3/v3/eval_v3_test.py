
import json
from bleu import corpus_bleu

# V3.2 完整测试集预测结果
path = "outputs/v3_layer4/v3_2_test_predictions.jsonl"

hyps = []
refs = []

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        hyps.append(record["caption"].lower().split())
        refs.append(record["reference"].lower().split())

print("Evaluating:", path)
print("Total samples:", len(hyps))

# 防止误把验证集结果当作测试集
assert len(hyps) == 4938, "测试集预测数量不正确！"

scores = corpus_bleu(hyps, refs)

for i, score in enumerate(scores, 1):
    print(f"BLEU-{i}: {score:.4f}%")
