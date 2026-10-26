import argparse, json
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
def tok(s): return str(s).lower().strip().split()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",default="outputs_v1/test_predictions.jsonl")
    a=ap.parse_args()
    refs=[]; hyps=[]
    for line in Path(a.predictions).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r=json.loads(line); refs.append([tok(r["reference"])]); hyps.append(tok(r["prediction"]))
    if not hyps: raise ValueError("预测文件为空")
    sm=SmoothingFunction().method1
    for n in range(1,5):
        weights=tuple(1/n if i<n else 0 for i in range(4))
        print(f"BLEU-{n}: {corpus_bleu(refs,hyps,weights=weights,smoothing_function=sm):.4f}")
    print("说明：这是基础 BLEU 实现；正式实验应固定评估工具与分词规则，并补充 METEOR、ROUGE-L、CIDEr-D 等指标。")
if __name__=="__main__": main()
