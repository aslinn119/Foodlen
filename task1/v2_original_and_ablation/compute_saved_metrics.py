"""Compute all required caption metrics from the already saved full-test predictions."""
import json
import sys
from pathlib import Path

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/"v1_cnn_gru"))
sys.path.insert(0,str(BASE/"v2_attention"))
from caption_metrics import compute_caption_metrics


def main():
    paths=[BASE/"results"/"v1_ablation_test.json",BASE/"results"/"v2_greedy_test.json",
           BASE/"results"/"v2_beam_test.json"]
    summaries={}
    for path in paths:
        data=json.loads(path.read_text(encoding="utf-8"))
        hyps=[r["hypothesis"].lower().split() for r in data["predictions"]]
        refs=[r["reference"].lower().split() for r in data["predictions"]]
        metrics=compute_caption_metrics(hyps,refs)
        data.update(rouge_l=metrics["rouge_l"],cider_d=metrics["cider_d"],
                    metric_details={"ROUGE-L":"COCO-caption LCS F-score, beta=1.2; one reference per image",
                                    "CIDEr-D":"COCO-caption TF-IDF 1-4 grams, sigma=6, scale=10; one reference per image"})
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
        summaries[path.stem]={"n":data["n"],"bleu1_to_4":data["bleu1_to_4"],
                              "rouge_l":data["rouge_l"],"cider_d":data["cider_d"]}
    out=BASE/"results"/"additional_caption_metrics.json"
    out.write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
