"""Consolidate measured validation and full-test BLEU results."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    results=BASE/"results";runs=BASE/"v2_attention"/"runs"
    audit=load(BASE/"checks"/"ready_data_audit.json")
    v2hist=load(runs/"v2"/"history.json");v1hist=load(runs/"v1_ablation"/"history.json")
    v1legacy=load(BASE/"baseline_reference"/"group_v1_reported_metrics.json")
    v1=load(results/"v1_ablation_test.json");v2g=load(results/"v2_greedy_test.json");v2b=load(results/"v2_beam_test.json")
    v2best=max(v2hist,key=lambda x:x["val_bleu4"]);v1best=max(v1hist,key=lambda x:x["val_bleu4"])
    obj={
        "dataset":{"split_sizes":{s:audit["splits"][s]["rows"] for s in ("train","val","test")},
                   "images":audit["image_count"],"vocabulary":audit["vocab_size"],"audit_pass":audit["pass"]},
        "metrics":{"BLEU":"Group-provided shared corpus BLEU implementation, percent scale",
                   "ROUGE-L":"COCO-caption LCS F-score, beta=1.2; one reference per image",
                   "CIDEr-D":"COCO-caption TF-IDF 1-4 grams, sigma=6, scale=10; one reference per image"},
        "validation":{"v2_best_epoch":v2best["epoch"],"v2_best_bleu1_to_4":v2best["val_bleu"],
                      "v1_ablation_best_epoch":v1best["epoch"],"v1_ablation_best_bleu1_to_4":v1best["val_bleu"]},
        "test":{"v1_group_reported":{"bleu1_to_4":v1legacy["bleu1_to_4"],"n":v1legacy["n"],
                                        "checkpoint_available":False,"verified":False},
                "v1_retrained_ablation":{"bleu1_to_4":v1["bleu1_to_4"],"rouge_l":v1["rouge_l"],"cider_d":v1["cider_d"],"n":v1["n"]},
                "v2_greedy":{"bleu1_to_4":v2g["bleu1_to_4"],"rouge_l":v2g["rouge_l"],"cider_d":v2g["cider_d"],"n":v2g["n"]},
                "v2_beam3":{"bleu1_to_4":v2b["bleu1_to_4"],"rouge_l":v2b["rouge_l"],"cider_d":v2b["cider_d"],"n":v2b["n"]}},
        "differences_bleu4":{"v2_greedy_minus_v1_retrained":v2g["bleu1_to_4"][3]-v1["bleu1_to_4"][3],
                              "v2_beam3_minus_v2_greedy":v2b["bleu1_to_4"][3]-v2g["bleu1_to_4"][3],
                              "v2_greedy_minus_v1_group_reported":v2g["bleu1_to_4"][3]-v1legacy["bleu1_to_4"][3]},
        "differences_additional_metrics":{"v2_greedy_minus_v1_retrained_rouge_l":v2g["rouge_l"]-v1["rouge_l"],
                                          "v2_greedy_minus_v1_retrained_cider_d":v2g["cider_d"]-v1["cider_d"],
                                          "v2_beam3_minus_v2_greedy_rouge_l":v2b["rouge_l"]-v2g["rouge_l"],
                                          "v2_beam3_minus_v2_greedy_cider_d":v2b["cider_d"]-v2g["cider_d"]},
        "baseline_note":"Historical group V1 score is included for context; its checkpoint was not supplied, so its exact configuration cannot be verified here."
    }
    out=results/"task1_results_summary.json";out.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(obj,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
