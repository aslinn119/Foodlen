"""验证 B-gen 输出后处理：原始、合并重复食材、合并后重量缩放。"""
import json, os, copy
import argparse
from collections import OrderedDict

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "final_results")

def merge_items(obj, largest_only=False):
    out = copy.deepcopy(obj)
    merged = OrderedDict()
    for item in out.get("items", []):
        name = item.get("name", "")
        weight = float(item["weight_g"])
        current = merged.get(name, 0.0)
        merged[name] = max(current, weight) if largest_only else current + weight
    out["items"] = [{"name": n, "weight_g": round(w, 1)} for n, w in merged.items()]
    return out

def scale_items(obj):
    out = copy.deepcopy(obj)
    total = max(float(out.get("total_weight_g", 0.0)), 0.0)
    current = sum(max(float(x.get("weight_g", 0.0)), 0.0) for x in out.get("items", []))
    if current > 0:
        factor = total / current
        for item in out["items"]:
            item["weight_g"] = round(max(float(item.get("weight_g", 0.0)), 0.0) * factor, 1)
    return out

def consistency(objs):
    diffs = [abs(sum(x["weight_g"] for x in o["items"]) - o["total_weight_g"])
             for o in objs if o.get("items")]
    return {"mean_abs_diff_g": sum(diffs) / max(len(diffs), 1),
            "within_5g_ratio": sum(d <= 5 for d in diffs) / max(len(diffs), 1),
            "duplicate_item_rate": sum(len({x['name'] for x in o['items']}) < len(o['items'])
                                        for o in objs) / max(len(objs), 1)}

def load(split):
    pred_path = os.path.join(OUT, f"modelBgen_{split}_preds.jsonl")
    ann_path = os.path.join(ROOT, "annotations", f"{split}.jsonl")
    with open(pred_path, encoding="utf-8") as stream:
        preds = [json.loads(line) for line in stream]
    with open(ann_path, encoding="utf-8") as stream:
        refs = [json.loads(line) for line in stream]
    if [p["dish_id"] for p in preds] != [r["dish_id"] for r in refs]:
        raise ValueError(f"{split}: predictions and annotations do not align")
    return [p["parsed"] for p in preds], refs


def ingredient_weight_errors(objs, refs):
    """Pool dish/name errors, summing repeated names on both sides."""
    matched_error = union_error = 0.0
    matched_count = union_count = 0
    for obj, ref in zip(objs, refs):
        pred = {i["name"]: i["weight_g"] for i in merge_items(obj)["items"]}
        true = {i["name"]: i["weight_g"] for i in merge_items(ref["target"])["items"]}
        for name in pred.keys() | true.keys():
            error = abs(pred.get(name, 0.0) - true.get(name, 0.0))
            union_error += error
            union_count += 1
            if name in pred and name in true:
                matched_error += error
                matched_count += 1
    return {
        "matched_count": matched_count,
        "union_count": union_count,
        "matched_mae_g": matched_error / matched_count if matched_count else None,
        "union_mae_g": union_error / union_count if union_count else None,
    }

def main():
    from metrics import evaluate_all
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-only", action="store_true",
                        help="Use only validation predictions for new experiments")
    args = parser.parse_args()
    result = {}
    splits = ("val",) if args.validation_only else ("val", "test")
    for split in splits:
        preds, refs = load(split)
        variants = {
            "raw": preds,
            "merge_duplicates": [merge_items(x) for x in preds],
            "merge_and_scale": [scale_items(merge_items(x)) for x in preds],
        }
        if args.validation_only:
            largest = [merge_items(x, largest_only=True) for x in preds]
            variants["keep_largest_duplicate"] = largest
            variants["keep_largest_and_scale"] = [scale_items(x) for x in largest]
        result[split] = {}
        for name, objs in variants.items():
            m = evaluate_all(objs, refs)
            result[split][name] = {
                "bleu_4": m["bleu_4"], "rouge_l": m["rouge_l"],
                "cider_d": m["cider_d"], "micro_f1": m["micro_f1"],
                "macro_f1": m["macro_f1"], "mean_mae": m["numeric"]["mean_mae"],
                "mean_mse": m["numeric"]["mean_mse"],
                "consistency": consistency(objs),
                "ingredient_weight": ingredient_weight_errors(objs, refs),
            }
    filename = ("bgen_item_weight_validation.json" if args.validation_only
                else "bgen_postprocess_comparison.json")
    path = os.path.join(OUT, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("saved:", path)

if __name__ == "__main__":
    main()
