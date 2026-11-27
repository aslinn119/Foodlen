"""Compare B-gen's training-mean and Model B fallbacks on validation only."""
import json
from pathlib import Path

from metrics import evaluate_all

ROOT = Path(__file__).resolve().parent


def read_rows(path):
    with (ROOT / path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(predictions, records):
    metrics = evaluate_all(predictions, records)
    return {
        "micro_f1": metrics["micro_f1"],
        "mean_mae": metrics["numeric"]["mean_mae"],
        "mean_mse": metrics["numeric"]["mean_mse"],
    }


def main():
    records = read_rows("annotations/val.jsonl")
    gen = read_rows("final_results/modelBgen_val_preds.jsonl")
    b = read_rows("final_results/modelB_val_preds.jsonl")
    ids = [r["dish_id"] for r in records]
    if len(set(ids)) != len(ids) or any([r["dish_id"] for r in rows] != ids for rows in (gen, b)):
        raise ValueError("Validation dish IDs differ across inputs")
    failed = [i for i, row in enumerate(gen) if row["parse_code"] == 0]
    raw = [r["parsed"] for r in gen]
    fallback = raw.copy()
    for i in failed:
        fallback[i] = b[i]["parsed"]
    result = {"split": "val", "n": len(records),
              "failure_dish_ids": [ids[i] for i in failed],
              "all": {"mean_fallback": summarize(raw, records),
                      "model_b_fallback": summarize(fallback, records)},
              "failed_only": None}
    if failed:
        refs = [records[i] for i in failed]
        result["failed_only"] = {
            "mean_fallback": summarize([raw[i] for i in failed], refs),
            "model_b_fallback": summarize([fallback[i] for i in failed], refs),
        }
    path = ROOT / "final_results/bgen_fallback_validation.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
