"""
eval_parse_stats.py —— P0② 复现核验与解析健康度总表

两部分：
1. 复现核验：从 final_results/ 已保存的预测明细（preds.jsonl）重新计算全部指标，
   与已保存的指标 json 比对，确认"保存预测 -> 指标"链路可复现。
2. 解析健康度总表：对 val/test × 模型 A/B/B-gen 统计
   strict JSON 率 / 修复率 / 正则兜底率 / 回退总率 / 空 items 率 / Schema 有效率 /
   分项-总重一致性误差；并导出字段级指标表 CSV 与 10 个失败样例。

输出：
  final_results/repro_check.json        复现核验结果（差异应为 0）
  final_results/parse_stats_all.json    三模型解析健康度
  final_results/field_metrics_all.csv   字段级 MSE/MAE/RMSE 总表
  final_results/failure_samples_10.jsonl 10 个失败样例（含失败原因标注）
"""

import os
import json
import csv
import math
import numpy as np

from metrics import evaluate_all, NUM_KEYS
from predict import validate_output, check_schema

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DATA_ROOT, "final_results")
MODELS = {"a": "modelA", "b": "modelB", "bgen": "modelBgen"}


def load_records(split):
    with open(os.path.join(DATA_ROOT, "annotations", f"{split}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_preds(split, tag):
    path = os.path.join(OUT, f"{MODELS[tag]}_{split}_preds.jsonl")
    with open(path, encoding="utf-8") as f:
        preds = [json.loads(line) for line in f]
    if [p["dish_id"] for p in preds] != [r["dish_id"] for r in load_records(split)]:
        raise ValueError(f"{split}/{tag}: 预测和标注的 dish_id 顺序不一致")
    return preds


def parse_stats(jsons, codes):
    n = len(codes)
    diffs = [abs(sum(it["weight_g"] for it in j["items"]) - j["total_weight_g"])
             for j in jsons if j["items"]]
    schema_ok = sum(1 for j in jsons if not check_schema(j)) / n
    return {
        "n": n,
        "strict_json_rate": sum(1 for c in codes if c == 1) / n,
        "repaired_rate": sum(1 for c in codes if c == 2) / n,
        "regex_fallback_rate": sum(1 for c in codes if c == 3) / n,
        "mean_fallback_rate": sum(1 for c in codes if c == 0) / n,
        "total_fallback_rate": 1.0 - sum(1 for c in codes if c == 1) / n,
        "empty_items_rate": sum(1 for j in jsons if not j["items"]) / n,
        "schema_valid_rate": schema_ok,
        "weight_consistency_mean_abs_diff_g": float(np.mean(diffs)) if diffs else 0.0,
        "weight_consistency_within_5g": float(np.mean([d <= 5.0 for d in diffs])) if diffs else 0.0,
    }


def repro_check():
    """从保存预测重算指标，与保存值比对。"""
    saved = {}
    for split in ["val", "test"]:
        with open(os.path.join(OUT, f"{split}_metrics.json"), encoding="utf-8") as f:
            saved[split] = json.load(f)
        with open(os.path.join(OUT, "bgen_metrics.json"), encoding="utf-8") as f:
            saved["bgen"] = json.load(f)

    report = {}
    for split in ["val", "test"]:
        records = load_records(split)
        report[split] = {}
        for tag, key in (("a", "modelA_cnn_gru"), ("b", "modelB_clip_lora"),
                         ("bgen", "modelBgen")):
            preds = load_preds(split, tag)
            m = evaluate_all([p["parsed"] for p in preds], records)
            s = saved["bgen"][split]["metrics"] if tag == "bgen" else saved[split][key]
            report[split][key] = metric_differences(m, s)
    return report


def metric_differences(computed, saved, prefix=""):
    """Compare all saved metrics, including per-field and aggregate errors."""
    differences = []
    for key in sorted(computed.keys() | saved.keys()):
        path = f"{prefix}{key}"
        if key not in computed or key not in saved:
            differences.append(path)
        elif isinstance(computed[key], dict) and isinstance(saved[key], dict):
            differences.extend(metric_differences(computed[key], saved[key], path + "."))
        elif not isinstance(computed[key], (int, float)) or not isinstance(saved[key], (int, float)):
            differences.append(path)
        elif not math.isclose(computed[key], saved[key], rel_tol=1e-9, abs_tol=1e-8):
            differences.append(path)
    return differences


def main():
    # ---------- 1. 复现核验 ----------
    rep = repro_check()
    ok = all(not differences for split in rep.values() for differences in split.values())
    with open(os.path.join(OUT, "repro_check.json"), "w", encoding="utf-8") as f:
        json.dump({"pass": ok, "detail": rep}, f, indent=2, ensure_ascii=False)
    print(f"复现核验: {'PASS —— 保存预测重算指标与保存值完全一致' if ok else 'FAIL，见 repro_check.json'}")

    # ---------- 2. 解析健康度总表 ----------
    stats = {}
    for split in ["val", "test"]:
        for tag in ["a", "b", "bgen"]:
            preds = load_preds(split, tag)
            codes = [p.get("parse_code", 1) for p in preds]
            stats[f"{split}_{tag}"] = parse_stats([p["parsed"] for p in preds], codes)
    with open(os.path.join(OUT, "parse_stats_all.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n解析健康度（strict / 回退 / 空items / Schema有效 / 分项-总重一致性±5g）:")
    for k, s in stats.items():
        print(f"  {k:10s} strict={s['strict_json_rate']*100:5.1f}%  fb={s['total_fallback_rate']*100:5.1f}%  "
              f"empty={s['empty_items_rate']*100:4.1f}%  schema={s['schema_valid_rate']*100:5.1f}%  "
              f"±5g={s['weight_consistency_within_5g']*100:5.1f}%  mean_diff={s['weight_consistency_mean_abs_diff_g']:.1f}g")

    # ---------- 3. 字段级指标总表 ----------
    rows = []
    for split in ["val", "test"]:
        records = load_records(split)
        for tag in ["a", "b", "bgen"]:
            preds = load_preds(split, tag)
            m = evaluate_all([p["parsed"] for p in preds], records)["numeric"]
            for k in NUM_KEYS + ["mean"]:
                d = m[k] if k != "mean" else {"mse": m["mean_mse"], "mae": m["mean_mae"], "rmse": m["mean_rmse"]}
                rows.append({"split": split, "model": tag, "field": k,
                             "mse": round(d["mse"], 2), "mae": round(d["mae"], 2), "rmse": round(d["rmse"], 2)})
    with open(os.path.join(OUT, "field_metrics_all.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["split", "model", "field", "mse", "mae", "rmse"])
        w.writeheader()
        w.writerows(rows)
    print("\n字段级指标表:", os.path.join(OUT, "field_metrics_all.csv"))

    # ---------- 4. 10 个失败样例（覆盖多种失败模式） ----------
    test_records = {r["dish_id"]: r for r in load_records("test")}
    candidates = []
    for tag in ["a", "b", "bgen"]:
        for p in load_preds("test", tag):
            j, code = p["parsed"], p.get("parse_code", 1)
            warns = validate_output(j)
            gt = test_records[p["dish_id"]]["target"]
            kcal_err = abs(j["total_calories_kcal"] - gt["total_calories_kcal"])
            reasons = []
            if code in (2, 3):
                reasons.append(f"JSON解析回退(code={code})")
            if code == 0:
                reasons.append("解析完全失败(均值回退)")
            if not j["items"]:
                reasons.append("空items")
            if any("相差超过 5g" in w for w in warns):
                reasons.append("分项-总重不一致>5g")
            if kcal_err > 200:
                reasons.append(f"热量误差{kcal_err:.0f}kcal")
            if reasons:
                candidates.append({"model": tag, "dish_id": p["dish_id"],
                                   "image": test_records[p["dish_id"]]["image"],
                                   "failure_reasons": reasons,
                                   "parse_code": code, "warnings": warns,
                                   "raw_text": p.get("text"), "pred": j, "gt": gt,
                                   "kcal_abs_err": round(kcal_err, 1)})
    # 去重并保证模型多样性：每模型至多 4 条，优先级：解析回退 > 空items > 一致性 > 大误差
    rank = {"a": 0, "bgen": 1, "b": 2}
    candidates.sort(key=lambda c: (0 if c["parse_code"] in (0, 2, 3) else 1,
                                   0 if not c["pred"]["items"] else 1,
                                   0 if any("相差超过 5g" in w for w in c["warnings"]) else 1,
                                   -c["kcal_abs_err"]))
    picks, seen_dish, per_model = [], set(), {"a": 0, "b": 0, "bgen": 0}
    for c in candidates:
        if per_model[c["model"]] >= 4:
            continue
        if c["dish_id"] in seen_dish and len(picks) >= 6:
            continue
        picks.append(c)
        seen_dish.add(c["dish_id"])
        per_model[c["model"]] += 1
        if len(picks) == 10:
            break
    with open(os.path.join(OUT, "failure_samples_10.jsonl"), "w", encoding="utf-8") as f:
        for c in picks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\n失败样例: {len(picks)} 条 ->", os.path.join(OUT, "failure_samples_10.jsonl"))
    for c in picks:
        print(f"  [{c['model']:5s}] {c['dish_id']}  {'; '.join(c['failure_reasons'])}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
