"""
eval_model_bgen.py —— B-gen（CLIP+LoRA 编码器 + 自回归字符解码器）统一评测

流程：缓存特征 -> 贪心解码生成 JSON 文本 -> json_parse 逐级解析
      -> metrics.evaluate_all 六项课程指标
      -> 解析健康度统计（严格解析率 / 修复率 / 回退率 / 空 items 率 / 分项-总重一致性）

输出：
  final_results/modelBgen_{split}_preds.jsonl   生成文本 + 解析结果
  final_results/bgen_metrics.json                val/test 全部指标
  final_results/bgen_summary.csv                 汇总表（供报告引用）
"""

import os
import json
import csv
import numpy as np
import torch
from tqdm import tqdm

from metrics import evaluate_all, NUM_KEYS
from json_parse import parse_nutrition_text
from model_bgen import build_char_vocab, decode_text, load_bgen_decoder, DATA_ROOT
from data_bgen import BGenSeqDataset

CKPT_DIR = os.path.join(DATA_ROOT, "checkpoints_bgen")
OUT_DIR = os.path.join(DATA_ROOT, "final_results")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_records(split):
    with open(os.path.join(DATA_ROOT, "annotations", f"{split}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def train_means():
    recs = load_records("train")
    v = np.array([[r["target"][k] for k in NUM_KEYS] for r in recs])
    return v.mean(axis=0)


def load_model():
    char2idx, idx2char = build_char_vocab()
    model = load_bgen_decoder(os.path.join(CKPT_DIR, "best_decoder.pt"), DEVICE)
    return model, char2idx, idx2char


@torch.no_grad()
def infer(split, model, char2idx, idx2char, mean_np, batch_size=128, beam=4):
    ds = BGenSeqDataset(split)
    feats = torch.from_numpy(ds.feats).to(DEVICE)
    texts, jsons, codes = [], [], []
    for i in tqdm(range(0, len(ds), batch_size), desc=f"B-gen 生成 {split} (beam={beam})"):
        seqs = model.generate(feats[i:i + batch_size], char2idx, beam=beam)
        for s in seqs:
            text = decode_text(s, idx2char)
            obj, code = parse_nutrition_text(text, fallback_means=mean_np.tolist())
            texts.append(text)
            jsons.append(obj)
            codes.append(code)
    return texts, jsons, codes


def parse_stats(jsons, codes):
    n = len(codes)
    strict = sum(1 for c in codes if c == 1) / n
    repaired = sum(1 for c in codes if c == 2) / n
    regex_fb = sum(1 for c in codes if c == 3) / n
    total_fb = 1.0 - strict
    empty_items = sum(1 for j in jsons if len(j["items"]) == 0) / n
    # 分项重量和 vs total_weight 的一致性（仅对非空 items 样本）
    diffs = []
    for j in jsons:
        if j["items"]:
            diffs.append(abs(sum(it["weight_g"] for it in j["items"]) - j["total_weight_g"]))
    consistency = {
        "covered": len(diffs), "n": n,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else 0.0,
        "max_abs_diff": float(np.max(diffs)) if diffs else 0.0,
        "within_5g_ratio": float(np.mean([d <= 5.0 for d in diffs])) if diffs else 0.0,
    }
    return {"strict_rate": strict, "repaired_rate": repaired, "regex_fallback_rate": regex_fb,
            "total_fallback_rate": total_fb, "empty_items_rate": empty_items,
            "weight_consistency": consistency}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=4, help="束宽；1 = 贪心解码")
    ap.add_argument("--tag", type=str, default="",
                    help="输出文件后缀，如 --tag beam4 生成 bgen_metrics_beam4.json（不覆盖默认结果）")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    mean_np = train_means()
    model, char2idx, idx2char = load_model()
    records_all = {}
    out = {}

    for split in ["val", "test"]:
        records = load_records(split)
        records_all[split] = records
        texts, jsons, codes = infer(split, model, char2idx, idx2char, mean_np, beam=args.beam)

        with open(os.path.join(OUT_DIR, f"modelBgen_{split}_preds{tag}.jsonl"), "w", encoding="utf-8") as f:
            for r, t, j, c in zip(records, texts, jsons, codes):
                f.write(json.dumps({"dish_id": r["dish_id"], "text": t,
                                    "parsed": j, "parse_code": c}, ensure_ascii=False) + "\n")

        m = evaluate_all(jsons, records)
        ps = parse_stats(jsons, codes)
        out[split] = {"metrics": m, "parse_stats": ps, "n": len(records)}

        print(f"\n########## B-gen {split} 集 ({len(records)} 样本) ##########")
        print(f"  BLEU-1={m['bleu_1']:.4f}  BLEU-4={m['bleu_4']:.4f}  "
              f"ROUGE-L={m['rouge_l']:.4f}  CIDEr-D={m['cider_d']:.4f}")
        print(f"  F1 micro={m['micro_f1']:.4f}  macro={m['macro_f1']:.4f}  "
              f"sample={m['sample_f1']:.4f}")
        print(f"  数值: mean_MSE={m['numeric']['mean_mse']:.1f}  "
              f"mean_MAE={m['numeric']['mean_mae']:.2f}  mean_RMSE={m['numeric']['mean_rmse']:.2f}")
        print(f"  解析: 严格={ps['strict_rate']*100:.1f}%  修复={ps['repaired_rate']*100:.1f}%  "
              f"正则兜底={ps['regex_fallback_rate']*100:.1f}%  空items={ps['empty_items_rate']*100:.1f}%")
        c = ps["weight_consistency"]
        print(f"  重量一致性: 覆盖{c['covered']}/{c['n']}  平均绝对差={c['mean_abs_diff']:.1f}g  "
              f"最大={c['max_abs_diff']:.1f}g  ±5g内={c['within_5g_ratio']*100:.1f}%")

    with open(os.path.join(OUT_DIR, f"bgen_metrics{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    rows = []
    for split in ["val", "test"]:
        m, ps = out[split]["metrics"], out[split]["parse_stats"]
        rows.append({"split": split, "model": "modelBgen_clip_lora_gen",
                     "bleu_1": round(m["bleu_1"], 4), "bleu_4": round(m["bleu_4"], 4),
                     "rouge_l": round(m["rouge_l"], 4), "cider_d": round(m["cider_d"], 4),
                     "micro_f1": round(m["micro_f1"], 4), "macro_f1": round(m["macro_f1"], 4),
                     "mean_mse": round(m["numeric"]["mean_mse"], 2),
                     "mean_mae": round(m["numeric"]["mean_mae"], 2),
                     "mae_weight": round(m["numeric"]["total_weight_g"]["mae"], 2),
                     "mae_calories": round(m["numeric"]["total_calories_kcal"]["mae"], 2),
                     "strict_json_rate": round(ps["strict_rate"], 4),
                     "fallback_rate": round(ps["total_fallback_rate"], 4),
                     "empty_items_rate": round(ps["empty_items_rate"], 4)})
    with open(os.path.join(OUT_DIR, f"bgen_summary{tag}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n指标已保存: bgen_metrics{tag}.json / bgen_summary{tag}.csv")


if __name__ == "__main__":
    main()
