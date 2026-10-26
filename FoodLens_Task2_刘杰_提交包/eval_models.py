"""
eval_models.py —— 任务二统一评测：模型 A / 模型 B 在 val、test 上的全部五项指标

输出：
  final_results/val_metrics.json / test_metrics.json   完整指标
  final_results/metrics_summary.csv                    汇总表（供报告引用）
  final_results/modelA_{split}_preds.jsonl             模型 A 生成文本 + 解析结果
  final_results/modelB_{split}_preds.jsonl             模型 B 结构化预测
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from metrics import evaluate_all, NUM_KEYS
from model_caption import (build_vocab_from_train, CaptionModel, DATA_ROOT)
from json_parse import parse_nutrition_text
from model_lora import CLIPLoRAMultiTask, load_ingredient_vocab, WEIGHT_NORM
from train_lora_mt import CLIPT_TF_EVAL

CKPT_A = os.path.join(DATA_ROOT, "checkpoints_caption")
CKPT_B = os.path.join(DATA_ROOT, "checkpoints_lora_mt")
OUT_DIR = os.path.join(DATA_ROOT, "final_results")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESNET_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_records(split):
    with open(os.path.join(DATA_ROOT, "annotations", f"{split}.jsonl"), "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def train_means():
    recs = load_records("train")
    v = np.array([[r["target"][k] for k in NUM_KEYS] for r in recs])
    return v.mean(axis=0), v.std(axis=0)


# =====================================================================
# 模型 A 推理：生成文本 -> 解析
# =====================================================================
class ImgDataset(Dataset):
    def __init__(self, records, transform):
        self.records, self.transform = records, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(os.path.join(DATA_ROOT, r["image"])).convert("RGB")
        return self.transform(img)


@torch.no_grad()
def infer_model_a(records, mean_np):
    vocab = build_vocab_from_train()
    ck = torch.load(os.path.join(CKPT_A, "best_model.pt"), map_location=DEVICE, weights_only=False)
    model = CaptionModel(ck["vocab_size"]).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()

    ds = ImgDataset(records, RESNET_TF)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    texts, jsons, parse_codes = [], [], []
    for imgs in tqdm(loader, desc="模型A 生成中"):
        imgs = imgs.to(DEVICE)
        seqs = model.generate(imgs, max_len=220)
        for s in seqs:
            text = vocab.decode(s)
            obj, code = parse_nutrition_text(text, fallback_means=mean_np.tolist())
            texts.append(text)
            jsons.append(obj)
            parse_codes.append(code)
    return texts, jsons, parse_codes


# =====================================================================
# 模型 B 推理：CLIP+LoRA 多任务
# =====================================================================
@torch.no_grad()
def infer_model_b(records, mean_np, std_np):
    vocab = load_ingredient_vocab()
    ck = torch.load(os.path.join(CKPT_B, "best_model.pt"), map_location=DEVICE, weights_only=False)
    model = CLIPLoRAMultiTask(num_targets=5, num_ingredients=len(vocab),
                              last_n=ck["lora_last_n"], rank=ck["rank"],
                              alpha=ck["alpha"]).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    t = ck["threshold"]

    ds = ImgDataset(records, CLIPT_TF_EVAL)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    jsons = []
    for imgs in tqdm(loader, desc="模型B 推理中"):
        imgs = imgs.to(DEVICE)
        out = model(imgs)
        nums = out["numeric"].cpu() * torch.tensor(std_np) + torch.tensor(mean_np)
        probs = torch.sigmoid(out["ingredient"]).cpu().numpy()
        wts = out["item_weight"].cpu().numpy() * WEIGHT_NORM
        for i in range(imgs.size(0)):
            n = {k: max(float(nums[i, j]), 0.0) for j, k in enumerate(NUM_KEYS)}
            idx = np.where(probs[i] >= t)[0][:16]
            items = []
            for j in idx:
                w = max(float(wts[i, j]), 0.0)
                items.append({"name": vocab[j], "weight_g": w})
            # 分项重量按回归的总重等比缩放，保证 items 之和与 total_weight 一致
            s = sum(it["weight_g"] for it in items)
            if s > 0:
                scale = n["total_weight_g"] / s
                for it in items:
                    it["weight_g"] = round(it["weight_g"] * scale, 1)
            else:
                items = []
            jsons.append({"items": items, **{k: round(n[k], 1) for k in NUM_KEYS}})
    return jsons


# =====================================================================
# 主流程
# =====================================================================
def main():
    mean_np, std_np = train_means()
    summary_rows = []

    for split in ["val", "test"]:
        records = load_records(split)
        print(f"\n########## {split} 集 ({len(records)} 样本) ##########")

        texts_a, jsons_a, codes_a = infer_model_a(records, mean_np)
        with open(os.path.join(OUT_DIR, f"modelA_{split}_preds.jsonl"), "w", encoding="utf-8") as f:
            for r, t, j, c in zip(records, texts_a, jsons_a, codes_a):
                f.write(json.dumps({"dish_id": r["dish_id"], "text": t,
                                    "parsed": j, "parse_code": c}, ensure_ascii=False) + "\n")
        strict = sum(1 for c in codes_a if c == 1)
        print(f"模型A 严格JSON解析率: {strict}/{len(codes_a)} = {strict/len(codes_a)*100:.1f}%")

        jsons_b = infer_model_b(records, mean_np, std_np)
        with open(os.path.join(OUT_DIR, f"modelB_{split}_preds.jsonl"), "w", encoding="utf-8") as f:
            for r, j in zip(records, jsons_b):
                f.write(json.dumps({"dish_id": r["dish_id"], "parsed": j}, ensure_ascii=False) + "\n")

        mA = evaluate_all(jsons_a, records)
        mB = evaluate_all(jsons_b, records)

        with open(os.path.join(OUT_DIR, f"{split}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"modelA_cnn_gru": mA, "modelB_clip_lora": mB}, f,
                      indent=2, ensure_ascii=False)

        print("\n指标                    模型A(CNN+GRU)   模型B(CLIP+LoRA)")
        for name, key in [("BLEU-1", "bleu_1"), ("BLEU-4", "bleu_4"),
                          ("ROUGE-L", "rouge_l"), ("CIDEr-D", "cider_d"),
                          ("F1(micro)", "micro_f1"), ("F1(macro)", "macro_f1")]:
            print(f"  {name:12s}        {mA[key]:.4f}          {mB[key]:.4f}")
        print("  数值 MSE/MAE（按字段）:")
        for k in NUM_KEYS:
            print(f"    {k:22s} A: MSE={mA['numeric'][k]['mse']:9.1f} MAE={mA['numeric'][k]['mae']:7.2f}"
                  f"   B: MSE={mB['numeric'][k]['mse']:9.1f} MAE={mB['numeric'][k]['mae']:7.2f}")
        print(f"    {'平均':22s} A: MSE={mA['numeric']['mean_mse']:9.1f} MAE={mA['numeric']['mean_mae']:7.2f}"
              f"   B: MSE={mB['numeric']['mean_mse']:9.1f} MAE={mB['numeric']['mean_mae']:7.2f}")

        for tag, m in [("A", mA), ("B", mB)]:
            summary_rows.append({
                "split": split, "model": f"model{tag}_" + ("cnn_gru" if tag == "A" else "clip_lora"),
                "bleu_1": round(m["bleu_1"], 4), "bleu_4": round(m["bleu_4"], 4),
                "rouge_l": round(m["rouge_l"], 4), "cider_d": round(m["cider_d"], 4),
                "micro_f1": round(m["micro_f1"], 4), "macro_f1": round(m["macro_f1"], 4),
                "mean_mse": round(m["numeric"]["mean_mse"], 2),
                "mean_mae": round(m["numeric"]["mean_mae"], 2),
                "mae_weight": round(m["numeric"]["total_weight_g"]["mae"], 2),
                "mae_calories": round(m["numeric"]["total_calories_kcal"]["mae"], 2),
            })

    import csv
    with open(os.path.join(OUT_DIR, "metrics_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n汇总表已保存: {os.path.join(OUT_DIR, 'metrics_summary.csv')}")


if __name__ == "__main__":
    main()
