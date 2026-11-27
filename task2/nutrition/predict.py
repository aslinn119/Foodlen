"""
predict.py —— Task 2 统一推理接口（组内集成入口）

predict_nutrition(image_path, model_id="b") -> dict

model_id:
  "a"    模型 A：CNN+GRU 生成式（任务一描述模型结构，ResNet18 整体表示 + GRU 解码）
  "b"    模型 B：CLIP + LoRA 参数高效微调多任务模型（结构化预测 + 程序组装）
  "bgen" 模型 B-gen：CLIP + LoRA（同 B 权重）+ 自回归字符解码器（生成式）

三个模型返回完全相同的 Schema：
{
  "items": [{"name": "bacon", "weight_g": 23.0}, ...],
  "total_weight_g": ...,
  "total_calories_kcal": ...,
  "protein_g": ...,
  "fat_g": ...,
  "carbohydrate_g": ...
}
数值单位固定：重量 g、热量 kcal、三大营养素 g，均为非负、一位小数。

模型按需懒加载（全局缓存），同进程内重复调用零额外加载开销。
verbose 版本额外返回生成文本 / 解析等级 / 字段一致性警告，供联调与错误分析。
"""

import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_KEYS = ["total_weight_g", "total_calories_kcal",
               "protein_g", "fat_g", "carbohydrate_g"]

MODEL_IDS = {"a", "b", "bgen"}
MODEL_NAMES = {
    "a": "模型A: CNN+GRU 生成式",
    "b": "模型B: CLIP+LoRA 结构化",
    "bgen": "模型B-gen: CLIP+LoRA+GRU 生成式",
}

# B-gen 推理配置：只影响解码，不需要重新训练权重。
BGEN_BEAM = 4
BGEN_LENGTH_PENALTY = 0.6
BGEN_CONSTRAIN_JSON = True

CLIP_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4815, 0.4578, 0.4082],
                         std=[0.2686, 0.2613, 0.2758]),
])
RESNET_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_loaders = {}


# =====================================================================
# 各模型懒加载
# =====================================================================
def _load_a():
    from model_caption import build_vocab_from_train, CaptionModel
    ck = torch.load(os.path.join(DATA_ROOT, "checkpoints_caption", "best_model.pt"),
                    map_location=DEVICE, weights_only=False)
    vocab = build_vocab_from_train()
    model = CaptionModel(ck["vocab_size"], pretrained=False).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return {"model": model, "vocab": vocab, "mean": _train_means()}


def _load_b(checkpoint_dir="checkpoints_lora_mt"):
    from model_lora import CLIPLoRAMultiTask, load_ingredient_vocab, WEIGHT_NORM
    ck = torch.load(os.path.join(DATA_ROOT, checkpoint_dir, "best_model.pt"),
                    map_location=DEVICE, weights_only=False)
    vocab = load_ingredient_vocab()
    model = CLIPLoRAMultiTask(num_targets=5, num_ingredients=len(vocab),
                              last_n=ck["lora_last_n"], rank=ck["rank"],
                              alpha=ck["alpha"], pretrained=False).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return {"model": model, "vocab": vocab, "threshold": ck["threshold"],
            "mean": torch.tensor(ck["mean"], dtype=torch.float32),
            "std": torch.tensor(ck["std"], dtype=torch.float32),
            "weight_norm": WEIGHT_NORM}


def _load_bgen():
    from model_bgen import build_char_vocab, load_bgen_decoder
    char2idx, idx2char = build_char_vocab()
    model = load_bgen_decoder(os.path.join(DATA_ROOT, "checkpoints_bgen", "best_decoder.pt"),
                              DEVICE)
    return {"model": model, "char2idx": char2idx, "idx2char": idx2char,
            "mean": _train_means()}


def _get(model_id):
    if model_id not in MODEL_IDS:
        raise ValueError(f"未知模型 id: {model_id!r}，可选 {sorted(MODEL_IDS)}")
    if model_id not in _loaders:
        _loaders[model_id] = {"a": _load_a, "b": _load_b, "bgen": _load_bgen}[model_id]()
    return _loaders[model_id]


def _train_means():
    return _train_targets().mean(axis=0)


def _train_targets():
    with open(os.path.join(DATA_ROOT, "annotations", "train.jsonl"), encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    return np.array([[r["target"][k] for k in TARGET_KEYS] for r in recs])


@torch.no_grad()
def predict_challenge_baseline(image_path, model_id):
    """Return the five nutrition values for mean, resnet18, or clip."""
    if model_id == "clip":
        result = predict_nutrition(image_path, "b", legacy=True)
        return {key: result[key] for key in TARGET_KEYS}
    if model_id not in ("mean", "resnet18"):
        raise ValueError(f"未知挑战基线: {model_id!r}")
    if model_id not in _loaders:
        targets = _train_targets()
        mean = targets.mean(axis=0)
        if model_id == "mean":
            _loaders[model_id] = (mean,)
        else:
            from model import NutritionRegressor
            model = NutritionRegressor(pretrained=False)
            model.load_state_dict(torch.load(
                os.path.join(DATA_ROOT, "checkpoints", "best_model.pt"),
                map_location="cpu", weights_only=True,
            ))
            _loaders[model_id] = (mean, targets.std(axis=0), model.to(DEVICE).eval())
    state = _loaders[model_id]
    if model_id == "mean":
        values = state[0]
    else:
        mean, std, model = state
        with Image.open(image_path) as image:
            x = RESNET_TF(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        values = model(x)[0].cpu().numpy() * std + mean
    return {key: float(value) for key, value in zip(TARGET_KEYS, values)}


# =====================================================================
# 各模型推理
# =====================================================================
@torch.no_grad()
def _infer_a(img, s):
    from json_parse import parse_nutrition_text
    x = RESNET_TF(img).unsqueeze(0).to(DEVICE)
    seq = s["model"].generate(x, max_len=220)[0]
    text = s["vocab"].decode(seq)
    obj, code = parse_nutrition_text(text, fallback_means=s["mean"].tolist())
    return obj, text, code


@torch.no_grad()
def _infer_b(img, s, calibrated=False, rank_overflow=True):
    from model_lora import WEIGHT_NORM
    x = CLIP_TF(img).unsqueeze(0).to(DEVICE)
    out = s["model"](x)
    nums = out["numeric"].cpu() * s["std"] + s["mean"]
    probs = torch.sigmoid(out["ingredient"]).cpu().numpy()[0]
    wts = out["item_weight"].cpu().numpy()[0] * WEIGHT_NORM
    n = {k: max(float(nums[0, j]), 0.0) for j, k in enumerate(TARGET_KEYS)}
    idx = select_b_ingredients(probs, 0.75 if calibrated else s["threshold"],
                               score_order=calibrated or rank_overflow)
    if s.get("require_items") and len(idx) == 0:
        idx = np.array([int(np.argmax(probs))])
    items = [{"name": s["vocab"][j], "weight_g": max(float(wts[j]), 0.0)} for j in idx]
    total_w = sum(it["weight_g"] for it in items)
    if s.get("require_items"):
        from constrained_caption import reconcile_weights
        obj = reconcile_weights({"items": items, **{k: round(n[k], 1) for k in TARGET_KEYS}})
        return obj, None, 1
    if total_w > 0:  # 分项重量等比缩放到与预测总重一致
        scale = n["total_weight_g"] / total_w
        for it in items:
            it["weight_g"] = round(it["weight_g"] * scale, 1)
    else:
        items = []
    obj = {"items": items, **{k: round(n[k], 1) for k in TARGET_KEYS}}
    return obj, None, 1


def select_b_ingredients(probs, threshold, *, score_order=True):
    idx = np.flatnonzero(probs >= threshold)
    if score_order and len(idx) > 16:
        idx = idx[np.argsort(-probs[idx], kind="stable")]
        idx = np.sort(idx[:16])  # retain vocabulary order in the output
    return idx[:16]


@torch.no_grad()
def _infer_bgen(img, s):
    from json_parse import parse_nutrition_text
    from model_bgen import decode_text
    enc = s["model"]
    f = enc.extract_features(CLIP_TF(img).unsqueeze(0).to(DEVICE))
    seq = enc.generate(
        f,
        s["char2idx"],
        beam=BGEN_BEAM,
        length_penalty=BGEN_LENGTH_PENALTY,
        constrain_json=BGEN_CONSTRAIN_JSON,
    )[0]
    text = decode_text(seq, s["idx2char"])
    obj, code = parse_nutrition_text(text, fallback_means=s["mean"].tolist())
    meta = {
        "beam": BGEN_BEAM,
        "length_penalty": BGEN_LENGTH_PENALTY,
        "constrain_json": BGEN_CONSTRAIN_JSON,
        "generated_chars": len(text),
    }
    return obj, text, code, meta


# =====================================================================
# 输出一致性检查（Schema / 单位 / 空 items / 分项-总重一致性）
# =====================================================================
def check_schema(nutrition):
    """按 schema.json 的硬性约束检查：字段齐全、类型正确、数值非负。
    返回违规列表；空列表 = Schema 有效。（分项-总重一致性等软约束不在此列）"""
    bad = []
    if not isinstance(nutrition, dict):
        return ["输出不是 dict"]
    allowed = set(TARGET_KEYS) | {"items"}
    extra = sorted(set(nutrition) - allowed)
    if extra:
        bad.append(f"存在未定义字段: {extra}")
    for k in TARGET_KEYS:
        if k not in nutrition:
            bad.append(f"缺失数值字段 {k}")
        else:
            try:
                if type(nutrition[k]) not in (int, float):
                    raise ValueError
                value = float(nutrition[k])
                if not np.isfinite(value):
                    bad.append(f"{k} 不是有限数值")
                elif value < 0:
                    bad.append(f"{k} 为负值")
            except (TypeError, ValueError):
                bad.append(f"{k} 不是数值")
    items = nutrition.get("items")
    if items is None or not isinstance(items, list):
        bad.append("items 缺失或不是列表")
    else:
        if len(items) > 16:
            bad.append(f"items 超过上限 16（实际 {len(items)}）")
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                bad.append(f"items[{i}] 不是对象")
                continue
            extra_item = sorted(set(it) - {"name", "weight_g"})
            if extra_item:
                bad.append(f"items[{i}] 存在未定义字段: {extra_item}")
            if not isinstance(it.get("name"), str) or not it["name"]:
                bad.append(f"items[{i}].name 缺失或非字符串")
            try:
                if type(it.get("weight_g")) not in (int, float):
                    raise ValueError
                value = float(it["weight_g"])
                if not np.isfinite(value):
                    bad.append(f"items[{i}].weight_g 不是有限数值")
                elif value < 0:
                    bad.append(f"items[{i}].weight_g 为负值")
            except (TypeError, ValueError):
                bad.append(f"items[{i}].weight_g 不是数值")
    return bad


def validate_output(nutrition):
    """返回全部警告列表（含 Schema 违规与软约束）；空列表表示完全合规。"""
    warns = check_schema(nutrition)
    if not isinstance(nutrition, dict):
        return warns
    items = nutrition.get("items")
    if isinstance(items, list):
        if len(items) == 0:
            warns.append("items 为空（未识别出任何食材）")
        s = sum(float(it.get("weight_g", 0.0)) for it in items
                if isinstance(it, dict) and it.get("weight_g") is not None) if items else 0.0
        try:
            tw = float(nutrition.get("total_weight_g", 0.0))
            if items and abs(s - tw) > 5.0:
                warns.append(f"分项重量和({s:.1f}g)与 total_weight_g({tw:.1f}g)相差超过 5g")
        except (TypeError, ValueError):
            pass
    return warns


# =====================================================================
# 统一入口
# =====================================================================
@torch.no_grad()
def predict_nutrition_verbose(image_path, model_id="b", *, bgen_deduplicate=False,
                              bgen_fallback_to_b=False, b_calibrated=False,
                              b_rank_overflow=None, legacy=False):
    """返回 {model, model_name, nutrition, text, parse_code, warnings}。"""
    if model_id not in MODEL_IDS:
        raise ValueError(f"未知模型 id: {model_id!r}，可选 {sorted(MODEL_IDS)}")
    if bgen_deduplicate and model_id != "bgen":
        raise ValueError("bgen_deduplicate 仅适用于 bgen")
    if bgen_fallback_to_b and model_id != "bgen":
        raise ValueError("bgen_fallback_to_b 仅适用于 bgen")
    if b_calibrated and model_id != "b":
        raise ValueError("b_calibrated 仅适用于 b")
    if b_rank_overflow and model_id != "b":
        raise ValueError("b_rank_overflow 仅适用于 b")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    if not legacy and model_id in ("a", "b"):
        if b_calibrated or b_rank_overflow is False:
            raise ValueError("Legacy B options require legacy=True")
        from predict_final import predict_nutrition_verbose as final_predict
        result = final_predict(image_path, model_id)
        result.update(model_name=MODEL_NAMES[model_id], warnings=validate_output(result["nutrition"]),
                      generation={"version": "20261119-v3", "policy": result["postprocess"]})
        return result
    s = _get(model_id)
    img = Image.open(image_path).convert("RGB")
    if model_id == "b":
        result = _infer_b(img, s, calibrated=b_calibrated, rank_overflow=b_rank_overflow is not False)
    else:
        result = {"a": _infer_a, "bgen": _infer_bgen}[model_id](img, s)
    if model_id == "bgen":
        obj, text, code, generation = result
        if code == 0 and bgen_fallback_to_b:
            obj, _, _ = _infer_b(img, _get("b"))
            generation["fallback_model"] = "b"
        elif bgen_deduplicate:
            from postprocess_bgen import merge_items
            obj = merge_items(obj, largest_only=True)
            generation["item_postprocess"] = "keep_largest_duplicate"
    else:
        obj, text, code = result
        generation = None
    return {
        "model": model_id,
        "model_name": MODEL_NAMES[model_id],
        "nutrition": obj,
        "text": text,               # 生成式模型（a/bgen）的原始 JSON 文本，b 为 None
        "parse_code": code,         # 1 严格 / 2 修复 / 3 正则兜底 / 0 失败 / b 恒为 1
        "warnings": validate_output(obj),
        "generation": generation,
    }


def predict_nutrition(image_path, model_id="b", *, bgen_deduplicate=False,
                      bgen_fallback_to_b=False, b_calibrated=False,
                      b_rank_overflow=None, legacy=False):
    """组内标准接口：输入图片路径，返回完整营养 JSON dict（三模型同 Schema）。"""
    return predict_nutrition_verbose(
        image_path, model_id, bgen_deduplicate=bgen_deduplicate,
        bgen_fallback_to_b=bgen_fallback_to_b,
        b_calibrated=b_calibrated,
        b_rank_overflow=b_rank_overflow, legacy=legacy,
    )["nutrition"]


if __name__ == "__main__":
    test_img = os.path.join(DATA_ROOT, "demo_images")
    sample = sorted(os.listdir(test_img))[0]
    sample_path = os.path.join(test_img, sample)
    for mid in ["a", "b", "bgen"]:
        r = predict_nutrition_verbose(sample_path, mid)
        print(f"===== {r['model_name']} =====")
        print(json.dumps(r["nutrition"], indent=2, ensure_ascii=False))
        if r["warnings"]:
            print("警告:", r["warnings"])
        print()
