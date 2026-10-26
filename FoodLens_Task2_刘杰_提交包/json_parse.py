"""
json_parse.py —— 把模型生成的文本稳健地解析成营养 JSON 结构

解析策略（逐级降级，保证任何输出都能得到合法结构）：
  1. 截取第一个 '{' 到最后一个 '}'，json.loads
  2. 常见修复：去尾逗号、单引号转双引号、True/False/None、缺失引号的键
  3. 正则兜底：直接抓 items 里的 name/weight_g 与 5 个数值字段
  4. 全部失败：空 items + train 均值（由调用方传入 fallback_means）
"""

import json
import re

NUM_KEYS = ["total_weight_g", "total_calories_kcal", "protein_g", "fat_g", "carbohydrate_g"]

_num_re = {k: re.compile(k + r'"\s*:\s*(-?\d+\.?\d*)') for k in NUM_KEYS}
_item_re = re.compile(r'"name"\s*:\s*"([^"]+)"[^}]*?"weight_g"\s*:\s*(-?\d+\.?\d*)')
_item_re2 = re.compile(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"weight_g"\s*:\s*(-?\d+\.?\d*)\s*\}')


def _fix_common(text):
    t = text.strip()
    t = re.sub(r",\s*([}\]])", r"\1", t)          # 去尾逗号
    t = t.replace("'", '"')
    t = re.sub(r"\bTrue\b", "true", t)
    t = re.sub(r"\bFalse\b", "false", t)
    t = re.sub(r"\bNone\b", "null", t)
    # 给未加引号的键补引号（JSON 风格的键）
    t = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)",
               lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}', t)
    return t


def parse_nutrition_text(text, fallback_means=None):
    """返回 (nutrition_dict, parse_ok)。parse_ok: 1=严格JSON 2=修复后 3=正则兜底 0=完全失败"""
    fb = dict(zip(NUM_KEYS, fallback_means)) if fallback_means is not None else \
        dict.fromkeys(NUM_KEYS, 0.0)

    if not text or "{" not in text:
        return _pack([], fb), 0

    candidate = text[text.find("{"):text.rfind("}") + 1]

    # 1. 严格解析
    try:
        obj = json.loads(candidate)
        return _validate(obj), 1
    except Exception:
        pass

    # 2. 修复后解析
    try:
        obj = json.loads(_fix_common(candidate))
        return _validate(obj), 2
    except Exception:
        pass

    # 3. 正则兜底
    items = []
    for m in _item_re2.finditer(candidate):
        w = float(m.group(2))
        if w >= 0:
            items.append({"name": m.group(1), "weight_g": w})
    nums = {}
    for k in NUM_KEYS:
        m = _num_re[k].search(candidate)
        nums[k] = float(m.group(1)) if m else fb[k]
    ok = 3 if (items or any(k in candidate for k in NUM_KEYS)) else 0
    if not items and not any(_num_re[k].search(candidate) for k in NUM_KEYS):
        return _pack([], fb), 0
    return _pack(items, nums), ok


def _validate(obj):
    """确保字段齐全、数值合法。"""
    if not isinstance(obj, dict):
        raise ValueError("not a dict")
    items = []
    for it in obj.get("items", []):
        if isinstance(it, dict) and "name" in it:
            try:
                w = max(float(it.get("weight_g", 0.0)), 0.0)
            except (TypeError, ValueError):
                w = 0.0
            items.append({"name": str(it["name"]), "weight_g": w})
    nums = {}
    for k in NUM_KEYS:
        try:
            nums[k] = max(float(obj.get(k, 0.0)), 0.0)
        except (TypeError, ValueError):
            nums[k] = 0.0
    return _pack(items, nums)


def _pack(items, nums):
    return {
        "items": items,
        "total_weight_g": float(nums.get("total_weight_g", 0.0)),
        "total_calories_kcal": float(nums.get("total_calories_kcal", 0.0)),
        "protein_g": float(nums.get("protein_g", 0.0)),
        "fat_g": float(nums.get("fat_g", 0.0)),
        "carbohydrate_g": float(nums.get("carbohydrate_g", 0.0)),
    }


if __name__ == "__main__":
    good = '{"items":[{"name":"bacon","weight_g":23.0}],"total_weight_g":23.0,' \
           '"total_calories_kcal":124.4,"protein_g":8.5,"fat_g":9.7,"carbohydrate_g":0.3}'
    print(parse_nutrition_text(good, [200, 250, 17, 12, 19]))
    broken = "{'items':[{'name':'bacon','weight_g':23.0},], 'total_weight_g':23.0}"
    print(parse_nutrition_text(broken, [200, 250, 17, 12, 19]))
    print(parse_nutrition_text('garbage output no json', [200, 250, 17, 12, 19]))
