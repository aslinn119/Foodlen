"""
metrics.py —— Task 2 统一评测指标（自包含实现，无第三方依赖）

评测标准（课程要求）：任务一指标 BLEU / ROUGE-L / CIDEr-D + F1、MSE

所有文本类指标都作用在「营养 JSON 的序列化文本」上：
  - 模型 A（图像描述模型）：直接生成 JSON 文本，与 target_text 对比
  - 模型 B（多模态预训练模型）：把预测的结构化结果按同一格式序列化后对比

tokenize: 字母数字连续段为词，其余单个符号（{ } [ ] " : , . 等）各为一个 token。
这样 JSON 结构符号参与匹配，数值按位拆开（64.0 -> 64 . 0），与 captioning 社区
对数字的处理方式一致，且两个模型完全同标准。
"""

import re
import math
import json
import numpy as np
from collections import Counter

# =====================================================================
# 1. tokenization
# =====================================================================

def tokenize(text):
    """把 JSON 文本切成 token 序列。"""
    return re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text)


def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# =====================================================================
# 2. BLEU（corpus 级，smoothing method 7，即 p_n = (matched+1)/(possible+1)）
# =====================================================================

def corpus_bleu(hyps, refs, max_n=4):
    """
    hyps: list[str]，refs: list[str]（一一对应，单参考）
    返回 dict: bleu_1..bleu_4（bleu_4 为通常所说的 BLEU）
    """
    assert len(hyps) == len(refs)
    p_ns, brev = [], 0.0
    c, r = 0, 0
    for n in range(1, max_n + 1):
        matched, possible = 0, 0
        for hyp, ref in zip(hyps, refs):
            ht, rt = tokenize(hyp), tokenize(ref)
            h_ng, r_ng = ngrams(ht, n), ngrams(rt, n)
            overlap = h_ng & r_ng
            matched += sum(overlap.values())
            possible += max(len(ht) - n + 1, 0)
        p_ns.append((matched + 1.0) / (possible + 1.0))
        if n == 1:
            for hyp, ref in zip(hyps, refs):
                c += len(tokenize(hyp))
                r += len(tokenize(ref))
    if c == 0:
        brev = 0.0
    elif c > r:
        brev = 1.0
    else:
        brev = math.exp(1.0 - float(r) / c)
    log_p = sum(math.log(p) for p in p_ns) / max_n
    out = {"brevity_penalty": brev}
    for i, p in enumerate(p_ns, start=1):
        out[f"bleu_{i}"] = p
    out["bleu_4"] = brev * math.exp(log_p)  #  headline BLEU（平滑后）
    return out


# =====================================================================
# 3. ROUGE-L（LCS F-measure，beta = 1.2，与 pycocoevalcap 一致）
# =====================================================================

def _lcs_len(a, b):
    """经典 DP 求 LCS 长度（token 序列短，直接 O(|a||b|)）。"""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def rouge_l(hyps, refs, beta=1.2):
    """返回 corpus 平均 ROUGE-L F。"""
    scores = []
    for hyp, ref in zip(hyps, refs):
        ht, rt = tokenize(hyp), tokenize(ref)
        if len(ht) == 0 or len(rt) == 0:
            scores.append(0.0)
            continue
        lcs = _lcs_len(ht, rt)
        prec = lcs / len(ht)
        rec = lcs / len(rt)
        if prec + rec == 0:
            scores.append(0.0)
        else:
            scores.append((1 + beta ** 2) * prec * rec / (rec + beta ** 2 * prec))
    return float(np.mean(scores))


# =====================================================================
# 4. CIDEr-D（单参考版本，DF 由评测语料自身的参考文本统计）
# =====================================================================

def _tf_idf_vectors(hyps_tokens, refs_tokens, n):
    """
    返回每个样本的 (hyp_vec, ref_vec)：
    vec[tuple_ngram] = tf * idf，idf = log(N / df)，df 由参考语料统计。
    CIDEr-D 对参考/hypothesis 的 n-gram count 做 clipping（以对方为上限）。
    """
    N = len(refs_tokens)
    df = Counter()
    for rt in refs_tokens:
        df.update(set(ngrams(rt, n).keys()))
    idf = {g: math.log(float(N) / max(df[g], 1)) for g in df}

    def vec(tokens, clip_with=None):
        ng = ngrams(tokens, n)
        if clip_with is not None:
            ng = Counter({g: min(c, clip_with.get(g, 0)) for g, c in ng.items()})
        return {g: c * idf.get(g, 0.0) for g, c in ng.items()}

    return [(vec(h, clip_with=ngrams(r, n)), vec(r, clip_with=ngrams(h, n)))
            for h, r in zip(hyps_tokens, refs_tokens)]


def cider_d(hyps, refs, max_n=4):
    """CIDEr-D：n=1..4 的 TF-IDF 余弦均值（x10），带长度指数惩罚。"""
    assert len(hyps) == len(refs)
    hyps_tokens = [tokenize(h) for h in hyps]
    refs_tokens = [tokenize(r) for r in refs]
    N = len(hyps)

    # 每个 n 预先算好全部样本的 TF-IDF 向量（df 由整个参考语料统计）
    per_n = [_tf_idf_vectors(hyps_tokens, refs_tokens, n) for n in range(1, max_n + 1)]

    scores = []
    for i in range(N):
        hyp, ref = hyps_tokens[i], refs_tokens[i]
        # CIDEr-D 长度惩罚：delta = |len(h)-len(r)| / len(r)
        delta = float(abs(len(hyp) - len(ref))) / max(len(ref), 1)
        val = 0.0
        for n in range(1, max_n + 1):
            vh, vr = per_n[n - 1][i]
            dot = sum(v * vr.get(g, 0.0) for g, v in vh.items())
            nh = math.sqrt(sum(v * v for v in vh.values()))
            nr = math.sqrt(sum(v * v for v in vr.values()))
            cos = (dot / (nh * nr)) if nh > 0 and nr > 0 else 0.0
            val += max(cos, 0.0) * 10.0
        val /= max_n
        val *= math.exp(-delta)  # 长度惩罚
        scores.append(val)
    return float(np.mean(scores))


# =====================================================================
# 5. 食材 F1（集合匹配，忽略顺序与重复）
# =====================================================================

def ingredient_f1(hyp_items, ref_items):
    """
    hyp_items / ref_items: list[str] 食材名列表。
    返回 dict: micro / macro / sample_mean 三套 F1，以及 precision / recall。
    """
    def to_set(items):
        return set(items)

    tp = fp = fn = 0
    per_class = {}  # class -> [tp, fp, fn]
    sample_f1s, sample_ps, sample_rs = [], [], []
    for h, r in zip(hyp_items, ref_items):
        hs, rs = to_set(h), to_set(r)
        hit = len(hs & rs)
        t, f_p, f_n = hit, len(hs) - hit, len(rs) - hit
        tp += t; fp += f_p; fn += f_n
        for name in hs | rs:
            c = per_class.setdefault(name, [0, 0, 0])
            if name in hs and name in rs:
                c[0] += 1
            elif name in hs:
                c[1] += 1
            else:
                c[2] += 1
        p = t / max(t + f_p, 1)
        rec = t / max(t + f_n, 1)
        f1 = 2 * p * rec / max(p + rec, 1e-9)
        sample_f1s.append(f1)
        sample_ps.append(p)
        sample_rs.append(rec)

    micro_p = tp / max(tp + fp, 1)
    micro_r = tp / max(tp + fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)

    class_f1s = []
    for t, f_p, f_n in per_class.values():
        p = t / max(t + f_p, 1)
        rec = t / max(t + f_n, 1)
        if p + rec > 0:
            class_f1s.append(2 * p * rec / (p + rec))
    macro_f1 = float(np.mean(class_f1s)) if class_f1s else 0.0

    return {
        "micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "sample_p": float(np.mean(sample_ps)),
        "sample_r": float(np.mean(sample_rs)),
        "sample_f1": float(np.mean(sample_f1s)),
    }


# =====================================================================
# 6. 数值回归指标（MSE / MAE / RMSE，按字段）
# =====================================================================

NUM_KEYS = ["total_weight_g", "total_calories_kcal",
            "protein_g", "fat_g", "carbohydrate_g"]


def numeric_metrics(hyps, refs):
    """
    hyps / refs: list[dict]，每个 dict 含 NUM_KEYS 五个数值字段。
    返回每个字段的 MSE / MAE / RMSE 以及总体平均。
    """
    H = np.array([[h[k] for k in NUM_KEYS] for h in hyps], dtype=np.float64)
    R = np.array([[r[k] for k in NUM_KEYS] for r in refs], dtype=np.float64)
    err = H - R
    out = {}
    for i, k in enumerate(NUM_KEYS):
        out[k] = {
            "mse": float(np.mean(err[:, i] ** 2)),
            "mae": float(np.mean(np.abs(err[:, i]))),
            "rmse": float(np.sqrt(np.mean(err[:, i] ** 2))),
        }
    out["mean_mse"] = float(np.mean(err ** 2))
    out["mean_mae"] = float(np.mean(np.abs(err)))
    out["mean_rmse"] = float(np.sqrt(np.mean(err ** 2)))
    return out


# =====================================================================
# 7. JSON 文本序列化（模型 B 用，与 target_text 同格式）
# =====================================================================

def nutrition_to_text(target):
    """把结构化营养 dict 序列化成与 target_text 完全相同的格式。"""
    obj = {
        "items": [
            {"name": it["name"], "weight_g": round(float(it["weight_g"]), 1)}
            for it in target["items"]
        ],
        "total_weight_g": round(float(target["total_weight_g"]), 1),
        "total_calories_kcal": round(float(target["total_calories_kcal"]), 1),
        "protein_g": round(float(target["protein_g"]), 1),
        "fat_g": round(float(target["fat_g"]), 1),
        "carbohydrate_g": round(float(target["carbohydrate_g"]), 1),
    }
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# =====================================================================
# 8. 汇总入口：给定预测的完整 JSON dict 列表，算全部指标
# =====================================================================

def evaluate_all(hyp_jsons, ref_records):
    """
    hyp_jsons: list[dict]，结构与 target 相同（含 items 和 5 个数值字段）
    ref_records: list[dict]，annotations 里的 target 字段
    返回完整指标 dict。
    """
    hyps_text = [nutrition_to_text(h) for h in hyp_jsons]
    refs_text = [r["target_text"] for r in ref_records]
    hyps_items = [[it["name"] for it in h["items"]] for h in hyp_jsons]
    refs_items = [[it["name"] for it in r["target"]["items"]] for r in ref_records]

    result = {}
    result.update(corpus_bleu(hyps_text, refs_text))
    result["rouge_l"] = rouge_l(hyps_text, refs_text)
    result["cider_d"] = cider_d(hyps_text, refs_text)
    result.update(ingredient_f1(hyps_items, refs_items))
    result["numeric"] = numeric_metrics(hyp_jsons, [r["target"] for r in ref_records])
    return result


if __name__ == "__main__":
    # 自测：完全相同的预测应得到 F1=1，BLEU/ROUGE/CIDEr 接近满分
    refs = [{"items": [{"name": "bacon", "weight_g": 23.0}],
             "total_weight_g": 23.0, "total_calories_kcal": 124.4,
             "protein_g": 8.5, "fat_g": 9.7, "carbohydrate_g": 0.3}]
    recs = [{"target": refs[0], "target_text": nutrition_to_text(refs[0])}]
    print("=== 自测：完美复制 ===")
    print(json.dumps(evaluate_all(refs, recs), indent=2, ensure_ascii=False))

    # 自测：全错
    bad = [{"items": [{"name": "salad", "weight_g": 200.0}],
            "total_weight_g": 200.0, "total_calories_kcal": 999.9,
            "protein_g": 0.0, "fat_g": 0.0, "carbohydrate_g": 0.0}]
    print("=== 自测：全错 ===")
    print(json.dumps(evaluate_all(bad, recs), indent=2, ensure_ascii=False))
