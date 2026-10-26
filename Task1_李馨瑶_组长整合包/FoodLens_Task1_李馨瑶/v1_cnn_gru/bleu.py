# -*- coding: utf-8 -*-
"""自包含的语料级 BLEU-1~4 评测（无第三方依赖，仅用于训练过程监控）。
正式指标请以 pycocoevalcap 为准（中期实验接入）。"""
from collections import Counter


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    """hyps/refs: token 列表的列表（单参考）。返回 bleu1..bleu4 列表。"""
    assert len(hyps) == len(refs)
    clip = [Counter() for _ in range(max_n)]
    total = [Counter() for _ in range(max_n)]
    hyp_len = ref_len = 0
    for hyp, ref in zip(hyps, refs):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            h, r = _ngrams(hyp, n), _ngrams(ref, n)
            total[n - 1].update(h)
            for g, c in h.items():
                clip[n - 1][g] += min(c, r.get(g, 0))
    p = []
    for n in range(max_n):
        s, t = sum(clip[n].values()), sum(total[n].values())
        p.append((s / t) if t > 0 else 0.0)
    if hyp_len == 0:
        return [0.0] * max_n
    bp = 1.0 if hyp_len > ref_len else pow(2.718281828, 1 - ref_len / hyp_len)
    out, cum = [], 1.0
    for n in range(max_n):
        cum *= max(p[n], 1e-16)
        out.append(100 * bp * (cum ** (1.0 / (n + 1))))
    return out
