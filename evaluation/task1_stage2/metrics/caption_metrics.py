"""Self-contained BLEU-adjacent caption metrics matching COCO caption conventions.

ROUGE-L uses whitespace tokens and beta=1.2. CIDEr-D uses 1--4-gram TF-IDF,
document frequency over references, sigma=6 length penalty, and the standard x10 scale.
Food500Cap_ready supplies one reference caption per image, so each image has one ref.
"""
from collections import Counter
import math

from bleu import corpus_bleu


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1)))


def _lcs_length(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b)+1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j-1]+1 if x == y else max(prev[j], cur[j-1]))
        prev = cur
    return prev[-1]


def rouge_l(hyp, ref, beta=1.2):
    if not hyp or not ref:
        return 0.0
    common = _lcs_length(hyp, ref)
    precision = common / len(hyp)
    recall = common / len(ref)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * precision * recall / (recall + b2 * precision)


def cider_d(hyps, refs, n=4, sigma=6.0):
    if len(hyps) != len(refs):
        raise ValueError("hypotheses and references must have matching lengths")
    if not hyps:
        return 0.0
    # Document frequency counts whether an n-gram occurs in any reference for an image.
    df = Counter()
    ref_grams = []
    for ref in refs:
        grams = [_ngrams(ref, k) for k in range(1, n+1)]
        ref_grams.append(grams)
        for k in range(n):
            df.update(set(grams[k]))
    log_n = math.log(len(refs))
    per_image = []
    for i, (hyp, grams_ref) in enumerate(zip(hyps, ref_grams)):
        grams_hyp = [_ngrams(hyp, k) for k in range(1, n+1)]
        hyp_len = len(hyp)
        ref_len = len(refs[i])
        delta = hyp_len - ref_len
        penalty = math.exp(-(delta*delta)/(2*sigma*sigma))
        ngram_scores = []
        for k in range(n):
            hw, rw = {}, {}
            for gram, count in grams_hyp[k].items():
                hw[gram] = count * (log_n - math.log(max(1, df[gram])))
            for gram, count in grams_ref[k].items():
                rw[gram] = count * (log_n - math.log(max(1, df[gram])))
            nh = math.sqrt(sum(v*v for v in hw.values()))
            nr = math.sqrt(sum(v*v for v in rw.values()))
            dot = sum(min(value, rw.get(gram, 0.0))*rw.get(gram, 0.0)
                      for gram, value in hw.items())
            sim = dot/(nh*nr) if nh and nr else 0.0
            ngram_scores.append(sim*penalty)
        per_image.append(10.0 * sum(ngram_scores)/n)
    return sum(per_image)/len(per_image)


def compute_caption_metrics(hyps, refs):
    return {"bleu1_to_4": corpus_bleu(hyps, refs),
            "rouge_l": sum(rouge_l(h, r) for h, r in zip(hyps, refs))/max(1, len(hyps)),
            "cider_d": cider_d(hyps, refs)}
