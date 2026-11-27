# -*- coding: utf-8 -*-
"""自包含的语料级 BLEU-1~4 评测（无第三方依赖，仅用于训练过程监控）。
正式指标请以 pycocoevalcap 为准（中期实验接入）。"""
from collections import Counter  # 导入计数器，用于统计 n-gram 出现次数


def _ngrams(tokens, n):
    # 从 token 序列中提取所有长度为 n 的连续片段（n-gram），并统计每种片段出现的次数
    # tokens[i:i + n] 是切片，取第 i 到第 i+n-1 个 token；tuple() 转成元组才能作为字典的键
    # range(len(tokens) - n + 1) 是所有合法起始位置（再往后切就不够 n 个了）
    # Counter(...) 返回 {n-gram: 出现次数} 的计数器
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    """hyps/refs: token 列表的列表（单参考）。返回 bleu1..bleu4 列表。"""
    # hyps: 模型生成的句子列表，每个句子是 token 列表；refs: 参考答案列表，一一对应
    assert len(hyps) == len(refs)  # 预测和参考数量必须一致，否则直接报错
    clip = [Counter() for _ in range(max_n)]   # clip[n]：全语料上被“裁剪”后的 n+1-gram 命中数（匹配上限）
    total = [Counter() for _ in range(max_n)]  # total[n]：全语料上预测句子的 n+1-gram 总数（分母）
    hyp_len = ref_len = 0  # 全语料的预测总长度、参考总长度（后面算长度惩罚 BP 用）
    for hyp, ref in zip(hyps, refs):  # 逐句遍历：hyp 是预测句，ref 是对应的参考句
        hyp_len += len(hyp)  # 累加预测句长度
        ref_len += len(ref)  # 累加参考句长度
        for n in range(1, max_n + 1):  # 依次计算 1-gram、2-gram、3-gram、4-gram
            h, r = _ngrams(hyp, n), _ngrams(ref, n)  # 分别统计预测句和参考句的 n-gram 计数
            total[n - 1].update(h)  # 把预测句的 n-gram 计数累加到全语料分母里
            for g, c in h.items():  # 遍历预测句里每种 n-gram（g 是片段，c 是出现次数）
                # 裁剪计数：命中数不能超过参考句中该片段的次数，防止预测靠重复刷分
                # min(c, r.get(g, 0))：预测出现 c 次，参考出现 r.get(g,0) 次（没出现则为 0），取较小者
                clip[n - 1][g] += min(c, r.get(g, 0))
    p = []  # 存放各阶 n-gram 的精确率 p_n = 命中数 / 总数
    for n in range(max_n):
        s, t = sum(clip[n].values()), sum(total[n].values())  # s=命中总数，t=预测总数
        p.append((s / t) if t > 0 else 0.0)  # 分母为 0（句子太短没有该阶 n-gram）时记为 0
    if hyp_len == 0:  # 整个预测语料为空（不可能输出任何东西）
        return [0.0] * max_n  # 直接返回全 0 分
    # 长度惩罚 BP（Brevity Penalty）：预测比参考长则不惩罚（1.0）；
    # 预测比参考短则按 e^(1 - ref_len/hyp_len) 惩罚，越短惩罚越重
    bp = 1.0 if hyp_len > ref_len else pow(2.718281828, 1 - ref_len / hyp_len)
    out, cum = [], 1.0  # out 存最终结果；cum 是 p_1..p_n 的累乘（几何平均的分子）
    for n in range(max_n):
        cum *= max(p[n], 1e-16)  # 累乘各阶精确率；用 1e-16 防止 p 为 0 时开方报错/全零
        # BLEU-(n+1) = 100 * BP * (p1*p2*...*pn+1)^(1/(n+1))，即前 n+1 阶精确率的几何平均再乘长度惩罚
        out.append(100 * bp * (cum ** (1.0 / (n + 1))))
    return out  # 返回 [bleu1, bleu2, bleu3, bleu4]，单位是百分制
