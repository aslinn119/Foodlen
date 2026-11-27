"""
model_bgen.py —— 模型 B-gen：CLIP（多模态预训练模型）+ LoRA 参数高效微调
                  + 自回归字符级解码器，直接「生成」营养 JSON 文本

课程合规定位（与模型 B 的结构化预测头路线互补）：
  - 骨干：CLIP ViT-B/32（视觉端）+ LoRA（最后 4 层注意力 q/k/v/out，rank=16）
          —— 与模型 B 完全同源，权重直接复用 checkpoints_lora_mt/best_model.pt
  - 解码：一层 GRU（hidden=512）以 [CLS] 视觉特征为初始隐状态，
          自回归逐字符生成完整 JSON 字符串（teacher forcing 训练，贪心解码推理）
  - 因此 B-gen 是「经过参数高效微调的多模态预训练模型 + 生成式语言解码」，
    同时满足课程任务二对②「多模态预训练模型」与「生成式输出」的双重要求。

字符级词表（48 个字符 + 3 个特殊符）：
  - 词表封闭，任何测试样本的食材名/数值都能逐字符生成，不存在 OOV
  - 数值按位生成，精度不受训练集数值词表覆盖限制
"""

import json
import os
import torch
import torch.nn as nn

from model_lora import CLIPLoRAMultiTask

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_MT = os.path.join(DATA_ROOT, "checkpoints_lora_mt", "best_model.pt")
VOCAB_PATH = os.path.join(DATA_ROOT, "checkpoints_bgen", "char_vocab.json")

PAD, BOS, EOS = 0, 1, 2
SPECIALS = ["<pad>", "<bos>", "<eos>"]
MAX_LEN = 720  # train 最长 target_text 686 字符 + 余量


# =====================================================================
# 字符词表（由训练集 target_text 构建，封闭集合）
# =====================================================================
def build_char_vocab():
    """返回 (char2idx, idx2char)。若已缓存则直接读取。"""
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return d["char2idx"], d["idx2char"]

    chars = set()
    with open(os.path.join(DATA_ROOT, "annotations", "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            chars.update(json.loads(line)["target_text"])
    idx2char = SPECIALS + sorted(chars)
    char2idx = {c: i for i, c in enumerate(idx2char)}

    os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump({"char2idx": char2idx, "idx2char": idx2char}, f, ensure_ascii=False, indent=1)
    return char2idx, idx2char


def encode_text(text, char2idx, add_bos=True, add_eos=True):
    ids = [char2idx[c] for c in text]  # 词表封闭，训练集已覆盖全部字符
    if add_bos:
        ids = [BOS] + ids
    if add_eos:
        ids = ids + [EOS]
    return ids


def decode_text(ids, idx2char):
    """token id 序列还原为字符串（遇到 EOS 截断，跳过特殊符）。"""
    out = []
    for i in ids:
        if i == EOS:
            break
        if i in (PAD, BOS):
            continue
        out.append(idx2char[i])
    return "".join(out)


# =====================================================================
# 模型：CLIP+LoRA 编码器（冻结，权重复用） + GRU 字符解码器（从头训练）
# =====================================================================
class BGenModel(nn.Module):
    def __init__(self, num_ingredients, emb_dim=128, hidden=512,
                 last_n=4, rank=16, alpha=16.0, dropout=0.1, *, pretrained=True):
        super().__init__()
        # 编码器：与模型 B 同构，加载其 LoRA 权重后整体冻结
        self.encoder = CLIPLoRAMultiTask(num_targets=5, num_ingredients=num_ingredients,
                                         last_n=last_n, rank=rank, alpha=alpha, pretrained=pretrained)
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 解码器：轻量 GRU（可训练参数约 1.4M）
        self.vocab_size = None  # build_char_vocab 后由外部赋入 set_vocab_size
        self.emb = nn.Embedding(1, emb_dim)   # 占位，set_vocab_size 后重建
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.gru = nn.GRU(emb_dim, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.h0_proj = nn.Linear(768, hidden)
        self.feat_proj = nn.Linear(768, emb_dim)   # 逐步注入视觉特征，增强条件约束
        self.out = nn.Linear(hidden, 1)       # 占位，set_vocab_size 后重建
        self._built = False

    def set_vocab_size(self, vocab_size):
        """词表大小确定后构建 embedding / 输出层（便于先建词表再建模型）。"""
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, self.emb_dim, padding_idx=PAD)
        self.out = nn.Linear(self.hidden, vocab_size)
        self._built = True

    # ---------- 编码（冻结，无梯度） ----------
    @torch.no_grad()
    def extract_features(self, pixel_values):
        """只取 [CLS] 视觉特征（供特征缓存与训练/评测复用）。"""
        self.eval()
        return self.encoder.clip(pixel_values=pixel_values).pooler_output

    # ---------- 解码 ----------
    def _step_inputs(self, feats, ids):
        """每个解码步的视觉条件注入：embedding + 特征投影。"""
        return self.dropout(self.emb(ids) + self.feat_proj(feats).unsqueeze(1))

    def forward(self, feats, ids_in, ss_prob=0.0):
        """teacher forcing 前向（可选 scheduled sampling 缓解曝光偏差）。

        feats:   [B, 768] 视觉特征
        ids_in:  [B, T]   输入 token（BOS 开头，右填充 PAD）
        ss_prob: 训练时以该概率改用模型自身上一步预测作为输入（scheduled sampling）
        返回 logits [B, T, V]
        """
        h = torch.tanh(self.h0_proj(feats)).unsqueeze(0)       # [1, B, H]
        T = ids_in.size(1)
        if (not self.training) or ss_prob <= 0.0:
            x = self._step_inputs(feats, ids_in)               # [B, T, E]
            out, _ = self.gru(x, h)
            return self.out(self.dropout(out))                 # [B, T, V]

        # scheduled sampling：逐步解码，按概率替换输入为自身预测
        B = ids_in.size(0)
        logits_all = []
        prev = ids_in[:, :1]                                   # BOS
        for t in range(T):
            out, h = self.gru(self._step_inputs(feats, prev), h)
            logits = self.out(self.dropout(out[:, -1]))        # [B, V]
            logits_all.append(logits)
            if t + 1 < T:
                gold_next = ids_in[:, t + 1:t + 2]             # 下一个真值 token
                pred = logits.argmax(dim=-1, keepdim=True)     # 模型对下一个 token 的预测
                use_ss = (torch.rand(B, 1, device=feats.device) < ss_prob) & \
                         (gold_next != PAD)
                prev = torch.where(use_ss, pred, gold_next)
        return torch.stack(logits_all, dim=1)                  # [B, T, V]

    @torch.no_grad()
    def generate(self, feats, char2idx, max_len=MAX_LEN, constrain_json=True,
                 beam=1, length_penalty=0.6):
        """自回归解码生成 JSON 字符序列。返回 list[list[int]]。

        beam=1：贪心解码（快）；beam>1：束搜索（累计对数概率，通常文本指标更优）。

        constrain_json：受限解码（JSON 数值语法约束，类 LLM 的 JSON mode）——
          - '.' 只允许出现在数字中间（前一位必须是数字且本数未用过 '.'）
          - '.' 之后只允许数字
          保证生成的数值字面量恒为合法 JSON number，不损失自回归生成性质。
        """
        self.eval()
        if beam <= 1:
            return self._generate_greedy(feats, char2idx, max_len, constrain_json)
        return self._generate_beam(feats, char2idx, max_len, constrain_json, beam,
                                   length_penalty)

    def _constrain(self, logits, last, dot_used, dot_id, digit_ids, not_digit):
        """数值语法受限解码（就地修改 logits）。last: [N] 上一 token，dot_used: [N] 状态。"""
        is_dot = last == dot_id
        is_digit = torch.isin(last, digit_ids)
        # 数值状态机：遇 '.' 置位，遇到数字/'.' 以外字符清零
        dot_used = torch.where(is_dot, torch.ones_like(dot_used), dot_used)
        dot_used = torch.where((~is_digit) & (~is_dot), torch.zeros_like(dot_used), dot_used)
        neg = float("-inf")
        # 1) '.' 之后只允许数字（及 EOS/PAD）
        logits[is_dot] = torch.where(
            not_digit.unsqueeze(0).expand_as(logits[is_dot]),
            torch.full_like(logits[is_dot], neg), logits[is_dot])
        # 2) 不允许 '.'：本数已用 或 前一位不是数字
        logits[dot_used | (~is_digit), dot_id] = neg
        # 3) 永不生成 PAD/BOS
        logits[:, PAD] = neg
        logits[:, BOS] = neg
        return dot_used

    @torch.no_grad()
    def _generate_greedy(self, feats, char2idx, max_len, constrain_json):
        B = feats.size(0)
        dev = feats.device
        h = torch.tanh(self.h0_proj(feats)).unsqueeze(0)       # [1, B, H]
        ids = torch.full((B, 1), BOS, dtype=torch.long, device=dev)
        finished = torch.zeros(B, dtype=torch.bool, device=dev)

        dot_id = char2idx["."]
        digit_ids = torch.tensor([char2idx[str(d)] for d in range(10)], device=dev)
        not_digit = torch.ones(self.vocab_size, dtype=torch.bool, device=dev)
        not_digit[digit_ids] = False
        not_digit[EOS] = False          # '.' 后允许 EOS（防御性；实际由数字收尾）
        not_digit[PAD] = False

        dot_used = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(max_len):
            x = self._step_inputs(feats, ids[:, -1:])          # [B, 1, E]
            out, h = self.gru(x, h)                            # [B, 1, H]
            logits = self.out(out[:, -1])                      # [B, V]

            if constrain_json:
                dot_used = self._constrain(logits, ids[:, -1], dot_used,
                                           dot_id, digit_ids, not_digit)

            nxt = logits.argmax(dim=-1)                        # [B]
            nxt = torch.where(finished, torch.full_like(nxt, PAD), nxt)
            ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            finished |= (nxt == EOS)
            if finished.all():
                break
        return [seq[1:].tolist() for seq in ids]               # 去掉 BOS

    @torch.no_grad()
    def _generate_beam(self, feats, char2idx, max_len, constrain_json, beam,
                       length_penalty=0.6):
        B = feats.size(0)
        K = beam
        dev = feats.device
        V = self.vocab_size
        neg = float("-inf")

        feats_k = feats.repeat_interleave(K, dim=0)            # [B*K, 768]
        h = torch.tanh(self.h0_proj(feats)).unsqueeze(0).repeat_interleave(K, dim=1)  # [1, B*K, H]
        ids = torch.full((B * K, 1), BOS, dtype=torch.long, device=dev)
        scores = torch.full((B, K), neg, device=dev)
        scores[:, 0] = 0.0                                       # 首步只有 beam0 合法
        scores = scores.view(B * K)
        finished = torch.zeros(B * K, dtype=torch.bool, device=dev)

        dot_id = char2idx["."]
        digit_ids = torch.tensor([char2idx[str(d)] for d in range(10)], device=dev)
        not_digit = torch.ones(V, dtype=torch.bool, device=dev)
        not_digit[digit_ids] = False
        not_digit[EOS] = False
        not_digit[PAD] = False

        dot_used = torch.zeros(B * K, dtype=torch.bool, device=dev)
        for _ in range(max_len):
            x = self._step_inputs(feats_k, ids[:, -1:])        # [B*K, 1, E]
            out, h = self.gru(x, h)                            # [1, B*K, H]
            logits = self.out(out[:, -1])                      # [B*K, V]
            if constrain_json:
                dot_used = self._constrain(logits, ids[:, -1], dot_used,
                                           dot_id, digit_ids, not_digit)
            logp = torch.log_softmax(logits, dim=-1)
            # 已结束 beam：只允许 PAD（logp=0，分数不再变化）
            logp[finished] = neg
            logp[finished, PAD] = 0.0

            total = (scores.unsqueeze(1) + logp).view(B, K * V)   # [B, K*V]
            # 长度归一化避免 beam search 偏爱过短序列。当前步所有未结束
            # beam 长度相同，已结束 beam 通过 PAD 保持分数不变。
            if length_penalty > 0:
                cur_len = ids.size(1)
                norm = ((5.0 + cur_len) / 6.0) ** length_penalty
                rank_scores = total / norm
            else:
                rank_scores = total
            _, top_idx = rank_scores.topk(K, dim=1)            # [B, K]
            top_scores = total.gather(1, top_idx)              # 保留未归一化累计分数
            src = (top_idx // V).long()                           # 来源 beam [B, K]
            tok = (top_idx % V).long()                            # 新 token [B, K]

            # gather 隐状态 / 序列 / 状态
            gi = src.unsqueeze(0).unsqueeze(-1).expand(1, B, K, h.size(-1))
            h = h.view(1, B, K, -1).gather(2, gi).view(1, B * K, -1)
            seq = ids.view(B, K, -1).gather(
                1, src.unsqueeze(-1).expand(B, K, ids.size(1)))
            ids = torch.cat([seq, tok.unsqueeze(-1)], dim=-1).view(B * K, -1)
            dot_used = dot_used.view(B, K).gather(1, src).view(B * K)
            finished = finished.view(B, K).gather(1, src).view(B * K) | (tok.view(B * K) == EOS)
            scores = top_scores.view(B * K)
            if finished.all():
                break

        best = ids.view(B, K, -1)[:, 0]                        # 累计对数概率最高的 beam
        return [seq[1:].tolist() for seq in best]              # 去掉 BOS


def load_bgen_decoder(path, device):
    """Load decoder parameters and use the Model B encoder used by the feature cache."""
    decoder = torch.load(path, map_location="cpu", weights_only=True)
    encoder = torch.load(CKPT_MT, map_location="cpu", weights_only=True)
    model = BGenModel(num_ingredients=encoder["num_ingredients"],
                      last_n=encoder["lora_last_n"], rank=encoder["rank"],
                      alpha=encoder["alpha"], emb_dim=decoder["emb_dim"],
                      hidden=decoder["hidden"], pretrained=False)
    model.set_vocab_size(decoder["vocab_size"])
    state = {key: value for key, value in decoder["model_state"].items()
             if not key.startswith("encoder.")}
    state.update({f"encoder.{key}": value for key, value in encoder["model_state"].items()})
    model.load_state_dict(state)
    return model.to(device).eval()


def load_bgen_encoder(device):
    """加载冻结的 CLIP+LoRA 编码器（复用模型 B 的多任务权重）。"""
    ck = torch.load(CKPT_MT, map_location=device, weights_only=False)
    model = BGenModel(num_ingredients=ck["num_ingredients"],
                      last_n=ck["lora_last_n"], rank=ck["rank"], alpha=ck["alpha"], pretrained=False)
    model.encoder.load_state_dict(ck["model_state"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


if __name__ == "__main__":
    char2idx, idx2char = build_char_vocab()
    print(f"字符词表大小: {len(idx2char)}")
    print(f"词表: {idx2char}")
    m = BGenModel(num_ingredients=151)
    m.set_vocab_size(len(idx2char))
    m.to("cuda" if torch.cuda.is_available() else "cpu")
    text = '{"items":[{"name":"bacon","weight_g":23.0}],' \
           '"total_weight_g":23.0,"total_calories_kcal":124.4}'
    ids = encode_text(text, char2idx)
    assert decode_text(ids, idx2char) == text, "编码/解码应无损互逆"
    print("编码/解码互逆自测通过")
    dev = next(m.parameters()).device
    feats = torch.randn(2, 768, device=dev)
    ids_t = torch.randint(3, len(idx2char), (2, 20), device=dev)
    m.train()
    logits = m(feats, ids_t, ss_prob=0.3)
    m.eval()
    seqs = m.generate(feats, char2idx, max_len=30)
    print("forward:", logits.shape, " generate:", len(seqs), "x", len(seqs[0]))
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"解码器可训练参数: {n_train/1e6:.3f}M")
