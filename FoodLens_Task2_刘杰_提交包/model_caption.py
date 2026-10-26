"""
model_caption.py —— 模型 A：任务一风格的图像描述模型（CNN 整体表示 + GRU 解码器）

结构对应任务一可选结构中的「CNN/ViT 整体表示 + GRU」：
  ResNet18（ImageNet 预训练，整体表示）→ 线性映射为 GRU 初始隐状态
  → GRU 逐 token 自回归生成营养 JSON 文本（target_text）

词表与 metrics.tokenize 共用同一分词规则，保证训练/评测一致。
"""

import json
import os
import re
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from metrics import tokenize

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
MAX_LEN = 220  # train 最长 token 序列约 190，留余量


# =====================================================================
# 词表
# =====================================================================
class Vocab:
    def __init__(self):
        self.word2idx = {w: i for i, w in enumerate(SPECIALS)}
        self.idx2word = list(SPECIALS)

    def build(self, texts):
        cnt = set()
        for t in texts:
            cnt.update(tokenize(t))
        for w in sorted(cnt):
            if w not in self.word2idx:
                self.word2idx[w] = len(self.idx2word)
                self.idx2word.append(w)
        return self

    def __len__(self):
        return len(self.idx2word)

    def encode(self, text, add_bos=True, add_eos=True):
        ids = [self.word2idx.get(t, UNK) for t in tokenize(text)]
        if add_bos:
            ids = [BOS] + ids
        if add_eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids, skip_special=True):
        """token 序列还原为字符串。

        tokenize 规则是「字母数字连续段 + 单字符符号」，切分无损，
        因此直接拼接即可还原出原始 JSON 文本（无多余空格）。
        """
        words = []
        for i in ids:
            if skip_special and i in (PAD, BOS, EOS):
                continue
            words.append(self.idx2word[i] if 0 <= i < len(self.idx2word) else "<unk>")
        return "".join(words)


def build_vocab_from_train():
    with open(os.path.join(DATA_ROOT, "annotations", "train.jsonl"), "r", encoding="utf-8") as f:
        texts = [json.loads(line)["target_text"] for line in f]
    return Vocab().build(texts)


# =====================================================================
# 模型：ResNet18 编码器 + GRU 解码器
# =====================================================================
class CaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, dropout=0.3):
        super().__init__()
        # ---- CNN 整体表示（去掉最后的分类 fc）----
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # [B,512,1,1]
        self.feat_dim = 512

        # ---- 图像特征 -> GRU 初始隐状态 ----
        self.img_proj = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(),
        )

        # ---- GRU 解码器 ----
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        nn.init.xavier_uniform_(self.img_proj[0].weight)
        nn.init.zeros_(self.img_proj[0].bias)

    def encode(self, images):
        """images: [B,3,H,W] -> h0: [1,B,hidden]"""
        feat = self.cnn(images).flatten(1)      # [B,512]
        h0 = self.img_proj(feat).unsqueeze(0)   # [1,B,hidden]
        return h0

    def forward(self, images, captions):
        """
        images:   [B,3,H,W]
        captions: [B,T] 已含 BOS、EOS（训练时输入到倒数第二个 token）
        """
        h = self.encode(images)
        emb = self.dropout(self.embedding(captions[:, :-1]))
        out, _ = self.gru(emb, h)               # [B,T-1,hidden]
        logits = self.fc_out(self.dropout(out)) # [B,T-1,vocab]
        return logits

    @torch.no_grad()
    def generate(self, images, max_len=MAX_LEN):
        """贪心解码，返回 token id 列表的列表。"""
        self.eval()
        B = images.size(0)
        h = self.encode(images)
        x = torch.full((B, 1), BOS, dtype=torch.long, device=images.device)
        seqs = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=images.device)
        for _ in range(max_len):
            emb = self.embedding(x[:, -1:])          # [B,1,emb]
            out, h = self.gru(emb, h)                # [B,1,hidden]
            logits = self.fc_out(out[:, -1])         # [B,vocab]
            nxt = logits.argmax(dim=-1)              # [B]
            for i in range(B):
                if not finished[i]:
                    seqs[i].append(nxt[i].item())
            finished |= nxt == EOS
            x = torch.cat([x, nxt.unsqueeze(1)], dim=1)
            if finished.all():
                break
        return seqs


if __name__ == "__main__":
    v = build_vocab_from_train()
    print("词表大小:", len(v))
    text = '{"items":[{"name":"bacon","weight_g":23.0}],"total_weight_g":23.0}'
    ids = v.encode(text)
    print("编码示例:", ids[:15], "...")
    print("解码回环:", v.decode(ids))
    m = CaptionModel(len(v))
    img = torch.randn(2, 3, 224, 224)
    cap = torch.randint(0, len(v), (2, 20))
    logits = m(img, cap)
    print("logits:", logits.shape)
    seqs = m.generate(img, max_len=30)
    print("生成示例:", v.decode(seqs[0])[:120])
