# -*- coding: utf-8 -*-
"""V1 模型：ResNet18 全局特征 + GRU 解码器（Show and Tell 风格基线）。"""
import torch
import torch.nn as nn
import torchvision


class EncoderCNN(nn.Module):
    """ResNet18 去掉分类层，输出 512 维全局图像特征。"""
    def __init__(self, embed_dim, finetune=True):
        super().__init__()
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # 输出 [B, 512, 1, 1]
        self.proj = nn.Linear(512, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        for p in self.backbone.parameters():
            p.requires_grad = finetune

    def forward(self, x):
        f = self.backbone(x).squeeze(-1).squeeze(-1)   # [B, 512]
        return torch.relu(self.bn(self.proj(f)))        # [B, embed_dim]


class DecoderGRU(nn.Module):
    """一层 GRU，用图像特征初始化隐状态，自回归生成 caption。"""
    def __init__(self, vocab_size, embed_dim, hidden_size, pad_id, num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(embed_dim, hidden_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, feats, caps, lengths):
        """Teacher forcing 训练：feats [B,E]，caps [B,L]（含 BOS 与 EOS）。"""
        h0 = torch.tanh(self.init_h(feats)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        embs = self.dropout(self.embedding(caps[:, :-1]))           # 输入去掉最后一个词
        packed = nn.utils.rnn.pack_padded_sequence(
            embs, [l - 1 for l in lengths], batch_first=True, enforce_sorted=True)
        out, _ = self.gru(packed, h0)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.fc(self.dropout(out))                            # [B, L-1, V]

    def step(self, words, h):
        embs = self.embedding(words).unsqueeze(1)                    # [B,1,E]
        out, h = self.gru(embs, h)
        return self.fc(out.squeeze(1)), h

    def init_hidden(self, feats):
        h = torch.tanh(self.init_h(feats))
        return h.unsqueeze(0).repeat(self.num_layers, 1, 1)


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512, pad_id=0,
                 enc_finetune=True, num_layers=1, dropout=0.5):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim, finetune=enc_finetune)
        self.decoder = DecoderGRU(vocab_size, embed_dim, hidden_size,
                                  pad_id, num_layers=num_layers, dropout=dropout)

    def forward(self, imgs, caps, lengths):
        feats = self.encoder(imgs)
        return self.decoder(feats, caps, lengths)

    @torch.no_grad()
    def greedy_decode(self, imgs, vocab, max_len=30):
        """贪心解码，返回 token id 序列列表（不含 BOS，截断于 EOS）。"""
        inv = {i: w for w, i in vocab.items()}
        device = next(self.parameters()).device
        feats = self.encoder(imgs)
        h = self.decoder.init_hidden(feats)
        words = torch.full((feats.size(0),), vocab["<BOS>"], dtype=torch.long, device=device)
        seqs = [[] for _ in range(feats.size(0))]
        finished = [False] * feats.size(0)
        for _ in range(max_len):
            logits, h = self.decoder.step(words, h)
            words = logits.argmax(-1)
            for i, w in enumerate(words.tolist()):
                if not finished[i]:
                    if w == vocab["<EOS>"]:
                        finished[i] = True
                    else:
                        seqs[i].append(w)
            if all(finished):
                break
        return [[inv.get(t, "<UNK>") for t in s] for s in seqs]

    @torch.no_grad()
    def beam_decode(self, img, vocab, beam_size=3, max_len=30):
        """单张图片的束搜索解码（batch=1），返回最优 token id 序列。"""
        device = next(self.parameters()).device
        feats = self.encoder(img.unsqueeze(0))                       # [1,E]
        h0 = self.decoder.init_hidden(feats)                         # [L,1,H]
        beams = [( [vocab["<BOS>"]], 0.0, h0 )]
        for _ in range(max_len):
            cand = []
            for seq, score, h in beams:
                if seq[-1] == vocab["<EOS>"]:
                    cand.append((seq, score, h))
                    continue
                word = torch.tensor([seq[-1]], device=device)
                logits, h_new = self.decoder.step(word, h)
                logp = torch.log_softmax(logits, -1).squeeze(0)
                top = torch.topk(logp, beam_size)
                for k in range(beam_size):
                    cand.append((seq + [top.indices[k].item()],
                                 score + top.values[k].item(), h_new))
            cand.sort(key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)  # 长度归一
            beams = cand[:beam_size]
            if all(s[-1] == vocab["<EOS>"] for s, _, _ in beams):
                break
        best = beams[0][0]
        if best and best[0] == vocab["<BOS>"]:
            best = best[1:]
        if vocab["<EOS>"] in best:
            best = best[: best.index(vocab["<EOS>"])]
        return best
