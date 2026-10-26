# -*- coding: utf-8 -*-
"""V1 模型：ResNet18 全局特征 + GRU 解码器（Show and Tell 风格基线）。"""
import torch
import torch.nn as nn
import torchvision  # 提供预训练的 ResNet18


class EncoderCNN(nn.Module):
    """ResNet18 去掉分类层，输出 512 维全局图像特征。"""
    def __init__(self, embed_dim, finetune=True):
        super().__init__()
        # 加载在 ImageNet 上预训练好的 ResNet18
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # 去掉最后一层全连接分类层，保留卷积主体，输出 [B, 512, 1, 1]
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # 输出 [B, 512, 1, 1]
        self.proj = nn.Linear(512, embed_dim)   # 把 512 维图像特征映射到词嵌入维度
        self.bn = nn.BatchNorm1d(embed_dim)     # 批归一化，稳定训练
        for p in self.backbone.parameters():
            p.requires_grad = finetune  # finetune=False 则冻结主干（不更新 ResNet 权重）

    def forward(self, x):
        f = self.backbone(x).squeeze(-1).squeeze(-1)   # [B,512,1,1] 去掉最后两个维度 -> [B, 512]
        return torch.relu(self.bn(self.proj(f)))        # 线性->BN->ReLU，输出 [B, embed_dim]


class DecoderGRU(nn.Module):
    """一层 GRU，用图像特征初始化隐状态，自回归生成 caption。"""
    def __init__(self, vocab_size, embed_dim, hidden_size, pad_id, num_layers=1, dropout=0.5):
        super().__init__()
        # 词嵌入层；padding_idx 让 PAD 的嵌入恒为 0 且不参与训练
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        # GRU 主体；只有层数>1 时层间 dropout 才生效
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)      # 用在嵌入和输出上的 dropout
        self.fc = nn.Linear(hidden_size, vocab_size)  # 把隐状态映射到词表大小的 logits
        self.init_h = nn.Linear(embed_dim, hidden_size)  # 用图像特征生成 GRU 初始隐状态
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, feats, caps, lengths):
        """Teacher forcing 训练：feats [B,E]，caps [B,L]（含 BOS 与 EOS）。"""
        # 图像特征 -> 初始隐状态 [num_layers, B, H]；tanh 把值压到 (-1,1)
        h0 = torch.tanh(self.init_h(feats)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        # Teacher forcing：输入是 caps 去掉最后一个词（即 BOS..倒数第二个词）
        # 模型要学会预测 caps[:, 1:]（第一个词..EOS）
        embs = self.dropout(self.embedding(caps[:, :-1]))           # 输入去掉最后一个词
        # 打包变长序列：l-1 是因为输入比原序列少一个词；要求已按长度降序
        packed = nn.utils.rnn.pack_padded_sequence(
            embs, [l - 1 for l in lengths], batch_first=True, enforce_sorted=True)
        out, _ = self.gru(packed, h0)  # GRU 前向，忽略 PAD 位置
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # 解包回 [B, L-1, H]
        return self.fc(self.dropout(out))                            # 映射到词表 [B, L-1, V]

    def step(self, words, h):
        # 解码时单步前向：输入当前词和当前隐状态，输出下一个词的 logits 和新隐状态
        embs = self.embedding(words).unsqueeze(1)                    # [B] -> [B,1,E]
        out, h = self.gru(embs, h)                                   # [B,1,H]
        return self.fc(out.squeeze(1)), h                            # logits [B,V]

    def init_hidden(self, feats):
        # 由图像特征生成初始隐状态（与 forward 里的 h0 相同，供解码用）
        h = torch.tanh(self.init_h(feats))
        return h.unsqueeze(0).repeat(self.num_layers, 1, 1)


class CaptionModel(nn.Module):
    """编码器 + 解码器 组合成的完整图像描述模型。"""
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512, pad_id=0,
                 enc_finetune=True, num_layers=1, dropout=0.5):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim, finetune=enc_finetune)
        self.decoder = DecoderGRU(vocab_size, embed_dim, hidden_size,
                                  pad_id, num_layers=num_layers, dropout=dropout)

    def forward(self, imgs, caps, lengths):
        # 训练用前向：图片 -> 特征 -> teacher forcing 计算每个位置的词 logits
        feats = self.encoder(imgs)
        return self.decoder(feats, caps, lengths)

    @torch.no_grad()  # 推理时不计算梯度，省显存提速
    def greedy_decode(self, imgs, vocab, max_len=30):
        """贪心解码，返回 token id 序列列表（不含 BOS，截断于 EOS）。"""
        inv = {i: w for w, i in vocab.items()}  # id -> 单词 的反向词表
        device = next(self.parameters()).device  # 模型所在设备（cuda/cpu）
        feats = self.encoder(imgs)               # 编码整个 batch 的图片
        h = self.decoder.init_hidden(feats)      # 初始隐状态
        # 第一个输入词全部设为 BOS
        words = torch.full((feats.size(0),), vocab["<BOS>"], dtype=torch.long, device=device)
        seqs = [[] for _ in range(feats.size(0))]      # 每张图的生成结果
        finished = [False] * feats.size(0)             # 每张图是否已生成 EOS
        for _ in range(max_len):  # 最多生成 max_len 个词
            logits, h = self.decoder.step(words, h)  # 单步前向
            words = logits.argmax(-1)              # 贪心：取概率最大的词
            for i, w in enumerate(words.tolist()):
                if not finished[i]:
                    if w == vocab["<EOS>"]:
                        finished[i] = True  # 遇到 EOS，该图生成结束（不记录 EOS 本身）
                    else:
                        seqs[i].append(w)
            if all(finished):  # 所有图都结束了就提前退出
                break
        # 把 id 序列翻回单词；查不到的 id 用 <UNK> 兜底
        return [[inv.get(t, "<UNK>") for t in s] for s in seqs]

    @torch.no_grad()
    def beam_decode(self, img, vocab, beam_size=3, max_len=30):
        """单张图片的束搜索解码（batch=1），返回最优 token id 序列。"""
        device = next(self.parameters()).device
        feats = self.encoder(img.unsqueeze(0))                       # 补 batch 维 -> [1,E]
        h0 = self.decoder.init_hidden(feats)                         # [L,1,H]
        # beam 里每个元素是 (当前序列, 累计对数概率, 隐状态)，初始只有 BOS
        beams = [( [vocab["<BOS>"]], 0.0, h0 )]
        for _ in range(max_len):
            cand = []  # 本步所有候选扩展
            for seq, score, h in beams:
                if seq[-1] == vocab["<EOS>"]:
                    cand.append((seq, score, h))  # 已结束的 beam 原样保留，不再扩展
                    continue
                word = torch.tensor([seq[-1]], device=device)  # 当前输入词
                logits, h_new = self.decoder.step(word, h)
                logp = torch.log_softmax(logits, -1).squeeze(0)  # 词表上的对数概率 [V]
                top = torch.topk(logp, beam_size)                # 取概率最高的 beam_size 个词
                for k in range(beam_size):
                    # 扩展：序列加新词，分数累加对数概率（概率乘积 -> 对数相加）
                    cand.append((seq + [top.indices[k].item()],
                                 score + top.values[k].item(), h_new))
            # 按 长度归一化 后的分数排序：score / len^0.7，避免偏向短句
            cand.sort(key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)  # 长度归一
            beams = cand[:beam_size]  # 只保留前 beam_size 条
            if all(s[-1] == vocab["<EOS>"] for s, _, _ in beams):
                break  # 所有 beam 都生成 EOS，提前结束
        best = beams[0][0]  # 分数最高的序列
        if best and best[0] == vocab["<BOS>"]:
            best = best[1:]  # 去掉开头的 BOS
        if vocab["<EOS>"] in best:
            best = best[: best.index(vocab["<EOS>"])]  # 截掉 EOS 及其后的内容
        return best
