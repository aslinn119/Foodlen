"""ResNet18 spatial features + additive attention + GRU caption decoder.

Training interface matches the supplied V1. Attention is aligned with generated
words; EOS is excluded from both the word and heatmap lists.
"""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class SpatialEncoder(nn.Module):
    def __init__(self, embed_dim=256, finetune=True, pretrained=True):
        super().__init__()
        network = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(network.children())[:-2])
        # Preserve V1's global projection for the initial decoder state.
        self.proj = nn.Linear(512, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = finetune

    def forward(self, images):
        grid = self.backbone(images)
        regions = grid.flatten(2).transpose(1, 2)  # B x 49 x 512 at 224 px
        global_embedding = torch.relu(self.bn(self.proj(regions.mean(1))))
        return regions, global_embedding


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size, feature_size=512, attention_size=256):
        super().__init__()
        self.feature_proj = nn.Linear(feature_size, attention_size, bias=False)
        self.hidden_proj = nn.Linear(hidden_size, attention_size, bias=False)
        self.score = nn.Linear(attention_size, 1, bias=False)

    def forward(self, regions, hidden, projected=None):
        projected = self.feature_proj(regions) if projected is None else projected
        energies = self.score(torch.tanh(projected + self.hidden_proj(hidden)[:, None])).squeeze(-1)
        weights = energies.softmax(dim=1)
        context = torch.bmm(weights[:, None], regions).squeeze(1)
        return context, weights


class AttentionDecoderGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512, pad_id=0,
                 num_layers=1, dropout=0.5, attention_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.attention = AdditiveAttention(hidden_size, attention_size=attention_size)
        self.gru = nn.GRU(embed_dim + 512, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(embed_dim, hidden_size)
        self.num_layers = num_layers

    def init_hidden(self, global_embedding):
        return torch.tanh(self.init_h(global_embedding)).unsqueeze(0).repeat(self.num_layers, 1, 1)

    def step(self, words, hidden, regions, projected=None):
        context, weights = self.attention(regions, hidden[-1], projected)
        embedding = self.dropout(self.embedding(words))
        inputs = torch.cat([embedding, context], dim=-1).unsqueeze(1)
        output, hidden = self.gru(inputs, hidden)
        return self.fc(self.dropout(output[:, 0])), hidden, weights

    def forward(self, regions, global_embedding, captions, lengths, return_attention=False):
        hidden = self.init_hidden(global_embedding)
        projected = self.attention.feature_proj(regions)
        logits, weights = [], []
        for timestep in range(captions.size(1) - 1):
            scores, hidden, alpha = self.step(captions[:, timestep], hidden, regions, projected)
            logits.append(scores)
            weights.append(alpha)
        scores = torch.stack(logits, dim=1)
        # PAD targets are ignored by CrossEntropyLoss, as in V1.
        if return_attention:
            return scores, torch.stack(weights, dim=1)
        return scores


class CaptionModelV2(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512, pad_id=0,
                 enc_finetune=True, num_layers=1, dropout=0.5,
                 attention_size=256, pretrained=True):
        super().__init__()
        self.encoder = SpatialEncoder(embed_dim, enc_finetune, pretrained)
        self.decoder = AttentionDecoderGRU(vocab_size, embed_dim, hidden_size,
                                          pad_id, num_layers, dropout, attention_size)

    def forward(self, images, captions, lengths, return_attention=False):
        regions, global_embedding = self.encoder(images)
        return self.decoder(regions, global_embedding, captions, lengths, return_attention)

    @torch.no_grad()
    def greedy_decode(self, images, vocab, max_len=30, return_attention=False):
        if self.training:
            raise RuntimeError('Call model.eval() before decoding.')
        regions, global_embedding = self.encoder(images)
        hidden = self.decoder.init_hidden(global_embedding)
        projected = self.decoder.attention.feature_proj(regions)
        inverse = {index: word for word, index in vocab.items()}
        words = torch.full((len(images),), vocab['<BOS>'], dtype=torch.long, device=images.device)
        sequences, maps = [[] for _ in images], [[] for _ in images]
        finished = [False] * len(images)
        for _ in range(max_len):
            logits, hidden, alpha = self.decoder.step(words, hidden, regions, projected)
            words = logits.argmax(-1)
            for index, word in enumerate(words.tolist()):
                if finished[index]:
                    continue
                if word == vocab['<EOS>']:
                    finished[index] = True
                else:
                    sequences[index].append(inverse.get(word, '<UNK>'))
                    maps[index].append(alpha[index].cpu())
            if all(finished):
                break
        maps = [torch.stack(a) if a else torch.empty(0, regions.size(1)) for a in maps]
        return (sequences, maps) if return_attention else sequences

    @torch.no_grad()
    def beam_decode(self, image, vocab, beam_size=3, max_len=30, length_penalty=0.7):
        """Return token IDs, following V1's beam API and scoring convention."""
        if self.training:
            raise RuntimeError('Call model.eval() before decoding.')
        regions, global_embedding = self.encoder(image.unsqueeze(0))
        return self.beam_decode_features(regions, global_embedding, vocab, beam_size,
                                         max_len, length_penalty)

    @torch.no_grad()
    def beam_decode_features(self, regions, global_embedding, vocab, beam_size=3,
                             max_len=30, length_penalty=0.7):
        """Beam-decode one pre-encoded image to support efficient batched evaluation."""
        if self.training:
            raise RuntimeError('Call model.eval() before decoding.')
        if beam_size < 1 or beam_size > len(vocab):
            raise ValueError('beam_size must lie between 1 and vocabulary size')
        if regions.size(0) != 1 or global_embedding.size(0) != 1:
            raise ValueError('beam_decode_features expects exactly one image')
        projected = self.decoder.attention.feature_proj(regions)
        beams = [([vocab['<BOS>']], 0., self.decoder.init_hidden(global_embedding))]
        for _ in range(max_len):
            candidates = []
            for sequence, score, hidden in beams:
                if sequence[-1] == vocab['<EOS>']:
                    candidates.append((sequence, score, hidden))
                    continue
                word = torch.tensor([sequence[-1]], device=regions.device)
                logits, new_hidden, _ = self.decoder.step(word, hidden, regions, projected)
                values, indices = logits.log_softmax(-1)[0].topk(beam_size)
                for value, index in zip(values.tolist(), indices.tolist()):
                    candidates.append((sequence + [index], score + value, new_hidden))
            candidates.sort(key=lambda b: b[1] / (len(b[0]) ** length_penalty), reverse=True)
            beams = candidates[:beam_size]
            if all(b[0][-1] == vocab['<EOS>'] for b in beams):
                break
        sequence = beams[0][0][1:]
        if vocab['<EOS>'] in sequence:
            sequence = sequence[:sequence.index(vocab['<EOS>'])]
        return sequence
