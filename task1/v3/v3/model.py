import math
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class SinusoidalPosition(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        divisor = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * divisor)
        pe[:, 1::2] = torch.cos(position * divisor[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, tokens):
        if tokens.shape[1] > self.pe.shape[1]:
            raise ValueError('Sequence exceeds positional encoding max_len')
        return tokens + self.pe[:, :tokens.shape[1]].to(dtype=tokens.dtype)


class FoodLensV3(nn.Module):
    """ResNet18 7x7 grid -> Transformer encoder -> autoregressive decoder."""
    def __init__(self, vocab_size, pad_id, d_model=256, nhead=4, enc_layers=2,
                 dec_layers=2, ff_dim=1024, dropout=0.1, pretrained=True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.visual_project = nn.Linear(512, d_model)
        self.visual_positions = nn.Parameter(torch.zeros(1, 49, d_model))
        nn.init.trunc_normal_(self.visual_positions, std=0.02)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.text_positions = SinusoidalPosition(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=False)
        self.output = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.pad_id = pad_id

    def freeze_cnn_except_layer4(self):
        """ResNet18 sequential index 7 is layer4; freeze earlier convolution stages."""
        for parameter in self.cnn.parameters():
            parameter.requires_grad = False
        for parameter in self.cnn[7].parameters():
            parameter.requires_grad = True
        self._keep_frozen_cnn_eval = True
        self.cnn[:7].eval()

    def train(self, mode=True):
        super().train(mode)
        # Frozen BatchNorm must not update its running mean/variance.
        if mode and getattr(self, '_keep_frozen_cnn_eval', False):
            self.cnn[:7].eval()
        return self

    def encode(self, images):
        grid = self.cnn(images)
        grid = nn.functional.adaptive_avg_pool2d(grid, (7, 7))
        tokens = grid.flatten(2).transpose(1, 2)  # B,49,512
        tokens = self.visual_project(tokens) + self.visual_positions
        return self.transformer.encoder(tokens)

    def decode(self, memory, input_ids):
        x = self.text_positions(self.embedding(input_ids) * math.sqrt(self.d_model))
        length = input_ids.shape[1]
        causal_mask = torch.triu(torch.ones(length, length, device=input_ids.device, dtype=torch.bool), diagonal=1)
        return self.output(self.transformer.decoder(
            tgt=x, memory=memory, tgt_mask=causal_mask,
            tgt_key_padding_mask=(input_ids == self.pad_id)))

    def forward(self, images, input_ids):
        return self.decode(self.encode(images), input_ids)

    @torch.no_grad()
    def greedy(self, images, bos_id, eos_id, max_len=35):
        self.eval()
        memory = self.encode(images)
        seq = torch.full((images.size(0), 1), bos_id, dtype=torch.long, device=images.device)
        finished = torch.zeros(images.size(0), dtype=torch.bool, device=images.device)
        for _ in range(max_len - 1):
            logits = self.decode(memory, seq)
            next_ids = logits[:, -1].argmax(-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, eos_id), next_ids)
            seq = torch.cat([seq, next_ids[:, None]], dim=1)
            finished |= next_ids.eq(eos_id)
            if finished.all():
                break
        return seq
