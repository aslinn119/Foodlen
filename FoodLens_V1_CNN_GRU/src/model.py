import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, feature_dim=256, pretrained=True, freeze=False):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
        except Exception as e:
            print(f"[提示] 预训练权重加载失败，改用随机初始化：{e}")
            net = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(512, feature_dim),
                                  nn.ReLU(), nn.Dropout(0.2))
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
    def forward(self, images):
        return self.proj(self.backbone(images))

class Decoder(nn.Module):
    def __init__(self, vocab_size, feature_dim=256, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.init_hidden = nn.Linear(feature_dim, hidden_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, features, input_ids):
        h0 = self.init_hidden(features).unsqueeze(0)
        out, _ = self.gru(self.dropout(self.embed(input_ids)), h0)
        return self.fc(self.dropout(out))
    @torch.no_grad()
    def generate(self, features, bos_id, eos_id, max_len=40):
        hidden = self.init_hidden(features).unsqueeze(0)
        current = torch.full((features.size(0), 1), bos_id, dtype=torch.long, device=features.device)
        finished = torch.zeros(features.size(0), dtype=torch.bool, device=features.device)
        result = []
        for _ in range(max_len - 1):
            out, hidden = self.gru(self.embed(current), hidden)
            nxt = self.fc(out[:, -1]).argmax(-1)
            nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
            result.append(nxt)
            finished |= nxt.eq(eos_id)
            current = nxt.unsqueeze(1)
            if finished.all():
                break
        return torch.stack(result, dim=1) if result else current[:, :0]

class ImageCaptioner(nn.Module):
    def __init__(self, vocab_size, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.encoder = Encoder(pretrained=pretrained, freeze=freeze_backbone)
        self.decoder = Decoder(vocab_size)
    def forward(self, images, input_ids):
        return self.decoder(self.encoder(images), input_ids)
