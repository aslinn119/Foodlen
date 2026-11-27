"""FoodLens V4: ImageNet-pretrained ViT-B/32 encoder + Transformer caption decoder."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torchvision.models import ViT_B_32_Weights, vit_b_32


class ViTCaptioner(nn.Module):
    """A ViT image encoder with an autoregressive Transformer text decoder.

    The torchvision ViT-B/32 encoder is frozen for the CPU-friendly baseline.
    Its 49 patch tokens plus the CLS token are projected to decoder dimension.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        max_len: int = 30,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.pretrained = bool(pretrained)

        weights = ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_32(weights=weights)
        # We use the sequence from vit.encoder, not the classification head.
        self.vit.heads = nn.Identity()
        for parameter in self.vit.parameters():
            parameter.requires_grad = False
        self.vit.eval()

        self.visual_projection = nn.Linear(768, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.input_dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_text_parameters()

    def _init_text_parameters(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.token_embedding.padding_idx].zero_()

    def train(self, mode: bool = True):
        """Keep the frozen pretrained ViT in eval mode while training the decoder."""
        super().train(mode)
        self.vit.eval()
        return self

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Return projected visual memory [B, 50, d_model]."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images shaped [B, 3, H, W], got {tuple(images.shape)}")
        with torch.no_grad():
            # torchvision ViT-B/32 expects 224x224 input and creates 49 patches.
            x = self.vit._process_input(images)
            batch_size = x.shape[0]
            cls_token = self.vit.class_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)  # [B, 50, 768]
            x = self.vit.encoder(x)               # [B, 50, 768]
        return self.visual_projection(x)          # [B, 50, d_model]

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        # True entries are masked by PyTorch Transformer APIs.
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def decode(self, memory: torch.Tensor, tokens_in: torch.Tensor) -> torch.Tensor:
        """Decode token prefixes; return logits [B, L, vocab_size]."""
        batch_size, length = tokens_in.shape
        if length > self.max_len:
            raise ValueError(f"Caption length {length} exceeds max_len={self.max_len}")
        positions = torch.arange(length, device=tokens_in.device).unsqueeze(0).expand(batch_size, -1)
        text = self.token_embedding(tokens_in) * math.sqrt(self.d_model)
        text = text + self.position_embedding(positions)
        text = self.input_dropout(text)

        tgt_mask = self.causal_mask(length, tokens_in.device)
        tgt_padding_mask = tokens_in.eq(self.pad_id)
        decoded = self.decoder(
            tgt=text,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.output_projection(decoded)

    def forward(self, images: torch.Tensor, tokens_in: torch.Tensor) -> torch.Tensor:
        memory = self.encode_image(images)
        return self.decode(memory, tokens_in)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int | None = None,
    ) -> torch.Tensor:
        """Greedy decoding. Returns token IDs including the initial BOS token."""
        self.eval()
        max_len = int(max_len or self.max_len)
        max_len = min(max_len, self.max_len)
        memory = self.encode_image(images)
        batch_size = images.shape[0]
        generated = torch.full(
            (batch_size, 1), int(bos_id), dtype=torch.long, device=images.device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=images.device)

        for _ in range(max_len - 1):
            logits = self.decode(memory, generated)[:, -1, :]
            logits[:, self.pad_id] = float("-inf")
            logits[:, int(bos_id)] = float("-inf")
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, self.pad_id), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(int(eos_id))
            if bool(finished.all()):
                break
        return generated

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Save only decoder/projection parameters; frozen ViT comes from torchvision weights."""
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("vit.")
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]):
        missing, unexpected = self.load_state_dict(state, strict=False)
        unexpected_non_vit = [key for key in unexpected if not key.startswith("vit.")]
        missing_non_vit = [key for key in missing if not key.startswith("vit.")]
        if unexpected_non_vit or missing_non_vit:
            raise RuntimeError(
                f"Checkpoint mismatch. Missing={missing_non_vit}, unexpected={unexpected_non_vit}"
            )
