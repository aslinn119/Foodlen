"""
model_lora.py —— 模型 B：CLIP（多模态预训练模型）+ LoRA 参数高效微调 + 多任务头

对应课程要求「经过全量或参数高效微调的多模态预训练模型」中的参数高效微调路线：
  - 骨干：CLIP ViT-B/32（ImageNet 图文对比预训练，本地 clip_model/）
  - 微调：冻结全部骨干参数，仅在最后 N 个 encoder 层的注意力
    q/k/v/out 四个投影上注入低秩矩阵 A/B（LoRA, rank=r, scale=alpha/r）
  - 任务头（多任务，共享 [CLS] 视觉表示）：
      1) 营养回归头：5 个数值字段（重量/热量/蛋白/脂肪/碳水）
      2) 食材多标签头：151 种训练集食材的 0/1 多标签分类（供 F1 评测）
      3) 分项重量头：每种食材的分量回归（供 JSON 的 items[].weight_g）
"""

import json
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
CLIP_PATH = os.path.join(DATA_ROOT, "clip_model")
INGREDIENTS_PATH = os.path.join(DATA_ROOT, "ingredients.json")

NUM_KEYS = ["total_weight_g", "total_calories_kcal", "protein_g", "fat_g", "carbohydrate_g"]
WEIGHT_NORM = 100.0   # 分项重量归一化系数（克）


# =====================================================================
# 食材词表（由训练集构建）
# =====================================================================
def build_ingredient_vocab():
    """返回按频率排序的 151 种训练集食材列表，并保存到 ingredients.json。"""
    if os.path.exists(INGREDIENTS_PATH):
        return json.load(open(INGREDIENTS_PATH, encoding="utf-8"))
    from collections import Counter
    cnt = Counter()
    with open(os.path.join(DATA_ROOT, "annotations", "train.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            for it in json.loads(line)["target"]["items"]:
                cnt[it["name"]] += 1
    vocab = [w for w, _ in cnt.most_common()]
    json.dump(vocab, open(INGREDIENTS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return vocab


# =====================================================================
# LoRA 线性层
# =====================================================================
class LoRALinear(nn.Module):
    """在原 Linear 旁并联低秩分支 B(Ax)，scale = alpha / rank。原始权重冻结。"""

    def __init__(self, linear, rank=16, alpha=16.0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight          # 冻结
        self.bias = linear.bias              # 冻结
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale
        return base + delta

    @property
    def num_lora_params(self):
        return self.lora_a.numel() + self.lora_b.numel()


def inject_lora(clip_model, last_n=4, rank=16, alpha=16.0):
    """把 CLIP 最后 last_n 个 encoder 层的注意力投影替换为 LoRALinear。"""
    layers = clip_model.vision_model.encoder.layers
    for layer in layers[-last_n:]:
        attn = layer.self_attn
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            setattr(attn, name, LoRALinear(getattr(attn, name), rank=rank, alpha=alpha))
    return clip_model


def count_lora_params(model):
    return sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)


# =====================================================================
# 多任务模型
# =====================================================================
class CLIPLoRAMultiTask(nn.Module):
    def __init__(self, num_targets=5, num_ingredients=151, last_n=4, rank=16, alpha=16.0):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(CLIP_PATH)
        for p in self.clip.parameters():
            p.requires_grad = False
        inject_lora(self.clip, last_n=last_n, rank=rank, alpha=alpha)

        hidden = self.clip.config.hidden_size  # 768
        self.norm_head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, num_targets))
        self.ingredient_head = nn.Linear(hidden, num_ingredients)
        self.item_weight_head = nn.Linear(hidden, num_ingredients)

    def forward(self, pixel_values):
        pooled = self.clip(pixel_values=pixel_values).pooler_output  # [B,768]
        return {
            "numeric": self.norm_head(pooled),          # [B,5] 归一化空间
            "ingredient": self.ingredient_head(pooled), # [B,C]
            "item_weight": self.item_weight_head(pooled),  # [B,C] 重量/WEIGHT_NORM
        }


def load_ingredient_vocab():
    return json.load(open(INGREDIENTS_PATH, encoding="utf-8"))


if __name__ == "__main__":
    torch.manual_seed(0)
    m = CLIPLoRAMultiTask(num_ingredients=len(build_ingredient_vocab()))
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    for k, v in out.items():
        print(k, v.shape)
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"总参数 {total/1e6:.1f}M，LoRA+头部可训练 {trainable/1e6:.3f}M "
          f"({trainable/total*100:.2f}%)，其中纯 LoRA {count_lora_params(m)/1e6:.3f}M")
