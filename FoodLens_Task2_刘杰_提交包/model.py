import torch
import torch.nn as nn
from torchvision import models

NUM_TARGETS = 5  # 重量、热量、蛋白、脂肪、碳水


class NutritionRegressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        # 替换最后一层：512 -> 5
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, NUM_TARGETS)

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = NutritionRegressor()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("输出形状:", out.shape)  # 应该是 [2, 5]
    print("参数量:", sum(p.numel() for p in model.parameters()) / 1e6, "M")