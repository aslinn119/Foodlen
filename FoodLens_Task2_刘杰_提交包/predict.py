"""
predict.py —— 与组长联调的统一推理接口

predict_nutrition(image_path) -> dict

使用模型 B（CLIP + LoRA 参数高效微调多任务模型，checkpoints_lora_mt/），
返回完整营养 JSON（含 items 食材列表），与预定义 Schema 完全一致：

{
  "items": [{"name": "bacon", "weight_g": 23.0}, ...],
  "total_weight_g": ...,
  "total_calories_kcal": ...,
  "protein_g": ...,
  "fat_g": ...,
  "carbohydrate_g": ...
}

demo_task2.py 是本接口的命令行演示封装。
"""

import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model_lora import CLIPLoRAMultiTask, load_ingredient_vocab, WEIGHT_NORM

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(DATA_ROOT, "checkpoints_lora_mt")

TARGET_KEYS = ["total_weight_g", "total_calories_kcal",
               "protein_g", "fat_g", "carbohydrate_g"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4815, 0.4578, 0.4082],
                         std=[0.2686, 0.2613, 0.2758]),
])

# 全局缓存：模型、词表、阈值、归一化统计
_state = {}


def _load():
    if not _state:
        ck = torch.load(os.path.join(CKPT_DIR, "best_model.pt"),
                        map_location=DEVICE, weights_only=False)
        vocab = load_ingredient_vocab()
        model = CLIPLoRAMultiTask(num_targets=5, num_ingredients=len(vocab),
                                  last_n=ck["lora_last_n"], rank=ck["rank"],
                                  alpha=ck["alpha"]).to(DEVICE)
        model.load_state_dict(ck["model_state"])
        model.eval()
        _state.update({
            "model": model,
            "vocab": vocab,
            "threshold": ck["threshold"],
            "mean": torch.tensor(ck["mean"], dtype=torch.float32),
            "std": torch.tensor(ck["std"], dtype=torch.float32),
        })
    return _state


@torch.no_grad()
def predict_nutrition(image_path: str) -> dict:
    """
    输入：菜品图片路径
    输出：完整营养 JSON dict（items + 5 个数值字段）
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    s = _load()
    img = Image.open(image_path).convert("RGB")
    x = _infer_transform(img).unsqueeze(0).to(DEVICE)

    out = s["model"](x)
    nums = out["numeric"].cpu() * s["std"] + s["mean"]
    probs = torch.sigmoid(out["ingredient"]).cpu().numpy()[0]
    wts = out["item_weight"].cpu().numpy()[0] * WEIGHT_NORM

    n = {k: max(float(nums[0, j]), 0.0) for j, k in enumerate(TARGET_KEYS)}

    idx = np.where(probs >= s["threshold"])[0][:16]
    items = []
    for j in idx:
        items.append({"name": s["vocab"][j], "weight_g": max(float(wts[j]), 0.0)})
    total_w = sum(it["weight_g"] for it in items)
    if total_w > 0:  # 分项重量等比缩放到与预测总重一致
        scale = n["total_weight_g"] / total_w
        for it in items:
            it["weight_g"] = round(it["weight_g"] * scale, 1)
    else:
        items = []

    return {
        "items": items,
        "total_weight_g": round(n["total_weight_g"], 1),
        "total_calories_kcal": round(n["total_calories_kcal"], 1),
        "protein_g": round(n["protein_g"], 1),
        "fat_g": round(n["fat_g"], 1),
        "carbohydrate_g": round(n["carbohydrate_g"], 1),
    }


if __name__ == "__main__":
    test_img = os.path.join(DATA_ROOT, "images", "test")
    sample = sorted(os.listdir(test_img))[0]
    sample_path = os.path.join(test_img, sample)
    print(f"测试图片: {sample_path}")
    print(json.dumps(predict_nutrition(sample_path), indent=2, ensure_ascii=False))
