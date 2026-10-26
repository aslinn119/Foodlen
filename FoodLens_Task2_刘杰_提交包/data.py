import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

# 回归目标（顺序固定，训练和评估必须一致）
TARGET_KEYS = [
    "total_weight_g",
    "total_calories_kcal",
    "protein_g",
    "fat_g",
    "carbohydrate_g",
]


class NutritionDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        ann_path = os.path.join(DATA_ROOT, "annotations", f"{split}.jsonl")
        with open(ann_path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img_path = os.path.join(DATA_ROOT, r["image"])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        targets = torch.tensor(
            [r["target"][k] for k in TARGET_KEYS], dtype=torch.float32
        )
        return img, targets


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_dataloaders(batch_size=32, num_workers=0):
    train_ds = NutritionDataset("train", transform=get_transforms(True))
    val_ds = NutritionDataset("val", transform=get_transforms(False))
    test_ds = NutritionDataset("test", transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)
    imgs, targets = next(iter(train_loader))
    print("图像 batch 形状:", imgs.shape)      # [4, 3, 224, 224]
    print("目标 batch 形状:", targets.shape)   # [4, 5]
    print("目标示例:", targets[0])