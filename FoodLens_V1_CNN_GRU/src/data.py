import json, re
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PATTERN = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*|[.,!?;:()]")

def tokenize(text):
    return PATTERN.findall(str(text).lower())

def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        vocab = json.load(f)
    if not isinstance(vocab, dict):
        raise ValueError("vocab.json 应为 token -> id 的字典")
    vocab = {str(k): int(v) for k, v in vocab.items()}
    for token in ("<PAD>", "<BOS>", "<EOS>", "<UNK>"):
        if token not in vocab:
            raise ValueError(f"vocab.json 中缺少 {token}，请检查特殊 token 的实际拼写")
    return vocab

def caption_to_ids(caption, vocab, max_len=40):
    ids = [vocab["<BOS>"]]
    for token in tokenize(caption):
        ids.append(vocab.get(token, vocab.get(token.lower(), vocab["<UNK>"])))
    ids.append(vocab["<EOS>"])
    return ids[:max_len]

def image_transform(image_size=224, train=False):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(ops)

class FoodCaptionDataset(Dataset):
    def __init__(self, root, split, vocab, max_len=40, image_size=224):
        self.root, self.vocab, self.max_len = Path(root), vocab, max_len
        ann = self.root / "annotations" / f"{split}.jsonl"
        if not ann.is_file():
            raise FileNotFoundError(ann)
        with ann.open(encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        self.transform = image_transform(image_size, split == "train")
    def __len__(self):
        return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        p = self.root / r["file_name"]
        with Image.open(p) as im:
            image = self.transform(im.convert("RGB"))
        ids = caption_to_ids(r["caption"], self.vocab, self.max_len)
        return image, torch.tensor(ids, dtype=torch.long), r["caption"]

def collate_fn(pad_id):
    def collate(batch):
        images, captions, raw = zip(*batch)
        padded = torch.full((len(captions), max(map(len, captions))), pad_id, dtype=torch.long)
        for i, c in enumerate(captions):
            padded[i, :len(c)] = c
        return torch.stack(images), padded, list(raw)
    return collate
