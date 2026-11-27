# -*- coding: utf-8 -*-
"""数据加载：Food500Cap_ready 数据集（jsonl 标注 + 图片目录）。"""
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# 特殊标记（与 vocab.json 一致）
PAD, BOS, EOS, UNK = "<PAD>", "<BOS>", "<EOS>", "<UNK>"


def load_vocab(path):
    vocab = json.load(open(path, encoding="utf-8"))
    assert vocab[PAD] == 0 and vocab[BOS] == 1 and vocab[EOS] == 2 and vocab[UNK] == 3
    return vocab


class FoodCapDataset(Dataset):
    """读取 annotations/{split}.jsonl，返回 (image_tensor, caption_ids, length, raw_caption)。"""
    def __init__(self, root, split, vocab, max_len=30, transform=None):
        self.root = root
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform
        ann = os.path.join(root, "annotations", split + ".jsonl")
        self.recs = [json.loads(l) for l in open(ann, encoding="utf-8")]

    def __len__(self):
        return len(self.recs)

    def encode(self, caption):
        ids = [self.vocab[BOS]]
        ids += [self.vocab.get(w, self.vocab[UNK]) for w in caption.lower().split()]
        ids = ids[: self.max_len - 1] + [self.vocab[EOS]]
        return ids

    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(os.path.join(self.root, r["file_name"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        ids = self.encode(r["caption"])
        return img, torch.tensor(ids, dtype=torch.long), len(ids), r["caption"]


def train_transforms():
    return T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def eval_transforms():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class CollateFn:
    """把变长 caption 填充成 batch，并按长度降序排列（便于 pack_padded_sequence）。
    用类而不用闭包：DataLoader 多进程 worker 需要能 pickle collate 函数。"""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        batch.sort(key=lambda x: x[2], reverse=True)
        imgs, seqs, lens, raw = zip(*batch)
        maxl = max(lens)
        padded = torch.full((len(seqs), maxl), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, : len(s)] = s
        return torch.stack(imgs), padded, torch.tensor(lens, dtype=torch.long), list(raw)


def make_collate(pad_id):
    return CollateFn(pad_id)
