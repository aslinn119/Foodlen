# -*- coding: utf-8 -*-
"""V1-compatible data reader, tokenizer and transforms used by FoodLens V4."""
import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

PAD, BOS, EOS, UNK = "<PAD>", "<BOS>", "<EOS>", "<UNK>"


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    assert vocab[PAD] == 0 and vocab[BOS] == 1 and vocab[EOS] == 2 and vocab[UNK] == 3
    return vocab


class FoodCapDataset(Dataset):
    """Read annotations/{split}.jsonl and return image, token IDs, length, raw caption."""
    def __init__(self, root, split, vocab, max_len=30, transform=None):
        self.root = root
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform
        ann = os.path.join(root, "annotations", split + ".jsonl")
        with open(ann, "r", encoding="utf-8") as f:
            self.recs = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.recs)

    def encode(self, caption):
        ids = [self.vocab[BOS]]
        ids += [self.vocab.get(word, self.vocab[UNK]) for word in caption.lower().split()]
        ids = ids[: self.max_len - 1] + [self.vocab[EOS]]
        return ids

    def __getitem__(self, index):
        record = self.recs[index]
        image_path = os.path.join(self.root, record["file_name"])
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        if self.transform:
            image = self.transform(image)
        ids = self.encode(record["caption"])
        return image, torch.tensor(ids, dtype=torch.long), len(ids), record["caption"]


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
    def __init__(self, pad_id):
        self.pad_id = int(pad_id)

    def __call__(self, batch):
        batch.sort(key=lambda item: item[2], reverse=True)
        images, sequences, lengths, captions = zip(*batch)
        max_length = max(lengths)
        padded = torch.full((len(sequences), max_length), self.pad_id, dtype=torch.long)
        for index, sequence in enumerate(sequences):
            padded[index, : len(sequence)] = sequence
        return torch.stack(images), padded, torch.tensor(lengths, dtype=torch.long), list(captions)


def make_collate(pad_id):
    return CollateFn(pad_id)
