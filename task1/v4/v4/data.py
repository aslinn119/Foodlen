"""V4 dataloader aligned with the project's V1 tokenizer and image transforms."""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from v1_data import (
    FoodCapDataset,
    load_vocab as load_vocab_json,
    train_transforms,
    eval_transforms,
    PAD,
    BOS,
    EOS,
    UNK,
)


def load_vocab(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        vocab = load_vocab_json(str(path))
    elif path.suffix.lower() in (".pt", ".pth"):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError("Expected a V1 checkpoint dictionary")
        raw = ckpt.get("vocab")
        if raw is None:
            raise KeyError(f"V1 checkpoint has no vocab key; available: {list(ckpt.keys())}")
        if isinstance(raw, dict) and "word2idx" in raw:
            raw = raw["word2idx"]
        if isinstance(raw, dict) and "stoi" in raw:
            raw = raw["stoi"]
        if not isinstance(raw, dict):
            raise ValueError("Unsupported checkpoint vocab format")
        vocab = {str(k): int(v) for k, v in raw.items()}
    else:
        raise ValueError("Supply V1 best_v1.pt or its original vocab.json")

    required = {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
    for token, value in required.items():
        if vocab.get(token) != value:
            raise ValueError(f"V1 vocab mismatch: {token} should equal {value}")
    if set(vocab.values()) != set(range(len(vocab))):
        raise ValueError("Vocab IDs must be consecutive")
    print(f"Using official V1 vocabulary: {len(vocab)} tokens")
    return vocab


def find_special(vocab, kind):
    return vocab[{"pad": PAD, "bos": BOS, "eos": EOS, "unk": UNK}[kind]]


class FoodCaptionDataset(Dataset):
    def __init__(self, root, split, vocab, max_len=30, limit=None, augment=None):
        self.split = split
        if augment is None:
            augment = split == "train"
        self.data = FoodCapDataset(
            root,
            split,
            vocab,
            max_len=max_len,
            transform=train_transforms() if augment else eval_transforms(),
        )
        self.indices = range(min(len(self.data), limit)) if limit is not None else range(len(self.data))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        image, ids, _, raw = self.data[idx]
        return image, ids, str(self.data.recs[idx]["file_name"]), raw


class CaptionCollate:
    def __init__(self, pad_id):
        self.pad_id = int(pad_id)

    def __call__(self, batch):
        images, sequences, names, captions = zip(*batch)
        tokens = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.pad_id
        )
        return torch.stack(images), tokens, list(names), list(captions)


def make_collate(pad_id):
    return CaptionCollate(pad_id)
