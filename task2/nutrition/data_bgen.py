"""
data_bgen.py —— B-gen 数据集：CLIP+LoRA 视觉特征缓存 + 字符级序列

特征缓存（一次性）：
  - 用冻结的 CLIP+LoRA 编码器（checkpoints_lora_mt 权重）前向三个 split，
    保存 [CLS] 池化特征（3258 x 768，共约 10MB）
  - 之后解码器训练/评测直接读缓存，无需重复跑图像骨干，秒级迭代

序列构建：
  - 输入 = <bos> + target_text 字符序列，监督 = 字符序列 + <eos>（右填充 <pad>）
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from model_bgen import build_char_vocab, encode_text, PAD, MAX_LEN, DATA_ROOT
from train_lora_mt import CLIPT_TF_EVAL

FEAT_DIR = os.path.join(DATA_ROOT, "checkpoints_bgen", "features")
os.makedirs(FEAT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_records(split):
    with open(os.path.join(DATA_ROOT, "annotations", f"{split}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def cache_features(split, force=False):
    """对 split 全量图像跑冻结 CLIP+LoRA 编码器，缓存池化特征。返回特征 npz 路径。"""
    path = os.path.join(FEAT_DIR, f"{split}.npz")
    if os.path.exists(path) and not force:
        return path

    from model_bgen import load_bgen_encoder
    model = load_bgen_encoder(DEVICE)

    records = load_records(split)
    feats, order = [], []
    bs = 128
    for i in tqdm(range(0, len(records), bs), desc=f"缓存 {split} 特征"):
        batch = records[i:i + bs]
        imgs = []
        for r in batch:
            img = Image.open(os.path.join(DATA_ROOT, r["image"])).convert("RGB")
            imgs.append(CLIPT_TF_EVAL(img))
        with torch.no_grad():
            f = model.extract_features(torch.stack(imgs).to(DEVICE))
        feats.append(f.cpu().numpy())
        order.extend(r["dish_id"] for r in batch)

    feats = np.concatenate(feats, axis=0).astype(np.float32)
    np.savez(path, feats=feats, dish_ids=np.array(order))
    print(f"{split}: {feats.shape} 已缓存 -> {path}")
    return path


class BGenSeqDataset(Dataset):
    """返回 (feat[768], ids[MAX_LEN] 含 BOS/EOS 并右填充, length)"""

    def __init__(self, split):
        char2idx, _ = build_char_vocab()
        self.char2idx = char2idx
        path = cache_features(split)
        z = np.load(path, allow_pickle=True)
        feat_map = {d: f for d, f in zip(z["dish_ids"].tolist(), z["feats"])}
        self.records = load_records(split)
        self.feats = np.stack([feat_map[r["dish_id"]] for r in self.records])
        self.seqs = [encode_text(r["target_text"], char2idx) for r in self.records]
        assert all(len(s) <= MAX_LEN for s in self.seqs), "存在超长序列，请调大 MAX_LEN"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        s = self.seqs[i]
        ids = torch.full((MAX_LEN,), PAD, dtype=torch.long)
        ids[:len(s)] = torch.tensor(s, dtype=torch.long)
        return torch.from_numpy(self.feats[i]), ids, len(s)


def collate(batch):
    feats = torch.stack([b[0] for b in batch])
    # 按 batch 内最长序列截断，减少无效计算
    max_len = max(b[2] for b in batch)
    ids = torch.stack([b[1][:max_len] for b in batch])
    lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return feats, ids, lengths


if __name__ == "__main__":
    for sp in ["train", "val", "test"]:
        cache_features(sp)
    ds = BGenSeqDataset("train")
    feats, ids, ln = ds[0]
    print("feat:", feats.shape, "ids:", ids.shape, "len:", ln)
    from model_bgen import decode_text, build_char_vocab
    _, idx2char = build_char_vocab()
    print("解码校验:", decode_text(ids[:ln].tolist(), idx2char)[:80], "...")
    loader = DataLoader(ds, batch_size=8, collate_fn=collate)
    f, i, l = next(iter(loader))
    print("batch:", f.shape, i.shape, l.shape)
