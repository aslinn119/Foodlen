# -*- coding: utf-8 -*-
"""数据加载：Food500Cap_ready 数据集（jsonl 标注 + 图片目录）。"""
import json  # 解析 jsonl 标注文件和 vocab.json
import os   # 拼接文件路径

import torch
from PIL import Image          # 读取图片
from torch.utils.data import Dataset  # PyTorch 数据集基类
import torchvision.transforms as T    # 图像预处理/数据增强

# 特殊标记（与 vocab.json 一致）：填充符、句首符、句尾符、未登录词
PAD, BOS, EOS, UNK = "<PAD>", "<BOS>", "<EOS>", "<UNK>"


def load_vocab(path):
    # 读取词表文件：{单词: id} 的字典
    vocab = json.load(open(path, encoding="utf-8"))
    # 校验 4 个特殊标记的 id 必须固定为 0/1/2/3（模型和填充逻辑依赖这个约定）
    assert vocab[PAD] == 0 and vocab[BOS] == 1 and vocab[EOS] == 2 and vocab[UNK] == 3
    return vocab


class FoodCapDataset(Dataset):
    """读取 annotations/{split}.jsonl，返回 (image_tensor, caption_ids, length, raw_caption)。"""
    def __init__(self, root, split, vocab, max_len=30, transform=None):
        self.root = root            # 数据集根目录
        self.vocab = vocab          # 词表
        self.max_len = max_len      # caption 最大长度（超过会截断）
        self.transform = transform  # 图像预处理流水线
        # 标注文件路径，如 annotations/train.jsonl
        ann = os.path.join(root, "annotations", split + ".jsonl")
        # jsonl 每行一条 JSON：{"file_name": 图片相对路径, "caption": 描述文本}
        self.recs = [json.loads(l) for l in open(ann, encoding="utf-8")]

    def __len__(self):
        return len(self.recs)  # 数据集大小 = 标注条数

    def encode(self, caption):
        # 把英文句子的文本转成 token id 序列
        ids = [self.vocab[BOS]]  # 开头加 BOS
        # 转小写、按空格分词，每个词查词表；查不到的词用 UNK 的 id
        ids += [self.vocab.get(w, self.vocab[UNK]) for w in caption.lower().split()]
        # 截断到 max_len-1（给 EOS 留一个位置），末尾加 EOS
        ids = ids[: self.max_len - 1] + [self.vocab[EOS]]
        return ids

    def __getitem__(self, i):
        # 取第 i 条数据
        r = self.recs[i]
        # 打开对应图片并统一转成 RGB 三通道（防止灰度图/带透明通道的图出错）
        img = Image.open(os.path.join(self.root, r["file_name"])).convert("RGB")
        if self.transform:
            img = self.transform(img)  # 做预处理（缩放/裁剪/归一化等）
        ids = self.encode(r["caption"])  # 文本转 id 序列
        # 返回：图片张量、id 序列张量、序列真实长度、原始文本（评测 BLEU 用）
        return img, torch.tensor(ids, dtype=torch.long), len(ids), r["caption"]


def train_transforms():
    # 训练集预处理：带数据增强，增加模型泛化能力
    return T.Compose([
        T.Resize(256),               # 短边缩放到 256
        T.RandomCrop(224),           # 随机裁剪 224x224（数据增强）
        T.RandomHorizontalFlip(),    # 随机水平翻转（数据增强）
        T.ToTensor(),                # 转成 [0,1] 的 Tensor，维度 [C,H,W]
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet 均值方差标准化
    ])


def eval_transforms():
    # 验证/测试集预处理：不做随机增强，保证结果可复现
    return T.Compose([
        T.Resize(256),               # 短边缩放到 256
        T.CenterCrop(224),           # 中心裁剪 224x224（确定性）
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class CollateFn:
    """把变长 caption 填充成 batch，并按长度降序排列（便于 pack_padded_sequence）。
    用类而不用闭包：DataLoader 多进程 worker 需要能 pickle collate 函数。"""

    def __init__(self, pad_id):
        self.pad_id = pad_id  # 填充用的 id（即 <PAD> 的 id，通常为 0）

    def __call__(self, batch):
        # batch 是若干条 (img, seq, len, raw)，先按序列长度从大到小排序
        # pack_padded_sequence 要求 batch 内按长度降序
        batch.sort(key=lambda x: x[2], reverse=True)
        imgs, seqs, lens, raw = zip(*batch)  # 拆成 4 组
        maxl = max(lens)  # 本 batch 内最长序列长度（以此为准对齐）
        # 先造一个全 PAD 的 [batch, maxl] 张量
        padded = torch.full((len(seqs), maxl), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, : len(s)] = s  # 每条序列填入前面，后面留 PAD
        # 返回：图片堆叠成 [B,C,H,W]、填充后的 id、各序列真实长度、原始文本列表
        return torch.stack(imgs), padded, torch.tensor(lens, dtype=torch.long), list(raw)


def make_collate(pad_id):
    # 工厂函数：对外提供一个简单的构造方式
    return CollateFn(pad_id)
