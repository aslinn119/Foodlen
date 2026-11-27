"""Evaluate Task 1 checkpoints on the untouched test split with the shared BLEU code."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "v1_cnn_gru"))
from data import FoodCapDataset, eval_transforms
from caption_metrics import compute_caption_metrics
from model import CaptionModel
from model_v2 import CaptionModelV2


class EvalRows(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return (*self.base[index], self.base.recs[index]["file_name"])


class EvalCollate:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, batch):
        batch.sort(key=lambda row: row[2], reverse=True)
        images, seqs, lengths, raw, names = zip(*batch)
        padded = torch.full((len(seqs), max(lengths)), self.pad, dtype=torch.long)
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = seq
        return torch.stack(images), padded, torch.tensor(lengths), list(raw), list(names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="0 evaluates the full test split")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    variant = ck.get("variant", "v1")
    vocab = ck["vocab"]
    args = ck["args"]
    cls = CaptionModelV2 if variant == "attention" else CaptionModel
    model = cls(len(vocab), args["embed_dim"], args["hidden_size"], vocab["<PAD>"],
                enc_finetune=True, **({"pretrained": False} if variant == "attention" else {})).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    dataset = FoodCapDataset(str(a.root), "test", vocab, args["max_len"], eval_transforms())
    limit = min(a.limit or len(dataset), len(dataset))
    if a.decode == "greedy":
        rows = EvalRows(dataset)
        loader = DataLoader(rows, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                            collate_fn=EvalCollate(vocab["<PAD>"]), pin_memory=device.type == "cuda")
        hyps, refs, filenames = [], [], []
        with torch.inference_mode():
            for images, _, _, raw, names in loader:
                pred = model.greedy_decode(images.to(device), vocab, args["max_len"])
                hyps.extend(pred)
                refs.extend([text.lower().split() for text in raw])
                n = len(hyps)
                filenames.extend(names)
                if n >= limit:
                    break
        hyps, refs = hyps[:limit], refs[:limit]
        filenames = filenames[:limit]
    else:
        hyps, refs, filenames = [], [], []
        rows = EvalRows(dataset)
        loader = DataLoader(rows, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                            collate_fn=EvalCollate(vocab["<PAD>"]), pin_memory=device.type == "cuda")
        with torch.inference_mode():
            inv = {idx: word for word, idx in vocab.items()}
            for images, _, _, raw, names in loader:
                regions, global_embedding = model.encoder(images.to(device, non_blocking=True))
                for j, text in enumerate(raw):
                    ids = model.beam_decode_features(regions[j:j+1], global_embedding[j:j+1],
                                                     vocab, a.beam_size, args["max_len"])
                    hyps.append([inv.get(idx, "<UNK>") for idx in ids])
                    refs.append(text.lower().split())
                    filenames.append(names[j])
                    if len(hyps) >= limit:
                        break
                if len(hyps) % 250 < len(raw):
                    print(f"beam test {len(hyps)}/{limit}", flush=True)
                if len(hyps) >= limit:
                    break
    metrics = compute_caption_metrics(hyps, refs)
    scores = metrics["bleu1_to_4"]
    result = {"checkpoint": str(a.checkpoint), "variant": variant, "decode": a.decode,
              "beam_size": a.beam_size if a.decode == "beam" else None,
              "split": "test", "n": len(hyps), "bleu1_to_4": scores,
              "rouge_l": metrics["rouge_l"], "cider_d": metrics["cider_d"],
              "predictions": [{"file_name": name, "reference": " ".join(ref),
                               "hypothesis": " ".join(hyp)}
                              for name, ref, hyp in zip(filenames, refs, hyps)]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
