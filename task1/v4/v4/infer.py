"""Generate captions for a FoodLens V4 split using greedy decoding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import FoodCaptionDataset, make_collate
from model import ViTCaptioner


def parse_args():
    parser = argparse.ArgumentParser(description="Infer with FoodLens V4")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=0, help="0 means full split")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default="outputs/v4_baseline/v4_predictions.jsonl")
    return parser.parse_args()


def decode_ids(ids, id_to_token, eos_id, pad_id, bos_id):
    words = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == eos_id:
            break
        if token_id in (pad_id, bos_id):
            continue
        words.append(id_to_token.get(token_id, "<UNK>"))
    return " ".join(words).strip()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = checkpoint["vocab"]
    config = checkpoint["config"]
    pad_id = int(config["pad_id"])
    bos_id = int(config["bos_id"])
    eos_id = int(config["eos_id"])
    max_len = int(args.max_len or config["max_len"])

    model = ViTCaptioner(
        vocab_size=len(vocab), pad_id=pad_id, max_len=int(config["max_len"]),
        d_model=int(config["d_model"]), nhead=int(config["nhead"]),
        num_decoder_layers=int(config["num_decoder_layers"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]), pretrained=bool(config["pretrained"]),
    )
    model.load_trainable_state_dict(checkpoint["trainable_state"])
    model.to(device).eval()

    limit = None if args.limit == 0 else args.limit
    dataset = FoodCaptionDataset(
        args.data_root, args.split, vocab, max_len=max_len, limit=limit, augment=False
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=make_collate(pad_id),
    )
    id_to_token = {int(index): token for token, index in vocab.items()}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as output_file, torch.no_grad():
        for images, _, names, references in loader:
            images = images.to(device)
            generated = model.generate(images, bos_id, eos_id, max_len=max_len).cpu()
            for name, reference, token_ids in zip(names, references, generated):
                caption = decode_ids(token_ids.tolist(), id_to_token, eos_id, pad_id, bos_id)
                record = {
                    "file_name": name,
                    "caption": caption,
                    "reference": reference,
                    "model_id": "v4_vit_b32_transformer",
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {output_path} ({count} predictions)")


if __name__ == "__main__":
    main()
