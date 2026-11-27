"""Train FoodLens V4 (frozen ImageNet ViT-B/32 + Transformer caption decoder)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import FoodCaptionDataset, load_vocab, find_special, make_collate
from model import ViTCaptioner


def parse_args():
    parser = argparse.ArgumentParser(description="Train FoodLens V4 ViT captioner")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--vocab", required=True, help="V1 best_v1.pt or vocab.json")
    parser.add_argument("--run-dir", default="outputs/v4_baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-limit", type=int, default=0, help="0 means full train split")
    parser.add_argument("--val-limit", type=int, default=0, help="0 means full val split")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Resume from outputs/.../last_v4.pt")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, captions, _, _ in loader:
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            tokens_in = captions[:, :-1]
            targets = captions[:, 1:]

            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images, tokens_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()

            valid_tokens = targets.ne(criterion.ignore_index).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    return total_loss / max(total_tokens, 1)


def save_checkpoint(path, model, optimizer, epoch, best_val, vocab, config):
    checkpoint = {
        "format_version": 1,
        "model_name": "vit_b_32_imagenet1k_v1_transformer_decoder",
        "trainable_state": model.trainable_state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val),
        "vocab": vocab,
        "config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(args.vocab)
    pad_id = find_special(vocab, "pad")
    bos_id = find_special(vocab, "bos")
    eos_id = find_special(vocab, "eos")
    train_limit = None if args.train_limit == 0 else args.train_limit
    val_limit = None if args.val_limit == 0 else args.val_limit

    train_set = FoodCaptionDataset(args.data_root, "train", vocab, args.max_len, train_limit, augment=True)
    val_set = FoodCaptionDataset(args.data_root, "val", vocab, args.max_len, val_limit, augment=False)
    collate = make_collate(pad_id)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    model = ViTCaptioner(
        vocab_size=len(vocab), pad_id=pad_id, max_len=args.max_len,
        d_model=512, nhead=8, num_decoder_layers=3,
        dim_feedforward=2048, dropout=0.1, pretrained=True,
    ).to(device)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    config = {
        "pretrained": True,
        "backbone": "torchvision ViT-B/32 ImageNet1K_V1",
        "d_model": 512,
        "nhead": 8,
        "num_decoder_layers": 3,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "max_len": args.max_len,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
    }

    start_epoch = 1
    best_val = float("inf")
    history_path = run_dir / "history.json"
    history = []

    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(checkpoint["trainable_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val_loss", float("inf")))
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"Resumed from epoch {start_epoch - 1}; next epoch {start_epoch}")

    print(
        f"device: {device} vocab_size: {len(vocab)} "
        f"train: {len(train_set)} val: {len(val_set)} "
        f"trainable_params: {sum(p.numel() for p in trainable_parameters):,}"
    )
    print("Backbone: pretrained ViT-B/32 (frozen); decoder and visual projection are trainable.")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)

        save_checkpoint(run_dir / "last_v4.pt", model, optimizer, epoch, min(best_val, val_loss), vocab, config)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(run_dir / "best_v4.pt", model, optimizer, epoch, best_val, vocab, config)
            print(f"Saved best checkpoint: {run_dir / 'best_v4.pt'}", flush=True)

    print(f"Training finished. Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
