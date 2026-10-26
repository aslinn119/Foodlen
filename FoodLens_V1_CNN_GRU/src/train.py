import argparse, json, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import load_vocab, FoodCaptionDataset, collate_fn
from model import ImageCaptioner

def epoch_run(model, loader, loss_fn, optimizer, device, pad, training):
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    with (torch.enable_grad() if training else torch.no_grad()):
        for images, caps, _ in tqdm(loader, desc="训练" if training else "验证"):
            images, caps = images.to(device), caps.to(device)
            x, y = caps[:, :-1], caps[:, 1:]
            logits = model(images, x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n = y.ne(pad).sum().item()
            total_loss += loss.item() * max(n, 1)
            total_tokens += n
    return total_loss / max(total_tokens, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-dir", default="outputs_v1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    root, out = Path(a.data_root), Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab = load_vocab(root / "vocab.json")
    pad, bos, eos = vocab["<PAD>"], vocab["<BOS>"], vocab["<EOS>"]
    print(f"词表大小={len(vocab)}；PAD={pad}, BOS={bos}, EOS={eos}, UNK={vocab['<UNK>']}")
    tr = FoodCaptionDataset(root, "train", vocab, a.max_len, a.image_size)
    va = FoodCaptionDataset(root, "val", vocab, a.max_len, a.image_size)
    collate = collate_fn(pad)
    train_loader = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers, collate_fn=collate)
    val_loader = DataLoader(va, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("运行设备：", device)
    if device.type == "cuda": print("GPU：", torch.cuda.get_device_name(0))
    model = ImageCaptioner(len(vocab), not a.no_pretrained, a.freeze_backbone).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad)
    optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=2)
    best, history = float("inf"), []
    for e in range(1, a.epochs+1):
        print(f"\\n===== Epoch {e}/{a.epochs} =====")
        tl = epoch_run(model, train_loader, loss_fn, optimizer, device, pad, True)
        vl = epoch_run(model, val_loader, loss_fn, None, device, pad, False)
        scheduler.step(vl)
        record = {"epoch":e, "train_loss":tl, "val_loss":vl, "lr":optimizer.param_groups[0]["lr"]}
        history.append(record)
        print(f"train_loss={tl:.4f} | val_loss={vl:.4f}")
        ckpt = {"epoch":e, "model_state_dict":model.state_dict(), "vocab":vocab,
                "config":vars(a), "val_loss":vl}
        torch.save(ckpt, out/"last_model.pt")
        if vl < best:
            best = vl
            torch.save(ckpt, out/"best_model.pt")
            print("已保存最佳模型")
        (out/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"训练完成，最佳验证集 loss={best:.4f}；输出目录：{out.resolve()}")

if __name__ == "__main__":
    main()
