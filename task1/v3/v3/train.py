import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import FoodCaptionDataset, load_vocab, find_special, make_collate
from model import FoodLensV3


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--vocab', required=True, help='V1 official vocabulary JSON or V1 checkpoint')
    p.add_argument('--run-dir', default='outputs/v3_smoke')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--train-limit', type=int, default=None, help='Omit for full train split')
    p.add_argument('--val-limit', type=int, default=None, help='Omit for full validation split')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--image-lr', type=float, default=1e-5)
    p.add_argument('--max-len', type=int, default=30)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--freeze-cnn', action='store_true')
    p.add_argument('--finetune-layer4', action='store_true', help='Freeze early ResNet stages; train only layer4')
    p.add_argument('--init-checkpoint', default=None, help='Initialize a fresh fine-tuning run from best_v3.pt (not optimizer resume)')
    p.add_argument('--resume', default=None, help='Path to last_v3.pt to resume from the next epoch')
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--enc-layers', type=int, default=2)
    p.add_argument('--dec-layers', type=int, default=2)
    args = p.parse_args()
    if args.finetune_layer4 and args.freeze_cnn:
        p.error('--finetune-layer4 and --freeze-cnn cannot be used together')
    if args.init_checkpoint and args.resume:
        p.error('--init-checkpoint and --resume are mutually exclusive')
    seed_everything(args.seed)
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    vocab = load_vocab(args.vocab)
    pad_id = find_special(vocab, 'pad')
    train_ds = FoodCaptionDataset(args.data_root, 'train', vocab, args.max_len, args.train_limit)
    val_ds = FoodCaptionDataset(args.data_root, 'val', vocab, args.max_len, args.val_limit)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=make_collate(pad_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=make_collate(pad_id))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device, 'vocab_size:', len(vocab), 'train:', len(train_ds), 'val:', len(val_ds), flush=True)
    model_cfg = dict(vocab_size=len(vocab), pad_id=pad_id, d_model=args.d_model,
                     nhead=args.nhead, enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                     pretrained=(not args.no_pretrained and not (args.init_checkpoint or args.resume)))
    model = FoodLensV3(**model_cfg).to(device)
    if args.freeze_cnn:
        for param in model.cnn.parameters(): param.requires_grad = False
        # No BN statistics updates in an entirely frozen CNN.
        model._keep_frozen_cnn_eval = True
    if args.finetune_layer4:
        model.freeze_cnn_except_layer4()
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get('vocab') != vocab:
            raise ValueError('Initial checkpoint vocabulary differs from official V1 vocabulary')
        expected_cfg = {**model_cfg, 'pretrained': False}
        if checkpoint.get('model_config') != expected_cfg:
            raise ValueError(f'Initial checkpoint architecture mismatch: {checkpoint.get("model_config")} vs {expected_cfg}')
        model.load_state_dict(checkpoint['model_state'], strict=True)
        print(f'Initialized model weights from {args.init_checkpoint}; new optimizer, epoch starts at 1', flush=True)
    non_cnn = [p for name, p in model.named_parameters() if not name.startswith('cnn.') and p.requires_grad]
    cnn = [p for p in model.cnn.parameters() if p.requires_grad]
    groups = [{'params':non_cnn, 'lr':args.lr}]
    if cnn: groups.append({'params':cnn, 'lr':args.image_lr})
    optimizer = torch.optim.AdamW(groups)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    history = []
    best_val = float('inf')
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
        # Load only checkpoints you created or otherwise trust.
        state = torch.load(resume_path, map_location=device, weights_only=False)
        required = {'model_state', 'optimizer_state', 'epoch', 'history', 'best_val', 'model_config', 'vocab'}
        missing = required - set(state)
        if missing:
            raise ValueError(f'Checkpoint cannot resume training; missing: {sorted(missing)}. Use last_v3.pt, not best_v3.pt.')
        if state['vocab'] != vocab:
            raise ValueError('Resume vocabulary does not match current V1 vocabulary')
        if state['model_config'] != {**model_cfg, 'pretrained': False}:
            raise ValueError('Resume model architecture/config differs from current settings')
        prev_args = state.get('args', {})
        for key in ['data_root', 'train_limit', 'val_limit', 'batch_size', 'freeze_cnn', 'finetune_layer4', 'max_len']:
            if prev_args.get(key) != getattr(args, key):
                raise ValueError(f'Resume configuration mismatch for {key}: '
                                 f'old={prev_args.get(key)!r}, new={getattr(args,key)!r}')
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        history = state['history']
        best_val = float(state['best_val'])
        start_epoch = int(state['epoch']) + 1
        print(f'Resumed from epoch {start_epoch - 1}, next epoch {start_epoch}', flush=True)
    if start_epoch > args.epochs:
        print(f'Already trained {start_epoch-1} epochs; --epochs is the total, not additional count.')
        return
    for epoch in range(start_epoch, args.epochs+1):
        totals = []
        model.train()
        if getattr(model, '_keep_frozen_cnn_eval', False):
            model.cnn[:7].eval() if args.finetune_layer4 else model.cnn.eval()
        for images, tokens, _, _ in train_loader:
            images = images.to(device); tokens = tokens.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, tokens[:, :-1])
            loss = criterion(logits.reshape(-1, len(vocab)), tokens[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(loss.item()))
        model.eval(); val_losses = []
        with torch.no_grad():
            for images, tokens, _, _ in val_loader:
                images = images.to(device); tokens = tokens.to(device)
                logits = model(images, tokens[:, :-1])
                loss = criterion(logits.reshape(-1, len(vocab)), tokens[:, 1:].reshape(-1))
                val_losses.append(float(loss.item()))
        train_loss = sum(totals)/max(1,len(totals))
        val_loss = sum(val_losses)/max(1,len(val_losses))
        history.append({'epoch':epoch,'train_loss':train_loss,'val_loss':val_loss})
        print(f'epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}', flush=True)
        (run_dir/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state':model.state_dict(), 'model_config':{**model_cfg,'pretrained':False},
                        'vocab':vocab, 'epoch':epoch, 'val_loss':val_loss, 'args':vars(args),
                        'preprocess':'V1 Resize(256) + train RandomCrop/Flip, eval CenterCrop(224)',
                        'tokenizer':'V1 caption.lower().split()'},run_dir/'best_v3.pt')
        # Save after each completed epoch, including optimizer state for real continuation.
        state = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                 'model_config': {**model_cfg, 'pretrained': False}, 'vocab': vocab,
                 'epoch': epoch, 'history': history, 'best_val': best_val,
                 'args': {**vars(args), 'resume': None}}
        tmp_path = run_dir/'last_v3.tmp'
        torch.save(state, tmp_path)
        tmp_path.replace(run_dir/'last_v3.pt')
    print('Saved:',run_dir/'best_v3.pt')

if __name__=='__main__': main()
