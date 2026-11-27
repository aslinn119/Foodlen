# FoodLens V4 — ViT-B/32 + Transformer Decoder

V4 uses torchvision's ImageNet-1K pretrained ViT-B/32 as a frozen visual encoder. The 49 patch tokens plus CLS token are projected from 768 to 512 dimensions and consumed by a 3-layer autoregressive Transformer decoder.

## Data compatibility

- Reuses the project's V1 vocabulary and tokenizer.
- Special IDs: PAD=0, BOS=1, EOS=2, UNK=3.
- Lowercase + whitespace tokenization; max caption length defaults to 30.
- 224x224 images, ImageNet normalization; random crop/flip for training and center crop for evaluation.

## Environment

Use the same PyTorch/torchvision environment as V3. `torchvision.models.vit_b_32` and the `ViT_B_32_Weights.IMAGENET1K_V1` weights are required. The first run downloads the pretrained weights (roughly several hundred MB); subsequent runs use the local Torch cache.

## Smoke test

Run from this `v4` directory:

```powershell
python train.py --data-root "C:\path\to\Food500Cap_ready" --vocab "C:\path\to\best_v1.pt" --run-dir "outputs\smoke" --epochs 1 --train-limit 8 --val-limit 8 --batch-size 2 --cpu-threads 4
```

The smoke test still uses pretrained ViT-B/32. It checks data loading, forward/backward, validation, and checkpoint writing; it is not a meaningful quality experiment.

## Full training

```powershell
python train.py --data-root "C:\path\to\Food500Cap_ready" --vocab "C:\path\to\best_v1.pt" --run-dir "outputs\v4_baseline" --epochs 5 --batch-size 4 --cpu-threads 4
```

`--train-limit 0` and `--val-limit 0` mean use the complete split. The default is 0.

## Resume

Wait for an epoch to finish, then interrupt. Resume from the latest epoch checkpoint:

```powershell
python train.py --data-root "C:\path\to\Food500Cap_ready" --vocab "C:\path\to\best_v1.pt" --run-dir "outputs\v4_baseline" --epochs 5 --batch-size 4 --cpu-threads 4 --resume "outputs\v4_baseline\last_v4.pt"
```

`--epochs` is the target total epoch count, not the number of additional epochs.

## Inference

```powershell
python infer.py --checkpoint "outputs\v4_baseline\best_v4.pt" --data-root "C:\path\to\Food500Cap_ready" --split val --limit 0 --batch-size 4 --max-len 30 --output "outputs\v4_baseline\v4_val_predictions.jsonl"
```

## Important notes

- V4 does not load V3's checkpoint: the visual architecture is different. It starts from ImageNet-pretrained ViT weights and trains a new caption decoder/projection.
- The ViT backbone is frozen in this CPU-friendly baseline. Only the visual projection and text decoder are trained.
- Checkpoints save only trainable parameters plus optimizer state and vocabulary. Inference recreates the frozen backbone from torchvision's pretrained weights, so the pretrained weights must be available in the local Torch cache or downloadable on first use.
- Do not use `--train-limit` or `--val-limit` for the final run; defaults use the full split.
