"""
User-Executable Training Pipeline for Equal-Interval Spatiotemporal Cyclone Sequences.
Supports Digital Typhoon WP and multi-frame satellite video datasets.

Usage:
  # Dry-run test (verifies full pipeline with synthetic data):
  python src/training/train_sequence.py --dry_run

  # Full training on modern-era Digital Typhoon WP dataset (2000-2023):
  python src/training/train_sequence.py `
      --data_dir data/sequences `
      --min_year 2000 `
      --frame_step 3 `
      --stride 2 `
      --img_size 256 `
      --epochs 15 `
      --batch_size 8 `
      --grad_accum_steps 2 `
      --backbone_lr 1.5e-5 `
      --lr 3.0e-4 `
      --consistency_weight 0.20 `
      --device cuda
"""

import os
import sys
import gc
import math
import copy
from pathlib import Path
import argparse
import time
import json
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_prep.dataset_sequence import CycloneSequenceDataset, WIND_MEAN, WIND_STD
from src.models.spatiotemporal_classifier import DualStreamSpatiotemporalCycloneModel, SpatiotemporalCycloneModel
from src.models.losses import FocalOrdinalLoss, WindCategoryConsistencyLoss
from src.evaluation.evaluate_sequence import run_sequence_evaluation

from tqdm import tqdm


class ModelEMA:
    """Exponential Moving Average of model parameters to achieve flatter loss basins and lower val loss."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(model_p.data, alpha=1.0 - d)

    def to(self, device):
        self.ema_model = self.ema_model.to(device)
        return self


def parse_args():
    parser = argparse.ArgumentParser(description="Train Spatiotemporal Cyclone Sequence Model")
    parser.add_argument("--data_dir", type=str, default="data/sequences", help="Path to sequence dataset")
    parser.add_argument("--track_csv", type=str, default=None, help="Path to track metadata CSV")
    parser.add_argument("--min_year", type=int, default=2000, help="Filter out pre-min_year satellite scans (default 2000 for modern high-precision era)")
    parser.add_argument("--max_year", type=int, default=None, help="Optional maximum year filter")
    parser.add_argument("--seq_length", type=int, default=4, help="Sequence length (consecutive frames)")
    parser.add_argument("--frame_step", type=int, default=3, help="Step between sequence frames (default 3 for 3-hour delta, spanning 9-12h)")
    parser.add_argument("--stride", type=int, default=2, help="Stride between sequence windows (default 2 for dense full-dataset training)")
    parser.add_argument("--img_size", type=int, default=256, help="Input spatial resolution (default 256 for fine eyewall resolution)")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max sample limit for fast experimentation")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (default 2 for Windows)")
    parser.add_argument("--spatial_backbone", type=str, default="convnext_tiny", help="Spatial backbone name")
    parser.add_argument("--temporal_engine", type=str, default="gru", choices=["gru", "transformer"], help="Temporal engine: 'gru' (Delta-BiGRU recurrent memory) or 'transformer'")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Temporal hidden feature dimension")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default 8 = 32 frames per batch, safe for 8GB GPU)")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps (effective batch size = batch_size * grad_accum_steps)")
    parser.add_argument("--use_focal_loss", action="store_true", default=True, help="Use Class-Balanced Focal Ordinal Loss for category classification")
    parser.add_argument("--focal_gamma", type=float, default=1.0, help="Focal loss focusing parameter gamma (default 1.0)")
    parser.add_argument("--ordinal_weight", type=float, default=0.08, help="Ordinal distance penalty weight (default 0.08)")
    parser.add_argument("--gaussian_sigma", type=float, default=0.40, help="Gaussian ordinal label smoothing sigma (default 0.40 to eliminate boundary loss spikes)")
    parser.add_argument("--consistency_weight", type=float, default=0.10, help="Wind-Category consistency regularization weight (default 0.10)")
    parser.add_argument("--wind_weight", type=float, default=0.40, help="Normalized wind regression loss weight (default 0.40)")
    parser.add_argument("--trend_weight", type=float, default=0.05, help="Trend classification loss weight (default 0.05)")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs (default 20)")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Linear LR warmup epochs (default 2)")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience: stop if val accuracy does not improve for N epochs (default 4)")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Maintain Exponential Moving Average of weights (decay=0.999) for evaluation")
    parser.add_argument("--use_tta", action="store_true", default=True, help="Use Test-Time Augmentation (TTA) with horizontal reflections during validation")
    parser.add_argument("--backbone_lr", type=float, default=1.5e-5, help="Spatial backbone learning rate (10x lower to preserve pre-trained features)")
    parser.add_argument("--lr", type=float, default=3.0e-4, help="Temporal recurrent & heads learning rate")
    parser.add_argument("--weight_decay", type=float, default=2e-4, help="Weight decay for regularization")
    parser.add_argument("--dropout", type=float, default=0.30, help="Dropout rate to prevent overfitting")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="models/sequence_model", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--dry_run", action="store_true", help="Run 2 mini-epochs with mock data to verify setup")
    parser.add_argument("--eval_after_train", action="store_true", default=True, help="Run complete evaluation suite after training")
    return parser.parse_args()


def train_epoch(
    epoch, epochs, model, dataloader, optimizer, scaler,
    criterion_cat, criterion_wind, criterion_trend, criterion_consistency,
    device, grad_accum_steps=2, consistency_weight=0.10, wind_weight=0.40, trend_weight=0.05,
    model_ema: Optional[ModelEMA] = None
):
    model.train()
    total_loss = 0.0
    correct_cat = 0
    total_samples = 0
    mae_wind = 0.0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{epochs:02d}] [Train]", dynamic_ncols=True, leave=False)

    for batch_idx, batch in enumerate(pbar):
        seq = batch["sequence"].to(device, non_blocking=True)
        cat = batch["category"].to(device, non_blocking=True)
        wind_kt = batch["wind_speed"].to(device, non_blocking=True)
        norm_wind = batch.get("norm_wind", (wind_kt - WIND_MEAN) / WIND_STD).to(device, non_blocking=True)
        trend = batch["trend"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda" if "cuda" in device.type else "cpu"):
            logits, pred_norm_wind, pred_trend = model(seq)
            loss_c = criterion_cat(logits, cat)
            loss_w = criterion_wind(pred_norm_wind, norm_wind)
            loss_t = criterion_trend(pred_trend, trend)
            
            raw_loss = loss_c + wind_weight * loss_w + trend_weight * loss_t
            if criterion_consistency is not None and consistency_weight > 0.0:
                loss_cons = criterion_consistency(logits, pred_norm_wind)
                raw_loss = raw_loss + consistency_weight * loss_cons

            loss = raw_loss / grad_accum_steps

        if scaler is not None and "cuda" in device.type:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    model_ema.update(model)
        else:
            loss.backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    model_ema.update(model)

        b_size = seq.size(0)
        total_loss += raw_loss.item() * b_size
        preds = torch.argmax(logits, dim=1)
        correct_cat += (preds == cat).sum().item()

        # De-normalize wind predictions back to real knots for MAE
        pred_wind_kt = pred_norm_wind * WIND_STD + WIND_MEAN
        mae_wind += torch.abs(pred_wind_kt - wind_kt).sum().item()
        total_samples += b_size

        pbar.set_postfix({
            "loss": f"{raw_loss.item():.3f}",
            "acc": f"{(correct_cat / max(1, total_samples)) * 100:.1f}%",
            "mae": f"{(mae_wind / max(1, total_samples)):.1f}kt"
        })

    avg_loss = total_loss / max(1, total_samples)
    acc = correct_cat / max(1, total_samples)
    avg_mae = mae_wind / max(1, total_samples)
    return avg_loss, acc, avg_mae



def evaluate(
    epoch, epochs, model, dataloader,
    criterion_cat, criterion_wind, criterion_trend, criterion_consistency,
    device, consistency_weight=0.20, wind_weight=0.40, trend_weight=0.05,
    use_tta: bool = False
):
    model.eval()
    total_loss = 0.0
    correct_cat = 0
    total_samples = 0
    mae_wind = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{epochs:02d}] [Val  ]", dynamic_ncols=True, leave=False)
    with torch.no_grad():
        for batch in pbar:
            seq = batch["sequence"].to(device, non_blocking=True)
            cat = batch["category"].to(device, non_blocking=True)
            wind_kt = batch["wind_speed"].to(device, non_blocking=True)
            norm_wind = batch.get("norm_wind", (wind_kt - WIND_MEAN) / WIND_STD).to(device, non_blocking=True)
            trend = batch["trend"].to(device, non_blocking=True)

            if use_tta:
                seq_flip = torch.flip(seq, dims=[-1])
                logits1, pred_norm_wind1, pred_trend1 = model(seq)
                logits2, pred_norm_wind2, pred_trend2 = model(seq_flip)
                logits = 0.5 * (logits1 + logits2)
                pred_norm_wind = 0.5 * (pred_norm_wind1 + pred_norm_wind2)
                pred_trend = 0.5 * (pred_trend1 + pred_trend2)
            else:
                logits, pred_norm_wind, pred_trend = model(seq)

            loss_c = criterion_cat(logits, cat)
            loss_w = criterion_wind(pred_norm_wind, norm_wind)
            loss_t = criterion_trend(pred_trend, trend)
            loss = loss_c + wind_weight * loss_w + trend_weight * loss_t
            if criterion_consistency is not None and consistency_weight > 0.0:
                loss_cons = criterion_consistency(logits, pred_norm_wind)
                loss = loss + consistency_weight * loss_cons

            b_size = seq.size(0)
            total_loss += loss.item() * b_size
            preds = torch.argmax(logits, dim=1)
            correct_cat += (preds == cat).sum().item()

            pred_wind_kt = pred_norm_wind * WIND_STD + WIND_MEAN
            mae_wind += torch.abs(pred_wind_kt - wind_kt).sum().item()
            total_samples += b_size

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "acc": f"{(correct_cat / max(1, total_samples)) * 100:.1f}%",
                "mae": f"{(mae_wind / max(1, total_samples)):.1f}kt"
            })

    avg_loss = total_loss / max(1, total_samples)
    acc = correct_cat / max(1, total_samples)
    avg_mae = mae_wind / max(1, total_samples)
    return avg_loss, acc, avg_mae


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"================================================================")
    print(f" Spatiotemporal Cyclone Sequence Training Engine (High-Precision)")
    print(f" Device: {device} | Sequence: {args.seq_length} frames x {args.frame_step}h step ({args.seq_length * args.frame_step}h total span)")
    print(f" Resolution: {args.img_size}x{args.img_size} | Stride: {args.stride}")
    print(f" Backbone: {args.spatial_backbone} | Temporal: {args.temporal_engine} (Cross-Attention Fusion)")
    if args.min_year:
        print(f" Dataset Era: {args.min_year} - {args.max_year if args.max_year else 'Present'} (Modern High-Precision Scans)")
    print(f" Differential LR: Backbone={args.backbone_lr:.1e} | Temporal/Heads={args.lr:.1e}")
    print(f"================================================================")

    # 1. Dataset Instantiation
    if args.dry_run:
        print("[DRY-RUN] Initializing synthetic 4-frame cyclone sequences...")
        train_ds = CycloneSequenceDataset(seq_length=args.seq_length, frame_step=args.frame_step, img_size=args.img_size, mock_num_samples=32, is_train=True)
        val_ds = CycloneSequenceDataset(seq_length=args.seq_length, frame_step=args.frame_step, img_size=args.img_size, mock_num_samples=16, is_train=False)
        epochs = 2
    else:
        full_ds = CycloneSequenceDataset(
            data_dir=args.data_dir,
            track_csv=args.track_csv,
            seq_length=args.seq_length,
            frame_step=args.frame_step,
            stride=args.stride,
            img_size=args.img_size,
            min_year=args.min_year,
            max_year=args.max_year,
            is_train=True
        )
        if len(full_ds) == 0:
            print(f"\n[ERROR] No valid cyclone sequence frames found in '{args.data_dir}' (min_year={args.min_year})!")
            print(f"  -> Please place your sequence dataset folders in '{args.data_dir}' or specify --data_dir <path>.")
            print("  -> To verify the training pipeline with synthetic sequences on CUDA, run with the '--dry_run' flag:")
            print("     python src/training/train_sequence.py --dry_run --device cuda\n")
            sys.exit(1)
        elif len(full_ds) < 2:
            print(f"\n[ERROR] Dataset in '{args.data_dir}' only contains {len(full_ds)} sequence. At least 2 sequences are required for train/val splitting.\n")
            sys.exit(1)

        if args.max_samples and len(full_ds) > args.max_samples:
            import random
            random.seed(42)
            sampled_indices = random.sample(range(len(full_ds)), args.max_samples)
            from torch.utils.data import Subset
            full_ds = Subset(full_ds, sampled_indices)
            print(f"[Sampling] Randomly sampled {args.max_samples} sequences.")

        val_size = max(1, int(len(full_ds) * 0.15))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])
        epochs = args.epochs

    # DataLoader setup: pin_memory=False avoids CachingHostAllocator leaks on Windows
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    print(f"Dataset Loaded: {len(train_ds)} train sequences | {len(val_ds)} val sequences")

    # 2. Model Instantiation
    model = SpatiotemporalCycloneModel(
        spatial_backbone=args.spatial_backbone,
        pretrained=not args.dry_run,
        num_classes=5,
        temporal_engine=args.temporal_engine,
        hidden_dim=args.hidden_dim,
        seq_length=args.seq_length
    ).to(device)

    # 3. Loss & Optimizer Setup
    if args.use_focal_loss:
        alpha_weights = torch.tensor([1.0, 1.05, 1.30, 1.15, 1.50], dtype=torch.float32).to(device)
        criterion_cat = FocalOrdinalLoss(
            num_classes=5,
            gamma=args.focal_gamma,
            alpha=alpha_weights,
            ordinal_weight=args.ordinal_weight,
            label_smoothing=0.01,
            gaussian_smoothing_sigma=args.gaussian_sigma
        )
        print(f"[Loss Setup] Using Class-Balanced Focal Ordinal Loss (gamma={args.focal_gamma}, ordinal_weight={args.ordinal_weight}, gaussian_sigma={args.gaussian_sigma})")
    else:
        criterion_cat = nn.CrossEntropyLoss(label_smoothing=0.01)

    criterion_cat = criterion_cat.to(device)
    criterion_wind = nn.SmoothL1Loss().to(device)
    criterion_trend = nn.CrossEntropyLoss().to(device)
    
    if args.consistency_weight > 0.0:
        criterion_consistency = WindCategoryConsistencyLoss().to(device)
        print(f"[Loss Setup] Using Wind-Category Consistency Regularization (weight={args.consistency_weight})")
    else:
        criterion_consistency = None

    # Parameter groups for Differential Learning Rates:
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "eye_expert" in name or "synoptic_expert" in name or "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay}
    ])

    # 2-epoch linear warmup followed by Cosine Annealing
    warmup_epochs = args.warmup_epochs
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if "cuda" in device.type else None

    # Exponential Moving Average (EMA) of weights
    model_ema = ModelEMA(model, decay=0.999).to(device) if args.use_ema else None
    if model_ema is not None:
        print("[Model EMA] Exponential Moving Average (decay=0.999) active for flat-basin generalization.")
    if args.use_tta:
        print("[TTA] Test-Time Augmentation (Horizontal Reflections) active during validation.")

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_acc = 0.0

    # 4. Checkpoint Resumption
    if args.resume and os.path.exists(args.resume):
        print(f"[Resume] Loading checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
            best_val_acc = float(ckpt.get("val_acc", 0.0))
            print(f"[Resume] Successfully resumed! Continuing from Epoch {start_epoch} (Previous Best Val Acc: {best_val_acc*100:.2f}%)")
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass

    history = []
    patience_counter = 0

    print("\nStarting Training Execution...")
    for epoch in range(start_epoch, epochs + 1):
        gc.collect()
        if "cuda" in device.type:
            torch.cuda.empty_cache()

        t0 = time.time()
        tr_loss, tr_acc, tr_mae = train_epoch(
            epoch, epochs, model, train_loader, optimizer, scaler,
            criterion_cat, criterion_wind, criterion_trend, criterion_consistency, device,
            grad_accum_steps=args.grad_accum_steps,
            consistency_weight=args.consistency_weight,
            wind_weight=args.wind_weight,
            trend_weight=args.trend_weight,
            model_ema=model_ema
        )
        eval_model = model_ema.ema_model if model_ema is not None else model
        val_loss, val_acc, val_mae = evaluate(
            epoch, epochs, eval_model, val_loader,
            criterion_cat, criterion_wind, criterion_trend, criterion_consistency, device,
            consistency_weight=args.consistency_weight,
            wind_weight=args.wind_weight,
            trend_weight=args.trend_weight,
            use_tta=args.use_tta
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc*100:.1f}%, MAE: {tr_mae:.1f}kt | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.1f}%, MAE: {val_mae:.1f}kt"
        )

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_mae": tr_mae,
            "val_loss": val_loss, "val_acc": val_acc, "val_mae": val_mae
        })

        if val_acc > best_val_acc and not args.dry_run:
            best_val_acc = val_acc
            patience_counter = 0
            save_state_dict = model_ema.ema_model.state_dict() if model_ema is not None else model.state_dict()
            ckpt = {
                "epoch": epoch,
                "model_state_dict": save_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_mae": val_mae,
                "seq_length": args.seq_length,
                "frame_step": args.frame_step,
                "img_size": args.img_size,
                "spatial_backbone": args.spatial_backbone,
                "temporal_engine": args.temporal_engine,
                "hidden_dim": args.hidden_dim,
                "use_ema": args.use_ema,
                "use_tta": args.use_tta
            }
            torch.save(ckpt, save_path / "best_sequence_model.pt")
            print(f"  --> Saved new best checkpoint (Val Acc: {val_acc*100:.2f}%)")
        elif not args.dry_run:
            patience_counter += 1
            print(f"  [Early Stopping Tracker] No val improvement for {patience_counter}/{args.patience} epochs (Best Val Acc: {best_val_acc*100:.2f}%)")
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping Triggered] Validation accuracy stopped improving. Stopping at Epoch {epoch} to prevent overfitting.")
                print(f"  --> Reverting to best checkpoint from Epoch {epoch - patience_counter} ({best_val_acc*100:.2f}% Val Acc).")
                break

        gc.collect()
        if "cuda" in device.type:
            torch.cuda.empty_cache()

    print("\nTraining Completed Successfully!")

    if args.dry_run:
        print("[DRY-RUN COMPLETE] Pipeline verified end-to-end! Ready for full training.")
        os._exit(0)
    elif args.eval_after_train:
        print("\n" + "=" * 64)
        print(" Running Post-Training Evaluation Suite on Validation Set...")
        print("=" * 64)
        best_ckpt_file = save_path / "best_sequence_model.pt"
        if best_ckpt_file.exists():
            ckpt = torch.load(str(best_ckpt_file), map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            print(f"Loaded best checkpoint from {best_ckpt_file} for evaluation.")
        run_sequence_evaluation(model, val_loader, device, save_path, use_tta=args.use_tta)

    os._exit(0)


if __name__ == "__main__":
    main()


