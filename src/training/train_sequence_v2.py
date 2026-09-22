"""
Version 2 Multi-Task Spatiotemporal Cyclone Training & Forecasting Pipeline.
Trains a unified spatiotemporal network to jointly predict:
  1. Intensity: 5-tier WMO Category + Sustained Wind (knots)
  2. Sensory: Minimum Central Surface Pressure (hPa)
  3. Evolvement: Track & Intensity Delta Forecast at +6h and +12h
  4. Danger Areas: 30-kt Gale & 50-kt Storm Wind Radii (km)
  5. Landfall: Coast Crossing Event & Landfall ETA (hours)

Usage:
  # Verify pipeline with synthetic data on CUDA:
  python src/training/train_sequence_v2.py --dry_run --device cuda

  # Full training inheriting from Version 1 pre-trained weights:
  python src/training/train_sequence_v2.py `
      --data_dir data/sequences `
      --init_from_v1 models/sequence_model/best_sequence_model.pt `
      --epochs 15 `
      --batch_size 8 `
      --grad_accum_steps 2 `
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

from src.data_prep.dataset_sequence_v2 import (
    CycloneSequenceDatasetV2,
    WIND_MEAN, WIND_STD,
    PRES_MEAN, PRES_STD,
    DANGER_R30_MAX, DANGER_R50_MAX
)
from src.models.spatiotemporal_forecaster import MultiTaskSpatiotemporalCycloneModel
from src.models.losses import FocalOrdinalLoss, WindCategoryConsistencyLoss
from src.evaluation.evaluate_sequence_v2 import run_v2_sequence_evaluation

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train Version 2 Multi-Task Spatiotemporal Cyclone Forecaster")
    parser.add_argument("--data_dir", type=str, default="data/sequences", help="Path to sequence dataset")
    parser.add_argument("--track_csv", type=str, default=None, help="Path to track metadata CSV")
    parser.add_argument("--min_year", type=int, default=2000, help="Filter out pre-min_year satellite scans (default 2000)")
    parser.add_argument("--max_year", type=int, default=None, help="Optional maximum year filter")
    parser.add_argument("--seq_length", type=int, default=4, help="Sequence length (consecutive frames)")
    parser.add_argument("--frame_step", type=int, default=3, help="Step between sequence frames (default 3)")
    parser.add_argument("--stride", type=int, default=2, help="Stride between sequence windows (default 2)")
    parser.add_argument("--img_size", type=int, default=256, help="Input spatial resolution (default 256)")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max sample limit for testing")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--spatial_backbone", type=str, default="convnext_tiny", help="Spatial backbone name")
    parser.add_argument("--temporal_engine", type=str, default="gru", choices=["gru", "transformer"])
    parser.add_argument("--hidden_dim", type=int, default=256, help="Temporal hidden feature dimension")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--use_focal_loss", action="store_true", default=True, help="Use Focal Ordinal Loss")
    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--ordinal_weight", type=float, default=0.08)

    # Multi-Task Loss Balancing Weights
    parser.add_argument("--wind_weight", type=float, default=0.20, help="Wind regression loss weight")
    parser.add_argument("--trend_weight", type=float, default=0.02, help="Trend loss weight")
    parser.add_argument("--pressure_weight", type=float, default=0.15, help="Sensory Central Pressure loss weight")
    parser.add_argument("--evolve_weight", type=float, default=0.20, help="Future evolvement (+6h/+12h) loss weight")
    parser.add_argument("--danger_weight", type=float, default=0.15, help="Danger area wind radii loss weight")
    parser.add_argument("--landfall_weight", type=float, default=0.10, help="Landfall probability & ETA loss weight")
    parser.add_argument("--consistency_weight", type=float, default=0.10, help="Wind-Category consistency weight")

    parser.add_argument("--epochs", type=int, default=15, help="Total training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Linear LR warmup epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--use_tta", action="store_true", default=True, help="Use TTA during validation")
    parser.add_argument("--backbone_lr", type=float, default=2.5e-5, help="Spatial backbone learning rate")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Temporal recurrent & multi-task heads learning rate")
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="models/sequence_model_v2", help="Directory to save V2 checkpoints")
    parser.add_argument("--init_from_v1", type=str, default=None, help="Optional Version 1 checkpoint to inherit pre-trained weights from")
    parser.add_argument("--resume", type=str, default=None, help="Path to V2 checkpoint to resume from")
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    parser.add_argument("--dry_run", action="store_true", help="Run 2 mini-epochs with mock multi-task data to verify pipeline")
    parser.add_argument("--eval_after_train", action="store_true", default=True)
    return parser.parse_args()


def train_epoch_v2(
    epoch, epochs, model, dataloader, optimizer, scaler,
    criterion_cat, criterion_wind, criterion_trend, criterion_pressure,
    criterion_evolve, criterion_danger, criterion_landfall,
    criterion_consistency, device, grad_accum_steps=2,
    wind_weight=0.20, trend_weight=0.02, pressure_weight=0.15,
    evolve_weight=0.20, danger_weight=0.15, landfall_weight=0.10,
    consistency_weight=0.10
):
    model.train()
    total_loss = 0.0
    correct_cat = 0
    total_samples = 0
    mae_wind = 0.0
    mae_pres = 0.0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{epochs:02d}] [V2 Train]", dynamic_ncols=True, leave=False)

    for batch_idx, batch in enumerate(pbar):
        seq = batch["sequence"].to(device, non_blocking=True)
        cat = batch["category"].to(device, non_blocking=True)
        wind_kt = batch["wind_speed"].to(device, non_blocking=True)
        norm_wind = batch["norm_wind"].to(device, non_blocking=True)
        trend = batch["trend"].to(device, non_blocking=True)
        norm_pres = batch["norm_pressure"].to(device, non_blocking=True)
        pres_hpa = batch["pressure"].to(device, non_blocking=True)
        evolve_target = batch["evolve_target"].to(device, non_blocking=True)
        danger_radii = batch["danger_radii"].to(device, non_blocking=True)
        landfall = batch["landfall"].to(device, non_blocking=True).unsqueeze(-1)
        landfall_eta = batch["landfall_eta"].to(device, non_blocking=True).unsqueeze(-1)

        with torch.amp.autocast(device_type="cuda" if "cuda" in device.type else "cpu"):
            out = model(seq)

            loss_c = criterion_cat(out["logits"], cat)
            loss_w = criterion_wind(out["pred_norm_wind"].squeeze(-1), norm_wind)
            loss_t = criterion_trend(out["pred_trend"], trend)
            loss_p = criterion_pressure(out["pred_norm_pressure"].squeeze(-1), norm_pres)
            loss_e = criterion_evolve(out["pred_evolve"], evolve_target)
            loss_d = criterion_danger(out["pred_danger_radii"], danger_radii)
            loss_l = criterion_landfall(out["pred_landfall"], landfall) + 0.5 * criterion_danger(out["pred_landfall_eta"], landfall_eta)

            raw_loss = (
                loss_c +
                wind_weight * loss_w +
                trend_weight * loss_t +
                pressure_weight * loss_p +
                evolve_weight * loss_e +
                danger_weight * loss_d +
                landfall_weight * loss_l
            )

            if criterion_consistency is not None and consistency_weight > 0.0:
                loss_cons = criterion_consistency(out["logits"], out["pred_norm_wind"])
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
        else:
            loss.backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        b_size = seq.size(0)
        total_loss += raw_loss.item() * b_size
        preds = torch.argmax(out["logits"], dim=1)
        correct_cat += (preds == cat).sum().item()

        # De-normalize for live display
        pred_wind_kt = out["pred_norm_wind"].squeeze(-1) * WIND_STD + WIND_MEAN
        mae_wind += torch.abs(pred_wind_kt - wind_kt).sum().item()

        pred_pres_hpa = out["pred_norm_pressure"].squeeze(-1) * PRES_STD + PRES_MEAN
        mae_pres += torch.abs(pred_pres_hpa - pres_hpa).sum().item()

        total_samples += b_size

        pbar.set_postfix({
            "loss": f"{raw_loss.item():.3f}",
            "acc": f"{(correct_cat / max(1, total_samples)) * 100:.1f}%",
            "w_mae": f"{(mae_wind / max(1, total_samples)):.1f}kt",
            "p_mae": f"{(mae_pres / max(1, total_samples)):.1f}hPa"
        })

    avg_loss = total_loss / max(1, total_samples)
    acc = correct_cat / max(1, total_samples)
    avg_w_mae = mae_wind / max(1, total_samples)
    avg_p_mae = mae_pres / max(1, total_samples)
    return avg_loss, acc, avg_w_mae, avg_p_mae


def evaluate_v2(
    epoch, epochs, model, dataloader,
    criterion_cat, criterion_wind, criterion_trend, criterion_pressure,
    criterion_evolve, criterion_danger, criterion_landfall,
    criterion_consistency, device,
    wind_weight=0.20, trend_weight=0.02, pressure_weight=0.15,
    evolve_weight=0.20, danger_weight=0.15, landfall_weight=0.10,
    consistency_weight=0.10, use_tta=True
):
    model.eval()
    total_loss = 0.0
    correct_cat = 0
    total_samples = 0
    mae_wind = 0.0
    mae_pres = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{epochs:02d}] [V2 Val ]", dynamic_ncols=True, leave=False)
    with torch.no_grad():
        for batch in pbar:
            seq = batch["sequence"].to(device, non_blocking=True)
            cat = batch["category"].to(device, non_blocking=True)
            wind_kt = batch["wind_speed"].to(device, non_blocking=True)
            norm_wind = batch["norm_wind"].to(device, non_blocking=True)
            trend = batch["trend"].to(device, non_blocking=True)
            norm_pres = batch["norm_pressure"].to(device, non_blocking=True)
            pres_hpa = batch["pressure"].to(device, non_blocking=True)
            evolve_target = batch["evolve_target"].to(device, non_blocking=True)
            danger_radii = batch["danger_radii"].to(device, non_blocking=True)
            landfall = batch["landfall"].to(device, non_blocking=True).unsqueeze(-1)
            landfall_eta = batch["landfall_eta"].to(device, non_blocking=True).unsqueeze(-1)

            if use_tta:
                seq_flip = torch.flip(seq, dims=[-1])
                o1 = model(seq)
                o2 = model(seq_flip)
                out = {}
                for k in o1:
                    out[k] = 0.5 * (o1[k] + o2[k])
            else:
                out = model(seq)

            loss_c = criterion_cat(out["logits"], cat)
            loss_w = criterion_wind(out["pred_norm_wind"].squeeze(-1), norm_wind)
            loss_t = criterion_trend(out["pred_trend"], trend)
            loss_p = criterion_pressure(out["pred_norm_pressure"].squeeze(-1), norm_pres)
            loss_e = criterion_evolve(out["pred_evolve"], evolve_target)
            loss_d = criterion_danger(out["pred_danger_radii"], danger_radii)
            loss_l = criterion_landfall(out["pred_landfall"], landfall) + 0.5 * criterion_danger(out["pred_landfall_eta"], landfall_eta)

            loss = (
                loss_c +
                wind_weight * loss_w +
                trend_weight * loss_t +
                pressure_weight * loss_p +
                evolve_weight * loss_e +
                danger_weight * loss_d +
                landfall_weight * loss_l
            )

            b_size = seq.size(0)
            total_loss += loss.item() * b_size
            preds = torch.argmax(out["logits"], dim=1)
            correct_cat += (preds == cat).sum().item()

            pred_wind_kt = out["pred_norm_wind"].squeeze(-1) * WIND_STD + WIND_MEAN
            mae_wind += torch.abs(pred_wind_kt - wind_kt).sum().item()

            pred_pres_hpa = out["pred_norm_pressure"].squeeze(-1) * PRES_STD + PRES_MEAN
            mae_pres += torch.abs(pred_pres_hpa - pres_hpa).sum().item()

            total_samples += b_size

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "acc": f"{(correct_cat / max(1, total_samples)) * 100:.1f}%",
                "w_mae": f"{(mae_wind / max(1, total_samples)):.1f}kt",
                "p_mae": f"{(mae_pres / max(1, total_samples)):.1f}hPa"
            })

    avg_loss = total_loss / max(1, total_samples)
    acc = correct_cat / max(1, total_samples)
    avg_w_mae = mae_wind / max(1, total_samples)
    avg_p_mae = mae_pres / max(1, total_samples)
    return avg_loss, acc, avg_w_mae, avg_p_mae


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"================================================================")
    print(f" Version 2 Spatiotemporal Multi-Task Cyclone Forecaster Engine")
    print(f" Device: {device} | Sequence: {args.seq_length} frames x {args.frame_step}h step")
    print(f" Tasks: Intensity + Central Pressure + +6h/+12h Evolve + Danger Radii + Landfall ETA")
    print(f" Spatial Backbone: {args.spatial_backbone} | Temporal: {args.temporal_engine}")
    print(f"================================================================")

    # 1. Dataset Instantiation
    if args.dry_run:
        print("[DRY-RUN] Initializing synthetic multi-task 4-frame sequences...")
        train_ds = CycloneSequenceDatasetV2(seq_length=args.seq_length, frame_step=args.frame_step, img_size=args.img_size, mock_num_samples=32, is_train=True)
        val_ds = CycloneSequenceDatasetV2(seq_length=args.seq_length, frame_step=args.frame_step, img_size=args.img_size, mock_num_samples=16, is_train=False)
        args.num_workers = 0
        epochs = 2
    else:
        full_ds = CycloneSequenceDatasetV2(
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
        if len(full_ds) < 2:
            print(f"\n[ERROR] Dataset in '{args.data_dir}' contains {len(full_ds)} sequence. Need at least 2 sequences.\n")
            sys.exit(1)

        val_size = max(1, int(len(full_ds) * 0.15))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(
            full_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        print(f"[Dataset Split] Seed={args.seed} | Train: {len(train_ds)} sequences | Val: {len(val_ds)} sequences")
        epochs = args.epochs

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    # 2. Model Instantiation
    model = MultiTaskSpatiotemporalCycloneModel(
        spatial_backbone=args.spatial_backbone,
        pretrained=not args.dry_run,
        num_classes=5,
        temporal_engine=args.temporal_engine,
        hidden_dim=args.hidden_dim,
        seq_length=args.seq_length,
        dropout=args.dropout
    ).to(device)

    # Inherit pre-trained weights from Version 1 if provided
    if args.init_from_v1 and os.path.exists(args.init_from_v1) and not args.dry_run:
        model.load_from_v1_checkpoint(args.init_from_v1, device)

    # 3. Loss & Optimizer Setup
    if args.use_focal_loss:
        alpha_weights = torch.tensor([1.0, 1.05, 1.30, 1.15, 1.50], dtype=torch.float32).to(device)
        criterion_cat = FocalOrdinalLoss(
            num_classes=5,
            gamma=args.focal_gamma,
            alpha=alpha_weights,
            ordinal_weight=args.ordinal_weight,
            label_smoothing=0.01,
            gaussian_smoothing_sigma=0.0
        ).to(device)
    else:
        criterion_cat = nn.CrossEntropyLoss(label_smoothing=0.01).to(device)

    criterion_wind = nn.SmoothL1Loss().to(device)
    criterion_trend = nn.CrossEntropyLoss().to(device)
    criterion_pressure = nn.SmoothL1Loss().to(device)
    criterion_evolve = nn.SmoothL1Loss().to(device)
    criterion_danger = nn.SmoothL1Loss().to(device)
    criterion_landfall = nn.BCEWithLogitsLoss().to(device)
    criterion_consistency = WindCategoryConsistencyLoss().to(device) if args.consistency_weight > 0.0 else None

    # Parameter groups for Differential Learning Rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "eye_expert" in name or "synoptic_expert" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay}
    ])

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_acc = 0.0

    # Checkpoint Resumption
    if args.resume and os.path.exists(args.resume):
        print(f"[Resume] Loading V2 checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        if "val_acc" in ckpt:
            best_val_acc = float(ckpt.get("val_acc", 0.0))
        if args.finetune:
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            epochs = start_epoch + args.epochs - 1
            print(f"[Fine-Tune V2] Resuming for {args.epochs} epochs: Epoch {start_epoch} -> {epochs}")
        elif "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
            if epochs < start_epoch:
                epochs = start_epoch
            print(f"[Resume V2] Successfully resumed! Continuing from Epoch {start_epoch} to {epochs}")
        if "optimizer_state_dict" in ckpt and not args.finetune:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass

    # LR Scheduler (Synced with start_epoch)
    if args.finetune:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        warmup_epochs = args.warmup_epochs
        def lr_lambda(step_idx):
            curr_ep = (start_epoch - 1) + step_idx
            if curr_ep < warmup_epochs:
                return float(curr_ep + 1) / float(max(1, warmup_epochs))
            progress = float(curr_ep - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = torch.amp.GradScaler("cuda") if "cuda" in device.type else None

    history_file = save_path / "training_history_v2.json"
    history = []
    if history_file.exists() and args.resume:
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
            history = [h for h in history if h.get("epoch", 0) < start_epoch]
        except Exception:
            history = []

    patience_counter = 0

    print("\nStarting Version 2 Multi-Task Training Execution...")
    for epoch in range(start_epoch, epochs + 1):
        gc.collect()
        if "cuda" in device.type:
            torch.cuda.empty_cache()

        t0 = time.time()
        tr_loss, tr_acc, tr_w_mae, tr_p_mae = train_epoch_v2(
            epoch, epochs, model, train_loader, optimizer, scaler,
            criterion_cat, criterion_wind, criterion_trend, criterion_pressure,
            criterion_evolve, criterion_danger, criterion_landfall,
            criterion_consistency, device,
            grad_accum_steps=args.grad_accum_steps,
            wind_weight=args.wind_weight,
            trend_weight=args.trend_weight,
            pressure_weight=args.pressure_weight,
            evolve_weight=args.evolve_weight,
            danger_weight=args.danger_weight,
            landfall_weight=args.landfall_weight,
            consistency_weight=args.consistency_weight
        )
        val_loss, val_acc, val_w_mae, val_p_mae = evaluate_v2(
            epoch, epochs, model, val_loader,
            criterion_cat, criterion_wind, criterion_trend, criterion_pressure,
            criterion_evolve, criterion_danger, criterion_landfall,
            criterion_consistency, device,
            wind_weight=args.wind_weight,
            trend_weight=args.trend_weight,
            pressure_weight=args.pressure_weight,
            evolve_weight=args.evolve_weight,
            danger_weight=args.danger_weight,
            landfall_weight=args.landfall_weight,
            consistency_weight=args.consistency_weight,
            use_tta=args.use_tta
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc*100:.1f}%, W_MAE: {tr_w_mae:.1f}kt, P_MAE: {tr_p_mae:.1f}hPa | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.1f}%, W_MAE: {val_w_mae:.1f}kt, P_MAE: {val_p_mae:.1f}hPa"
        )

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_w_mae": tr_w_mae, "train_p_mae": tr_p_mae,
            "val_loss": val_loss, "val_acc": val_acc, "val_w_mae": val_w_mae, "val_p_mae": val_p_mae
        })
        try:
            with open(save_path / "training_history_v2.json", "w") as f:
                json.dump(history, f, indent=2)
        except Exception:
            pass

        if val_acc > best_val_acc and not args.dry_run:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_w_mae": val_w_mae,
                "val_p_mae": val_p_mae,
                "seq_length": args.seq_length,
                "frame_step": args.frame_step,
                "img_size": args.img_size,
                "spatial_backbone": args.spatial_backbone,
                "temporal_engine": args.temporal_engine,
                "hidden_dim": args.hidden_dim,
                "use_tta": args.use_tta
            }
            torch.save(ckpt, save_path / "best_sequence_model_v2.pt")
            print(f"  --> Saved new best V2 checkpoint (Val Acc: {val_acc*100:.2f}%)")
        elif not args.dry_run:
            patience_counter += 1
            print(f"  [Early Stopping Tracker] No val improvement for {patience_counter}/{args.patience} epochs (Best Val Acc: {best_val_acc*100:.2f}%)")
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping Triggered] Stopping at Epoch {epoch} to prevent overfitting.")
                break

        gc.collect()
        if "cuda" in device.type:
            torch.cuda.empty_cache()

    print("\nVersion 2 Training Completed Successfully!")

    if args.dry_run:
        print("[DRY-RUN COMPLETE] Version 2 pipeline verified end-to-end! Ready for training.")
        del train_loader, val_loader, model, optimizer
        gc.collect()
        if "cuda" in device.type:
            torch.cuda.empty_cache()
        return
    elif args.eval_after_train:
        print("\n" + "=" * 64)
        print(" Running Post-Training Version 2 Evaluation Suite...")
        print("=" * 64)
        best_ckpt_file = save_path / "best_sequence_model_v2.pt"
        if best_ckpt_file.exists():
            ckpt = torch.load(str(best_ckpt_file), map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        run_v2_sequence_evaluation(model, val_loader, device, save_path, use_tta=args.use_tta)

    del train_loader, val_loader, model, optimizer
    gc.collect()
    if "cuda" in device.type:
        torch.cuda.empty_cache()
    return


if __name__ == "__main__":
    main()
