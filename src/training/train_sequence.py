"""
User-Executable Training Pipeline for Equal-Interval Spatiotemporal Cyclone Sequences.
Supports Digital Typhoon WP and multi-frame satellite video datasets.

Usage:
  # Dry-run test (verifies full pipeline with synthetic data):
  python src/training/train_sequence.py --dry_run

  # Full training on downloaded Digital Typhoon WP dataset:
  python src/training/train_sequence.py \
      --data_dir data/digital_typhoon_wp \
      --track_csv data/digital_typhoon_wp/metadata.csv \
      --epochs 20 \
      --batch_size 16 \
      --device cuda
"""

import os
import sys
from pathlib import Path
import argparse
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_prep.dataset_sequence import CycloneSequenceDataset
from src.models.spatiotemporal_classifier import DualStreamSpatiotemporalCycloneModel, SpatiotemporalCycloneModel
from src.evaluation.evaluate_sequence import run_sequence_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Train Spatiotemporal Cyclone Sequence Model")
    parser.add_argument("--data_dir", type=str, default="data/sequences", help="Path to sequence dataset")
    parser.add_argument("--track_csv", type=str, default=None, help="Path to track metadata CSV")
    parser.add_argument("--seq_length", type=int, default=4, help="Sequence length (consecutive frames)")
    parser.add_argument("--spatial_backbone", type=str, default="convnext_tiny", help="Spatial backbone name")
    parser.add_argument("--temporal_engine", type=str, default="transformer", choices=["transformer", "gru"])
    parser.add_argument("--hidden_dim", type=int, default=256, help="Temporal hidden feature dimension")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="models/sequence_model", help="Directory to save checkpoints")
    parser.add_argument("--dry_run", action="store_true", help="Run 2 mini-epochs with mock data to verify setup")
    parser.add_argument("--eval_after_train", action="store_true", default=True, help="Run complete evaluation suite after training")
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, scaler, criterion_cat, criterion_wind, criterion_trend, device):
    model.train()
    total_loss = 0.0
    correct_cat = 0
    total_samples = 0
    mae_wind = 0.0

    for batch in dataloader:
        seq = batch["sequence"].to(device)
        cat = batch["category"].to(device)
        wind = batch["wind_speed"].to(device)
        trend = batch["trend"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda" if "cuda" in device.type else "cpu"):
            logits, pred_wind, pred_trend = model(seq)
            loss_c = criterion_cat(logits, cat)
            loss_w = criterion_wind(pred_wind, wind)
            loss_t = criterion_trend(pred_trend, trend)
            loss = loss_c + 0.05 * loss_w + 0.30 * loss_t

        if scaler is not None and "cuda" in device.type:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        total_loss += loss.item() * seq.size(0)
        preds = torch.argmax(logits, dim=1)
        correct_cat += (preds == cat).sum().item()
        mae_wind += torch.abs(pred_wind - wind).sum().item()
        total_samples += seq.size(0)

    avg_loss = total_loss / max(1, total_samples)
    acc = correct_cat / max(1, total_samples)
    avg_mae = mae_wind / max(1, total_samples)
    return avg_loss, acc, avg_mae


def evaluate(model, dataloader, criterion_cat, criterion_wind, criterion_trend, device):
    model.eval()
    total_loss = 0.0
    correct_cat = 0
    total_samples = 0
    mae_wind = 0.0

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["sequence"].to(device)
            cat = batch["category"].to(device)
            wind = batch["wind_speed"].to(device)
            trend = batch["trend"].to(device)

            logits, pred_wind, pred_trend = model(seq)
            loss_c = criterion_cat(logits, cat)
            loss_w = criterion_wind(pred_wind, wind)
            loss_t = criterion_trend(pred_trend, trend)
            loss = loss_c + 0.05 * loss_w + 0.30 * loss_t

            total_loss += loss.item() * seq.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_cat += (preds == cat).sum().item()
            mae_wind += torch.abs(pred_wind - wind).sum().item()
            total_samples += seq.size(0)

    avg_loss = total_loss / max(1, total_samples)
    acc = correct_cat / max(1, total_samples)
    avg_mae = mae_wind / max(1, total_samples)
    return avg_loss, acc, avg_mae


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"================================================================")
    print(f" Spatiotemporal Cyclone Sequence Training Engine")
    print(f" Device: {device} | Sequence Length: {args.seq_length} frames")
    print(f" Backbone: {args.spatial_backbone} | Temporal: {args.temporal_engine}")
    print(f"================================================================")

    # 1. Dataset Instantiation
    if args.dry_run:
        print("[DRY-RUN] Initializing synthetic 4-frame cyclone sequences...")
        train_ds = CycloneSequenceDataset(seq_length=args.seq_length, mock_num_samples=32, is_train=True)
        val_ds = CycloneSequenceDataset(seq_length=args.seq_length, mock_num_samples=16, is_train=False)
        epochs = 2
    else:
        full_ds = CycloneSequenceDataset(
            data_dir=args.data_dir,
            track_csv=args.track_csv,
            seq_length=args.seq_length,
            is_train=True
        )
        val_size = max(1, int(len(full_ds) * 0.15))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])
        epochs = args.epochs

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

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
    criterion_cat = nn.CrossEntropyLoss(label_smoothing=0.01)
    criterion_wind = nn.SmoothL1Loss()
    criterion_trend = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if "cuda" in device.type else None

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = []

    print("\nStarting Training Execution...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_mae = train_epoch(
            model, train_loader, optimizer, scaler,
            criterion_cat, criterion_wind, criterion_trend, device
        )
        val_loss, val_acc, val_mae = evaluate(
            model, val_loader,
            criterion_cat, criterion_wind, criterion_trend, device
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
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_mae": val_mae,
                "seq_length": args.seq_length,
                "spatial_backbone": args.spatial_backbone,
                "temporal_engine": args.temporal_engine,
                "hidden_dim": args.hidden_dim
            }
            torch.save(ckpt, save_path / "best_sequence_model.pt")
            print(f"  --> Saved new best checkpoint (Val Acc: {val_acc*100:.2f}%)")

    print("\nTraining Completed Successfully!")
    if args.dry_run:
        print("[DRY-RUN COMPLETE] Pipeline verified end-to-end! Ready for full training.")
    elif args.eval_after_train:
        print("\n" + "=" * 64)
        print(" Running Post-Training Evaluation Suite on Validation Set...")
        print("=" * 64)
        best_ckpt_file = save_path / "best_sequence_model.pt"
        if best_ckpt_file.exists():
            ckpt = torch.load(str(best_ckpt_file), map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            print(f"Loaded best checkpoint from {best_ckpt_file} for evaluation.")
        run_sequence_evaluation(model, val_loader, device, save_path)


if __name__ == "__main__":
    main()
