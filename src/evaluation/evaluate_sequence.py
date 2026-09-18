"""
Comprehensive Evaluation & Diagnostic Suite for Spatiotemporal Cyclone Sequence Models.
Generates:
  1. Detailed Confusion Matrices (Normalized & Raw counts)
  2. Per-Class Precision / Recall / F1 Breakdown Charts
  3. Sustained Wind Speed Regression Scatter & Residual Diagnostics (R², MAE, RMSE)
  4. Multi-Class ROC Curves with AUC Scores
  5. Intensity Tendency / Trend Verification Matrix
  6. Structured evaluation_summary.json
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import argparse
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import label_binarize

from src.data_prep.dataset_sequence import WIND_MEAN, WIND_STD

# Styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14
})

WMO_CATEGORIES = [
    "Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe / Super"
]

TREND_LABELS = ["Weakening", "Steady", "Intensifying"]


def run_sequence_evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    use_tta: bool = False,
    wind_fusion: bool = False
) -> Dict[str, Any]:
    """Runs complete evaluation on the dataset and generates publication-quality plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "evaluation_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_cat_preds = []
    all_cat_targets = []
    all_cat_probs = []

    all_wind_preds = []
    all_wind_targets = []

    all_trend_preds = []
    all_trend_targets = []

    tta_str = " (with TTA Horizontal Reflections)" if use_tta else ""
    fusion_str = " + Wind-Informed Fusion" if wind_fusion else ""
    print(f"[Evaluation] Running inference across validation/test sequence split{tta_str}{fusion_str}...")

    # Category centers and spreads for wind-informed Gaussian prior
    centers = torch.tensor([25.0, 41.0, 56.0, 77.0, 115.0], device=device)
    sigmas = torch.tensor([7.0, 6.0, 6.5, 9.0, 16.0], device=device)

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["sequence"].to(device)
            cat = batch["category"].to(device)
            wind_kt = batch["wind_speed"].to(device)
            trend = batch["trend"].to(device)

            # Forward pass
            if use_tta:
                seq_flip = torch.flip(seq, dims=[-1])
                logits1, pred_norm_wind1, pred_trend1 = model(seq)
                logits2, pred_norm_wind2, pred_trend2 = model(seq_flip)
                logits = 0.5 * (logits1 + logits2)
                pred_norm_wind = 0.5 * (pred_norm_wind1 + pred_norm_wind2)
                pred_trend = 0.5 * (pred_trend1 + pred_trend2)
            else:
                logits, pred_norm_wind, pred_trend = model(seq)

            probs = F.softmax(logits, dim=1)

            # De-normalize wind predictions to knots
            pred_wind_kt = pred_norm_wind * WIND_STD + WIND_MEAN

            # Multi-Task Joint Inference: Refine ambiguous category boundaries using continuous wind
            if wind_fusion:
                diff = (pred_wind_kt.unsqueeze(1) - centers.unsqueeze(0)) / (sigmas.unsqueeze(0) + 1e-6)
                log_lik = -0.5 * (diff ** 2)
                fused_log_probs = torch.log(probs + 1e-8) + 0.30 * log_lik
                cat_preds = torch.argmax(fused_log_probs, dim=1)
            else:
                cat_preds = torch.argmax(probs, dim=1)

            all_cat_probs.append(probs.cpu().numpy())
            all_cat_preds.append(cat_preds.cpu().numpy())
            all_cat_targets.append(cat.cpu().numpy())

            all_wind_preds.append(pred_wind_kt.cpu().numpy())
            all_wind_targets.append(wind_kt.cpu().numpy())

            all_trend_preds.append(torch.argmax(pred_trend, dim=1).cpu().numpy())
            all_trend_targets.append(trend.cpu().numpy())


    y_true_cat = np.concatenate(all_cat_targets)
    y_pred_cat = np.concatenate(all_cat_preds)
    y_probs_cat = np.concatenate(all_cat_probs, axis=0)

    y_true_wind = np.concatenate(all_wind_targets)
    y_pred_wind = np.concatenate(all_wind_preds)

    y_true_trend = np.concatenate(all_trend_targets)
    y_pred_trend = np.concatenate(all_trend_preds)

    # 1. Classification Metrics
    report = classification_report(
        y_true_cat, y_pred_cat,
        target_names=[WMO_CATEGORIES[i] for i in sorted(np.unique(y_true_cat))],
        output_dict=True,
        zero_division=0
    )
    acc = float(np.mean(y_true_cat == y_pred_cat))
    macro_f1 = float(report["macro avg"]["f1-score"])
    weighted_f1 = float(report["weighted avg"]["f1-score"])

    # 2. Wind Regression Metrics
    mae_kt = float(mean_absolute_error(y_true_wind, y_pred_wind))
    rmse_kt = float(np.sqrt(mean_squared_error(y_true_wind, y_pred_wind)))
    r2 = float(r2_score(y_true_wind, y_pred_wind)) if len(y_true_wind) > 1 and np.var(y_true_wind) > 1e-4 else 0.0

    # 3. Trend Metrics
    trend_acc = float(np.mean(y_true_trend == y_pred_trend))

    print("\n==================== Evaluation Results ====================")
    print(f" Category Accuracy  : {acc*100:.2f}% | Macro F1: {macro_f1:.4f}")
    print(f" Wind Speed MAE     : {mae_kt:.2f} knots ({mae_kt * 1.852:.2f} km/h)")
    print(f" Wind Speed RMSE    : {rmse_kt:.2f} knots | R²: {r2:.4f}")
    print(f" Trend Accuracy     : {trend_acc*100:.2f}%")
    print(f"============================================================")

    # -------------------------------------------------------------
    # PLOT 1: Confusion Matrices (Raw & Normalized)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    unique_cats = sorted(list(set(y_true_cat) | set(y_pred_cat)))
    cat_names = [WMO_CATEGORIES[c] for c in unique_cats]

    cm_raw = confusion_matrix(y_true_cat, y_pred_cat, labels=unique_cats)
    cm_norm = confusion_matrix(y_true_cat, y_pred_cat, labels=unique_cats, normalize="true")

    sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=cat_names, yticklabels=cat_names)
    axes[0].set_title("Sequence Model: Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted WMO Tier")
    axes[0].set_ylabel("True Ground Truth WMO Tier")

    sns.heatmap(cm_norm * 100, annot=True, fmt=".1f", cmap="Blues", cbar=True, ax=axes[1],
                xticklabels=cat_names, yticklabels=cat_names)
    axes[1].set_title("Sequence Model: Confusion Matrix (Normalized %)", fontweight="bold")
    axes[1].set_xlabel("Predicted WMO Tier")
    axes[1].set_ylabel("True Ground Truth WMO Tier")

    plt.tight_layout()
    plt.savefig(plots_dir / "sequence_confusion_matrix.png", dpi=250)
    plt.close()

    # -------------------------------------------------------------
    # PLOT 2: Per-Class Precision, Recall, F1 Bar Chart
    # -------------------------------------------------------------
    per_class_data = []
    for c_name in cat_names:
        if c_name in report:
            per_class_data.append({
                "Category": c_name,
                "Precision": report[c_name]["precision"] * 100,
                "Recall": report[c_name]["recall"] * 100,
                "F1-Score": report[c_name]["f1-score"] * 100
            })
    if per_class_data:
        df_pc = pd.DataFrame(per_class_data).melt(id_vars="Category", var_name="Metric", value_name="Percentage")
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df_pc, x="Category", y="Percentage", hue="Metric", palette="crest")
        plt.title("Spatiotemporal Model: Per-Class Performance Metrics Breakdown", fontweight="bold")
        plt.ylabel("Score (%)")
        plt.ylim(0, 105)
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 1),
                            ha="center", va="bottom", fontsize=8)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "sequence_per_class_metrics_bar.png", dpi=250)
        plt.close()

    # -------------------------------------------------------------
    # PLOT 3: Wind Speed Regression Diagnostics
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    v_min = min(y_true_wind.min(), y_pred_wind.min()) - 5
    v_max = max(y_true_wind.max(), y_pred_wind.max()) + 5

    axes[0].scatter(y_true_wind, y_pred_wind, alpha=0.6, color="#0284c7", edgecolors="none", s=35)
    axes[0].plot([v_min, v_max], [v_min, v_max], "r--", lw=2, label="Identity (y = x)")
    axes[0].set_title(f"True vs Predicted Wind Speed (MAE: {mae_kt:.1f} kt, R²: {r2:.3f})", fontweight="bold")
    axes[0].set_xlabel("True Sustained Wind Speed (knots)")
    axes[0].set_ylabel("Estimated Sustained Wind Speed (knots)")
    axes[0].set_xlim(v_min, v_max)
    axes[0].set_ylim(v_min, v_max)
    axes[0].legend()

    residuals = y_pred_wind - y_true_wind
    sns.histplot(residuals, kde=True, color="#0f766e", ax=axes[1])
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Wind Speed Residual Error Distribution (kt)", fontweight="bold")
    axes[1].set_xlabel("Residual (Predicted - True in knots)")

    plt.tight_layout()
    plt.savefig(plots_dir / "sequence_wind_regression_diagnostics.png", dpi=250)
    plt.close()

    # -------------------------------------------------------------
    # PLOT 4: Multi-Class ROC Curves
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    try:
        y_bin = label_binarize(y_true_cat, classes=list(range(5)))
        for i, c_name in enumerate(WMO_CATEGORIES):
            if i in unique_cats and np.sum(y_bin[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs_cat[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"{c_name} (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Spatiotemporal Model: Multi-Class ROC-AUC Curves", fontweight="bold")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plots_dir / "sequence_roc_auc_curves.png", dpi=250)
        plt.close()
    except Exception as e:
        print(f"[Evaluation] Note on ROC generation: {e}")

    # -------------------------------------------------------------
    # PLOT 5: Intensity Trend Confusion Matrix
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    cm_trend = confusion_matrix(y_true_trend, y_pred_trend, labels=[0, 1, 2], normalize="true")
    sns.heatmap(cm_trend * 100, annot=True, fmt=".1f", cmap="Greens",
                xticklabels=TREND_LABELS, yticklabels=TREND_LABELS)
    plt.title("Intensity Trend Classification: Weakening vs Steady vs Intensifying (%)", fontweight="bold")
    plt.xlabel("Predicted Tendency")
    plt.ylabel("True Tendency")
    plt.tight_layout()
    plt.savefig(plots_dir / "sequence_intensity_trend_matrix.png", dpi=250)
    plt.close()

    summary = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "wind_mae_knots": mae_kt,
        "wind_mae_kmh": mae_kt * 1.852,
        "wind_rmse_knots": rmse_kt,
        "wind_r2": r2,
        "trend_accuracy": trend_acc,
        "classification_report": report
    }

    summary_file = output_dir / "sequence_evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"[Evaluation] All diagnostic charts saved to: {plots_dir}")
    print(f"[Evaluation] Summary metrics saved to: {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Spatiotemporal Cyclone Model Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="models/sequence_model/best_sequence_model.pt")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="models/sequence_model")
    parser.add_argument("--mock_samples", type=int, default=100)
    parser.add_argument("--use_tta", action="store_true", default=True, help="Use Test-Time Augmentation (horizontal reflections)")
    parser.add_argument("--wind_fusion", action="store_true", default=False, help="Use continuous wind likelihood to calibrate category boundary predictions")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation fraction to evaluate from data_dir (default 0.15, 0.0 for full data)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic evaluation split")
    args = parser.parse_args()

    device = torch.device(args.device)
    from src.data_prep.dataset_sequence import CycloneSequenceDataset
    from src.models.spatiotemporal_classifier import DualStreamSpatiotemporalCycloneModel

    if args.data_dir and os.path.exists(args.data_dir):
        full_ds = CycloneSequenceDataset(data_dir=args.data_dir, is_train=False)
        if args.val_split > 0.0:
            val_size = max(1, int(len(full_ds) * args.val_split))
            train_size = len(full_ds) - val_size
            _, eval_ds = torch.utils.data.random_split(
                full_ds,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(args.seed)
            )
            print(f"[Evaluation Split] Using deterministic val split of {len(eval_ds)} samples (seed={args.seed})")
        else:
            eval_ds = full_ds
    else:
        eval_ds = CycloneSequenceDataset(mock_num_samples=args.mock_samples, is_train=False)

    eval_loader = DataLoader(eval_ds, batch_size=16, shuffle=False)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = DualStreamSpatiotemporalCycloneModel(
            spatial_backbone=ckpt.get("spatial_backbone", "convnext_tiny"),
            temporal_engine=ckpt.get("temporal_engine", "gru"),
            hidden_dim=ckpt.get("hidden_dim", 256),
            seq_length=ckpt.get("seq_length", 4),
            pretrained=False
        ).to(device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"[Evaluation] Loaded weights from {ckpt_path}")
    else:
        model = DualStreamSpatiotemporalCycloneModel(pretrained=False).to(device)
        print(f"[Evaluation] Checkpoint {ckpt_path} not found; evaluating uninitialized weights.")

    run_sequence_evaluation(model, eval_loader, device, Path(args.output_dir), use_tta=args.use_tta, wind_fusion=args.wind_fusion)


if __name__ == "__main__":
    main()
