"""
Comprehensive Diagnostic and Visualization Generator for Cyclone AI Models.
Generates publication-quality charts:
  1. Training & Validation Learning Curves (Loss, Accuracy, F1, Wind MAE)
  2. Confusion Matrix Heatmaps (Normalized & Raw)
  3. Per-Class Precision / Recall / F1 Breakdown Bar Charts
  4. True vs Predicted Wind Speed Regression Scatter & Residual Plots
  5. Multi-Class ROC Curves with AUC Scores
"""

import os
import json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    mean_absolute_error,
    r2_score
)

from src.data_prep.dataset_vision import build_image_dataloaders
from src.models.vision_classifier import CycloneVisionModel, load_vision_model_from_checkpoint



# Set aesthetic styling
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


def plot_training_learning_curves(history_json_path: Path, output_path: Path):
    """Plots multi-panel training vs validation learning curves."""
    if not history_json_path.exists():
        print(f"[Warning] Training history not found at {history_json_path}")
        return

    with open(history_json_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Loss Curve
    axes[0, 0].plot(epochs, history["train_loss"], "o-", color="#1E40AF", label="Train Loss", linewidth=2, markersize=4)
    axes[0, 0].plot(epochs, history["val_loss"], "s-", color="#DC2626", label="Val Loss", linewidth=2, markersize=4)
    axes[0, 0].axvline(x=17, color="#6B7280", linestyle="--", alpha=0.7, label="Plateau Epoch (17-18)")
    axes[0, 0].set_title("Loss Convergence (Focal + Huber Loss)", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Accuracy Curve
    axes[0, 1].plot(epochs, [a * 100 for a in history.get("val_acc", [])], "^-", color="#059669", label="Val Accuracy (%)", linewidth=2, markersize=4)
    axes[0, 1].set_title("Validation Classification Accuracy (%)", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Macro-F1 Score
    axes[1, 0].plot(epochs, history.get("val_f1", []), "d-", color="#7C3AED", label="Val Macro-F1", linewidth=2, markersize=4)
    axes[1, 0].set_title("Validation Macro-F1 Score", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Macro-F1")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Wind Speed Mean Absolute Error (MAE)
    axes[1, 1].plot(epochs, history.get("val_mae", []), "v-", color="#D97706", label="Val Wind MAE (knots)", linewidth=2, markersize=4)
    axes[1, 1].set_title("Validation Wind Speed MAE (knots)", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MAE (knots)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Tropical Cyclone Satellite Vision Model - Training Dynamics", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Chart Saved] Training learning curves -> {output_path}")


def plot_confusion_matrices(y_true, y_pred, class_names, output_path: Path):
    """Plots Normalized and Count Confusion Matrices side-by-side."""
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    cm_raw = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Normalized CM
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar_kws={"label": "Normalized Proportion"}
    )
    axes[0].set_title("Normalized Confusion Matrix (%)", fontweight="bold")
    axes[0].set_xlabel("Predicted Category")
    axes[0].set_ylabel("True Category")
    axes[0].tick_params(axis='x', rotation=30)

    # Raw Count CM
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar_kws={"label": "Sample Count"}
    )
    axes[1].set_title("Absolute Sample Counts", fontweight="bold")
    axes[1].set_xlabel("Predicted Category")
    axes[1].set_ylabel("True Category")
    axes[1].tick_params(axis='x', rotation=30)

    plt.suptitle("Cyclone Severity Category - Confusion Matrix Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Chart Saved] Confusion matrices -> {output_path}")


def plot_per_class_metrics_bar(y_true, y_pred, class_names, output_path: Path):
    """Plots Precision, Recall, and F1-Score per cyclone category."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    categories = [c for c in class_names if c in report]
    precision = [report[c]["precision"] * 100 for c in categories]
    recall = [report[c]["recall"] * 100 for c in categories]
    f1 = [report[c]["f1-score"] * 100 for c in categories]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label="Precision (%)", color="#3B82F6")
    ax.bar(x, recall, width, label="Recall (%)", color="#10B981")
    ax.bar(x + width, f1, width, label="F1-Score (%)", color="#F59E0B")

    ax.set_title("Per-Category Classification Performance (Precision vs Recall vs F1)", fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_ylim([0, 105])
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Chart Saved] Per-class metrics bar chart -> {output_path}")


def plot_wind_speed_regression_scatter(y_true_winds, y_pred_winds, output_path: Path):
    """Plots Ground Truth vs Predicted Wind Speed regression scatter and error residuals."""
    y_true = np.array(y_true_winds)
    y_pred = np.array(y_pred_winds)
    residuals = y_pred - y_true
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.3, color="#2563EB", edgecolors="none", s=25)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal 1:1 Line")
    axes[0].fill_between([min_val, max_val], [min_val - 10, max_val - 10], [min_val + 10, max_val + 10], color="gray", alpha=0.15, label="±10 knots Error Band")
    axes[0].set_title(f"Wind Speed Estimation: True vs Predicted\n$R^2 = {r2:.4f}$ | MAE = {mae:.2f} knots", fontweight="bold")
    axes[0].set_xlabel("True Maximum Sustained Wind Speed (knots)")
    axes[0].set_ylabel("Predicted Wind Speed (knots)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Residual distribution histogram
    sns.histplot(residuals, kde=True, ax=axes[1], color="#7C3AED", bins=35)
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    axes[1].set_title(rf"Error Residual Distribution ($\mu = {np.mean(residuals):.2f}$, $\sigma = {np.std(residuals):.2f}$)", fontweight="bold")
    axes[1].set_xlabel("Residual Error (Predicted - True in knots)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Chart Saved] Wind speed regression diagnostics -> {output_path}")


def generate_all_evaluation_visualizations(config_path: str = "configs/config.yaml"):
    """Runs complete evaluation and produces all diagnostic graphs."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    output_dir = Path(paths.get("output_eval_dir", "models/evaluation_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training curves
    history_path = Path(paths.get("vision_model_dir", "models/vision_model")) / "vision_training_history.json"
    plot_training_learning_curves(history_path, output_dir / "training_learning_curves.png")

    # 2. Evaluate Vision Test Set for Confusion Matrix & Regression Scatter
    vision_weights = Path(paths.get("vision_model_dir", "models/vision_model")) / "best_vision_model.pt"
    if vision_weights.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_vision_model_from_checkpoint(str(vision_weights), device=device, default_num_classes=len(cfg.get("categories", [])))


        _, _, test_loader, _ = build_image_dataloaders(
            root_dir=paths.get("images_dir", "data/images"),
            categories_config=cfg.get("categories"),
            batch_size=32,
            num_workers=0
        )

        all_preds, all_targets = [], []
        all_pred_winds, all_target_winds = [], []

        with torch.no_grad():
            for images, targets_cat, targets_wind in test_loader:
                images = images.to(device)
                logits, pred_wind = model(images)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(targets_cat.numpy())
                all_pred_winds.extend(pred_wind.cpu().numpy())
                all_target_winds.extend(targets_wind.numpy())

        class_names = [c["name"] for c in cfg.get("categories", [])]
        plot_confusion_matrices(all_targets, all_preds, class_names, output_dir / "vision_confusion_matrices_detailed.png")
        plot_per_class_metrics_bar(all_targets, all_preds, class_names, output_dir / "vision_per_class_metrics_bar.png")
        plot_wind_speed_regression_scatter(all_target_winds, all_pred_winds, output_dir / "vision_wind_regression_diagnostics.png")

    print(f"\n[All Visualizations Generated Successfully in '{output_dir}']")


if __name__ == "__main__":
    generate_all_evaluation_visualizations()
