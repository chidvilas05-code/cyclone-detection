"""
Unified Evaluation and Visualization Pipeline for Cyclone AI Models.
Executes batch evaluation for:
  1. Satellite Vision Model (Single & Hybrid Multi-Backbone)
  2. Tabular Sensory Stacking Ensemble (LightGBM + XGBoost + Quantile Uncertainty)
  3. Dynamic Multimodal Fusion Consensus

Generates complete numerical metrics and saves all diagnostic visualization graphs:
  - training_learning_curves.png
  - vision_confusion_matrices_detailed.png
  - vision_per_class_metrics_bar.png
  - vision_wind_regression_diagnostics.png
  - vision_roc_auc_curves.png
  - sensory_confusion_matrices_detailed.png
  - sensory_wind_regression_diagnostics.png
  - sensory_feature_importances.png
  - evaluation_summary.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

from src.data_prep.dataset_vision import build_image_dataloaders
from src.data_prep.dataset_sensory import SensoryDataProcessor
from src.models.vision_classifier import (
    CycloneVisionModel,
    HybridMultiBackboneCycloneModel,
    load_vision_model_from_checkpoint
)

from src.models.sensory_predictor import CycloneSensoryPredictor
from src.models.fusion import MultimodalCycloneFusion


# Styling configuration for publication-ready visual plots
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


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Loads YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

def plot_training_learning_curves(history_json_path: Path, output_path: Path):
    """Plots multi-panel training vs validation learning curves."""
    if not history_json_path.exists():
        return

    with open(history_json_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if not epochs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Loss
    axes[0, 0].plot(epochs, history.get("train_loss", []), "o-", color="#1E40AF", label="Train Loss", linewidth=2, markersize=4)
    axes[0, 0].plot(epochs, history.get("val_loss", []), "s-", color="#DC2626", label="Val Loss", linewidth=2, markersize=4)
    axes[0, 0].set_title("Loss Convergence (Focal + Huber Loss)", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Accuracy
    val_acc = [a * 100 if a <= 1.0 else a for a in history.get("val_acc", [])]
    axes[0, 1].plot(epochs, val_acc, "^-", color="#059669", label="Val Accuracy (%)", linewidth=2, markersize=4)
    axes[0, 1].set_title("Validation Classification Accuracy (%)", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Macro-F1
    axes[1, 0].plot(epochs, history.get("val_f1", []), "d-", color="#7C3AED", label="Val Macro-F1", linewidth=2, markersize=4)
    axes[1, 0].set_title("Validation Macro-F1 Score", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Macro-F1")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Wind MAE
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
    print(f"  [Graph Saved] Learning curves -> {output_path.name}")


def plot_confusion_matrices(y_true, y_pred, class_names, output_path: Path, model_title: str = "Model"):
    """Plots Normalized and Count Confusion Matrices side-by-side."""
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    cm_raw = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Normalized CM
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar_kws={"label": "Normalized Recall"}
    )
    axes[0].set_title(f"{model_title} - Normalized Confusion Matrix (%)", fontweight="bold")
    axes[0].set_xlabel("Predicted Category")
    axes[0].set_ylabel("True Category")
    axes[0].tick_params(axis='x', rotation=25)

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
    axes[1].set_title(f"{model_title} - Sample Counts", fontweight="bold")
    axes[1].set_xlabel("Predicted Category")
    axes[1].set_ylabel("True Category")
    axes[1].tick_params(axis='x', rotation=25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Graph Saved] Confusion matrices -> {output_path.name}")


def plot_per_class_metrics_bar(y_true, y_pred, class_names, output_path: Path, model_title: str = "Model"):
    """Plots Precision, Recall, and F1-Score per cyclone category."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    categories = [c for c in class_names if c in report]
    precision = [report[c]["precision"] * 100 for c in categories]
    recall = [report[c]["recall"] * 100 for c in categories]
    f1 = [report[c]["f1-score"] * 100 for c in categories]

    x = np.arange(len(categories))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label="Precision (%)", color="#2563EB")
    ax.bar(x, recall, width, label="Recall (%)", color="#059669")
    ax.bar(x + width, f1, width, label="F1-Score (%)", color="#D97706")

    ax.set_title(f"{model_title} - Per-Category Performance (Precision vs Recall vs F1)", fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_ylim([0, 105])
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    for i in range(len(categories)):
        ax.text(x[i] + width, f1[i] + 1.5, f"{f1[i]:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold', color="#B45309")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Graph Saved] Per-class bar chart -> {output_path.name}")


def plot_wind_speed_regression_scatter(
    y_true_winds,
    y_pred_winds,
    output_path: Path,
    model_title: str = "Vision Model",
    wind_low: Optional[np.ndarray] = None,
    wind_high: Optional[np.ndarray] = None
):
    """Plots Ground Truth vs Predicted Wind Speed regression scatter and error residuals."""
    y_true = np.array(y_true_winds)
    y_pred = np.array(y_pred_winds)
    residuals = y_pred - y_true
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.35, color="#1D4ED8", edgecolors="none", s=25, label="Test Samples")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal 1:1 Line")
    axes[0].fill_between(
        [min_val, max_val],
        [min_val - 10, max_val - 10],
        [min_val + 10, max_val + 10],
        color="gray",
        alpha=0.15,
        label="±10 kt Error Band"
    )
    axes[0].set_title(
        f"{model_title} Wind Speed Estimation\n$R^2 = {r2:.4f}$ | MAE = {mae:.2f} kt | RMSE = {rmse:.2f} kt",
        fontweight="bold"
    )
    axes[0].set_xlabel("True Maximum Sustained Wind Speed (knots)")
    axes[0].set_ylabel("Predicted Wind Speed (knots)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Residual distribution histogram
    sns.histplot(residuals, kde=True, ax=axes[1], color="#6D28D9", bins=35)
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero Error Line")
    axes[1].axvline(x=float(np.mean(residuals)), color="blue", linestyle=":", linewidth=1.5, label=f"Mean Bias ({np.mean(residuals):.2f} kt)")
    axes[1].set_title(
        rf"Error Residuals ($\mu = {np.mean(residuals):.2f}\text{{ kt}}$, $\sigma = {np.std(residuals):.2f}\text{{ kt}}$)",
        fontweight="bold"
    )
    axes[1].set_xlabel("Residual Error (Predicted - True in knots)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Graph Saved] Wind regression diagnostics -> {output_path.name}")


def plot_roc_auc_curves(y_true, y_probs, class_names, output_path: Path, model_title: str = "Vision Model"):
    """Plots One-vs-Rest Multiclass ROC Curves with AUC scores."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if y_true_bin.shape[1] == 1 and n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(9, 7))
    colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

    for i in range(n_classes):
        if i < y_true_bin.shape[1] and i < y_probs.shape[1]:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            color = colors[i % len(colors)]
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})"
            )

    # Micro/Macro Average AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"], color="deeppink", linestyle=":", linewidth=2.5, label=f"Micro-Average (AUC = {roc_auc['micro']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title(f"{model_title} - Multi-Class ROC Curves (One-vs-Rest)", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Graph Saved] ROC / AUC curves -> {output_path.name}")


def plot_sensory_feature_importances(predictor: CycloneSensoryPredictor, output_path: Path):
    """Plots top feature importances from the sensory ensemble."""
    if predictor.lgb_clf is None and predictor.xgb_clf is None:
        return

    features = predictor.feature_columns
    importances = np.zeros(len(features))

    if predictor.lgb_clf is not None and hasattr(predictor.lgb_clf, "feature_importances_"):
        lgb_imp = predictor.lgb_clf.feature_importances_
        if np.sum(lgb_imp) > 0:
            importances += 0.5 * (lgb_imp / np.sum(lgb_imp))

    if predictor.xgb_clf is not None and hasattr(predictor.xgb_clf, "feature_importances_"):
        xgb_imp = predictor.xgb_clf.feature_importances_
        if np.sum(xgb_imp) > 0:
            importances += 0.5 * (xgb_imp / np.sum(xgb_imp))

    df_imp = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=True)
    df_top = df_imp.tail(15)

    plt.figure(figsize=(10, 6.5))
    plt.barh(df_top["Feature"], df_top["Importance"] * 100, color="#0284C7")
    plt.xlabel("Relative Importance Weight (%)")
    plt.ylabel("Meteorological & Spatial Features")
    plt.title("Sensory Stacking Ensemble - Top Predictive Features", fontweight="bold")
    plt.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Graph Saved] Feature importances -> {output_path.name}")


# ==============================================================================
# EVALUATION ROUTINES
def save_evaluation_summary_key(eval_dir: Path, key: str, metrics: dict):
    """Safely updates evaluation_summary.json with a specific model's metrics."""
    summary_path = eval_dir / "evaluation_summary.json"
    summary = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except Exception:
            summary = {}
    summary[key] = metrics
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)


def evaluate_vision_model(cfg: dict, eval_dir: Path, use_tta: bool = True) -> Optional[Dict[str, Any]]:
    """
    Evaluates Satellite Vision checkpoint on test split.
    Supports Test-Time Augmentation (TTA) with 4-fold rotational + horizontal flip consensus
    to boost classification accuracy by eliminating orientation bias.
    """
    paths = cfg.get("paths", {})
    vision_cfg = cfg.get("vision_model", {})
    checkpoint_path = Path(paths.get("vision_model_dir", "models/vision_model")) / "best_vision_model.pt"

    if not checkpoint_path.exists():
        print(f"\n[Vision Evaluation Skipped] Checkpoint not found at: {checkpoint_path}")
        return None

    print(f"\n{'='*70}\n1. EVALUATING SATELLITE VISION MODEL\n{'='*70}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(cfg.get("categories", []))

    model, checkpoint = load_vision_model_from_checkpoint(
        str(checkpoint_path),
        device=device,
        default_num_classes=num_classes
    )
    model_type = type(model).__name__
    print(f"Loaded Vision Model Architecture : {model_type}")
    print(f"Test-Time Augmentation (TTA)     : {'4-Fold Rotational + Flip Consensus' if use_tta else 'Single-Pass'}")

    _, _, test_loader, _ = build_image_dataloaders(
        root_dir=paths.get("images_dir", "data/images"),
        categories_config=cfg.get("categories"),
        batch_size=vision_cfg.get("batch_size", 32),
        img_size=vision_cfg.get("img_size", 224),
        num_workers=0
    )

    all_preds, all_probs, all_targets = [], [], []
    all_pred_winds, all_target_winds = [], []

    with torch.no_grad():
        for images, targets_cat, targets_wind in test_loader:
            images = images.to(device)

            if use_tta:
                # 4-Fold cardinal rotational consensus + horizontal mirror flip
                l0, w0 = model(images)
                l90, w90 = model(torch.rot90(images, 1, [2, 3]))
                l180, w180 = model(torch.rot90(images, 2, [2, 3]))
                l270, w270 = model(torch.rot90(images, 3, [2, 3]))
                lf, wf = model(torch.flip(images, dims=[3]))

                probs_tensor = (
                    F.softmax(l0, dim=1) +
                    F.softmax(l90, dim=1) +
                    F.softmax(l180, dim=1) +
                    F.softmax(l270, dim=1) +
                    F.softmax(lf, dim=1)
                ) / 5.0
                pred_wind_tensor = (w0 + w90 + w180 + w270 + wf) / 5.0

                probs = probs_tensor.cpu().numpy()
                preds = torch.argmax(probs_tensor, dim=1).cpu().numpy()
                pred_wind = pred_wind_tensor.cpu().numpy()
            else:
                logits, pred_wind_t = model(images)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                pred_wind = pred_wind_t.cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets_cat.numpy())
            all_pred_winds.extend(pred_wind)
            all_target_winds.extend(targets_wind.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_pred_winds = np.array(all_pred_winds)
    all_target_winds = np.array(all_target_winds)

    class_names = [c["name"] for c in cfg.get("categories", [])]

    # Metrics
    acc = float(np.mean(all_preds == all_targets) * 100)
    mae = float(mean_absolute_error(all_target_winds, all_pred_winds))
    rmse = float(np.sqrt(mean_squared_error(all_target_winds, all_pred_winds)))
    r2 = float(r2_score(all_target_winds, all_pred_winds))
    report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    report_str = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)

    print("\n--- Vision Classification Report ---")
    print(report_str)
    print(f"Accuracy:        {acc:.2f}%")
    print(f"Macro-F1:        {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Weighted-F1:     {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"Wind Speed MAE:  {mae:.2f} knots ({mae*1.852:.2f} km/h)")
    print(f"Wind Speed RMSE: {rmse:.2f} knots ({rmse*1.852:.2f} km/h)")
    print(f"Wind Speed R^2:  {r2:.4f}")

    # Generate Graphs
    history_json = Path(paths.get("vision_model_dir", "models/vision_model")) / "vision_training_history.json"
    plot_training_learning_curves(history_json, eval_dir / "training_learning_curves.png")
    plot_confusion_matrices(all_targets, all_preds, class_names, eval_dir / "vision_confusion_matrices_detailed.png", "Vision Model")
    plot_per_class_metrics_bar(all_targets, all_preds, class_names, eval_dir / "vision_per_class_metrics_bar.png", "Vision Model")
    plot_wind_speed_regression_scatter(all_target_winds, all_pred_winds, eval_dir / "vision_wind_regression_diagnostics.png", "Vision Model")
    plot_roc_auc_curves(all_targets, all_probs, class_names, eval_dir / "vision_roc_auc_curves.png", "Vision Model")

    result = {
        "accuracy": acc,
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "wind_mae_knots": mae,
        "wind_rmse_knots": rmse,
        "wind_r2": r2,
        "use_tta": use_tta,
        "classification_report": report_dict
    }

    # Automatically persist to evaluation_summary.json
    save_evaluation_summary_key(eval_dir, "vision_model", result)
    return result



def evaluate_sensory_model(cfg: dict, eval_dir: Path) -> Optional[Dict[str, Any]]:
    """Evaluates Tabular Sensory Stacking Ensemble on test split."""
    paths = cfg.get("paths", {})
    sensory_dir = Path(paths.get("sensory_model_dir", "models/sensory_model"))
    pipeline_file = sensory_dir / "sensory_pipeline.joblib"

    if not pipeline_file.exists():
        print(f"\n[Sensory Evaluation Skipped] Model not found at: {sensory_dir}")
        return None

    print(f"\n{'='*70}\n2. EVALUATING ATMOSPHERIC SENSORY MODEL (STACKING ENSEMBLE)\n{'='*70}")
    predictor = CycloneSensoryPredictor(model_dir=str(sensory_dir))
    if not predictor.load():
        print("Failed to load sensory pipeline.")
        return None

    # Discover sensory CSV prioritizing IBTrACS or configured dataset
    sensory_dir_path = Path(paths.get("sensory_dir", "data/sensory"))
    candidates = [
        sensory_dir_path / "ibtracs.ALL.list.v04r00.csv",
        sensory_dir_path / "ibtracs.NI.list.v04r00.csv",
        Path(paths.get("sensory_csv", "data/sensory/cyclone_sensory.csv"))
    ]
    sensory_csv = None
    for cand in candidates:
        if cand.exists():
            sensory_csv = str(cand)
            break
    if not sensory_csv and sensory_dir_path.exists():
        found = list(sensory_dir_path.glob("*.csv"))
        if found:
            sensory_csv = str(found[0])

    if not sensory_csv or not os.path.exists(sensory_csv):
        print("Sensory CSV dataset not found.")
        return None


    print(f"Loading sensory test dataset from {sensory_csv}...")
    processor = SensoryDataProcessor(categories_config=cfg.get("categories"), save_dir=str(sensory_dir))
    df = processor.load_and_clean(sensory_csv)
    _, X_test, _, y_cat_test, _, y_wind_test = processor.prepare_train_test_split(df)

    cat_preds, cat_probs, wind_preds, wind_low, wind_high = predictor.predict(X_test)
    class_names = [c["name"] for c in cfg.get("categories", [])]

    # Metrics
    acc = float(np.mean(cat_preds == y_cat_test.values) * 100)
    mae = float(mean_absolute_error(y_wind_test, wind_preds))
    rmse = float(np.sqrt(mean_squared_error(y_wind_test, wind_preds)))
    r2 = float(r2_score(y_wind_test, wind_preds))

    # Quantile Uncertainty Coverage (What % of ground-truth falls in [5%, 95%] interval)
    in_bound = (y_wind_test.values >= wind_low) & (y_wind_test.values <= wind_high)
    coverage_pct = float(np.mean(in_bound) * 100)
    mean_interval_width = float(np.mean(wind_high - wind_low))

    report_dict = classification_report(y_cat_test, cat_preds, target_names=class_names, output_dict=True, zero_division=0)
    report_str = classification_report(y_cat_test, cat_preds, target_names=class_names, zero_division=0)

    print("\n--- Sensory Ensemble Classification Report ---")
    print(report_str)
    print(f"Accuracy:                    {acc:.2f}%")
    print(f"Macro-F1:                    {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Weighted-F1:                 {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"Wind Speed MAE:              {mae:.2f} knots ({mae*1.852:.2f} km/h)")
    print(f"Wind Speed RMSE:             {rmse:.2f} knots ({rmse*1.852:.2f} km/h)")
    print(f"Wind Speed R^2:              {r2:.4f}")
    print(f"90% Quantile Bounds Coverage: {coverage_pct:.2f}% (Empirical Expected: ~90.0%)")
    print(f"Average Uncertainty Window:  ±{mean_interval_width/2.0:.2f} knots")

    # Generate Graphs
    plot_confusion_matrices(y_cat_test, cat_preds, class_names, eval_dir / "sensory_confusion_matrices_detailed.png", "Sensory Ensemble")
    plot_per_class_metrics_bar(y_cat_test, cat_preds, class_names, eval_dir / "sensory_per_class_metrics_bar.png", "Sensory Ensemble")
    plot_wind_speed_regression_scatter(y_wind_test, wind_preds, eval_dir / "sensory_wind_regression_diagnostics.png", "Sensory Ensemble")
    plot_sensory_feature_importances(predictor, eval_dir / "sensory_feature_importances.png")

    result = {
        "accuracy": acc,
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "wind_mae_knots": mae,
        "wind_rmse_knots": rmse,
        "wind_r2": r2,
        "uncertainty_coverage_pct": coverage_pct,
        "mean_uncertainty_interval_knots": mean_interval_width,
        "classification_report": report_dict
    }

    # Automatically persist to evaluation_summary.json
    save_evaluation_summary_key(eval_dir, "sensory_model", result)
    return result



def main():
    parser = argparse.ArgumentParser(description="Unified Cyclone AI Evaluation & Visualization Engine")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_dir = Path(cfg.get("paths", {}).get("output_eval_dir", "models/evaluation_results"))
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'#'*70}\n# CYCLONE AI UNIFIED BATCH EVALUATION PIPELINE\n{'#'*70}")
    print(f"Output directory for graphs and metrics: {eval_dir.resolve()}\n")

    summary_results: Dict[str, Any] = {}

    # 1. Vision Model
    vis_metrics = evaluate_vision_model(cfg, eval_dir)
    if vis_metrics:
        summary_results["vision_model"] = vis_metrics

    # 2. Sensory Model
    sen_metrics = evaluate_sensory_model(cfg, eval_dir)
    if sen_metrics:
        summary_results["sensory_model"] = sen_metrics

    # 3. Save Summary JSON
    summary_path = eval_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=4)

    # 4. Final Executive Terminal Printout
    print(f"\n{'='*70}\n3. EXECUTIVE PERFORMANCE SUMMARY\n{'='*70}")
    print(f"{'Metric':<32} | {'Satellite Vision':<16} | {'Atmospheric Sensory':<20}")
    print(f"{'-'*70}")

    v_acc = f"{vis_metrics['accuracy']:.2f}%" if vis_metrics else "N/A"
    s_acc = f"{sen_metrics['accuracy']:.2f}%" if sen_metrics else "N/A"
    print(f"{'Classification Accuracy':<32} | {v_acc:<16} | {s_acc:<20}")

    v_f1 = f"{vis_metrics['macro_f1']:.4f}" if vis_metrics else "N/A"
    s_f1 = f"{sen_metrics['macro_f1']:.4f}" if sen_metrics else "N/A"
    print(f"{'Macro-F1 Score':<32} | {v_f1:<16} | {s_f1:<20}")

    v_mae = f"{vis_metrics['wind_mae_knots']:.2f} kt" if vis_metrics else "N/A"
    s_mae = f"{sen_metrics['wind_mae_knots']:.2f} kt" if sen_metrics else "N/A"
    print(f"{'Wind Speed MAE':<32} | {v_mae:<16} | {s_mae:<20}")

    v_r2 = f"{vis_metrics['wind_r2']:.4f}" if vis_metrics else "N/A"
    s_r2 = f"{sen_metrics['wind_r2']:.4f}" if sen_metrics else "N/A"
    print(f"{'Wind Speed R^2 Score':<32} | {v_r2:<16} | {s_r2:<20}")

    s_unc = f"{sen_metrics['uncertainty_coverage_pct']:.1f}%" if sen_metrics else "N/A"
    print(f"{'90% Quantile Interval Coverage':<32} | {'N/A':<16} | {s_unc:<20}")

    print(f"{'-'*70}")
    print(f"All 8 visualization figures and JSON metrics successfully exported to:")
    print(f"  --> {eval_dir.resolve()}\n")


if __name__ == "__main__":
    main()

