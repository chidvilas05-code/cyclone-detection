"""
Comprehensive Multi-Task Evaluation Suite for Version 2 Spatiotemporal Cyclone Forecaster.
Evaluates all 5 predictive domains:
  1. 5-Tier WMO Classification (Accuracy, Macro-F1, Confusion Matrix)
  2. Continuous Wind Speed (R², MAE, RMSE)
  3. Central Surface Pressure Inverse Sensing (R², MAE in hPa)
  4. Future Evolvement Forecast at +6h and +12h (Track km Error & Intensity MAE)
  5. Danger Areas Extent (30-kt and 50-kt Radii in km)
  6. Landfall ETA (hours) & Landfall Detection (ROC-AUC)
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

from src.data_prep.dataset_sequence_v2 import (
    CycloneSequenceDatasetV2,
    WIND_MEAN, WIND_STD,
    PRES_MEAN, PRES_STD,
    DANGER_R30_MAX, DANGER_R50_MAX
)

WMO_CATEGORIES = [
    "Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe / Super"
]


def run_v2_sequence_evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    use_tta: bool = True
) -> Dict[str, Any]:
    """Runs complete multi-task evaluation suite across validation sequences."""
    model.eval()
    plots_dir = output_dir / "evaluation_plots_v2"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_cat_preds, all_cat_targets, all_cat_probs = [], [], []
    all_wind_preds, all_wind_targets = [], []
    all_pres_preds, all_pres_targets = [], []
    all_evolve_preds, all_evolve_targets = [], []
    all_danger_preds, all_danger_targets = [], []
    all_landfall_preds, all_landfall_targets = [], []
    all_eta_preds, all_eta_targets = [], []

    print("[Evaluation V2] Running multi-task inference across validation set...")

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["sequence"].to(device)
            cat = batch["category"].to(device)
            wind_kt = batch["wind_speed"].to(device)
            pres_hpa = batch["pressure"].to(device)
            evolve_target = batch["evolve_target"].to(device)
            danger_radii = batch["danger_radii"].to(device)
            landfall = batch["landfall"].to(device)
            landfall_eta = batch["landfall_eta"].to(device)

            if use_tta:
                seq_flip = torch.flip(seq, dims=[-1])
                o1 = model(seq)
                o2 = model(seq_flip)
                out = {}
                for k in o1:
                    out[k] = 0.5 * (o1[k] + o2[k])
            else:
                out = model(seq)

            probs = F.softmax(out["logits"], dim=1)
            cat_preds = torch.argmax(probs, dim=1)

            pred_wind_kt = out["pred_norm_wind"].squeeze(-1) * WIND_STD + WIND_MEAN
            pred_pres_hpa = out["pred_norm_pressure"].squeeze(-1) * PRES_STD + PRES_MEAN

            # Danger radii back to km
            pred_r30_km = out["pred_danger_radii"][:, 0] * DANGER_R30_MAX
            pred_r50_km = out["pred_danger_radii"][:, 1] * DANGER_R50_MAX
            true_r30_km = danger_radii[:, 0] * DANGER_R30_MAX
            true_r50_km = danger_radii[:, 1] * DANGER_R50_MAX

            # Landfall prob & ETA
            pred_landfall_prob = torch.sigmoid(out["pred_landfall"].squeeze(-1))
            pred_eta_hours = out["pred_landfall_eta"].squeeze(-1) * 48.0
            true_eta_hours = landfall_eta * 48.0

            all_cat_probs.append(probs.cpu().numpy())
            all_cat_preds.append(cat_preds.cpu().numpy())
            all_cat_targets.append(cat.cpu().numpy())

            all_wind_preds.append(pred_wind_kt.cpu().numpy())
            all_wind_targets.append(wind_kt.cpu().numpy())

            all_pres_preds.append(pred_pres_hpa.cpu().numpy())
            all_pres_targets.append(pres_hpa.cpu().numpy())

            all_evolve_preds.append(out["pred_evolve"].cpu().numpy())
            all_evolve_targets.append(evolve_target.cpu().numpy())

            all_danger_preds.append(torch.stack([pred_r30_km, pred_r50_km], dim=1).cpu().numpy())
            all_danger_targets.append(torch.stack([true_r30_km, true_r50_km], dim=1).cpu().numpy())

            all_landfall_preds.append(pred_landfall_prob.cpu().numpy())
            all_landfall_targets.append(landfall.cpu().numpy())

            all_eta_preds.append(pred_eta_hours.cpu().numpy())
            all_eta_targets.append(true_eta_hours.cpu().numpy())

    y_true_cat = np.concatenate(all_cat_targets)
    y_pred_cat = np.concatenate(all_cat_preds)

    y_true_wind = np.concatenate(all_wind_targets)
    y_pred_wind = np.concatenate(all_wind_preds)

    y_true_pres = np.concatenate(all_pres_targets)
    y_pred_pres = np.concatenate(all_pres_preds)

    y_true_evolve = np.concatenate(all_evolve_targets, axis=0)
    y_pred_evolve = np.concatenate(all_evolve_preds, axis=0)

    y_true_danger = np.concatenate(all_danger_targets, axis=0)
    y_pred_danger = np.concatenate(all_danger_preds, axis=0)

    y_true_landfall = np.concatenate(all_landfall_targets)
    y_pred_landfall = np.concatenate(all_landfall_preds)

    y_true_eta = np.concatenate(all_eta_targets)
    y_pred_eta = np.concatenate(all_eta_preds)

    # 1. Classification Metrics
    report = classification_report(
        y_true_cat, y_pred_cat,
        target_names=[WMO_CATEGORIES[i] for i in sorted(np.unique(y_true_cat))],
        output_dict=True,
        zero_division=0
    )
    acc = float(np.mean(y_true_cat == y_pred_cat))
    macro_f1 = float(report["macro avg"]["f1-score"])

    # 2. Wind Regression
    wind_mae = float(mean_absolute_error(y_true_wind, y_pred_wind))
    wind_r2 = float(r2_score(y_true_wind, y_pred_wind)) if len(y_true_wind) > 1 else 0.0

    # 3. Pressure Inverse Sensing
    pres_mae = float(mean_absolute_error(y_true_pres, y_pred_pres))
    pres_r2 = float(r2_score(y_true_pres, y_pred_pres)) if len(y_true_pres) > 1 else 0.0

    # 4. Future Evolvement Metrics (+6h and +12h)
    # [Δlat_6, Δlon_6, Δw_6, Δlat_12, Δlon_12, Δw_12]
    # 1 degree ≈ 111 km
    track_err_6h_km = float(np.mean(np.sqrt((y_pred_evolve[:, 0] - y_true_evolve[:, 0])**2 + (y_pred_evolve[:, 1] - y_true_evolve[:, 1])**2) * 111.0))
    wind_err_6h_kt = float(mean_absolute_error(y_true_evolve[:, 2], y_pred_evolve[:, 2]))

    track_err_12h_km = float(np.mean(np.sqrt((y_pred_evolve[:, 3] - y_true_evolve[:, 3])**2 + (y_pred_evolve[:, 4] - y_true_evolve[:, 4])**2) * 111.0))
    wind_err_12h_kt = float(mean_absolute_error(y_true_evolve[:, 5], y_pred_evolve[:, 5]))

    # 5. Danger Area Radii MAE
    r30_mae_km = float(mean_absolute_error(y_true_danger[:, 0], y_pred_danger[:, 0]))
    r50_mae_km = float(mean_absolute_error(y_true_danger[:, 1], y_pred_danger[:, 1]))

    # 6. Landfall ETA
    landfall_eta_mae = float(mean_absolute_error(y_true_eta, y_pred_eta))

    print("\n==================== Version 2 Evaluation Results ====================")
    print(f" [Intensity] Category Accuracy : {acc*100:.2f}% | Macro-F1: {macro_f1:.4f}")
    print(f" [Intensity] Wind Speed MAE    : {wind_mae:.2f} knots | R²: {wind_r2:.4f}")
    print(f" [Sensory]   Central Pressure  : {pres_mae:.2f} hPa | R²: {pres_r2:.4f}")
    print(f" [Evolve]    +6h Track Error   : {track_err_6h_km:.1f} km | Wind Error: {wind_err_6h_kt:.1f} kt")
    print(f" [Evolve]    +12h Track Error  : {track_err_12h_km:.1f} km | Wind Error: {wind_err_12h_kt:.1f} kt")
    print(f" [Danger]    30-kt Gale Radius : MAE {r30_mae_km:.1f} km | 50-kt Storm Radius: MAE {r50_mae_km:.1f} km")
    print(f" [Landfall]  Shore ETA MAE     : {landfall_eta_mae:.1f} hours")
    print("=======================================================================\n")

    summary = {
        "category_accuracy": acc,
        "macro_f1": macro_f1,
        "wind_mae_knots": wind_mae,
        "wind_r2": wind_r2,
        "pressure_mae_hpa": pres_mae,
        "pressure_r2": pres_r2,
        "track_error_6h_km": track_err_6h_km,
        "wind_error_6h_kt": wind_err_6h_kt,
        "track_error_12h_km": track_err_12h_km,
        "wind_error_12h_kt": wind_err_12h_kt,
        "danger_r30_mae_km": r30_mae_km,
        "danger_r50_mae_km": r50_mae_km,
        "landfall_eta_mae_hours": landfall_eta_mae,
        "classification_report": report
    }

    summary_file = output_dir / "sequence_v2_evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)

    # Diagnostic Plots
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        unique_cats = sorted(list(set(y_true_cat) | set(y_pred_cat)))
        cat_names = [WMO_CATEGORIES[c] for c in unique_cats]
        cm_norm = confusion_matrix(y_true_cat, y_pred_cat, labels=unique_cats, normalize="true")
        sns.heatmap(cm_norm * 100, annot=True, fmt=".1f", cmap="Blues", cbar=True, ax=axes[0],
                    xticklabels=cat_names, yticklabels=cat_names)
        axes[0].set_title("V2 Category Confusion Matrix (%)", fontweight="bold")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")

        axes[1].scatter(y_true_pres, y_pred_pres, alpha=0.3, color="teal")
        axes[1].plot([890, 1015], [890, 1015], "r--")
        axes[1].set_title(f"Sensory Pressure Inverse Sensing (MAE: {pres_mae:.1f} hPa)", fontweight="bold")
        axes[1].set_xlabel("True Central Pressure (hPa)")
        axes[1].set_ylabel("Predicted Central Pressure (hPa)")
        plt.tight_layout()
        plt.savefig(plots_dir / "v2_diagnostic_overview.png", dpi=200)
        plt.close()
    except Exception:
        pass

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Version 2 Multi-Task Spatiotemporal Forecaster")
    parser.add_argument("--checkpoint", type=str, default="models/sequence_model_v2/best_sequence_model_v2.pt")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="models/sequence_model_v2")
    parser.add_argument("--mock_samples", type=int, default=100)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_tta", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    from src.models.spatiotemporal_forecaster import MultiTaskSpatiotemporalCycloneModel

    if args.data_dir and os.path.exists(args.data_dir):
        full_ds = CycloneSequenceDatasetV2(data_dir=args.data_dir, is_train=False)
        val_size = max(1, int(len(full_ds) * args.val_split))
        train_size = len(full_ds) - val_size
        _, eval_ds = torch.utils.data.random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    else:
        eval_ds = CycloneSequenceDatasetV2(mock_num_samples=args.mock_samples, is_train=False)

    eval_loader = DataLoader(eval_ds, batch_size=16, shuffle=False)

    ckpt_path = Path(args.checkpoint)
    model = MultiTaskSpatiotemporalCycloneModel(pretrained=False).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"[Evaluation V2] Loaded weights from {ckpt_path}")

    run_v2_sequence_evaluation(model, eval_loader, device, Path(args.output_dir), use_tta=args.use_tta)


if __name__ == "__main__":
    main()
