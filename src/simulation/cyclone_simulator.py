"""
Physical Cyclone Occurrence Simulator & Comprehensive Testbench.
Simulates randomized, unseen cyclone occurrences across multiple global ocean basins
(Bay of Bengal, Arabian Sea, Western Pacific, South China Sea, etc.)
and rigorously evaluates all model performance aspects:
  1. WMO Category Classification (Accuracy, Macro-F1, Confusion Matrix)
  2. Sustained Wind Speed (MAE in knots/kmh, RMSE, R²)
  3. Central Surface Pressure Inverse Sensing (MAE in hPa, R²)
  4. Future Trajectory Evolvement (+6h and +12h Track Error in km & Wind Error in kt)
  5. Danger Area Wind Radii (R30 gale & R50 storm extent in km)
  6. Landfall Probability & Shore Arrival ETA Error (hours)
  7. High-Threat / Dangerous Storms Subset (Category 3, 4, 5 & Rapid Intensification)
  8. Geographic Ocean Basin Breakdown
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score
)

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

WMO_CATEGORIES = [
    "Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe / Super"
]


def classify_ocean_basin(lat: float, lon: float) -> str:
    """Identifies the geographic ocean basin from cyclone coordinates."""
    if lat < 0:
        return "Southern Ocean"
    elif 50.0 <= lon <= 78.0 and 5.0 <= lat <= 26.0:
        return "Arabian Sea"
    elif 78.1 <= lon <= 99.0 and 5.0 <= lat <= 25.0:
        return "Bay of Bengal"
    elif 99.1 <= lon <= 120.0 and 0.0 <= lat <= 26.0:
        return "South China Sea"
    elif 120.1 <= lon <= 180.0 and 5.0 <= lat <= 38.0:
        return "Western North Pacific"
    elif lon < 0:
        return "Atlantic / East Pacific"
    else:
        return "North Indian / Tropical Ocean"


def run_cyclone_simulation_test(
    model_checkpoint: str,
    data_dir: str = "data/sequences",
    num_simulations: int = 100,
    threat_only: bool = False,
    basin_filter: Optional[str] = None,
    seed: int = 42,
    device_str: str = "cuda",
    output_dir_str: str = "models/sequence_model_v2/simulation_results"
) -> Dict[str, Any]:
    """
    Simulates a sequence of randomized cyclone occurrences from holdout storm data
    and tests all performance dimensions of the model.
    """
    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("  Physical Cyclone Simulation & Comprehensive Testbench")
    print(f"  Target Model Checkpoint: {model_checkpoint}")
    print(f"  Device: {device} | Total Simulated Occurrences: {num_simulations}")
    if threat_only:
        print("  Focus Mode: High-Threat Cyclones (Category 3, 4, 5 Only)")
    if basin_filter:
        print(f"  Basin Filter: {basin_filter}")
    print("=" * 74)

    # 1. Instantiate and Load Model (Offline with pretrained=False)
    model = MultiTaskSpatiotemporalCycloneModel(
        pretrained=False,
        num_classes=5,
        hidden_dim=256,
        seq_length=4,
        eye_crop_ratio=0.25
    ).to(device)

    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_checkpoint}")

    print(f"[Loading] Loading model weights from {model_checkpoint}...")
    ckpt = torch.load(model_checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("[Ready] Model loaded cleanly into evaluation mode.\n")

    # 2. Index Dataset
    print(f"[Dataset] Indexing holdout sequence dataset from {data_dir}...")
    t0 = time.time()
    full_ds = CycloneSequenceDatasetV2(data_dir=data_dir, is_train=False)
    print(f"[Dataset] Indexed {len(full_ds)} sequence windows in {time.time() - t0:.2f}s.")

    # 3. Simulate Random Cyclone Occurrences
    rng = np.random.RandomState(seed)
    candidate_indices = []

    for i, s in enumerate(full_ds.samples):
        lat = s.get("lat", 15.0)
        lon = s.get("lng", 130.0)
        basin = classify_ocean_basin(lat, lon)
        cat = s["category"]

        if threat_only and cat < 3:
            continue
        if basin_filter and basin.lower() != basin_filter.lower():
            continue
        candidate_indices.append(i)

    if len(candidate_indices) == 0:
        raise ValueError("No cyclone occurrences found matching the specified filter criteria.")

    num_samples = min(num_simulations, len(candidate_indices))
    selected_indices = rng.choice(candidate_indices, size=num_samples, replace=False).tolist()
    print(f"[Simulation] Selected {num_samples} randomized occurrences across candidate pool of {len(candidate_indices)}.\n")

    sim_subset = Subset(full_ds, selected_indices)
    dataloader = DataLoader(sim_subset, batch_size=8, shuffle=False, num_workers=0)

    # 4. Run Multi-Aspect Evaluation
    records = []
    print("[Testing] Evaluating model across all simulated occurrences...")

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["sequence"].to(device)
            cat_true = batch["category"].numpy()
            wind_true = batch["wind_speed"].numpy()
            pres_true = batch["pressure"].numpy()
            trend_true = batch["trend"].numpy()
            evolve_true = batch["evolve_target"].numpy()
            danger_radii_true = batch["danger_radii"].numpy()
            landfall_true = batch["landfall"].numpy()
            landfall_eta_true = batch["landfall_eta"].numpy() * 48.0
            lats = batch["lat"].numpy()
            lngs = batch["lng"].numpy()
            storm_ids = batch["storm_id"]

            # Test-Time Augmentation (TTA)
            seq_flip = torch.flip(seq, dims=[-1])
            amp_dtype = torch.bfloat16 if ("cuda" in device.type and torch.cuda.is_bf16_supported()) else torch.float16
            with torch.amp.autocast(device_type="cuda" if "cuda" in device.type else "cpu", dtype=amp_dtype):
                o1 = model(seq)
                o2 = model(seq_flip)
                out = {k: 0.5 * (o1[k] + o2[k]) for k in o1}

            # De-normalize predictions
            pred_logits = out["logits"].float().cpu()
            pred_cats = torch.argmax(pred_logits, dim=-1).numpy()
            pred_cat_probs = F.softmax(pred_logits, dim=-1).numpy()

            pred_norm_wind = out["pred_norm_wind"].squeeze(-1).float().cpu().numpy()
            pred_winds = pred_norm_wind * WIND_STD + WIND_MEAN

            pred_norm_pres = out["pred_norm_pressure"].squeeze(-1).float().cpu().numpy()
            pred_pressures = pred_norm_pres * PRES_STD + PRES_MEAN

            pred_trends = torch.argmax(out["pred_trend"].float().cpu(), dim=-1).numpy()
            pred_evolve = out["pred_evolve"].float().cpu().numpy()

            pred_danger = out["pred_danger_radii"].float().cpu().numpy()
            pred_r30_km = np.clip(pred_danger[:, 0] * DANGER_R30_MAX, 0.0, DANGER_R30_MAX)
            pred_r50_km = np.clip(pred_danger[:, 1] * DANGER_R50_MAX, 0.0, DANGER_R50_MAX)

            pred_landfall_prob = torch.sigmoid(out["pred_landfall"].squeeze(-1).float().cpu()).numpy()
            pred_landfall_eta = np.clip(out["pred_landfall_eta"].squeeze(-1).float().cpu().numpy() * 48.0, 0.0, 48.0)

            b_len = len(cat_true)
            for b in range(b_len):
                # +6h and +12h Track error in km
                d_lat_6_true = evolve_true[b, 0]
                d_lon_6_true = evolve_true[b, 1]
                d_lat_6_pred = pred_evolve[b, 0]
                d_lon_6_pred = pred_evolve[b, 1]
                track_err_6h_km = math.sqrt(
                    ((d_lat_6_pred - d_lat_6_true) * 111.0)**2 +
                    ((d_lon_6_pred - d_lon_6_true) * 100.0)**2
                )

                d_lat_12_true = evolve_true[b, 3]
                d_lon_12_true = evolve_true[b, 4]
                d_lat_12_pred = pred_evolve[b, 3]
                d_lon_12_pred = pred_evolve[b, 4]
                track_err_12h_km = math.sqrt(
                    ((d_lat_12_pred - d_lat_12_true) * 111.0)**2 +
                    ((d_lon_12_pred - d_lon_12_true) * 100.0)**2
                )

                d_wind_6_err = abs(pred_evolve[b, 2] - evolve_true[b, 2])
                d_wind_12_err = abs(pred_evolve[b, 5] - evolve_true[b, 5])

                r30_true_km = danger_radii_true[b, 0] * DANGER_R30_MAX
                r50_true_km = danger_radii_true[b, 1] * DANGER_R50_MAX

                is_ri = (evolve_true[b, 5] >= 15.0)  # Rapid Intensification: +15 kt in 12h
                basin_name = classify_ocean_basin(lats[b], lngs[b])

                records.append({
                    "storm_id": storm_ids[b],
                    "basin": basin_name,
                    "lat": float(lats[b]),
                    "lon": float(lngs[b]),
                    "is_ri": bool(is_ri),
                    "cat_true": int(cat_true[b]),
                    "cat_pred": int(pred_cats[b]),
                    "cat_correct": bool(pred_cats[b] == cat_true[b]),
                    "cat_probs": pred_cat_probs[b].tolist(),
                    "wind_true": float(wind_true[b]),
                    "wind_pred": float(pred_winds[b]),
                    "wind_error": float(abs(pred_winds[b] - wind_true[b])),
                    "pressure_true": float(pres_true[b]),
                    "pressure_pred": float(pred_pressures[b]),
                    "pressure_error": float(abs(pred_pressures[b] - pres_true[b])),
                    "trend_true": int(trend_true[b]),
                    "trend_pred": int(pred_trends[b]),
                    "trend_correct": bool(pred_trends[b] == trend_true[b]),
                    "track_err_6h_km": float(track_err_6h_km),
                    "track_err_12h_km": float(track_err_12h_km),
                    "d_wind_6h_err_kt": float(d_wind_6_err),
                    "d_wind_12h_err_kt": float(d_wind_12_err),
                    "r30_true_km": float(r30_true_km),
                    "r30_pred_km": float(pred_r30_km[b]),
                    "r30_err": float(abs(pred_r30_km[b] - r30_true_km)),
                    "r50_true_km": float(r50_true_km),
                    "r50_pred_km": float(pred_r50_km[b]),
                    "r50_err": float(abs(pred_r50_km[b] - r50_true_km)),
                    "landfall_true": float(landfall_true[b]),
                    "landfall_pred_prob": float(pred_landfall_prob[b]),
                    "landfall_pred_binary": int(pred_landfall_prob[b] >= 0.5),
                    "landfall_eta_true": float(landfall_eta_true[b]),
                    "landfall_eta_pred": float(pred_landfall_eta[b]),
                    "landfall_eta_err": float(abs(pred_landfall_eta[b] - landfall_eta_true[b]))
                })

    df = pd.DataFrame(records)

    # 5. Compute Quantitative Metrics Across All Domains
    acc = accuracy_score(df["cat_true"], df["cat_pred"])
    macro_f1 = f1_score(df["cat_true"], df["cat_pred"], average="macro", zero_division=0)
    weighted_f1 = f1_score(df["cat_true"], df["cat_pred"], average="weighted", zero_division=0)

    wind_mae = mean_absolute_error(df["wind_true"], df["wind_pred"])
    wind_rmse = math.sqrt(mean_squared_error(df["wind_true"], df["wind_pred"]))
    wind_r2 = r2_score(df["wind_true"], df["wind_pred"])

    pres_mae = mean_absolute_error(df["pressure_true"], df["pressure_pred"])
    pres_r2 = r2_score(df["pressure_true"], df["pressure_pred"])

    track_6h_mae = df["track_err_6h_km"].mean()
    track_12h_mae = df["track_err_12h_km"].mean()
    wind_6h_mae = df["d_wind_6h_err_kt"].mean()
    wind_12h_mae = df["d_wind_12h_err_kt"].mean()

    r30_mae = df["r30_err"].mean()
    r50_mae = df["r50_err"].mean()

    landfall_acc = accuracy_score(df["landfall_true"], df["landfall_pred_binary"])
    landfall_eta_mae = df[df["landfall_true"] > 0.5]["landfall_eta_err"].mean() if (df["landfall_true"] > 0.5).sum() > 0 else 0.0

    trend_acc = accuracy_score(df["trend_true"], df["trend_pred"])

    # High-Threat Subset Performance (Category 3, 4, 5 / >= 64 knots)
    threat_df = df[df["cat_true"] >= 3]
    threat_acc = accuracy_score(threat_df["cat_true"], threat_df["cat_pred"]) if len(threat_df) > 0 else 0.0
    threat_wind_mae = mean_absolute_error(threat_df["wind_true"], threat_df["wind_pred"]) if len(threat_df) > 0 else 0.0
    threat_pres_mae = mean_absolute_error(threat_df["pressure_true"], threat_df["pressure_pred"]) if len(threat_df) > 0 else 0.0

    # Basin Breakdown Summary
    basin_summary = {}
    for basin, b_df in df.groupby("basin"):
        basin_summary[basin] = {
            "samples": len(b_df),
            "category_accuracy": float(accuracy_score(b_df["cat_true"], b_df["cat_pred"])),
            "wind_mae_knots": float(mean_absolute_error(b_df["wind_true"], b_df["wind_pred"])),
            "pressure_mae_hpa": float(mean_absolute_error(b_df["pressure_true"], b_df["pressure_pred"])),
            "track_error_6h_km": float(b_df["track_err_6h_km"].mean())
        }

    summary = {
        "num_simulations": len(df),
        "overall_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "wind_mae_knots": float(wind_mae),
        "wind_mae_kmh": float(wind_mae * 1.852),
        "wind_rmse_knots": float(wind_rmse),
        "wind_r2": float(wind_r2),
        "pressure_mae_hpa": float(pres_mae),
        "pressure_r2": float(pres_r2),
        "track_error_6h_km": float(track_6h_mae),
        "track_error_12h_km": float(track_12h_mae),
        "wind_delta_6h_err_kt": float(wind_6h_mae),
        "wind_delta_12h_err_kt": float(wind_12h_mae),
        "danger_r30_mae_km": float(r30_mae),
        "danger_r50_mae_km": float(r50_mae),
        "landfall_accuracy": float(landfall_acc),
        "landfall_eta_mae_hours": float(landfall_eta_mae),
        "trend_accuracy": float(trend_acc),
        "threat_subset": {
            "num_threat_cyclones": int(len(threat_df)),
            "threat_accuracy": float(threat_acc),
            "threat_wind_mae_kt": float(threat_wind_mae),
            "threat_pressure_mae_hpa": float(threat_pres_mae)
        },
        "basin_breakdown": basin_summary,
        "classification_report": classification_report(
            df["cat_true"], df["cat_pred"], labels=[0, 1, 2, 3, 4], target_names=WMO_CATEGORIES, output_dict=True, zero_division=0
        )
    }

    # Save Results JSON
    summary_path = output_dir / "simulation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 6. Generate Visual Diagnostic Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    sns.set_theme(style="whitegrid")

    # Plot 1: Confusion Matrix
    cm = confusion_matrix(df["cat_true"], df["cat_pred"], labels=[0, 1, 2, 3, 4])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
                xticklabels=[c[:8] for c in WMO_CATEGORIES],
                yticklabels=[c[:8] for c in WMO_CATEGORIES])
    axes[0, 0].set_title(f"WMO Tier Classification (Acc: {acc*100:.1f}%)", fontweight="bold")
    axes[0, 0].set_xlabel("Predicted Tier")
    axes[0, 0].set_ylabel("True Tier")

    # Plot 2: Wind Speed Scatter (Predicted vs True)
    ax_w = axes[0, 1]
    scatter = ax_w.scatter(df["wind_true"], df["wind_pred"], c=df["cat_true"], cmap="plasma", alpha=0.75, edgecolors="k", linewidth=0.5, s=45)
    lims = [15.0, max(140.0, df["wind_true"].max() + 5.0)]
    ax_w.plot(lims, lims, "r--", linewidth=1.5, label="Ideal 1:1")
    ax_w.axhline(64.0, color="orange", linestyle=":", alpha=0.7, label="Severe Threshold (64 kt)")
    ax_w.set_title(f"Sustained Wind Speed (MAE: {wind_mae:.2f} kt, R²: {wind_r2:.3f})", fontweight="bold")
    ax_w.set_xlabel("True Wind Speed (knots)")
    ax_w.set_ylabel("Predicted Wind Speed (knots)")
    ax_w.legend(loc="upper left", fontsize=8)

    # Plot 3: Central Pressure Scatter
    ax_p = axes[0, 2]
    ax_p.scatter(df["pressure_true"], df["pressure_pred"], c="teal", alpha=0.7, edgecolors="k", linewidth=0.5, s=40)
    p_min = min(df["pressure_true"].min(), df["pressure_pred"].min()) - 5.0
    p_max = max(df["pressure_true"].max(), df["pressure_pred"].max()) + 5.0
    p_lims = [p_min, p_max]
    ax_p.plot(p_lims, p_lims, "r--", linewidth=1.5, label="Ideal 1:1")
    ax_p.set_title(f"Central Surface Pressure (MAE: {pres_mae:.2f} hPa, R²: {pres_r2:.3f})", fontweight="bold")
    ax_p.set_xlabel("True Central Pressure (hPa)")
    ax_p.set_ylabel("Predicted Central Pressure (hPa)")
    ax_p.legend(loc="upper left", fontsize=8)

    # Plot 4: Track Evolvement Error Distribution (+6h vs +12h)
    ax_t = axes[1, 0]
    sns.boxplot(data=[df["track_err_6h_km"], df["track_err_12h_km"]], palette=["#4CAF50", "#2196F3"], ax=ax_t)
    ax_t.set_xticks([0, 1])
    ax_t.set_xticklabels(["+6h Horizon", "+12h Horizon"])
    ax_t.set_ylabel("Displacement Error (km)")
    ax_t.set_title(f"Trajectory Forecast (+6h: {track_6h_mae:.1f} km, +12h: {track_12h_mae:.1f} km)", fontweight="bold")

    # Plot 5: Danger Wind Radii Error (R30 & R50)
    ax_d = axes[1, 1]
    sns.boxplot(data=[df["r30_err"], df["r50_err"]], palette=["#FF9800", "#E91E63"], ax=ax_d)
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(["R30 Gale Radius", "R50 Storm Radius"])
    ax_d.set_ylabel("Radii Error (km)")
    ax_d.set_title(f"Danger Area Extent (R30: {r30_mae:.1f} km, R50: {r50_mae:.1f} km)", fontweight="bold")

    # Plot 6: Basin Breakdown Performance Bar
    ax_b = axes[1, 2]
    basin_accs = [basin_summary[b]["category_accuracy"] * 100 for b in basin_summary]
    basin_labels = [b.replace(" ", "\n") for b in basin_summary]
    bars = ax_b.bar(basin_labels, basin_accs, color="#673AB7", alpha=0.85)
    ax_b.set_ylabel("Accuracy (%)")
    ax_b.set_ylim(0, 105)
    ax_b.set_title("Cross-Basin Accuracy Generalization", fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "simulation_performance_dashboard.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Dashboard] Saved visual performance dashboard to: {plot_path}")
    print(f"[Metrics] Saved JSON summary to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Physical Cyclone Simulation & Multi-Aspect Testing Suite")
    parser.add_argument("--checkpoint", type=str, default="models/sequence_model_v2/best_sequence_model_v2.pt",
                        help="Path to trained V2 checkpoint to test")
    parser.add_argument("--data_dir", type=str, default="data/sequences",
                        help="Path to sequence dataset")
    parser.add_argument("--num_simulations", type=int, default=100,
                        help="Number of randomized cyclone occurrences to simulate")
    parser.add_argument("--threat_only", action="store_true", default=False,
                        help="Evaluate only dangerous high-threat storms (Category 3, 4, 5)")
    parser.add_argument("--basin", type=str, default=None,
                        help="Optional ocean basin filter (e.g. 'Bay of Bengal', 'Arabian Sea', 'Western North Pacific')")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for repeatable random occurrence selection")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default="models/sequence_model_v2/simulation_results",
                        help="Output directory for simulation benchmark results")
    args = parser.parse_args()

    results = run_cyclone_simulation_test(
        model_checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        num_simulations=args.num_simulations,
        threat_only=args.threat_only,
        basin_filter=args.basin,
        seed=args.seed,
        device_str=args.device,
        output_dir_str=args.output_dir
    )

    print("\n" + "=" * 74)
    print("  SIMULATION BENCHMARK SUMMARY (RANDOM UNSEEN CYCLONE OCCURRENCES)")
    print("=" * 74)
    print(f"Total Cyclones Simulated:        {results['num_simulations']}")
    print(f"WMO Category Accuracy:           {results['overall_accuracy']*100:.2f}% (Macro-F1: {results['macro_f1']:.3f})")
    print(f"Wind Speed MAE:                  {results['wind_mae_knots']:.2f} knots ({results['wind_mae_kmh']:.2f} km/h) | R²: {results['wind_r2']:.3f}")
    print(f"Central Surface Pressure MAE:    {results['pressure_mae_hpa']:.2f} hPa | R²: {results['pressure_r2']:.3f}")
    print(f"Future Track Error (+6h / +12h): {results['track_error_6h_km']:.1f} km / {results['track_error_12h_km']:.1f} km")
    print(f"Danger Radii MAE (R30 / R50):    {results['danger_r30_mae_km']:.1f} km / {results['danger_r50_mae_km']:.1f} km")
    print(f"Landfall Detection Accuracy:     {results['landfall_accuracy']*100:.2f}% | Arrival ETA MAE: {results['landfall_eta_mae_hours']:.2f} hours")
    print("-" * 74)
    threat = results["threat_subset"]
    print(f"High-Threat Subset (Cat 3, 4, 5): {threat['num_threat_cyclones']} storms tested")
    print(f"  -> High-Threat Category Acc:   {threat['threat_accuracy']*100:.2f}%")
    print(f"  -> High-Threat Wind MAE:       {threat['threat_wind_mae_kt']:.2f} knots")
    print(f"  -> High-Threat Pressure MAE:   {threat['threat_pressure_mae_hpa']:.2f} hPa")
    print("=" * 74)


if __name__ == "__main__":
    main()
