"""
Advanced Training Script for Cyclone Tabular Sensory and Atmospheric Intelligence.
Trains a Multi-Model Stacking Ensemble (LightGBM + XGBoost + Quantile Uncertainty Bounds).
Supports the full 700,000+ record Global NOAA IBTrACS dataset.
"""

import os
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from src.data_prep.dataset_sensory import SensoryDataProcessor
from src.models.sensory_predictor import CycloneSensoryPredictor


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_sensory_csv(paths_cfg: dict) -> Path:
    """Discovers sensory CSV from specified path or directory scan, prioritizing global IBTrACS."""
    sensory_dir = Path(paths_cfg.get("sensory_dir", "data/sensory"))
    
    # Check for full global or regional IBTrACS first
    ibtracs_candidates = [
        sensory_dir / "ibtracs.ALL.list.v04r00.csv",
        sensory_dir / "ibtracs.NI.list.v04r00.csv",
        sensory_dir / "ibtracs.csv"
    ]
    for cand in ibtracs_candidates:
        if cand.exists():
            return cand

    # Check for default cyclone_sensory.csv
    specified_csv = Path(paths_cfg.get("sensory_csv", "data/sensory/cyclone_sensory.csv"))
    if specified_csv.exists():
        return specified_csv

    if sensory_dir.exists():
        found = list(sensory_dir.glob("*.csv"))
        if found:
            return found[0]

    raise FileNotFoundError(
        f"No sensory CSV file found in '{sensory_dir}'. "
        f"Please download IBTrACS or the sensory dataset and place the CSV in '{sensory_dir}'."
    )


def train_sensory_model(config_path: str = "configs/config.yaml", custom_csv: str = None):
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    sensory_cfg = cfg.get("sensory_model", {})

    csv_path = Path(custom_csv) if custom_csv else find_sensory_csv(paths)
    save_dir = paths.get("sensory_model_dir", "models/sensory_model")

    print(f"\n{'='*60}")
    print(f"Cyclone Sensory Multi-Model Stacking Ensemble Pipeline")
    print(f"{'='*60}")
    print(f"Input Dataset    : {csv_path.name}")
    print(f"Ensemble Stack   : LightGBM (Leaf-wise) + XGBoost (Depth-wise) + 90% Quantiles")
    print(f"Model Output Dir : {save_dir}")
    print(f"{'='*60}\n")

    # 1. Load and Clean Data
    processor = SensoryDataProcessor(
        categories_config=cfg.get("categories"),
        save_dir=save_dir
    )
    df_clean = processor.load_and_clean(str(csv_path))
    print(f"Sanitized Records Loaded : {len(df_clean):,}")

    # 2. Prepare Train/Test Splits
    X_train, X_test, y_cat_train, y_cat_test, y_wind_train, y_wind_test = processor.prepare_train_test_split(df_clean)
    print(f"Physical Feature Space   : {processor.feature_columns}")
    print(f"Training Samples         : {len(X_train):,}")
    print(f"Testing Samples          : {len(X_test):,}\n")

    # 3. Train Multi-Model Stacking Predictor
    predictor = CycloneSensoryPredictor(
        model_dir=save_dir,
        model_type="ensemble"
    )
    predictor.categories_config = cfg.get("categories")

    print("Training Multi-Model Stacking Ensemble & Quantile Uncertainty Bounds...")
    predictor.fit(
        X_train=X_train,
        y_cat_train=y_cat_train,
        y_wind_train=y_wind_train,
        X_val=X_test,
        y_cat_val=y_cat_test,
        y_wind_val=y_wind_test,
        params=sensory_cfg.get("lgbm_params")
    )
    print("Training Complete!\n")

    # 4. Evaluation
    cat_preds, cat_probs, wind_preds, wind_low, wind_high = predictor.predict(X_test)

    acc = accuracy_score(y_cat_test, cat_preds)
    f1 = f1_score(y_cat_test, cat_preds, average="macro", zero_division=0)
    mae = mean_absolute_error(y_wind_test, wind_preds)
    rmse = np.sqrt(mean_squared_error(y_wind_test, wind_preds))
    r2 = r2_score(y_wind_test, wind_preds)
    avg_uncertainty = float(np.mean((wind_high - wind_low) / 2.0))

    print(f"{'='*60}")
    print("Stacking Ensemble Test Set Performance")
    print(f"{'='*60}")
    print(f"Classification Accuracy : {acc:.2%}")
    print(f"Classification Macro-F1 : {f1:.4f}")
    print(f"Wind Speed MAE          : {mae:.2f} knots")
    print(f"Wind Speed RMSE         : {rmse:.2f} knots")
    print(f"Wind Speed R² Score     : {r2:.4f}")
    print(f"Mean 90% Confidence Band: ± {avg_uncertainty:.2f} knots")
    print(f"{'='*60}\n")

    target_names = [c["name"] for c in cfg.get("categories", [])]
    unique_labels = sorted(np.unique(np.concatenate([y_cat_test, cat_preds])))
    active_target_names = [target_names[i] for i in unique_labels if i < len(target_names)]

    print("Detailed Classification Report:")
    print(classification_report(y_cat_test, cat_preds, target_names=active_target_names, zero_division=0))

    # ---------------- Automatic Post-Training Evaluation & Graph Generation ----------------
    print(f"\n{'='*70}\nAUTO-GENERATING SENSORY DIAGNOSTIC EVALUATION & VISUALIZATION GRAPHS\n{'='*70}")
    eval_dir = Path(paths.get("output_eval_dir", "models/evaluation_results"))
    eval_dir.mkdir(parents=True, exist_ok=True)
    from src.evaluation.evaluate_all import evaluate_sensory_model
    evaluate_sensory_model(cfg, eval_dir)
    print(f"Sensory evaluation figures and metrics successfully updated in: {eval_dir.resolve()}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cyclone Sensory Stacking Ensemble")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--csv", type=str, default=None, help="Custom path to sensory CSV")
    args = parser.parse_args()

    train_sensory_model(config_path=args.config, custom_csv=args.csv)
