"""
Advanced Meteorological Sensory & Atmospheric Data Processor.
Supports full NOAA Global IBTrACS v4 (700,000+ track records), SHIPS diagnostics, and Kaggle datasets.
Applies domain physics transformations: Pressure Deficit, Coriolis Parameter, Thermodynamic MPI,
Distance to Land Decays, and Quantile Targets.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# Comprehensive Canonical Feature Alias Mappings for Global IBTrACS & Atmospheric Datasets
ALIAS_MAP = {
    "central_pressure": [
        "wmo_pres", "usa_pres", "reunion_pres", "tokyo_pres", "newdelhi_pres", "bom_pres", "cma_pres",
        "pressure", "central_pressure", "min_pressure", "pres", "slp", "barometric_pressure"
    ],
    "wind_speed": [
        "wmo_wind", "usa_wind", "reunion_wind", "tokyo_wind", "newdelhi_wind", "bom_wind", "cma_wind",
        "wind_speed", "wind", "intensity", "vmax", "max_wind", "knots", "kt"
    ],
    "sst": ["sst", "sea_surface_temp", "sea_surface_temperature", "ocean_temp", "temp", "sst_c"],
    "vertical_wind_shear": ["shear", "wind_shear", "vws", "vertical_wind_shear", "shear_kt"],
    "relative_humidity": ["rh", "relative_humidity", "humidity", "rh_700", "rh_500", "rhum"],
    "translation_speed": ["storm_speed", "translation_speed", "speed", "storm_dir_speed", "motion_speed"],
    "distance_to_land": ["dist2land", "distance_to_land", "dist_to_land", "landfall_dist"],
    "latitude": ["lat", "latitude"],
    "longitude": ["lon", "long", "longitude"],
}


class SensoryDataProcessor:
    """Preprocesses sensory & atmospheric features for multi-model stacking ensembles."""

    def __init__(
        self,
        categories_config: Optional[List[Dict[str, Any]]] = None,
        save_dir: str = "models/sensory_model"
    ):
        self.categories_config = categories_config or [
            {"id": 0, "name": "Depression", "min_knots": 0, "max_knots": 33},
            {"id": 1, "name": "Cyclonic Storm", "min_knots": 34, "max_knots": 47},
            {"id": 2, "name": "Severe Cyclonic Storm", "min_knots": 48, "max_knots": 63},
            {"id": 3, "name": "Very Severe Cyclonic Storm", "min_knots": 64, "max_knots": 89},
            {"id": 4, "name": "Extremely Severe / Super Cyclone", "min_knots": 90, "max_knots": 250},
        ]
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.feature_columns: List[str] = []
        self.fitted = False

    def _resolve_column(self, df: pd.DataFrame, alias_key: str) -> Optional[str]:
        """Finds column in dataframe matching known aliases."""
        df_cols_lower = {col.lower(): col for col in df.columns}
        for alias in ALIAS_MAP.get(alias_key, [alias_key]):
            if alias.lower() in df_cols_lower:
                return df_cols_lower[alias.lower()]
        return None

    def load_and_clean(self, csv_path: str) -> pd.DataFrame:
        """Loads and sanitizes cyclone sensory CSV (auto-handles multi-line IBTrACS headers)."""
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"Sensory file not found at '{csv_path}'")

        print(f"[Sensory Loader] Ingesting: {csv_file.name} (File Size: {csv_file.stat().st_size / (1024**2):.1f} MB)")

        # IBTrACS files have units in the second line (e.g. 'kt', 'mb')
        try:
            sample_df = pd.read_csv(csv_file, nrows=3)
            first_val = str(sample_df.iloc[0].values[0])
            if any(unit in first_val.lower() for unit in ["kt", "mb", "hpa", "deg", "year"]):
                df = pd.read_csv(csv_file, skiprows=[1], low_memory=False)
            else:
                df = pd.read_csv(csv_file, low_memory=False)
        except Exception:
            df = pd.read_csv(csv_file, low_memory=False)

        # Standardize matching columns
        clean_dict = {}
        for canonical_name in ALIAS_MAP.keys():
            matched_col = self._resolve_column(df, canonical_name)
            if matched_col:
                series = pd.to_numeric(df[matched_col], errors="coerce")
                clean_dict[canonical_name] = series

        if not clean_dict:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                raise ValueError("No numeric sensory columns could be parsed from the provided CSV.")
            return numeric_df

        clean_df = pd.DataFrame(clean_dict)

        # If wind speed is missing, derive via Atkinson-Holliday relation from pressure
        if "wind_speed" not in clean_df or clean_df["wind_speed"].dropna().empty:
            if "central_pressure" in clean_df:
                delta_p = np.maximum(0, 1013.25 - clean_df["central_pressure"])
                clean_df["wind_speed"] = 6.7 * (delta_p ** 0.644)
            else:
                raise ValueError("Dataset must contain wind_speed or central_pressure to derive intensity.")

        # Drop invalid wind rows
        clean_df = clean_df.dropna(subset=["wind_speed"])
        clean_df = clean_df[clean_df["wind_speed"] > 0].copy()

        # If SST is missing (e.g. IBTrACS raw), derive physically realistic SST proxy from latitude
        if "sst" not in clean_df or clean_df["sst"].dropna().empty:
            if "latitude" in clean_df:
                lat_abs = clean_df["latitude"].abs()
                # Equatorial warm pool proxy: 29.5°C at equator, decaying towards poles
                clean_df["sst"] = np.clip(30.0 - 0.25 * (lat_abs ** 1.1), 20.0, 31.5)

        # If Wind Shear is missing, provide climatological distribution
        if "vertical_wind_shear" not in clean_df or clean_df["vertical_wind_shear"].dropna().empty:
            if "latitude" in clean_df:
                clean_df["vertical_wind_shear"] = np.clip(8.0 + 0.4 * clean_df["latitude"].abs(), 4.0, 45.0)

        # If distance_to_land is missing, provide open-ocean default (300 km)
        if "distance_to_land" not in clean_df or clean_df["distance_to_land"].dropna().empty:
            clean_df["distance_to_land"] = 300.0

        # Compute Category Label
        def map_category(wind):
            for cat in self.categories_config:
                if cat["min_knots"] <= wind <= cat["max_knots"]:
                    return cat["id"]
            return len(self.categories_config) - 1

        clean_df["category"] = clean_df["wind_speed"].apply(map_category)

        # Meteorological Physics Feature Engineering
        clean_df = self.engineer_physics_features(clean_df)
        return clean_df

    @staticmethod
    def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derives advanced thermodynamic and kinematic meteorological predictors."""
        df = df.copy()

        # 1. Barometric Pressure Deficit (Tangential Acceleration Driver)
        if "central_pressure" in df:
            df["pressure_deficit"] = np.maximum(0.0, 1013.25 - df["central_pressure"])
            df["empirical_potential_wind"] = 6.7 * (df["pressure_deficit"] ** 0.644)

        # 2. Coriolis Parameter (Planetary Vorticity)
        if "latitude" in df:
            omega = 7.2921e-5
            lat_rad = np.radians(df["latitude"].abs())
            df["coriolis_f"] = 2 * omega * np.sin(lat_rad) * 1e4
            df["distance_from_equator"] = df["latitude"].abs()

        # 3. Thermodynamic Maximum Potential Intensity (MPI Proxy)
        if "pressure_deficit" in df and "sst" in df:
            thermal_potential = np.maximum(0.0, df["sst"] - 26.0)
            df["ocean_heat_index"] = df["pressure_deficit"] * thermal_potential
            df["thermodynamic_mpi"] = 55.0 * np.sqrt(thermal_potential / 3.0 + 1e-5) + 0.35 * df["pressure_deficit"]

        # 4. Ventilation / Shear Ratio
        if "vertical_wind_shear" in df and "sst" in df:
            df["shear_sst_ratio"] = df["vertical_wind_shear"] / np.maximum(1.0, df["sst"])

        # 5. Proximity to Land Decay Factor
        if "distance_to_land" in df:
            df["land_friction_decay"] = np.exp(-df["distance_to_land"].clip(lower=0.0) / 150.0)

        return df

    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Splits into Train and Test sets with stratified category sampling."""
        exclude = {"category", "wind_speed"}
        self.feature_columns = [col for col in df.columns if col not in exclude]

        X = df[self.feature_columns].copy()
        for col in self.feature_columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        y_cat = df["category"]
        y_wind = df["wind_speed"]

        X_train, X_test, y_cat_train, y_cat_test, y_wind_train, y_wind_test = train_test_split(
            X, y_cat, y_wind, test_size=test_size, random_state=random_state, stratify=y_cat
        )

        self.fitted = True
        metadata = {
            "feature_columns": self.feature_columns,
            "categories_config": self.categories_config
        }
        joblib.dump(metadata, self.save_dir / "sensory_metadata.joblib")

        return X_train, X_test, y_cat_train, y_cat_test, y_wind_train, y_wind_test

    def transform_single_input(self, input_dict: Dict[str, float]) -> np.ndarray:
        """Transforms a single observation dictionary with physics feature engineering."""
        metadata = joblib.load(self.save_dir / "sensory_metadata.joblib")
        feature_cols = metadata["feature_columns"]

        df_single = pd.DataFrame([input_dict])
        df_engineered = self.engineer_physics_features(df_single)

        values = []
        for col in feature_cols:
            val = df_engineered[col].iloc[0] if col in df_engineered else np.nan
            values.append(val)

        return np.array(values, dtype=np.float32).reshape(1, -1)
