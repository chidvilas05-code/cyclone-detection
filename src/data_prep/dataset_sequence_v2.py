"""
Version 2 Spatiotemporal Cyclone Sequence Dataset.
Extracts multi-task supervision targets from paired Digital Typhoon and IBTrACS archives:
  1. Current WMO Category (5 tiers)
  2. Sustained Wind Speed (knots)
  3. Minimum Central Surface Pressure (hPa)
  4. Eye Geographic Coordinates (lat, lng)
  5. Future Evolvement Vector at +6h and +12h (Δlat, Δlon, Δwind)
  6. Coast Landfall Status & Landfall ETA (hours)
  7. Danger Area Extent Radii (30-kt gale & 50-kt storm radii)
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import random
import math

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.data_prep.dataset_sequence import CoherentSequenceTransform, WIND_MEAN, WIND_STD

# Normalization constants
PRES_MEAN = 988.0
PRES_STD = 22.0
COORD_DELTA_STD = 1.5     # degrees latitude/longitude per 6-12h
DANGER_R30_MAX = 500.0    # km
DANGER_R50_MAX = 250.0    # km


class CycloneSequenceDatasetV2(Dataset):
    """
    Version 2 Spatiotemporal Dataset.
    Loads 4-frame temporal sequences and constructs paired multi-task targets:
      - Category & Continuous Wind
      - Central Pressure (Atmospheric Sensory Target)
      - Future Motion & Intensity Forecast (+6h, +12h)
      - Landfall ETA & Coast Danger
      - Wind Radii Danger Buffer Zones (R30, R50)
    """
    def __init__(
        self,
        data_dir: Optional[str] = None,
        track_csv: Optional[str] = None,
        seq_length: int = 4,
        frame_step: int = 3,
        stride: int = 2,
        img_size: int = 256,
        min_year: Optional[int] = 2000,
        max_year: Optional[int] = None,
        is_train: bool = True,
        mock_num_samples: Optional[int] = None
    ):
        self.seq_length = seq_length
        self.frame_step = max(1, frame_step)
        self.stride = max(1, stride)
        self.img_size = img_size
        self.min_year = min_year
        self.max_year = max_year
        self.is_train = is_train
        self.transform = CoherentSequenceTransform(img_size=img_size, is_train=is_train)
        self.samples: List[Dict[str, Any]] = []

        if mock_num_samples is not None:
            self._create_mock_dataset(mock_num_samples, img_size)
        elif data_dir is not None and os.path.exists(data_dir):
            self._index_dataset(Path(data_dir), track_csv)
        else:
            self._create_mock_dataset(100, img_size)

    def _create_mock_dataset(self, n_samples: int, img_size: int):
        """Generates realistic synthetic multi-task sequences for dry-runs and pipeline testing."""
        for i in range(n_samples):
            base_wind = float(np.random.uniform(25.0, 110.0))
            wind_trend = float(np.random.uniform(-4.0, 5.0))
            winds = [max(15.0, base_wind + wind_trend * step * self.frame_step) for step in range(self.seq_length)]
            curr_wind = winds[-1]

            if curr_wind < 34: cat = 0
            elif curr_wind < 48: cat = 1
            elif curr_wind < 64: cat = 2
            elif curr_wind < 90: cat = 3
            else: cat = 4

            # Atkinson-Holliday pressure relationship
            curr_pressure = max(900.0, 1010.0 - ((curr_wind / 6.7) ** (1.0 / 0.644)))

            delta_w = curr_wind - winds[0]
            if delta_w < -8.0: trend = 0
            elif delta_w > 8.0: trend = 2
            else: trend = 1

            # Simulated future evolution (+6h, +12h)
            d_lat_6 = float(np.random.uniform(-0.8, 1.2))
            d_lon_6 = float(np.random.uniform(-1.5, 0.5))
            d_wind_6 = float(np.random.uniform(-6.0, 8.0))

            d_lat_12 = d_lat_6 * 1.9 + float(np.random.uniform(-0.3, 0.3))
            d_lon_12 = d_lon_6 * 1.9 + float(np.random.uniform(-0.3, 0.3))
            d_wind_12 = d_wind_6 * 1.8 + float(np.random.uniform(-4.0, 4.0))

            r30 = min(DANGER_R30_MAX, max(0.0, curr_wind * 3.5 + np.random.uniform(-20, 20)))
            r50 = min(DANGER_R50_MAX, max(0.0, (curr_wind - 35.0) * 2.0)) if curr_wind >= 48 else 0.0
            landfall = 1.0 if np.random.random() < 0.15 else 0.0
            landfall_eta = float(np.random.uniform(6.0, 36.0)) if landfall > 0.5 else 48.0

            self.samples.append({
                "type": "mock",
                "storm_id": f"MOCK_{i:04d}",
                "year": 2023,
                "winds": winds,
                "category": cat,
                "target_wind": curr_wind,
                "norm_wind": (curr_wind - WIND_MEAN) / WIND_STD,
                "pressure": curr_pressure,
                "norm_pressure": (curr_pressure - PRES_MEAN) / PRES_STD,
                "target_trend": trend,
                "evolve_target": [d_lat_6, d_lon_6, d_wind_6, d_lat_12, d_lon_12, d_wind_12],
                "danger_radii": [r30 / DANGER_R30_MAX, r50 / DANGER_R50_MAX],
                "landfall": landfall,
                "landfall_eta": landfall_eta / 48.0,
                "lat": float(np.random.uniform(10.0, 35.0)),
                "lng": float(np.random.uniform(120.0, 150.0)),
                "seed": i + (0 if self.is_train else 10000)
            })

    def _index_dataset(self, data_path: Path, track_csv: Optional[str]):
        """
        Indexes storm sequence frames paired with multi-task metadata:
          data_dir/
            ├── image_png/{storm_id}/
            └── metadata/metadata/{storm_id}.csv
        """
        possible_img_roots = [
            data_path / "image_png" / "image_png",
            data_path / "image_png",
            data_path / "images",
            data_path
        ]
        img_root = None
        for pr in possible_img_roots:
            if pr.exists() and any(d.is_dir() and (any(d.glob("*.png")) or any(d.glob("*.jpg"))) for d in pr.iterdir() if d.is_dir()):
                img_root = pr
                break
        if img_root is None:
            img_root = data_path

        possible_meta_roots = [
            data_path / "metadata" / "metadata",
            data_path / "metadata",
            data_path
        ]
        meta_root = None
        for pm in possible_meta_roots:
            if pm.exists() and any(f.suffix == ".csv" for f in pm.iterdir() if f.is_file()):
                meta_root = pm
                break

        storm_dirs = [d for d in img_root.iterdir() if d.is_dir()]

        for s_dir in sorted(storm_dirs):
            s_year = 2000
            try:
                s_year = int(s_dir.name[:4])
                if self.min_year is not None and s_year < self.min_year:
                    continue
                if self.max_year is not None and s_year > self.max_year:
                    continue
            except Exception:
                pass

            img_files = sorted(list(s_dir.glob("*.png")) + list(s_dir.glob("*.jpg")))
            if len(img_files) < self.seq_length:
                continue

            # Load storm metadata CSV
            df_meta = None
            if meta_root:
                m_csv = meta_root / f"{s_dir.name}.csv"
                if m_csv.exists():
                    try:
                        df_meta = pd.read_csv(m_csv)
                    except Exception:
                        df_meta = None

            if df_meta is None or len(df_meta) == 0:
                continue

            # Create quick lookup by image base name
            file_to_idx = {}
            for row_idx, row in df_meta.iterrows():
                f_raw = str(row.get("file_1", ""))
                f_clean = f_raw.replace(".h5", ".png").replace(".h5", ".jpg")
                file_to_idx[f_clean] = row_idx
                file_to_idx[f_raw] = row_idx

            step = self.frame_step
            min_span = (self.seq_length - 1) * step + 1

            if len(img_files) >= min_span:
                for start_idx in range(0, len(img_files) - min_span + 1, self.stride):
                    window_files = [img_files[start_idx + i * step] for i in range(self.seq_length)]
                    self._append_v2_sample(window_files, df_meta, file_to_idx, s_dir.name, s_year)
            else:
                adaptive_step = max(1, (len(img_files) - 1) // (self.seq_length - 1))
                span = (self.seq_length - 1) * adaptive_step + 1
                for start_idx in range(0, len(img_files) - span + 1, self.stride):
                    window_files = [img_files[start_idx + i * adaptive_step] for i in range(self.seq_length)]
                    self._append_v2_sample(window_files, df_meta, file_to_idx, s_dir.name, s_year)

    def _append_v2_sample(
        self,
        window_files: List[Path],
        df_meta: pd.DataFrame,
        file_to_idx: Dict[str, int],
        storm_id: str,
        year: int
    ):
        winds, pressures, lats, lngs = [], [], [], []
        curr_file = window_files[-1]
        curr_row_idx = file_to_idx.get(curr_file.name, file_to_idx.get(curr_file.stem, None))

        if curr_row_idx is None:
            curr_row_idx = min(len(df_meta) - 1, len(df_meta) // 2)

        curr_row = df_meta.iloc[curr_row_idx]

        # 1. Current Intensity & Pressure
        curr_wind = float(curr_row.get("wind", 0.0))
        curr_pres = float(curr_row.get("pressure", 1010.0))
        curr_lat = float(curr_row.get("lat", 20.0))
        curr_lng = float(curr_row.get("lng", 130.0))

        if curr_wind <= 0.0 and curr_pres < 1010.0:
            curr_wind = max(15.0, 6.7 * ((1010.0 - curr_pres) ** 0.644))
        elif curr_wind <= 0.0:
            curr_wind = 25.0

        if curr_pres >= 1013.0 or curr_pres <= 870.0:
            curr_pres = max(890.0, 1010.0 - ((curr_wind / 6.7) ** (1.0 / 0.644)))

        # Category
        if curr_wind < 34.0: curr_cat = 0
        elif curr_wind < 48.0: curr_cat = 1
        elif curr_wind < 64.0: curr_cat = 2
        elif curr_wind < 90.0: curr_cat = 3
        else: curr_cat = 4

        # 2. Window Trend
        first_file = window_files[0]
        first_idx = file_to_idx.get(first_file.name, max(0, curr_row_idx - (self.seq_length - 1) * self.frame_step))
        first_wind = float(df_meta.iloc[first_idx].get("wind", curr_wind))
        if first_wind <= 0: first_wind = curr_wind

        delta_w = curr_wind - first_wind
        if delta_w < -8.0: trend = 0
        elif delta_w > 8.0: trend = 2
        else: trend = 1

        # 3. Future Evolvement (+6h and +12h forecast)
        step_6h = 6
        step_12h = 12

        idx_6h = min(len(df_meta) - 1, curr_row_idx + step_6h)
        idx_12h = min(len(df_meta) - 1, curr_row_idx + step_12h)

        row_6h = df_meta.iloc[idx_6h]
        row_12h = df_meta.iloc[idx_12h]

        d_lat_6 = float(row_6h.get("lat", curr_lat)) - curr_lat
        d_lon_6 = float(row_6h.get("lng", curr_lng)) - curr_lng
        w_6h = float(row_6h.get("wind", curr_wind))
        if w_6h <= 0: w_6h = curr_wind
        d_wind_6 = w_6h - curr_wind

        d_lat_12 = float(row_12h.get("lat", curr_lat)) - curr_lat
        d_lon_12 = float(row_12h.get("lng", curr_lng)) - curr_lng
        w_12h = float(row_12h.get("wind", curr_wind))
        if w_12h <= 0: w_12h = curr_wind
        d_wind_12 = w_12h - curr_wind

        # 4. Danger Areas (30-kt and 50-kt radii)
        r30_nm = float(curr_row.get("long30", 0.0))
        r50_nm = float(curr_row.get("long50", 0.0))
        r30_km = r30_nm * 1.852
        r50_km = r50_nm * 1.852

        # 5. Landfall & Time to Reach Shore
        landfall_flag = 1.0 if float(curr_row.get("landfall", 0.0)) > 0.5 else 0.0
        
        landfall_eta_hours = 48.0
        future_slice = df_meta.iloc[curr_row_idx : min(len(df_meta), curr_row_idx + 48)]
        if not future_slice.empty:
            landfall_indices = future_slice[future_slice["landfall"] > 0.5].index
            if len(landfall_indices) > 0:
                landfall_eta_hours = float(landfall_indices[0] - curr_row_idx)

        self.samples.append({
            "type": "files",
            "storm_id": storm_id,
            "year": year,
            "paths": [str(p) for p in window_files],
            "category": curr_cat,
            "target_wind": curr_wind,
            "norm_wind": (curr_wind - WIND_MEAN) / WIND_STD,
            "pressure": curr_pres,
            "norm_pressure": (curr_pres - PRES_MEAN) / PRES_STD,
            "target_trend": trend,
            "evolve_target": [d_lat_6, d_lon_6, d_wind_6, d_lat_12, d_lon_12, d_wind_12],
            "danger_radii": [min(1.0, r30_km / DANGER_R30_MAX), min(1.0, r50_km / DANGER_R50_MAX)],
            "landfall": landfall_flag,
            "landfall_eta": min(1.0, landfall_eta_hours / 48.0),
            "lat": curr_lat,
            "lng": curr_lng
        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        if item["type"] == "mock":
            k = self.seq_length
            h = self.img_size
            w = self.img_size
            seq_tensor = torch.zeros(k, 3, h, w, dtype=torch.float32)

            center = h // 2
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            mask = np.exp(-dist / (h / 4.0))

            for t_idx in range(k):
                frame_np = np.stack([mask, mask * 0.9, mask * 1.1], axis=0).astype(np.float32)
                seq_tensor[t_idx] = torch.from_numpy(frame_np)

            return {
                "sequence": seq_tensor,
                "category": torch.tensor(item["category"], dtype=torch.long),
                "wind_speed": torch.tensor(item["target_wind"], dtype=torch.float32),
                "norm_wind": torch.tensor(item["norm_wind"], dtype=torch.float32),
                "trend": torch.tensor(item["target_trend"], dtype=torch.long),
                "pressure": torch.tensor(item["pressure"], dtype=torch.float32),
                "norm_pressure": torch.tensor(item["norm_pressure"], dtype=torch.float32),
                "evolve_target": torch.tensor(item["evolve_target"], dtype=torch.float32),
                "danger_radii": torch.tensor(item["danger_radii"], dtype=torch.float32),
                "landfall": torch.tensor(item["landfall"], dtype=torch.float32),
                "landfall_eta": torch.tensor(item["landfall_eta"], dtype=torch.float32),
                "lat": torch.tensor(item["lat"], dtype=torch.float32),
                "lng": torch.tensor(item["lng"], dtype=torch.float32),
                "storm_id": item["storm_id"]
            }

        raw_images = []
        for path_str in item["paths"]:
            try:
                img = Image.open(path_str).convert("RGB")
            except Exception:
                img = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
            raw_images.append(img)

        seq_tensor = self.transform(raw_images)

        return {
            "sequence": seq_tensor,
            "category": torch.tensor(item["category"], dtype=torch.long),
            "wind_speed": torch.tensor(item["target_wind"], dtype=torch.float32),
            "norm_wind": torch.tensor(item["norm_wind"], dtype=torch.float32),
            "trend": torch.tensor(item["target_trend"], dtype=torch.long),
            "pressure": torch.tensor(item["pressure"], dtype=torch.float32),
            "norm_pressure": torch.tensor(item["norm_pressure"], dtype=torch.float32),
            "evolve_target": torch.tensor(item["evolve_target"], dtype=torch.float32),
            "danger_radii": torch.tensor(item["danger_radii"], dtype=torch.float32),
            "landfall": torch.tensor(item["landfall"], dtype=torch.float32),
            "landfall_eta": torch.tensor(item["landfall_eta"], dtype=torch.float32),
            "lat": torch.tensor(item["lat"], dtype=torch.float32),
            "lng": torch.tensor(item["lng"], dtype=torch.float32),
            "storm_id": item["storm_id"]
        }
