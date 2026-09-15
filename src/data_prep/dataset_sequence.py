"""
Equal-Interval Spatiotemporal Cyclone Sequence Dataset.
Constructs temporal sequence windows [t-K+1, ..., t] from satellite storm archives
(e.g., Digital Typhoon WP 1-hour / 30-minute intervals) for live decoding and forecasting.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import random

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class CoherentSequenceTransform:
    """
    Applies identical spatial augmentations across all K frames in a sequence
    to preserve physical rotation, vortex structure, and motion dynamics.
    """
    def __init__(
        self,
        img_size: int = 224,
        is_train: bool = True,
        rotation_degrees: float = 180.0
    ):
        self.img_size = img_size
        self.is_train = is_train
        self.rotation_degrees = rotation_degrees
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, img_sequence: List[Image.Image]) -> torch.Tensor:
        # Generate random parameters once per sequence
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees) if self.is_train else 0.0
        h_flip = (random.random() > 0.5) if self.is_train else False
        v_flip = (random.random() > 0.5) if self.is_train else False

        processed_frames = []
        for img in img_sequence:
            img = img.convert("RGB")
            img = TF.resize(img, (self.img_size, self.img_size))

            if self.is_train:
                if angle != 0.0:
                    img = TF.rotate(img, angle)
                if h_flip:
                    img = TF.hflip(img)
                if v_flip:
                    img = TF.vflip(img)

            t = TF.to_tensor(img)
            t = self.normalize(t)
            processed_frames.append(t)

        # Output shape: (K, 3, H, W)
        return torch.stack(processed_frames, dim=0)


class CycloneSequenceDataset(Dataset):
    """
    Spatiotemporal Dataset ingesting consecutive multi-frame windows of tropical cyclones.
    Compatible with Digital Typhoon WP (1-hour / 30-min intervals) and track archives.
    """
    def __init__(
        self,
        data_dir: Optional[str] = None,
        track_csv: Optional[str] = None,
        seq_length: int = 4,
        stride: int = 1,
        img_size: int = 224,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        is_train: bool = True,
        mock_num_samples: Optional[int] = None
    ):
        self.seq_length = seq_length
        self.stride = stride
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
        """Generates realistic synthetic sequences for dry-runs and integration tests."""
        for i in range(n_samples):
            base_wind = float(np.random.uniform(25.0, 110.0))
            wind_trend = float(np.random.uniform(-4.0, 5.0))
            winds = [max(15.0, base_wind + wind_trend * step) for step in range(self.seq_length)]
            curr_wind = winds[-1]

            if curr_wind < 34: cat = 0
            elif curr_wind < 48: cat = 1
            elif curr_wind < 64: cat = 2
            elif curr_wind < 90: cat = 3
            else: cat = 4

            delta = curr_wind - winds[0]
            if delta < -5.0: trend = 0
            elif delta > 5.0: trend = 2
            else: trend = 1

            self.samples.append({
                "type": "mock",
                "storm_id": f"MOCK_{i:04d}",
                "year": 2023,
                "winds": winds,
                "category": cat,
                "target_wind": curr_wind,
                "target_trend": trend,
                "seed": i + (0 if self.is_train else 10000)
            })

    def _index_dataset(self, data_path: Path, track_csv: Optional[str]):
        """
        Indexes storm sequence paths from Digital Typhoon WP directory structure:
        data_dir/
          ├── image_png/image_png/{storm_id}/ (or directly {storm_id}/)
          └── metadata/metadata/{storm_id}.csv
        """
        # 1. Resolve image root directory
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

        # 2. Resolve metadata root directory
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
        total_indexed_sequences = 0

        for s_dir in sorted(storm_dirs):
            # Apply year filtering if specified (e.g. min_year=2000, min_year=2015)
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

            # Check for matching metadata CSV
            meta_map = {}
            if meta_root:
                m_csv = meta_root / f"{s_dir.name}.csv"
                if m_csv.exists():
                    try:
                        df = pd.read_csv(m_csv)
                        for _, row in df.iterrows():
                            # Map filename to wind
                            f_name = str(row.get("file_1", "")).replace(".h5", ".png")
                            w_val = float(row.get("wind", 0.0))
                            p_val = float(row.get("pressure", 1010.0))
                            # Atkinson-Holliday pressure-wind relationship if wind is 0
                            if w_val <= 0.0 and p_val < 1010.0:
                                w_val = max(15.0, 6.7 * ((1010.0 - p_val) ** 0.644))
                            elif w_val <= 0.0:
                                w_val = 25.0
                            meta_map[f_name] = w_val
                    except Exception:
                        pass

            for start_idx in range(0, len(img_files) - self.seq_length + 1, self.stride):
                window_files = img_files[start_idx : start_idx + self.seq_length]
                
                # Determine wind speeds across the window
                winds = []
                for p in window_files:
                    w = meta_map.get(p.name, None)
                    if w is None:
                        w = 45.0  # Default nominal tropical storm wind
                    winds.append(w)

                curr_wind = float(winds[-1])
                
                # Category assignment (WMO 5 tiers)
                if curr_wind < 34.0:
                    curr_cat = 0
                elif curr_wind < 48.0:
                    curr_cat = 1
                elif curr_wind < 64.0:
                    curr_cat = 2
                elif curr_wind < 90.0:
                    curr_cat = 3
                else:
                    curr_cat = 4

                # Intensity trend calculation
                delta_w = curr_wind - winds[0]
                if delta_w < -5.0:
                    trend = 0  # Weakening
                elif delta_w > 5.0:
                    trend = 2  # Intensifying
                else:
                    trend = 1  # Steady

                self.samples.append({
                    "type": "files",
                    "storm_id": s_dir.name,
                    "year": s_year,
                    "paths": [str(p) for p in window_files],
                    "category": curr_cat,
                    "target_wind": curr_wind,
                    "target_trend": trend
                })
                total_indexed_sequences += 1

    def __len__(self) -> int:
        return len(self.samples)

    def _render_mock_cyclone_frame(self, wind: float, step: int, seed: int) -> Image.Image:
        np.random.seed(seed + step * 37)
        canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        center = (112, 112)
        angle = step * 25.0
        for arm in range(3):
            arm_angle = angle + arm * 120
            pts = []
            for r in range(15, 95, 5):
                theta = np.deg2rad(arm_angle + r * 2.2)
                x = int(center[0] + r * np.cos(theta))
                y = int(center[1] + r * np.sin(theta))
                pts.append([x, y])
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (180, 200, 255), 7)
        eye_r = max(5, int(18 - (wind / 15.0)))
        cv2.circle(canvas, center, int(eye_r * 2.2), (235, 240, 255), -1)
        cv2.circle(canvas, center, eye_r, (25, 25, 30), -1)
        cv2.GaussianBlur(canvas, (11, 11), 3.0, dst=canvas)
        return Image.fromarray(canvas)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        if sample["type"] == "mock":
            frames = []
            for step, w_val in enumerate(sample["winds"]):
                frame = self._render_mock_cyclone_frame(w_val, step, sample["seed"])
                frames.append(frame)
        else:
            frames = [Image.open(p) for p in sample["paths"]]

        seq_tensor = self.transform(frames)

        return {
            "sequence": seq_tensor,
            "category": torch.tensor(sample["category"], dtype=torch.long),
            "wind_speed": torch.tensor(sample["target_wind"], dtype=torch.float32),
            "trend": torch.tensor(sample["target_trend"], dtype=torch.long)
        }
