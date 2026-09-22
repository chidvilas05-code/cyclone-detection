"""
Version 2 Multi-Task Spatiotemporal Cyclone Forecaster Architecture.
Inherits the proven Dual-Stream ConvNeXt + Cross-Attention + Delta-BiGRU foundation
and equips it with 5 dedicated physical and hazard forecasting heads:
  1. WMO Category Classifier (5 tiers)
  2. Normalized Sustained Wind Speed (knots)
  3. Minimum Central Surface Pressure (hPa)
  4. Future Evolvement Vector at +6h and +12h (Δlat, Δlon, Δwind)
  5. Danger Area Extent Radii (30-kt gale & 50-kt storm radii)
  6. Landfall Probability & Shore ETA
"""

import math
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.spatiotemporal_classifier import CrossAttentionStreamFusion, PositionalEncoding


class MultiTaskSpatiotemporalCycloneModel(nn.Module):
    """
    Version 2 Multi-Task Spatiotemporal Forecaster.
    Transforms equal-interval satellite sequences into complete atmospheric,
    intensity, trajectory, and disaster impact intelligence.
    """
    def __init__(
        self,
        spatial_backbone: Optional[str] = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = 5,
        temporal_engine: str = "gru",
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
        seq_length: int = 4,
        eye_crop_ratio: float = 0.50
    ):
        super().__init__()
        self.eye_crop_ratio = eye_crop_ratio
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.temporal_engine_type = temporal_engine

        # Backbone instantiation helper
        def make_backbone(name: str):
            try:
                import timm
                b = timm.create_model(name, pretrained=pretrained, num_classes=0)
                dim = b.num_features
                return b, dim
            except Exception:
                from torchvision.models import resnet50, ResNet50_Weights
                w = ResNet50_Weights.DEFAULT if pretrained else None
                b = resnet50(weights=w)
                dim = b.fc.in_features
                b.fc = nn.Identity()
                return b, dim

        # 1. Spatial Dual-Stream Encoders
        self.eye_expert, eye_dim = make_backbone(spatial_backbone)
        self.synoptic_expert, synoptic_dim = make_backbone(spatial_backbone)

        # 2. Eyewall Organization Gate
        self.eye_gate = nn.Sequential(
            nn.Linear(eye_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 3. Interactive Cross-Attention Fusion
        self.cross_scale_fusion = CrossAttentionStreamFusion(
            eye_dim=eye_dim,
            synoptic_dim=synoptic_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout / 2.0
        )

        # 4. Explicit Temporal Difference Projection
        # Fuses [current state z_t, step delta (z_t - z_t-1), window delta (z_t - z_0)]
        self.change_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout / 2.0)
        )

        # 5. Temporal Sequence Engine (Delta-BiGRU / Transformer)
        if temporal_engine.lower() == "transformer":
            self.pos_encoder = PositionalEncoding(d_model=hidden_dim, max_len=16)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.pos_encoder = nn.Identity()
            self.temporal_encoder = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )

        # 6. Temporal Attention Aggregator
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # ============================================================
        # Multi-Task Prediction Heads (Attached to Latent Vector z_t)
        # ============================================================

        # Head 1: WMO Category (5 tiers)
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

        # Head 2: Sustained Wind Speed (knots)
        self.wind_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.wind_head[-1].bias.data.fill_(0.0)

        # Head 3: Intensification Trend (Weakening / Steady / Intensifying)
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3)
        )

        # Head 4: Central Pressure (hPa) - Atmospheric Sensory Head
        self.pressure_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.pressure_head[-1].bias.data.fill_(0.0)

        # Head 5: Future Evolvement (+6h and +12h Forecast)
        # Outputs: [Δlat_6h, Δlon_6h, Δwind_6h, Δlat_12h, Δlon_12h, Δwind_12h]
        self.evolve_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2.0),
            nn.Linear(128, 6)
        )
        self.evolve_head[-1].bias.data.fill_(0.0)

        # Head 6: Danger Area Extent Radii [Norm_R30, Norm_R50]
        self.danger_area_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()  # outputs in [0, 1] normalized by MAX radii
        )

        # Head 7: Landfall Probability & ETA [P_landfall (logit), Norm_ETA (0-1)]
        self.landfall_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        )

    def extract_core_eye_crop(self, x: torch.Tensor) -> torch.Tensor:
        """Dynamically centers and crops the eyewall region."""
        N, C, H, W = x.shape
        crop_h = int(H * self.eye_crop_ratio)
        crop_w = int(W * self.eye_crop_ratio)
        start_y = (H - crop_h) // 2
        start_x = (W - crop_w) // 2
        core = x[:, :, start_y:start_y + crop_h, start_x:start_x + crop_w]
        return F.interpolate(core, size=(H, W), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, K, C, H, W)
        Returns:
          Dict containing:
            'logits': (B, 5) Category logits
            'pred_norm_wind': (B, 1) Normalized sustained wind
            'pred_trend': (B, 3) Intensity trend logits
            'pred_norm_pressure': (B, 1) Normalized central pressure
            'pred_evolve': (B, 6) [Δlat_6, Δlon_6, Δw_6, Δlat_12, Δlon_12, Δw_12]
            'pred_danger_radii': (B, 2) [Norm_R30, Norm_R50]
            'pred_landfall': (B, 1) Landfall probability logit
            'pred_landfall_eta': (B, 1) Landfall ETA
        """
        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)

        # 1. Dual-Stream Spatial Encoding
        eye_crop = self.extract_core_eye_crop(x_flat)
        f_eye = self.eye_expert(eye_crop)
        f_synoptic = self.synoptic_expert(x_flat)

        # Dynamic Eyewall Organization Gating
        gate = self.eye_gate(f_eye)
        f_eye_gated = f_eye * gate

        # Bidirectional Cross-Attention
        f_fused = self.cross_scale_fusion(f_eye_gated, f_synoptic)  # (B*K, hidden_dim)
        seq_features = f_fused.view(B, K, self.hidden_dim)

        # 2. Explicit Physical Difference Trajectory
        z_t = seq_features
        z_curr = z_t[:, -1:, :]
        z_step_diff = z_t - torch.roll(z_t, shifts=1, dims=1)
        z_step_diff[:, 0, :] = 0.0
        z_window_diff = z_t - z_t[:, 0:1, :]

        augmented_seq = torch.cat([z_t, z_step_diff, z_window_diff], dim=-1)
        projected_seq = self.change_projection(augmented_seq)  # (B, K, hidden_dim)

        # 3. Temporal Sequence Processing
        if self.temporal_engine_type.lower() == "transformer":
            pe_seq = self.pos_encoder(projected_seq)
            temporal_out = self.temporal_encoder(pe_seq)
        else:
            temporal_out, _ = self.temporal_encoder(projected_seq)

        # 4. Temporal Attention Aggregation
        attn_weights = F.softmax(self.temporal_attn(temporal_out), dim=1)
        pooled = (temporal_out * attn_weights).sum(dim=1)  # (B, hidden_dim)

        # 5. Multi-Task Output Predictions
        logits = self.classifier_head(pooled)
        pred_norm_wind = self.wind_head(pooled)
        pred_trend = self.trend_head(pooled)
        pred_norm_pressure = self.pressure_head(pooled)
        pred_evolve = self.evolve_head(pooled)
        pred_danger = self.danger_area_head(pooled)
        landfall_raw = self.landfall_head(pooled)

        pred_landfall_logit = landfall_raw[:, 0:1]
        pred_landfall_eta = F.relu(landfall_raw[:, 1:2])

        return {
            "logits": logits,
            "pred_norm_wind": pred_norm_wind,
            "pred_trend": pred_trend,
            "pred_norm_pressure": pred_norm_pressure,
            "pred_evolve": pred_evolve,
            "pred_danger_radii": pred_danger,
            "pred_landfall": pred_landfall_logit,
            "pred_landfall_eta": pred_landfall_eta
        }

    def load_from_v1_checkpoint(self, checkpoint_path: str, device: torch.device):
        """Loads shared spatial and temporal weights from a Version 1 checkpoint."""
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            print(f"[Model V2] Checkpoint {checkpoint_path} not found. Starting with fresh weights.")
            return

        ckpt = torch.load(str(ckpt_p), map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Filter out keys that match exact shapes
        model_dict = self.state_dict()
        transferred = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                transferred[k] = v

        model_dict.update(transferred)
        self.load_state_dict(model_dict)
        print(f"[Model V2] Successfully inherited {len(transferred)} layer parameters from Version 1 ({checkpoint_path})!")
