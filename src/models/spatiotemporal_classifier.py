"""
Dual-Stream Spatiotemporal Cyclone Sequence Model Architecture.
Covers BOTH:
  1. Micro-Core Eye Stream: Zooms into the central 50% eyewall & CDO for each frame in the sequence.
  2. Synoptic Full-Picture Stream: Ingests the complete 1000 km outer rainband and synoptic field.
Followed by cross-scale attention gating and a Temporal Transformer to model intensification dynamics.
"""

import math
from typing import Tuple, Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal temporal positional encoding for satellite sequence steps."""
    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class DualStreamSpatiotemporalCycloneModel(nn.Module):
    """
    Dual-Stream Spatiotemporal Architecture:
      - Stream 1 (Core Eye Expert): Ingests the 50% central eye / eyewall zoom for each frame.
      - Stream 2 (Synoptic Expert): Ingests the full 100% synoptic satellite image.
      - Dynamic Cross-Scale Gating per frame: Reconciles eye vs synoptic features based on storm maturity.
      - Temporal Transformer: Models sequential time dynamics [t-K+1, ..., t] across both streams.
      - Tri-Head Predictor: Category (5 tiers) + Wind Speed (kt) + Intensity Trend (Weakening/Steady/Intensifying).
    """
    def __init__(
        self,
        eye_backbone_name: str = "convnext_tiny",
        synoptic_backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = 5,
        temporal_engine: str = "transformer",
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.20,
        seq_length: int = 4,
        eye_crop_ratio: float = 0.50
    ):
        super().__init__()
        self.eye_crop_ratio = eye_crop_ratio
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.temporal_engine_type = temporal_engine

        # Helper to instantiate backbones
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

        # Stream 1: Core Eye Expert (CNN)
        self.eye_expert, eye_dim = make_backbone(eye_backbone_name)

        # Stream 2: Synoptic Global Expert (Full Picture)
        self.synoptic_expert, synoptic_dim = make_backbone(synoptic_backbone_name)

        # Adaptive Eye Gating per frame
        self.eye_gate = nn.Sequential(
            nn.Linear(eye_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Cross-Scale Spatial Fusion to project into temporal hidden_dim
        self.cross_scale_fusion = nn.Sequential(
            nn.Linear(eye_dim + synoptic_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout / 2.0)
        )

        # Temporal Sequence Engine
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

        # Temporal Attention Context Aggregator
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Multi-Task Predictor Heads
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.regression_head[-1].bias.data.fill_(50.0)

        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3)
        )

    def extract_central_eye_crop(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (N, C, H, W) where N = B * K
        Extracts the central 50% eyewall & CDO crop and upsamples back to (H, W).
        """
        h, w = x.shape[2], x.shape[3]
        crop_h = int(h * self.eye_crop_ratio)
        crop_w = int(w * self.eye_crop_ratio)
        ch_start = (h - crop_h) // 2
        cw_start = (w - crop_w) // 2
        eye_sub = x[:, :, ch_start : ch_start + crop_h, cw_start : cw_start + crop_w]
        return F.interpolate(eye_sub, size=(h, w), mode="bilinear", align_corners=False)

    def extract_spatiotemporal_features(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: Shape (B, K, C, H, W)
        Returns fused temporal feature sequence: Shape (B, K, hidden_dim)
        """
        B, K, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * K, C, H, W)

        # 1. Full Synoptic Picture Stream
        f_synoptic = self.synoptic_expert(x_flat)  # (B*K, synoptic_dim)

        # 2. Core Eye Zoom Stream
        x_eye = self.extract_central_eye_crop(x_flat)
        f_eye = self.eye_expert(x_eye)              # (B*K, eye_dim)

        # 3. Dynamic Eye Gating per frame
        alpha = 0.30 + 0.70 * self.eye_gate(f_eye)  # (B*K, 1)
        f_gated_eye = f_eye * alpha

        # 4. Cross-Scale Gated Fusion
        f_combined = torch.cat([f_gated_eye, f_synoptic], dim=-1)
        z_flat = self.cross_scale_fusion(f_combined) # (B*K, hidden_dim)

        return z_flat.view(B, K, self.hidden_dim)

    def forward(
        self,
        x_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass over sequence window covering both eye and full picture.
        Returns:
          logits: (B, num_classes)
          wind_speed: (B,)
          trend_logits: (B, 3)
        """
        # 1. Dual-stream spatial feature extraction per timestamp
        z_seq = self.extract_spatiotemporal_features(x_seq)  # (B, K, hidden_dim)

        # 2. Temporal modeling across timestamps
        if self.temporal_engine_type.lower() == "transformer":
            z_seq = self.pos_encoder(z_seq)
            temporal_out = self.temporal_encoder(z_seq)       # (B, K, hidden_dim)
        else:
            temporal_out, _ = self.temporal_encoder(z_seq)    # (B, K, hidden_dim)

        # 3. Temporal Context Aggregation
        attn_weights = F.softmax(self.temporal_attn(temporal_out), dim=1)  # (B, K, 1)
        context_vector = torch.sum(attn_weights * temporal_out, dim=1)     # (B, hidden_dim)

        final_frame_feat = temporal_out[:, -1, :]
        fused_summary = 0.5 * context_vector + 0.5 * final_frame_feat

        # 4. Multi-Task Heads
        logits = self.classifier_head(fused_summary)
        wind_speed = self.regression_head(fused_summary).squeeze(-1)
        trend_logits = self.trend_head(fused_summary)

        return logits, wind_speed, trend_logits


# Backward compatible alias
SpatiotemporalCycloneModel = DualStreamSpatiotemporalCycloneModel
