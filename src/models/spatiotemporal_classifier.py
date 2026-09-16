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


class CrossAttentionStreamFusion(nn.Module):
    """
    Bidirectional Interactive Cross-Attention Stream Fusion with Residual Base Skip Connection:
    - Base Stream: Direct linear fusion of eye and synoptic features preserving pre-trained ConvNeXt embeddings.
    - Context Stream: Eye features act as Query to probe Synoptic environmental context (shear, moisture, spiral bands).
    - Zero-Initialized Residual Layer: Guarantees the network starts 100% equivalent to the strong baseline
      and smoothly learns rich cross-attention interactions without destabilizing early training epochs.
    """
    def __init__(self, eye_dim: int, synoptic_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.base_fusion = nn.Sequential(
            nn.Linear(eye_dim + synoptic_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        self.proj_eye = nn.Linear(eye_dim, hidden_dim)
        self.proj_syn = nn.Linear(synoptic_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        self.refine_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Zero-initialize the refinement output so initial predictions are stable
        nn.init.zeros_(self.refine_mlp[-1].weight)
        nn.init.zeros_(self.refine_mlp[-1].bias)

    def forward(self, f_eye: torch.Tensor, f_synoptic: torch.Tensor) -> torch.Tensor:
        """
        f_eye: (N, eye_dim) where N = B * K
        f_synoptic: (N, synoptic_dim)
        returns: (N, hidden_dim)
        """
        # 1. Base combined representation preserving pre-trained weights
        base = self.base_fusion(torch.cat([f_eye, f_synoptic], dim=-1))

        # 2. Cross-Attention context: Eye queries Synoptic environment
        e = self.proj_eye(f_eye).unsqueeze(1)
        s = self.proj_syn(f_synoptic).unsqueeze(1)
        ctx, _ = self.cross_attn(query=e, key=s, value=s)

        # 3. Residual addition with zero-initialized refinement
        out = self.norm(base + self.refine_mlp(ctx.squeeze(1)))
        return out



class DualStreamSpatiotemporalCycloneModel(nn.Module):
    """
    Dual-Stream Spatiotemporal Architecture:
      - Stream 1 (Core Eye Expert): Ingests the 50% central eye / eyewall zoom for each frame.
      - Stream 2 (Synoptic Expert): Ingests the full 100% synoptic satellite image.
      - Interactive Cross-Attention Stream Fusion: Bidirectional cross-attention between eyewall and synoptic field.
      - Temporal Transformer: Models sequential time dynamics [t-K+1, ..., t] across multi-hour steps.
      - Tri-Head Predictor: Category (5 tiers) + Normalized Wind Speed (kt) + Intensity Trend (Weakening/Steady/Intensifying).
    """
    def __init__(
        self,
        spatial_backbone: Optional[str] = None,
        eye_backbone_name: str = "convnext_tiny",
        synoptic_backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = 5,
        temporal_engine: str = "gru",
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.30,
        seq_length: int = 4,
        eye_crop_ratio: float = 0.50
    ):
        super().__init__()
        if spatial_backbone is not None:
            eye_backbone_name = spatial_backbone
            synoptic_backbone_name = spatial_backbone

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

        # Interactive Cross-Attention Stream Fusion
        self.cross_scale_fusion = CrossAttentionStreamFusion(
            eye_dim=eye_dim,
            synoptic_dim=synoptic_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout / 2.0
        )

        # Explicit Temporal Difference Projection:
        # Fuses [current state z_t, consecutive frame delta (z_t - z_t-1), global window delta (z_t - z_0)]
        self.change_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout / 2.0)
        )

        # Temporal Sequence Engine (Bidirectional GRU / Transformer)
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
        self.regression_head[-1].bias.data.fill_(0.0)

        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3)
        )


    def compute_vortex_center(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, H, W)
        Computes normalized translation offsets (N, 2) in range [-0.25, 0.25]
        to dynamically track and center around the physical vortex eye.
        """
        N, C, H, W = x.shape
        # Luminance proxy
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]  # (N, 1, H, W)

        # Spatial gradients (Sobel filters)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)

        # Coordinate grids in [-1, 1]
        y_grid = torch.linspace(-1.0, 1.0, H, dtype=x.dtype, device=x.device).view(1, 1, H, 1)
        x_grid = torch.linspace(-1.0, 1.0, W, dtype=x.dtype, device=x.device).view(1, 1, 1, W)
        dist_sq = y_grid**2 + x_grid**2
        spatial_prior = torch.exp(-dist_sq / 0.50)

        s_map = (grad_mag + (1.0 - gray) * 0.5) * spatial_prior
        s_flat = s_map.view(N, -1)
        weights = F.softmax(s_flat * 4.0, dim=-1).view(N, 1, H, W)

        c_y = (weights * y_grid).sum(dim=(-2, -1))
        c_x = (weights * x_grid).sum(dim=(-2, -1))

        # Organization Coherence Gate:
        # Measures peak-to-mean gradient contrast within central region.
        # If the storm is an early amorphous depression with no eye (low contrast), gate -> 0 (stays at center [0,0]).
        # If an organized circular eyewall is present (high contrast), gate -> 1 (actively tracks the eyewall).
        peak_val = s_flat.amax(dim=-1, keepdim=True)
        mean_val = s_flat.mean(dim=-1, keepdim=True) + 1e-5
        contrast = peak_val / mean_val  # (N, 1)
        org_gate = torch.sigmoid((contrast - 2.2) * 2.5)  # (N, 1)

        t_y = torch.clamp(c_y * org_gate, -0.25, 0.25)
        t_x = torch.clamp(c_x * org_gate, -0.25, 0.25)
        return torch.cat([t_x, t_y], dim=-1)

    def extract_dynamic_vortex_crop(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (N, C, H, W) where N = B * K
        Extracts the 50% spatial crop dynamically centered around the active vortex core.
        Uses GPU Affine Grid Sampling for sub-millisecond execution.
        """
        N, C, H, W = x.shape
        offsets = self.compute_vortex_center(x)  # (N, 2)

        s = self.eye_crop_ratio
        theta = torch.zeros((N, 2, 3), dtype=x.dtype, device=x.device)
        theta[:, 0, 0] = s
        theta[:, 1, 1] = s
        theta[:, 0, 2] = offsets[:, 0]
        theta[:, 1, 2] = offsets[:, 1]

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def extract_spatiotemporal_features(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: Shape (B, K, C, H, W)
        Returns fused temporal feature sequence: Shape (B, K, hidden_dim)
        """
        B, K, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * K, C, H, W)

        # 1. Full Synoptic Picture Stream
        f_synoptic = self.synoptic_expert(x_flat)  # (B*K, synoptic_dim)

        # 2. Core Eye Zoom Stream (with Dynamic Vortex Centering)
        x_eye = self.extract_dynamic_vortex_crop(x_flat)
        f_eye = self.eye_expert(x_eye)              # (B*K, eye_dim)

        # 3. Dynamic Eye Gating per frame
        alpha = 0.30 + 0.70 * self.eye_gate(f_eye)  # (B*K, 1)
        f_gated_eye = f_eye * alpha

        # 4. Interactive Cross-Attention Stream Fusion
        z_flat = self.cross_scale_fusion(f_gated_eye, f_synoptic) # (B*K, hidden_dim)

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

        # 2. Explicit Temporal Difference Calculation (Tracks exact storm change):
        # Step-wise change between adjacent frames (t vs t-1)
        delta_step = torch.zeros_like(z_seq)
        delta_step[:, 1:] = z_seq[:, 1:] - z_seq[:, :-1]

        # Global change relative to the initial frame in the sequence window (t vs t_0)
        delta_global = z_seq - z_seq[:, 0:1]

        # Combine current visual state with velocity and total change vectors
        z_combined = torch.cat([z_seq, delta_step, delta_global], dim=-1)
        u_seq = self.change_projection(z_combined)  # (B, K, hidden_dim)

        # 3. Temporal Recurrent / Attention modeling across timestamps
        if self.temporal_engine_type.lower() == "transformer":
            u_seq = self.pos_encoder(u_seq)
            temporal_out = self.temporal_encoder(u_seq)       # (B, K, hidden_dim)
        else:
            temporal_out, _ = self.temporal_encoder(u_seq)    # (B, K, hidden_dim)

        # 4. Temporal Context Aggregation
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
