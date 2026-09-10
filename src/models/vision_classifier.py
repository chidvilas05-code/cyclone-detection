"""
Satellite Vision Transfer Learning Model for Tropical Cyclone Classification & Intensity Estimation.
Features:
  - Pretrained Backbones: ConvNeXt-Tiny, EfficientNet-B2, ResNet-50
  - Dual-Head: Multi-class Category Classification + Continuous Wind Speed Regression
  - Explainable AI: Built-in Grad-CAM for visualizing spiral rainbands and central eye wall focus.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Class-Balanced Cross-Entropy / Focal Loss with Label Smoothing.
    When gamma=0.0, delivers full, un-dampened cross-entropy gradient flow across all confidence levels,
    preventing gradient starvation and enabling validation accuracy to push past 90% toward 95%.
    """
    def __init__(self, gamma: float = 0.0, alpha: Optional[torch.Tensor] = None, label_smoothing: float = 0.03):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.alpha.to(inputs.device) if self.alpha is not None else None
        ce_loss = F.cross_entropy(inputs, targets, weight=weights, reduction="none", label_smoothing=self.label_smoothing)

        if self.gamma > 0.0:
            p_t = torch.exp(-ce_loss).clamp(min=1e-6, max=1.0)
            focal_loss = ((1.0 - p_t) ** self.gamma) * ce_loss
            return focal_loss.mean()

        return ce_loss.mean()


class CycloneVisionModel(nn.Module):
    """
    Dual-Head Transfer Learning architecture for Tropical Cyclones.
    Extracts high-level cyclonic spatial patterns (spiral density, eye wall circularity,
    central dense overcast) from satellite imagery.
    """
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = 5,
        dropout_rate: float = 0.40
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Try loading via timm, with graceful fallback to torchvision
        try:
            import timm
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
            self.is_timm = True
        except Exception:
            import torchvision.models as tv_models
            self.is_timm = False
            if "resnet" in backbone_name:
                weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
                base = tv_models.resnet50(weights=weights)
                in_features = base.fc.in_features
                base.fc = nn.Identity()
                self.backbone = base
            else:
                weights = tv_models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
                base = tv_models.efficientnet_b2(weights=weights)
                in_features = base.classifier[1].in_features
                base.classifier = nn.Identity()
                self.backbone = base

        # Shared latent representation layer
        self.feature_layer = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0)
        )

        # Head 1: Cyclone Severity Classification (Logits)
        self.classifier_head = nn.Linear(256, num_classes)

        # Head 2: Intensity Wind Speed Regression (Knots)
        self.regression_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        latent = self.feature_layer(features)
        
        logits = self.classifier_head(latent)
        wind_speed = self.regression_head(latent).squeeze(-1)

        return logits, wind_speed

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """Returns the final convolutional feature layer for Grad-CAM gradient hooks."""
        if hasattr(self.backbone, "stages"):
            # ConvNeXt final stage
            return self.backbone.stages[-1]
        elif hasattr(self.backbone, "layers"):
            # Swin Transformer final layer
            return self.backbone.layers[-1]
        elif hasattr(self.backbone, "conv_head"):
            # EfficientNet final conv
            return self.backbone.conv_head
        elif hasattr(self.backbone, "layer4"):
            # ResNet layer 4
            return self.backbone.layer4[-1]
        elif hasattr(self.backbone, "features"):
            return self.backbone.features[-1]
        else:
            last_conv = None
            for module in self.backbone.modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
                    last_conv = module
            return last_conv


class HybridMultiBackboneCycloneModel(nn.Module):
    """
    Multi-Expert Hybrid Transfer Learning Network.
    Combines:
      1. CNN Backbone (ConvNeXt-Tiny): Extracts local eyewall sharpness & fine convective gradients.
      2. Vision Transformer (Swin-T): Extracts global long-range spiral arm curvature (1000km synoptic context).
    Merges both feature streams via gated cross-modal fusion.
    """
    def __init__(
        self,
        cnn_name: str = "convnext_tiny",
        transformer_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        num_classes: int = 5,
        dropout_rate: float = 0.40
    ):
        super().__init__()
        import timm

        # Expert 1: Local Eyewall CNN
        self.cnn_expert = timm.create_model(cnn_name, pretrained=pretrained, num_classes=0)
        cnn_dim = self.cnn_expert.num_features

        # Expert 2: Global Spiral Transformer
        self.transformer_expert = timm.create_model(transformer_name, pretrained=pretrained, num_classes=0)
        transformer_dim = self.transformer_expert.num_features

        # Gated Cross-Modal Fusion Bottleneck
        total_in_dim = cnn_dim + transformer_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_in_dim, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0)
        )

        # Dual Prediction Heads
        self.classifier_head = nn.Linear(256, num_classes)
        self.regression_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_local = self.cnn_expert(x)
        f_global = self.transformer_expert(x)

        # Joint multi-scale latent representation
        f_fused = torch.cat([f_local, f_global], dim=-1)
        latent = self.fusion_layer(f_fused)

        logits = self.classifier_head(latent)
        wind_speed = self.regression_head(latent).squeeze(-1)
        return logits, wind_speed

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """Returns the CNN stage for fine Grad-CAM eyewall rendering."""
        if hasattr(self.cnn_expert, "stages"):
            return self.cnn_expert.stages[-1]
        return list(self.cnn_expert.children())[-1]


class CentralEyeCoreModel(nn.Module):
    """
    Specialized Deep Architecture for Central Dense Overcast (CDO) and Inner Eye Wall.
    Focuses specifically on:
      1. Eye warm-core temperature vs cold cloud-top thermal contrast (T_eye - T_eyewall)
      2. Eyewall circularity, thickness, and azimuthal symmetry
      3. Stadium effect eyewall slope detection
    """
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = 5,
        crop_ratio: float = 0.50,  # Focus on inner 50% central eye area
        dropout_rate: float = 0.40
    ):
        super().__init__()
        import timm
        self.crop_ratio = crop_ratio
        self.backbone_name = backbone_name
        self.eye_backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        in_features = self.eye_backbone.num_features

        # Central core feature refinement
        self.eye_feature_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate / 2.0)
        )

        self.classifier_head = nn.Linear(256, num_classes)
        self.regression_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.regression_head[-1].bias.data.fill_(50.0)

    def extract_central_eye_crop(
        self,
        x: torch.Tensor,
        eye_coords: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """Extracts and upscales the core eyewall region using differentiable bilinear interpolation."""
        h, w = x.shape[2], x.shape[3]
        if eye_coords is not None:
            cx = int(eye_coords[0] * w)
            cy = int(eye_coords[1] * h)
        else:
            cx, cy = w // 2, h // 2

        crop_h = int(h * self.crop_ratio)
        crop_w = int(w * self.crop_ratio)

        ch_start = max(0, min(h - crop_h, cy - crop_h // 2))
        ch_end = ch_start + crop_h
        cw_start = max(0, min(w - crop_w, cx - crop_w // 2))
        cw_end = cw_start + crop_w

        self.last_crop_box = (ch_start, ch_end, cw_start, cw_end)
        eye_crop = x[:, :, ch_start:ch_end, cw_start:cw_end]
        return F.interpolate(eye_crop, size=(h, w), mode="bilinear", align_corners=False)

    def forward(
        self,
        x: torch.Tensor,
        eye_coords: Optional[Tuple[float, float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eye_zoomed = self.extract_central_eye_crop(x, eye_coords=eye_coords)
        features = self.eye_backbone(eye_zoomed)
        latent = self.eye_feature_head(features)
        logits = self.classifier_head(latent)
        wind_speed = self.regression_head(latent).squeeze(-1)
        return logits, wind_speed

    def get_target_layer_for_gradcam(self) -> nn.Module:
        if hasattr(self.eye_backbone, "stages"):
            return self.eye_backbone.stages[-1]
        return list(self.eye_backbone.children())[-1]


class DualStreamEyeAndSynopticModel(nn.Module):
    """
    Dual-Stream Architecture:
      - Stream 1 (Core Eye): Zooms into the inner 50% central eye/eyewall.
      - Stream 2 (Synoptic Full-View): Ingests the complete 1000 km synoptic cloud field.
      - Cross-Scale Gated Fusion: Combines eyewall intensity metrics with synoptic feeder band curvature.
    """
    def __init__(
        self,
        eye_backbone_name: str = "convnext_tiny",
        synoptic_backbone_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        num_classes: int = 5,
        crop_ratio: float = 0.50,
        dropout_rate: float = 0.40
    ):
        super().__init__()
        import timm
        self.crop_ratio = crop_ratio

        # Stream 1: Core Eye Expert (CNN)
        self.eye_expert = timm.create_model(eye_backbone_name, pretrained=pretrained, num_classes=0)
        eye_dim = self.eye_expert.num_features

        # Stream 2: Synoptic Global Expert (Transformer)
        self.synoptic_expert = timm.create_model(synoptic_backbone_name, pretrained=pretrained, num_classes=0)
        synoptic_dim = self.synoptic_expert.num_features

        # Adaptive Eye Organization Gate: Dynamically computes attention based on eye development
        self.eye_gate = nn.Sequential(
            nn.Linear(eye_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Gated Cross-Scale Fusion with LayerNorm for stable variance across train/eval modes
        # Gated Cross-Scale Fusion: BatchNorm1d standardizes ConvNeXt + Swin feature scales
        total_dim = eye_dim + synoptic_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate / 2.0)
        )

        self.classifier_head = nn.Linear(256, num_classes)
        self.regression_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # Initialize regression head bias near dataset mean wind speed (~50 knots)
        # to prevent massive initial SmoothL1 error spikes
        self.regression_head[-1].bias.data.fill_(50.0)

    def extract_central_eye_crop(
        self,
        x: torch.Tensor,
        eye_coords: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """Extracts and upscales the core eyewall region using differentiable bilinear interpolation."""
        h, w = x.shape[2], x.shape[3]
        if eye_coords is not None:
            cx = int(eye_coords[0] * w)
            cy = int(eye_coords[1] * h)
        else:
            cx, cy = w // 2, h // 2

        crop_h = int(h * self.crop_ratio)
        crop_w = int(w * self.crop_ratio)

        ch_start = max(0, min(h - crop_h, cy - crop_h // 2))
        ch_end = ch_start + crop_h
        cw_start = max(0, min(w - crop_w, cx - crop_w // 2))
        cw_end = cw_start + crop_w

        self.last_crop_box = (ch_start, ch_end, cw_start, cw_end)
        eye_crop = x[:, :, ch_start:ch_end, cw_start:cw_end]
        return F.interpolate(eye_crop, size=(h, w), mode="bilinear", align_corners=False)

    def forward(
        self,
        x: torch.Tensor,
        eye_coords: Optional[Tuple[float, float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eye_zoom = self.extract_central_eye_crop(x, eye_coords=eye_coords)
        f_eye = self.eye_expert(eye_zoom)
        f_synoptic = self.synoptic_expert(x)

        # Dynamic Eye Gating: scales eye feature weight based on inner organization
        alpha = 0.3 + 0.7 * self.eye_gate(f_eye)
        f_gated_eye = f_eye * alpha

        f_fused = torch.cat([f_gated_eye, f_synoptic], dim=-1)
        latent = self.fusion(f_fused)

        logits = self.classifier_head(latent)
        wind_speed = self.regression_head(latent).squeeze(-1)
        return logits, wind_speed

    def get_target_layer_for_gradcam(self) -> nn.Module:
        if hasattr(self.eye_expert, "stages"):
            return self.eye_expert.stages[-1]

        return list(self.eye_expert.children())[-1]


# Alias for explicit naming
IntegratedDualStreamCycloneModel = DualStreamEyeAndSynopticModel



class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for visual explanation.
    Highlights cloud spirals, eye wall symmetry, and convective bursts driving the AI decision.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or model.get_target_layer_for_gradcam()
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        if self.target_layer is not None:
            self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
            self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        eye_coords: Optional[Tuple[float, float]] = None,
        *args,
        **kwargs
    ) -> np.ndarray:
        """
        Generates 2D Grad-CAM heatmap array normalized between 0 and 1.
        input_tensor: Shape (1, 3, H, W)
        """
        self.model.eval()
        self.model.zero_grad()

        # Check if model supports eye_coords
        import inspect
        sig = inspect.signature(self.model.forward)
        if "eye_coords" in sig.parameters:
            logits, _ = self.model(input_tensor, eye_coords=eye_coords)
        else:
            logits, _ = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        if self.gradients is None or self.activations is None:
            return np.zeros((h, w), dtype=np.float32)

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = F.relu(cam)

        crop_box = getattr(self.model, "last_crop_box", None)
        if crop_box is not None:
            ch_start, ch_end, cw_start, cw_end = crop_box
            sub_h = max(1, ch_end - ch_start)
            sub_w = max(1, cw_end - cw_start)
            cam_sub = cam.unsqueeze(0).unsqueeze(0)
            cam_sub = F.interpolate(cam_sub, size=(sub_h, sub_w), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

            full_cam = np.zeros((h, w), dtype=np.float32)
            full_cam[ch_start:ch_end, cw_start:cw_end] = cam_sub
            # Smooth transition so heatmap blurs naturally into convective rainbands
            full_cam = cv2.GaussianBlur(full_cam, (25, 25), 0)
            cam = full_cam
        else:
            cam = cam.unsqueeze(0).unsqueeze(0)
            cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
            cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()


def load_vision_model_from_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    default_num_classes: int = 5
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Robustly instantiates and loads any Cyclone vision model architecture:
      - CentralEyeCoreModel (Eye-specific)
      - DualStreamEyeAndSynopticModel (Dual-stream Core-Eye + Synoptic)
      - HybridMultiBackboneCycloneModel (CNN + Swin-T)
      - CycloneVisionModel (Standard ConvNeXt / EfficientNet / ResNet)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    num_classes = checkpoint.get("num_classes", default_num_classes)
    backbone_name = str(checkpoint.get("backbone", "convnext_tiny")).lower()
    state_keys = list(state_dict.keys())

    # Detect model type
    if backbone_name in ["dual_stream_eye", "eye_dual", "dual_stream"] or ("eye_expert" in state_keys and "synoptic_expert" in state_keys):
        model = DualStreamEyeAndSynopticModel(
            eye_backbone_name=checkpoint.get("eye_backbone", "convnext_tiny"),
            synoptic_backbone_name=checkpoint.get("synoptic_backbone", "swin_tiny_patch4_window7_224"),
            pretrained=False,
            num_classes=num_classes
        ).to(device)
    elif backbone_name in ["eye_core", "central_eye", "eye_only"] or any(k.startswith("eye_backbone") for k in state_keys):
        model = CentralEyeCoreModel(
            backbone_name=checkpoint.get("backbone_name", "convnext_tiny"),
            pretrained=False,
            num_classes=num_classes
        ).to(device)
    elif (
        checkpoint.get("is_hybrid", False)
        or backbone_name in ["hybrid", "dual_backbone", "hybrid_expert"]
        or any(k.startswith("cnn_expert") or k.startswith("transformer_expert") for k in state_keys)
    ):
        cnn_name = checkpoint.get("cnn_name", "convnext_tiny")
        transformer_name = checkpoint.get("transformer_name", "swin_tiny_patch4_window7_224")
        model = HybridMultiBackboneCycloneModel(
            cnn_name=cnn_name,
            transformer_name=transformer_name,
            pretrained=False,
            num_classes=num_classes
        ).to(device)
    else:
        model = CycloneVisionModel(
            backbone_name=backbone_name,
            pretrained=False,
            num_classes=num_classes
        ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


