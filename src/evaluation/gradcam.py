"""
Visual Explainability Utilities for Tropical Cyclone Cloud Band Interpretability.
Generates thermal heatmaps pinpointing convective cores, rainband curvature, and eye structures.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for visual explanation.
    Highlights convective cloud bands, eye wall symmetry, and storm core structures driving AI decisions.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_target_layer_for_gradcam"):
            self.target_layer = model.get_target_layer_for_gradcam()
        else:
            # Fallback to the deepest convolutional layer in model
            conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            self.target_layer = conv_layers[-1] if conv_layers else None

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

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        eye_coords: Optional[Tuple[float, float]] = None,
        *args,
        **kwargs
    ) -> np.ndarray:
        """
        Generates 2D Grad-CAM heatmap array normalized between 0.0 and 1.0.
        input_tensor: Shape (1, 3, H, W)
        """
        self.model.eval()
        self.model.zero_grad()

        import inspect
        sig = inspect.signature(self.model.forward)
        if "eye_coords" in sig.parameters:
            output = self.model(input_tensor, eye_coords=eye_coords)
        else:
            output = self.model(input_tensor)

        logits = output[0] if isinstance(output, tuple) else output

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

        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def preprocess_image_for_model(
    pil_image: Image.Image,
    img_size: int = 224,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Preprocesses a PIL image for model evaluation and Grad-CAM."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    img_rgb = pil_image.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    return tensor


def overlay_heatmap_on_image(
    original_pil: Image.Image,
    cam_2d: np.ndarray,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET
) -> Image.Image:
    """
    Overlays a 2D Grad-CAM heatmap onto the original satellite image.
    Uses attention-weighted blending so only active convective storm areas glow,
    leaving clear background ocean and outer bands pristine without blue cast.
    """
    orig_np = np.array(original_pil.convert("RGB"))
    h, w = orig_np.shape[:2]

    # Resize CAM to match original image dimensions
    cam_resized = cv2.resize(cam_2d.astype(np.float32), (w, h))

    # Convert to 8-bit heatmap
    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Attention-weighted mask
    weight = np.clip((cam_resized - 0.15) / 0.85, 0.0, 1.0)
    weight = np.expand_dims(weight, axis=2)

    blended_np = np.uint8(
        orig_np * (1.0 - weight * alpha) + heatmap_rgb * (weight * alpha)
    )

    return Image.fromarray(blended_np)


def generate_cyclone_gradcam(
    model: torch.nn.Module,
    pil_image: Image.Image,
    target_class: Optional[int] = None,
    eye_coords: Optional[Tuple[float, float]] = None,
    img_size: int = 224
) -> Tuple[Image.Image, np.ndarray, int, float, float, np.ndarray]:
    """
    Unified end-to-end Grad-CAM runner for cyclone models.
    """
    device = next(model.parameters()).device
    input_tensor = preprocess_image_for_model(pil_image, img_size=img_size, device=device)

    model.eval()
    gradcam_engine = GradCAM(model)

    with torch.enable_grad():
        input_tensor.requires_grad_(True)
        import inspect
        sig = inspect.signature(model.forward)
        if "eye_coords" in sig.parameters:
            output = model(input_tensor, eye_coords=eye_coords)
        else:
            output = model(input_tensor)

        logits = output[0] if isinstance(output, tuple) else output
        pred_wind_tensor = output[1] if isinstance(output, tuple) and len(output) > 1 else torch.tensor([0.0])

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        pred_wind = float(pred_wind_tensor.detach().cpu().reshape(-1)[0])

        cam_sig = inspect.signature(gradcam_engine.generate_cam)
        cam_kwargs = {}
        if "eye_coords" in cam_sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in cam_sig.parameters.values()):
            cam_kwargs["eye_coords"] = eye_coords

        try:
            cam_2d = gradcam_engine.generate_cam(
                input_tensor,
                target_class=target_class or pred_class,
                **cam_kwargs
            )
        except TypeError:
            cam_2d = gradcam_engine.generate_cam(
                input_tensor,
                target_class=target_class or pred_class
            )

    gradcam_engine.remove_hooks()
    blended = overlay_heatmap_on_image(pil_image, cam_2d, alpha=0.55)

    return blended, cam_2d, pred_class, confidence, pred_wind, probs
