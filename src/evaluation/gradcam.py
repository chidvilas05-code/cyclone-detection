"""
Visual Explainability Utilities for Tropical Cyclone Cloud Band Interpretability.
Generates thermal heatmaps pinpointing convective cores, rainband curvature, and eye structures.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

from src.models.vision_classifier import CycloneVisionModel, GradCAM


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

    # Attention threshold mask: only blend where CAM has positive attention!
    # Values < 0.12 are background; values >= 0.12 smoothly ramp up blending weight
    mask_weight = np.clip((cam_resized - 0.12) / 0.88, 0.0, 1.0)[:, :, np.newaxis] * alpha

    # Blend
    blended = np.uint8((1.0 - mask_weight) * orig_np.astype(np.float32) + mask_weight * heatmap_rgb.astype(np.float32))
    return Image.fromarray(blended)


def generate_cyclone_gradcam(
    model: CycloneVisionModel,
    pil_image: Image.Image,
    target_class: Optional[int] = None,
    img_size: int = 224,
    eye_coords: Optional[Tuple[float, float]] = None
) -> Tuple[Image.Image, np.ndarray, int, float, float, np.ndarray]:
    """
    Executes forward pass and Grad-CAM backpropagation.
    Returns:
      blended_image: PIL Image with heatmap overlay
      cam_2d: 2D numpy array of activation intensities
      predicted_class: Integer ID
      confidence: Float (0.0 to 1.0)
      pred_wind: Estimated wind speed in knots
      probs: Array of softmax probabilities across all 5 classes
    """
    device = next(model.parameters()).device
    input_tensor = preprocess_image_for_model(pil_image, img_size=img_size, device=device)

    model.eval()
    gradcam_engine = GradCAM(model)

    with torch.enable_grad():
        input_tensor.requires_grad_(True)
        # Check if model accepts eye_coords
        import inspect
        sig = inspect.signature(model.forward)
        if "eye_coords" in sig.parameters:
            logits, pred_wind_tensor = model(input_tensor, eye_coords=eye_coords)
        else:
            logits, pred_wind_tensor = model(input_tensor)

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        pred_wind = float(pred_wind_tensor.detach().cpu().reshape(-1)[0])

        import inspect
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
