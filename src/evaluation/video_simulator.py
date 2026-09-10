"""
Video Simulation & Real-Time Satellite Tracking Engine for Tropical Cyclones.
Enables real-time cyclone detection, eye formation tracking, and intensity forecasting
from satellite time-lapse video streams (.mp4, .avi, .mov, .webm).
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import time

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import pandas as pd

from src.models.vision_classifier import CycloneVisionModel
from src.evaluation.gradcam import generate_cyclone_gradcam, preprocess_image_for_model


CATEGORY_COLORS = {
    0: (16, 185, 129),   # Emerald Green: Depression
    1: (245, 158, 11),   # Amber: Cyclonic Storm
    2: (249, 115, 22),   # Orange: Severe Cyclonic Storm
    3: (239, 68, 68),    # Red: Very Severe Cyclonic Storm
    4: (159, 18, 57),    # Rose/Crimson: Super Cyclone
}


def generate_demo_cyclone_video(
    output_path: str = "data/demo_cyclone_timelapse.mp4",
    num_frames: int = 40,
    fps: int = 5
) -> str:
    """
    Generates a realistic multi-frame satellite time-lapse video from
    the Cyclone_Images.h5 archive or synthetic vortex dynamics.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    h5_candidates = list(Path("data/images").glob("*.h5"))
    frames = []

    if h5_candidates:
        try:
            import h5py
            with h5py.File(h5_candidates[0], "r") as f:
                key = "Images" if "Images" in f.keys() else list(f.keys())[0]
                total = f[key].shape[0]
                indices = np.linspace(0, min(total - 1, 1000), num=num_frames, dtype=int)
                for idx in indices:
                    raw = f[key][idx]
                    if raw.ndim == 3 and raw.shape[-1] >= 2:
                        ir = raw[:, :, 0]
                        wv = raw[:, :, 1]
                        diff = ir - wv
                        def norm(ch):
                            c_min, c_max = ch.min(), ch.max()
                            return np.uint8(255 * (ch - c_min) / (c_max - c_min + 1e-6))
                        comp = np.stack([norm(diff), norm(wv), norm(ir)], axis=-1)
                    else:
                        ir = raw[:, :, 0] if raw.ndim == 3 else raw
                        c_min, c_max = ir.min(), ir.max()
                        ir_u8 = np.uint8(255 * (ir - c_min) / (c_max - c_min + 1e-6))
                        comp = np.repeat(ir_u8[:, :, np.newaxis], 3, axis=-1)
                    resized = cv2.resize(comp, (224, 224), interpolation=cv2.INTER_CUBIC)
                    frames.append(resized)
        except Exception as e:
            print(f"[Video Generator] Note on H5 reading: {e}")

    # Fallback: Synthetic spiraling cyclone loop if H5 unavailable
    if not frames:
        for i in range(num_frames):
            angle = i * (360.0 / num_frames)
            canvas = np.zeros((224, 224, 3), dtype=np.uint8)
            center = (112, 112)
            # Draw synthetic spiral arms
            for arm in range(3):
                arm_angle = angle + arm * 120
                pts = []
                for r in range(15, 100, 4):
                    theta = np.deg2rad(arm_angle + r * 2.5)
                    x = int(center[0] + r * np.cos(theta))
                    y = int(center[1] + r * np.sin(theta))
                    pts.append([x, y])
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (180, 200, 255), 6)
            # Central dense overcast / eyewall
            cv2.circle(canvas, center, 22, (230, 240, 255), -1)
            cv2.circle(canvas, center, 8, (15, 15, 20), -1)  # Eye
            cv2.GaussianBlur(canvas, (11, 11), 3, dst=canvas)
            frames.append(canvas)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (224, 224))
    for f in frames:
        writer.write(f)
    writer.release()
    return str(out_file)


def locate_cyclone_eye(
    frame_bgr: np.ndarray,
    prev_eye_center: Optional[Tuple[int, int]] = None,
    contrast_threshold: float = 16.0
) -> Tuple[Tuple[int, int], float, bool]:
    """
    Dynamically pinpoints the exact coordinates of the cyclone eye or circulation vortex.
    
    Combines:
    1. Multi-scale spiral gradient curvature accumulator to locate the rotational circulation center (LLCC/ULCC).
    2. Local eye-depression basin search within the circulation core.
    3. Fully enclosed azimuthal eyewall validation (eliminates false positives from watermarks and ocean borders).
    4. Temporal consistency smoothing with prev_eye_center.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Downsample for fast, robust spiral gradient curvature accumulator
    scale = 0.5
    small = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
    blurred_small = cv2.GaussianBlur(small, (9, 9), 2.0)
    
    gx = cv2.Sobel(blurred_small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred_small, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    
    # Only spiral clouds (ignore dark ocean and flat areas)
    cloud_mask = (blurred_small >= 115) & (mag >= 3.0)
    valid_y, valid_x = np.where(cloud_mask)
    if len(valid_x) == 0:
        return (w // 2, h // 2), 0.0, False
        
    gx_v = gx[valid_y, valid_x] / (mag[valid_y, valid_x] + 1e-6)
    gy_v = gy[valid_y, valid_x] / (mag[valid_y, valid_x] + 1e-6)
    
    # Subsample for high real-time throughput
    step = max(1, len(valid_x) // 1200)
    vx = valid_x[::step]
    vy = valid_y[::step]
    nx = gx_v[::step]
    ny = gy_v[::step]
    
    sh, sw = small.shape
    acc = np.zeros((sh, sw), dtype=np.float32)
    
    for x_i, y_i, dx_i, dy_i in zip(vx, vy, nx, ny):
        for d in range(12, 110, 6):
            for sign in [1, -1]:
                cx = int(x_i + sign * d * dx_i)
                cy = int(y_i + sign * d * dy_i)
                if 0 <= cx < sw and 0 <= cy < sh:
                    acc[cy, cx] += 1.0 / (d ** 0.5)
                    
    # Smooth accumulator map
    acc = cv2.GaussianBlur(acc, (13, 13), 2.5)
    
    # Proximity bias toward previous eye location to prevent tracking hops
    if prev_eye_center is not None:
        px_s, py_s = int(prev_eye_center[0] * scale), int(prev_eye_center[1] * scale)
        y_g, x_g = np.ogrid[:sh, :sw]
        dist_prev = np.sqrt((x_g - px_s) ** 2 + (y_g - py_s) ** 2)
        proximity = np.clip(1.0 - dist_prev / 80.0, 0.0, 1.0)
        acc = acc * (1.0 + 0.35 * proximity)
        
    _, _, _, max_loc = cv2.minMaxLoc(acc)
    vortex_x = int(max_loc[0] / scale)
    vortex_y = int(max_loc[1] / scale)
    
    # 2. Local Eye Cavity Refinement:
    # Within a 40px radius of the vortex center, search for an enclosed circular eye depression
    box_r = 38
    rx_min = max(0, vortex_x - box_r)
    rx_max = min(w, vortex_x + box_r)
    ry_min = max(0, vortex_y - box_r)
    ry_max = min(h, vortex_y + box_r)
    
    patch = gray[ry_min:ry_max, rx_min:rx_max]
    patch_blurred = cv2.GaussianBlur(patch, (5, 5), 1.0)
    
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(patch_blurred)
    candidate_eye_x = rx_min + min_l[0]
    candidate_eye_y = ry_min + min_l[1]
    
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    sin_a, cos_a = np.sin(angles), np.cos(angles)
    
    is_open_eye = False
    eye_score = 0.0
    final_x, final_y = vortex_x, vortex_y
    
    for r in [8, 12, 16, 20]:
        sx = np.clip((candidate_eye_x + r * cos_a).astype(int), 0, w - 1)
        sy = np.clip((candidate_eye_y + r * sin_a).astype(int), 0, h - 1)
        ring = gray[sy, sx]
        min_ring = float(np.min(ring))
        mean_ring = float(np.mean(ring))
        c_val = float(patch_blurred[min_l[1], min_l[0]])
        
        if min_ring >= 115 and (mean_ring - c_val) >= 20:
            is_open_eye = True
            eye_score = (mean_ring - c_val) * 1.5 + mean_ring
            final_x, final_y = candidate_eye_x, candidate_eye_y
            break
            
    # Temporal smoothing
    if prev_eye_center is not None:
        smooth_x = int(0.70 * final_x + 0.30 * prev_eye_center[0])
        smooth_y = int(0.70 * final_y + 0.30 * prev_eye_center[1])
        final_x, final_y = smooth_x, smooth_y
        
    return (final_x, final_y), eye_score, is_open_eye


def annotate_frame_hud(
    frame_pil: Image.Image,
    category_name: str,
    cat_id: int,
    wind_speed: float,
    confidence: float,
    frame_idx: int,
    total_frames: int,
    eye_center: Optional[Tuple[int, int]] = None,
    is_open_eye: bool = False,
    eye_score: float = 0.0
) -> Image.Image:
    """
    Overlays a professional operational HUD onto the video frame:
    - Bounding brackets centered dynamically on the detected cyclone eye/vortex
    - Red crosshair and targeting reticle directly on the eye core
    - Color-coded severity badge
    - Real-time wind speed (knots and km/h)
    - AI Confidence and frame timestamp
    """
    img = frame_pil.convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 1. Top HUD Status Banner
    color = CATEGORY_COLORS.get(cat_id, (16, 185, 129))
    draw.rectangle([0, 0, w, 44], fill=(15, 23, 42, 210))  # Slate dark transparent banner
    draw.rectangle([0, 42, w, 44], fill=color + (255,))     # Severity accent line

    # Category Text & Wind Speed
    wind_kmh = wind_speed * 1.852
    hud_text_left = f"{category_name.upper()}"
    draw.text((10, 8), hud_text_left, fill=color + (255,))
    draw.text((10, 24), f"AI Conf: {confidence*100:.1f}% | Frame: {frame_idx:03d}/{total_frames:03d}", fill=(203, 213, 225, 255))
    draw.text((w - 110, 8), f"{wind_speed:.1f} kt", fill=(255, 255, 255, 255))
    draw.text((w - 110, 24), f"{wind_kmh:.0f} km/h", fill=(148, 163, 184, 255))

    # 2. Dynamic Eyewall Core / Eye Bounding Brackets
    if eye_center is not None:
        eye_x, eye_y = eye_center
    else:
        eye_x, eye_y = w // 2, h // 2

    box_size = int(min(w, h) * 0.28)
    half = box_size // 2
    cw_start = max(8, eye_x - half)
    cw_end = min(w - 8, eye_x + half)
    ch_start = max(48, eye_y - half)
    ch_end = min(h - 8, eye_y + half)

    # Target brackets
    bracket_len = 22
    bracket_color = (56, 189, 248, 230)  # Sky blue

    # Top-Left
    draw.line([(cw_start, ch_start), (cw_start + bracket_len, ch_start)], fill=bracket_color, width=3)
    draw.line([(cw_start, ch_start), (cw_start, ch_start + bracket_len)], fill=bracket_color, width=3)
    # Top-Right
    draw.line([(cw_end, ch_start), (cw_end - bracket_len, ch_start)], fill=bracket_color, width=3)
    draw.line([(cw_end, ch_start), (cw_end, ch_start + bracket_len)], fill=bracket_color, width=3)
    # Bottom-Left
    draw.line([(cw_start, ch_end), (cw_start + bracket_len, ch_end)], fill=bracket_color, width=3)
    draw.line([(cw_start, ch_end), (cw_start, ch_end - bracket_len)], fill=bracket_color, width=3)
    # Bottom-Right
    draw.line([(cw_end, ch_end), (cw_end - bracket_len, ch_end)], fill=bracket_color, width=3)
    draw.line([(cw_end, ch_end), (cw_end, ch_end - bracket_len)], fill=bracket_color, width=3)

    if is_open_eye:
        # Precision Crosshair and Reticle on the Eye
        reticle_color = (239, 68, 68, 240)  # Crimson
        draw.ellipse([eye_x - 14, eye_y - 14, eye_x + 14, eye_y + 14], outline=reticle_color, width=2)
        draw.line([(eye_x - 8, eye_y), (eye_x + 8, eye_y)], fill=reticle_color, width=2)
        draw.line([(eye_x, eye_y - 8), (eye_x, eye_y + 8)], fill=reticle_color, width=2)
        tag_text = f"CYCLONE EYE ({eye_x}, {eye_y})"
        draw.text((cw_start + 4, max(46, ch_start - 14)), tag_text, fill=(56, 189, 248, 255))
    else:
        draw.ellipse([eye_x - 8, eye_y - 8, eye_x + 8, eye_y + 8], outline=(56, 189, 248, 180), width=1)
        tag_text = f"VORTEX CORE ({eye_x}, {eye_y})"
        draw.text((cw_start + 4, max(46, ch_start - 14)), tag_text, fill=(56, 189, 248, 200))

    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


def process_video_frame(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    categories_config: List[Dict[str, Any]],
    device: torch.device,
    frame_idx: int = 1,
    total_frames: int = 1,
    generate_gradcam: bool = False,
    img_size: int = 224,
    prev_eye_center: Optional[Tuple[int, int]] = None,
    **kwargs
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Ingests a single video frame, computes model inference,
    locates the cyclone eye/vortex dynamically, and returns an annotated HUD frame.
    """
    h_orig, w_orig = frame_bgr.shape[:2]

    # 1. Locate Cyclone Eye dynamically from physical satellite albedo/contrast
    detected_center, eye_score, is_open_eye = locate_cyclone_eye(frame_bgr, prev_eye_center=prev_eye_center)

    # 2. Temporal exponential smoothing to prevent tracking jitter
    if prev_eye_center is not None:
        alpha = 0.75
        smooth_x = int(alpha * detected_center[0] + (1 - alpha) * prev_eye_center[0])
        smooth_y = int(alpha * detected_center[1] + (1 - alpha) * prev_eye_center[1])
        final_eye_center = (smooth_x, smooth_y)
    else:
        final_eye_center = detected_center

    norm_eye_x = final_eye_center[0] / float(w_orig)
    norm_eye_y = final_eye_center[1] / float(h_orig)
    eye_coords = (norm_eye_x, norm_eye_y)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame_rgb)

    if generate_gradcam:
        blended, cam_2d, cat_id, conf, pred_wind, probs = generate_cyclone_gradcam(
            model, pil_frame, img_size=img_size, eye_coords=eye_coords
        )
        cat_id = int(np.squeeze(cat_id))
        conf = float(np.squeeze(conf))
        pred_wind = float(np.squeeze(pred_wind))
        display_frame = blended
    else:
        input_tensor = preprocess_image_for_model(pil_frame, img_size=img_size, device=device)
        model.eval()
        with torch.no_grad():
            import inspect
            sig = inspect.signature(model.forward)
            if "eye_coords" in sig.parameters:
                logits, pred_wind_tensor = model(input_tensor, eye_coords=eye_coords)
            else:
                logits, pred_wind_tensor = model(input_tensor)

            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            cat_id = int(np.argmax(probs))
            conf = float(probs[cat_id])
            pred_wind = float(pred_wind_tensor.detach().cpu().reshape(-1)[0])
        display_frame = pil_frame

    if isinstance(categories_config, list):
        cat_info = categories_config[min(cat_id, len(categories_config) - 1)]
        cat_name = cat_info["name"] if isinstance(cat_info, dict) else str(cat_info)
    elif isinstance(categories_config, dict):
        cat_info = categories_config.get(cat_id, categories_config.get(str(cat_id), f"Category {cat_id}"))
        cat_name = cat_info["name"] if isinstance(cat_info, dict) else str(cat_info)
    else:
        cat_name = f"Category {cat_id}"

    annotated = annotate_frame_hud(
        display_frame,
        category_name=cat_name,
        cat_id=cat_id,
        wind_speed=pred_wind,
        confidence=conf,
        frame_idx=frame_idx,
        total_frames=total_frames,
        eye_center=final_eye_center,
        is_open_eye=is_open_eye,
        eye_score=eye_score
    )

    record = {
        "frame": frame_idx,
        "category_id": cat_id,
        "category_name": cat_name,
        "wind_speed_knots": np.round(pred_wind, 1),
        "wind_speed_kmh": np.round(pred_wind * 1.852, 1),
        "confidence": np.round(conf * 100, 1),
        "eye_x": final_eye_center[0],
        "eye_y": final_eye_center[1],
        "is_open_eye": is_open_eye,
        "eye_score": np.round(eye_score, 1),
        "probabilities": probs.tolist()
    }

    return annotated, record
