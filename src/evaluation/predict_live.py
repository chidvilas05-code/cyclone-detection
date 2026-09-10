"""
Live Real-Time Single-Sample / Batch Inference Script for Tropical Cyclones.
Run from CLI to evaluate live satellite screenshots and real-time atmospheric readings.
"""

import sys
import argparse
from pathlib import Path
import yaml
from PIL import Image
import torch

from src.models.vision_classifier import CycloneVisionModel, HybridMultiBackboneCycloneModel, load_vision_model_from_checkpoint
from src.models.sensory_predictor import CycloneSensoryPredictor
from src.models.fusion import MultimodalCycloneFusion

from src.evaluation.gradcam import generate_cyclone_gradcam


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def predict_live(
    image_path: str = None,
    central_pressure: float = None,
    sst: float = None,
    vertical_wind_shear: float = None,
    relative_humidity: float = None,
    translation_speed: float = None,
    latitude: float = None,
    longitude: float = None,
    config_path: str = "configs/config.yaml",
    save_gradcam_path: str = "live_gradcam_output.png"
):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Vision Model if image provided
    vision_prediction = None
    if image_path and Path(image_path).exists():
        vision_ckpt = Path(cfg["paths"]["vision_model_dir"]) / "best_vision_model.pt"
        if vision_ckpt.exists():
            vision_model, _ = load_vision_model_from_checkpoint(
                str(vision_ckpt),
                device=device,
                default_num_classes=len(cfg.get("categories", []))
            )

            pil_img = Image.open(image_path).convert("RGB")

            blended_img, cam_2d, cat_id, conf, pred_wind, probs = generate_cyclone_gradcam(
                vision_model, pil_img, img_size=cfg["vision_model"]["img_size"]
            )
            cat_cfg = cfg["categories"][cat_id]
            vision_prediction = {
                "category_id": cat_id,
                "category_name": cat_cfg["name"],
                "confidence": conf,
                "wind_speed_knots": pred_wind,
                "probabilities": probs.tolist()
            }
            # Save Grad-CAM heatmap
            blended_img.save(save_gradcam_path)
            print(f"[Grad-CAM] Explainability heatmap saved to: {save_gradcam_path}")
        else:
            print(f"[Warning] Vision model weights not found at {vision_ckpt}")

    # 2. Load Sensory Model if sensor parameters provided
    sensory_prediction = None
    if central_pressure is not None or sst is not None or latitude is not None:
        sensory_predictor = CycloneSensoryPredictor(model_dir=cfg["paths"]["sensory_model_dir"])
        if sensory_predictor.load():
            sensor_dict = {
                "central_pressure": central_pressure or 990.0,
                "sst": sst or 29.0,
                "vertical_wind_shear": vertical_wind_shear or 12.0,
                "relative_humidity": relative_humidity or 75.0,
                "translation_speed": translation_speed or 14.0,
                "latitude": latitude or 15.0,
                "longitude": longitude or 85.0
            }
            sensory_prediction = sensory_predictor.predict_single(sensor_dict)

    # 3. Dynamic Multimodal Fusion
    fusion_engine = MultimodalCycloneFusion(
        vision_weight=cfg["fusion"]["vision_weight"],
        sensory_weight=cfg["fusion"]["sensory_weight"],
        categories_config=cfg["categories"]
    )
    result = fusion_engine.fuse(vision_pred=vision_prediction, sensory_pred=sensory_prediction)

    # Print Formatted Live Advisory
    print("\n" + "="*65)
    print("      AI TROPICAL CYCLONE EARLY WARNING - LIVE ADVISORY")
    print("="*65)
    print(f" Modality Evaluated   : {result['modality_used']}")
    print(f" Cyclone Status       : {'CYCLONE DETECTED' if result['wind_speed_knots'] >= 34 else 'TROPICAL DEPRESSION / DISTURBANCE'}")
    print(f" Predicted Category   : {result['category_name']} ({result.get('category_code', 'N/A')})")
    print(f" Estimated Wind Speed : {result['wind_speed_knots']} knots ({result['wind_speed_kmh']} km/h)")
    print(f" System Confidence    : {result['confidence']}%")
    print(f" Diagnostic Summary   : {result['rationale']}")
    print("="*65 + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Real-time Tropical Cyclone Evaluation")
    parser.add_argument("--image", type=str, default=None, help="Path to live satellite screenshot/image (PNG/JPG)")
    parser.add_argument("--pressure", type=float, default=None, help="Live central barometric pressure in hPa/mb (e.g. 975.0)")
    parser.add_argument("--sst", type=float, default=None, help="Sea Surface Temperature in °C (e.g. 29.5)")
    parser.add_argument("--shear", type=float, default=None, help="Vertical Wind Shear in knots (e.g. 10.0)")
    parser.add_argument("--rh", type=float, default=None, help="Relative Humidity at 700 hPa % (e.g. 80.0)")
    parser.add_argument("--speed", type=float, default=None, help="Forward translation speed in knots (e.g. 12.0)")
    parser.add_argument("--lat", type=float, default=None, help="Current storm latitude in °N (e.g. 16.5)")
    parser.add_argument("--lon", type=float, default=None, help="Current storm longitude in °E (e.g. 88.2)")
    parser.add_argument("--out", type=str, default="live_gradcam_output.png", help="Output path for Grad-CAM image")
    args = parser.parse_args()

    predict_live(
        image_path=args.image,
        central_pressure=args.pressure,
        sst=args.sst,
        vertical_wind_shear=args.shear,
        relative_humidity=args.rh,
        translation_speed=args.speed,
        latitude=args.lat,
        longitude=args.lon,
        save_gradcam_path=args.out
    )
