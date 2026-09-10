"""
Multimodal Decision Fusion Module.
Synthesizes satellite cloud pattern representations and physical atmospheric sensor data
into an actionable early-warning consensus.
"""

from typing import Dict, Any, Optional, List
import numpy as np


class MultimodalCycloneFusion:
    """
    Ensemble engine that blends image-based spatial features with
    atmospheric sensor measurements (SST, central pressure, wind shear).
    """
    def __init__(
        self,
        vision_weight: float = 0.60,
        sensory_weight: float = 0.40,
        categories_config: Optional[List[Dict[str, Any]]] = None
    ):
        self.vision_weight = vision_weight
        self.sensory_weight = sensory_weight
        self.categories_config = categories_config or [
            {"id": 0, "name": "Depression", "code": "D", "min_knots": 0, "max_knots": 33},
            {"id": 1, "name": "Cyclonic Storm", "code": "CS", "min_knots": 34, "max_knots": 47},
            {"id": 2, "name": "Severe Cyclonic Storm", "code": "SCS", "min_knots": 48, "max_knots": 63},
            {"id": 3, "name": "Very Severe Cyclonic Storm", "code": "VSCS", "min_knots": 64, "max_knots": 89},
            {"id": 4, "name": "Extremely Severe / Super Cyclone", "code": "SuCS", "min_knots": 90, "max_knots": 250},
        ]

    def fuse(
        self,
        vision_pred: Optional[Dict[str, Any]] = None,
        sensory_pred: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Computes the fused consensus prediction.
        Handles:
          - Both modalities present
          - Only Vision present
          - Only Sensory present
        """
        # Case 1: Both modalities available (Dynamic Uncertainty-Weighted Fusion)
        if vision_pred is not None and sensory_pred is not None:
            vis_probs = np.array(vision_pred.get("probabilities", [0.2] * len(self.categories_config)))
            sen_probs = np.array(sensory_pred.get("probabilities", [0.2] * len(self.categories_config)))

            vis_conf = float(vision_pred.get("confidence", 0.5))
            sen_conf = float(sensory_pred.get("confidence", 0.5))

            # Dynamic confidence-weighted squared blend (uncertainty minimization)
            score_vis = (vis_conf ** 2) * self.vision_weight
            score_sen = (sen_conf ** 2) * self.sensory_weight
            denom = score_vis + score_sen
            if denom > 1e-6:
                w_vis = score_vis / denom
                w_sen = score_sen / denom
            else:
                w_vis, w_sen = 0.5, 0.5

            # Blended class probability distribution
            fused_probs = (w_vis * vis_probs) + (w_sen * sen_probs)
            fused_cat_id = int(np.argmax(fused_probs))
            confidence = float(np.max(fused_probs))

            # Blended wind speed
            vis_wind = vision_pred.get("wind_speed_knots", 0.0)
            sen_wind = sensory_pred.get("wind_speed_knots", 0.0)
            fused_wind = float((w_vis * vis_wind) + (w_sen * sen_wind))

            cat_info = self.categories_config[fused_cat_id]

            # Diagnostic summary
            rationale = (
                f"Dynamic uncertainty consensus: Satellite Image (weight: {w_vis:.1%}, conf: {vis_conf:.1%}) "
                f"blended with Atmospheric Sensors (weight: {w_sen:.1%}, conf: {sen_conf:.1%}). "
                f"Visual wind estimate: {vis_wind:.1f} kt, Sensor wind estimate: {sen_wind:.1f} kt."
            )

            return {
                "category_id": fused_cat_id,
                "category_name": cat_info["name"],
                "category_code": cat_info.get("code", f"CAT_{fused_cat_id}"),
                "wind_speed_knots": round(fused_wind, 1),
                "wind_speed_kmh": round(fused_wind * 1.852, 1),
                "confidence": round(confidence * 100, 2),
                "probabilities": fused_probs.tolist(),
                "dynamic_weights": {"vision_weight": round(w_vis, 3), "sensory_weight": round(w_sen, 3)},
                "modality_used": "Dynamic Multimodal (Vision + Atmospheric)",
                "rationale": rationale,
                "vision_detail": vision_pred,
                "sensory_detail": sensory_pred
            }

        # Case 2: Vision Only
        elif vision_pred is not None:
            cat_id = vision_pred.get("category_id", 0)
            cat_info = self.categories_config[cat_id]
            wind_kt = vision_pred.get("wind_speed_knots", 0.0)
            conf = vision_pred.get("confidence", 0.0)

            return {
                "category_id": cat_id,
                "category_name": cat_info["name"],
                "category_code": cat_info.get("code", f"CAT_{cat_id}"),
                "wind_speed_knots": round(wind_kt, 1),
                "wind_speed_kmh": round(wind_kt * 1.852, 1),
                "confidence": round(conf * 100, 2) if conf <= 1.0 else round(conf, 2),
                "probabilities": vision_pred.get("probabilities", []),
                "modality_used": "Satellite Imagery Only",
                "rationale": "Prediction derived solely from satellite cloud pattern deep feature extraction.",
                "vision_detail": vision_pred,
                "sensory_detail": None
            }

        # Case 3: Sensory Only
        elif sensory_pred is not None:
            cat_id = sensory_pred.get("category_id", 0)
            cat_info = self.categories_config[cat_id]
            wind_kt = sensory_pred.get("wind_speed_knots", 0.0)
            conf = sensory_pred.get("confidence", 0.0)

            return {
                "category_id": cat_id,
                "category_name": cat_info["name"],
                "category_code": cat_info.get("code", f"CAT_{cat_id}"),
                "wind_speed_knots": round(wind_kt, 1),
                "wind_speed_kmh": round(wind_kt * 1.852, 1),
                "confidence": round(conf * 100, 2) if conf <= 1.0 else round(conf, 2),
                "probabilities": sensory_pred.get("probabilities", []),
                "modality_used": "Atmospheric Sensory Only",
                "rationale": "Prediction derived solely from atmospheric pressure and environmental measurements.",
                "vision_detail": None,
                "sensory_detail": sensory_pred
            }

        else:
            return {
                "category_id": 0,
                "category_name": "No Data Provided",
                "category_code": "N/A",
                "wind_speed_knots": 0.0,
                "wind_speed_kmh": 0.0,
                "confidence": 0.0,
                "probabilities": [],
                "modality_used": "None",
                "rationale": "Please provide either a satellite image or sensor readings.",
                "vision_detail": None,
                "sensory_detail": None
            }
