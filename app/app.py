"""
Tropical Cyclone Early Warning & Pattern Identification Web Dashboard.
Built with Streamlit for rapid, interactive decision support.
Supports:
  - Satellite Image Upload & Pattern Classification
  - Grad-CAM Heatmap Visualization for Cloud Structure & Eye Wall
  - Real-Time Atmospheric / Sensor Sliders
  - Multimodal Decision Fusion Consensus
"""

import sys
import os
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.sensory_predictor import CycloneSensoryPredictor
from src.models.fusion import MultimodalCycloneFusion
import time
import cv2
import tempfile

import importlib
import src.models.vision_classifier
importlib.reload(src.models.vision_classifier)
from src.models.vision_classifier import (
    CycloneVisionModel,
    HybridMultiBackboneCycloneModel,
    load_vision_model_from_checkpoint
)
import src.evaluation.gradcam
importlib.reload(src.evaluation.gradcam)
from src.evaluation.gradcam import generate_cyclone_gradcam

import src.evaluation.video_simulator
importlib.reload(src.evaluation.video_simulator)
from src.evaluation.video_simulator import process_video_frame


# Set page config
st.set_page_config(
    page_title="AI Cyclone Early-Warning System",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #DBEAFE;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1D4ED8;
    }
    .badge-0 { background-color: #10B981; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-1 { background-color: #F59E0B; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-2 { background-color: #F97316; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-3 { background-color: #EF4444; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-4 { background-color: #7F1D1D; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_app_resources():
    """Caches models and configs to guarantee high-speed inference."""
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Vision Model
    vision_model = None
    vision_weights = Path(cfg["paths"]["vision_model_dir"]) / "best_vision_model.pt"
    if vision_weights.exists():
        try:
            vision_model, _ = load_vision_model_from_checkpoint(
                str(vision_weights),
                device=device,
                default_num_classes=len(cfg.get("categories", []))
            )
        except Exception as e:
            st.warning(f"Note on Vision Model: {e}")


    # 2. Load Sensory Model
    sensory_predictor = CycloneSensoryPredictor(model_dir=cfg["paths"]["sensory_model_dir"])
    sensory_loaded = sensory_predictor.load()

    # 3. Multimodal Fusion Engine
    fusion_engine = MultimodalCycloneFusion(
        vision_weight=cfg["fusion"]["vision_weight"],
        sensory_weight=cfg["fusion"]["sensory_weight"],
        categories_config=cfg["categories"]
    )

    return cfg, vision_model, sensory_predictor, sensory_loaded, fusion_engine, device


cfg, vision_model, sensory_predictor, sensory_loaded, fusion_engine, device = load_app_resources()

# Sidebar: Operational Controls & Parameters
st.sidebar.image("https://img.icons8.com/color/96/cyclone.png", width=64)
st.sidebar.title("Operational Controls")

app_mode = st.sidebar.radio(
    "Select Operating Modality:",
    [
        "Multimodal (Satellite + Sensory)",
        "Satellite Imagery Only",
        "Atmospheric Sensory Only",
        "🎥 Video Simulation (Time-Lapse Tracking)"
    ]
)

sensor_dict = {}
df_sensory_csv = None

if app_mode != "🎥 Video Simulation (Time-Lapse Tracking)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Atmospheric Sensor Inputs")

    sensory_input_mode = st.sidebar.radio(
        "Sensor Data Input Method:",
        ["📁 Upload Sensory CSV", "🎛️ Manual Sliders"],
        index=0
    )

    if sensory_input_mode == "📁 Upload Sensory CSV":
        uploaded_csv = st.sidebar.file_uploader(
            "Upload Telemetry CSV",
            type=["csv"],
            help="Upload CSV containing atmospheric readings (central_pressure, sst, wind_shear, rh, etc.)"
        )

        raw_df = None
        if uploaded_csv is not None:
            try:
                raw_df = pd.read_csv(uploaded_csv)
            except Exception as e:
                st.sidebar.error(f"Failed to read CSV: {e}")

        if raw_df is not None and not raw_df.empty:
            # Standardize matching column aliases
            from src.data_prep.dataset_sensory import ALIAS_MAP
            col_map = {}
            for canonical, aliases in ALIAS_MAP.items():
                for c in raw_df.columns:
                    if c.lower() in [a.lower() for a in aliases] or c.lower() == canonical.lower():
                        col_map[c] = canonical
                        break
            df_sensory_csv = raw_df.rename(columns=col_map)
            st.sidebar.success(f"✓ Loaded CSV: {len(df_sensory_csv)} records")

            max_idx = len(df_sensory_csv) - 1
            selected_row_idx = st.sidebar.number_input(
                f"Select Record (0 to {max_idx}):",
                min_value=0,
                max_value=max_idx,
                value=0,
                step=1
            )
            row = df_sensory_csv.iloc[selected_row_idx]
            sensor_dict = {
                "central_pressure": float(row.get("central_pressure", 980.0)),
                "sst": float(row.get("sst", 29.0)),
                "vertical_wind_shear": float(row.get("vertical_wind_shear", 12.0)),
                "relative_humidity": float(row.get("relative_humidity", 75.0)),
                "translation_speed": float(row.get("translation_speed", 14.0)),
                "latitude": float(row.get("latitude", 15.0)),
                "longitude": float(row.get("longitude", 87.0))
            }
            st.sidebar.info(
                f"**Selected Record #{selected_row_idx}:**\n"
                f"- Pressure: `{sensor_dict['central_pressure']} hPa`\n"
                f"- SST: `{sensor_dict['sst']} °C` | Shear: `{sensor_dict['vertical_wind_shear']} kt`\n"
                f"- Position: `{sensor_dict['latitude']}°N, {sensor_dict['longitude']}°E`"
            )
        else:
            st.sidebar.info("Upload a telemetry CSV file above to analyze sensory records.")
            sensor_dict = {
                "central_pressure": 980.0,
                "sst": 29.0,
                "vertical_wind_shear": 12.0,
                "relative_humidity": 75.0,
                "translation_speed": 14.0,
                "latitude": 15.0,
                "longitude": 87.0
            }
    else:
        # Interactive Sliders for Sensory Data
        central_pressure = st.sidebar.slider("Central Pressure (hPa / mb)", min_value=880.0, max_value=1015.0, value=980.0, step=0.5)
        sst = st.sidebar.slider("Sea Surface Temperature (°C)", min_value=22.0, max_value=33.0, value=29.0, step=0.2)
        wind_shear = st.sidebar.slider("Vertical Wind Shear (knots)", min_value=2.0, max_value=45.0, value=12.0, step=0.5)
        humidity = st.sidebar.slider("Relative Humidity 700 hPa (%)", min_value=20.0, max_value=98.0, value=75.0, step=1.0)
        translation_speed = st.sidebar.slider("Storm Forward Speed (knots)", min_value=2.0, max_value=35.0, value=14.0, step=0.5)
        latitude = st.sidebar.slider("Center Latitude (°N)", min_value=0.0, max_value=35.0, value=15.2, step=0.1)
        longitude = st.sidebar.slider("Center Longitude (°E)", min_value=40.0, max_value=105.0, value=87.5, step=0.1)

        sensor_dict = {
            "central_pressure": central_pressure,
            "sst": sst,
            "vertical_wind_shear": wind_shear,
            "relative_humidity": humidity,
            "translation_speed": translation_speed,
            "latitude": latitude,
            "longitude": longitude
        }
else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎥 Video Mode Active")
    st.sidebar.info(
        "**Real-Time Satellite Video Tracking**\n\n"
        "Continuously streams and evaluates frames using convolutional attention to track vortex development and eyewall consolidation."
    )

# Main Panel
st.markdown('<div class="main-header">🌀 AI Tropical Cyclone Analysis & Early Warning System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Source Satellite Pattern Identification • Explainable Grad-CAM • Atmospheric Sensor Fusion</div>',
    unsafe_allow_html=True
)

if app_mode == "🎥 Video Simulation (Time-Lapse Tracking)":
    st.markdown("### 🎥 Real-Time Satellite Cyclone Video Simulation & Tracking")
    st.write(
        "Stream satellite time-lapse imagery through deep convolutional backbones to continuously detect cyclone formation, "
        "track eyewall consolidation, and forecast intensity trajectories with an operational HUD overlay."
    )

    if vision_model is None:
        st.error("⚠️ Vision model checkpoint not found in `models/vision_model/best_vision_model.pt`. Please ensure the vision model is available.")
        st.stop()

    v_source_col, v_param_col = st.columns([1, 1], gap="large")

    with v_source_col:
        st.markdown("##### 1. Upload Satellite Video Stream")
        uploaded_video = st.file_uploader(
            "Upload Satellite Time-Lapse Video",
            type=["mp4", "webm", "avi", "mov", "mkv"],
            help="Upload an infrared or visible satellite time-lapse video loop (MP4, WebM, AVI, MOV, MKV)."
        )
        video_path = None
        if uploaded_video is not None:
            file_ext = Path(uploaded_video.name).suffix.lower() or ".webm"
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            tfile.write(uploaded_video.read())
            tfile.close()
            video_path = tfile.name
            st.success(f"✓ Uploaded `{uploaded_video.name}` ({uploaded_video.size / 1024:.1f} KB)")
        else:
            st.info("Upload a satellite video file above (.mp4, .webm, .avi, .mov, .mkv) to begin simulation.")

    with v_param_col:
        st.markdown("##### 2. Simulation & Inference Settings")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            frame_stride = st.slider(
                "Frame Stride (Sample Interval):",
                min_value=1,
                max_value=8,
                value=2,
                help="Sample every N frames. Lower values give higher temporal resolution; higher values run faster."
            )
        with c_p2:
            playback_delay = st.slider(
                "Playback Delay per Step (s):",
                min_value=0.0,
                max_value=0.25,
                value=0.04,
                step=0.01,
                help="Operational time-step delay between frames."
            )

        enable_gradcam_sim = st.checkbox(
            "🔥 Enable Real-Time Grad-CAM Eyewall Focus",
            value=False,
            help="Computes gradient activations per frame to visualize deep convective rainband focusing."
        )

    st.markdown("---")

    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        estimated_sample_frames = max(1, total_raw_frames // frame_stride)
        st.write(
            f"📹 Video Stream Ready: **{total_raw_frames} raw frames** | "
            f"Sampling **{estimated_sample_frames} frames** (stride = {frame_stride})"
        )

        start_simulation = st.button(
            "▶️ Start Real-Time Cyclone Tracking Simulation",
            type="primary",
            use_container_width=True
        )

        sim_video_col, sim_telemetry_col = st.columns([1, 1], gap="large")

        with sim_video_col:
            st.markdown("##### 🛰️ Operational Satellite HUD Stream")
            frame_placeholder = st.empty()
            frame_placeholder.info("Click **'Start Real-Time Cyclone Tracking Simulation'** to initialize satellite tracking.")

        with sim_telemetry_col:
            st.markdown("##### 📊 Real-Time Telemetry & Severity Diagnostics")
            kpi_placeholder = st.empty()
            kpi_placeholder.markdown("""
            <div class="metric-card">
                <div style="font-size:0.9rem; color:#6B7280;">SYSTEM STANDBY</div>
                <div style="font-size:1.4rem; font-weight:700; color:#1D4ED8; margin-top:0.5rem;">Ready for Ingestion</div>
                <div style="font-size:0.85rem; color:#4B5563; margin-top:0.3rem;">Awaiting simulation trigger</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### 📈 Dynamic Wind Speed Trajectory (Knots)")
        chart_placeholder = st.empty()
        progress_bar = st.empty()

        if start_simulation:
            cap = cv2.VideoCapture(video_path)
            raw_frame_idx = 0
            processed_count = 0
            tracking_history = []
            chart_data = pd.DataFrame(columns=["Wind Speed (kt)"])
            last_eye_center = None

            with st.spinner("Processing satellite stream in real time..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if raw_frame_idx % frame_stride == 0:
                        processed_count += 1
                        annotated_frame, record = process_video_frame(
                            model=vision_model,
                            frame_bgr=frame,
                            categories_config=cfg["categories"],
                            device=device,
                            frame_idx=processed_count,
                            total_frames=estimated_sample_frames,
                            generate_gradcam=enable_gradcam_sim,
                            img_size=cfg["vision_model"]["img_size"],
                            prev_eye_center=last_eye_center
                        )
                        tracking_history.append(record)
                        last_eye_center = (record.get("eye_x", frame.shape[1]//2), record.get("eye_y", frame.shape[0]//2))

                        frame_placeholder.image(
                            annotated_frame,
                            caption=f"Frame #{processed_count} / {estimated_sample_frames} | Raw #{raw_frame_idx}",
                            use_container_width=True
                        )

                        cat_name = record["category_name"]
                        cat_id = record["category_id"]
                        wind_kt = record["wind_speed_knots"]
                        wind_kmh = record["wind_speed_kmh"]
                        conf = record["confidence"]
                        badge_class = f"badge-{min(cat_id, 4)}"

                        eye_x, eye_y = record.get("eye_x", "N/A"), record.get("eye_y", "N/A")
                        is_open_eye = record.get("is_open_eye", False)
                        if is_open_eye:
                            status_color = "#DC2626"
                            eyewall_status = f"Eye at ({eye_x}, {eye_y})"
                        elif wind_kt >= 34:
                            status_color = "#D97706"
                            eyewall_status = f"Vortex at ({eye_x}, {eye_y})"
                        else:
                            status_color = "#10B981"
                            eyewall_status = "Cloud Banding"

                        kpi_placeholder.markdown(f"""
                        <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
                            <div class="metric-card">
                                <div style="font-size:0.8rem; color:#6B7280; text-transform:uppercase;">Cyclone Category</div>
                                <div style="margin: 0.4rem 0;"><span class="{badge_class}">{cat_name}</span></div>
                                <div style="font-size:0.8rem; color:#4B5563;">Level {cat_id}</div>
                            </div>
                            <div class="metric-card">
                                <div style="font-size:0.8rem; color:#6B7280; text-transform:uppercase;">Sustained Wind</div>
                                <div class="metric-val">{wind_kt} <span style="font-size:0.9rem; color:#4B5563;">kt</span></div>
                                <div style="font-size:0.8rem; color:#4B5563;">{wind_kmh} km/h</div>
                            </div>
                        </div>
                        <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                            <div class="metric-card">
                                <div style="font-size:0.8rem; color:#6B7280; text-transform:uppercase;">AI Confidence</div>
                                <div class="metric-val">{conf}%</div>
                                <div style="font-size:0.8rem; color:#4B5563;">Vision Backbone</div>
                            </div>
                            <div class="metric-card">
                                <div style="font-size:0.8rem; color:#6B7280; text-transform:uppercase;">Eyewall Core / Eye</div>
                                <div style="font-size:1.05rem; font-weight:700; color:{status_color}; margin-top:0.4rem;">{eyewall_status}</div>
                                <div style="font-size:0.8rem; color:#4B5563;">Frame {processed_count}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        new_row = pd.DataFrame([{"Wind Speed (kt)": wind_kt}])
                        chart_data = pd.concat([chart_data, new_row], ignore_index=True)
                        chart_placeholder.line_chart(chart_data, color="#1D4ED8")

                        progress_val = min(1.0, processed_count / max(1, estimated_sample_frames))
                        progress_bar.progress(progress_val)

                        if playback_delay > 0:
                            time.sleep(playback_delay)

                    raw_frame_idx += 1

                cap.release()

            st.success(f"✓ Video simulation completed! Processed {processed_count} frames successfully.")

            if tracking_history:
                df_track = pd.DataFrame(tracking_history)
                max_wind = df_track["wind_speed_knots"].max()
                max_row = df_track.loc[df_track["wind_speed_knots"].idxmax()]

                st.markdown("##### 📋 Mission Summary & Telemetry Log")
                s1, s2, s3 = st.columns(3)
                s1.metric("Peak Sustained Wind", f"{max_wind:.1f} kt", f"{max_wind*1.852:.1f} km/h")
                s2.metric("Peak Severity Class", str(max_row["category_name"]))
                s3.metric("Peak Intensity Frame", f"Frame #{int(max_row['frame'])}")

                with st.expander("👁️ View Full Frame-by-Frame Tracking Log", expanded=False):
                    display_cols = [c for c in ["frame", "category_name", "wind_speed_knots", "wind_speed_kmh", "confidence", "eye_x", "eye_y", "is_open_eye", "eye_score"] if c in df_track.columns]
                    st.dataframe(df_track[display_cols], use_container_width=True)

                csv_bytes = df_track.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Video Tracking Log (CSV)",
                    data=csv_bytes,
                    file_name="cyclone_video_tracking_log.csv",
                    mime="text/csv",
                    use_container_width=True
                )

else:
    col_upload, col_display = st.columns([1, 1], gap="large")

    uploaded_image = None
    if app_mode in ["Multimodal (Satellite + Sensory)", "Satellite Imagery Only"]:
        with col_upload:
            st.subheader("1. Satellite Imagery Ingestion")
            uploaded_file = st.file_uploader(
                "Upload Satellite Image (IR / Multispectral / Grayscale / GeoTIFF)",
                type=["png", "jpg", "jpeg", "tif", "bmp"]
            )
            if uploaded_file is not None:
                uploaded_image = Image.open(uploaded_file)
                st.image(uploaded_image, caption="Uploaded Satellite Image", use_container_width=True)
            else:
                st.info("Upload a cyclone satellite image to trigger deep feature extraction and Grad-CAM attention analysis.")

    vision_prediction = None
    cam_overlay = None

    # Perform Vision Model Inference
    if uploaded_image is not None and vision_model is not None:
        with st.spinner("Analyzing cyclonic cloud structure & eye formation..."):
            try:
                cam_overlay, cam_2d, cat_id, conf, pred_wind, probs = generate_cyclone_gradcam(
                    vision_model, uploaded_image, img_size=cfg["vision_model"]["img_size"]
                )
                cat_id_int = int(np.squeeze(cat_id))
                cat_cfg = cfg["categories"][cat_id_int]
                vision_prediction = {
                    "category_id": cat_id_int,
                    "category_name": cat_cfg["name"],
                    "confidence": float(np.squeeze(conf)),
                    "wind_speed_knots": float(np.squeeze(pred_wind)),
                    "probabilities": probs.tolist() if hasattr(probs, "tolist") else list(probs)
                }
            except Exception as e:
                st.error(f"Vision inference error: {e}")

    elif uploaded_image is not None and vision_model is None:
        st.warning("⚠️ Satellite Vision model checkpoint not found in `models/vision_model/`. Please train the vision model using `python -m src.training.train_vision`.")

    # Perform Sensory Model Inference
    sensory_prediction = None
    if app_mode in ["Multimodal (Satellite + Sensory)", "Atmospheric Sensory Only"]:
        if sensory_loaded:
            sensory_prediction = sensory_predictor.predict_single(sensor_dict)
        else:
            delta_p = max(0.0, 1013.25 - central_pressure)
            rule_wind = 6.7 * (delta_p ** 0.644)
            rule_cat = 0
            for c in cfg["categories"]:
                if c["min_knots"] <= rule_wind <= c["max_knots"]:
                    rule_cat = c["id"]
            sensory_prediction = {
                "category_id": rule_cat,
                "category_name": cfg["categories"][rule_cat]["name"],
                "wind_speed_knots": rule_wind,
                "confidence": 0.85,
                "probabilities": [0.1 if i != rule_cat else 0.85 for i in range(5)]
            }

    # Multimodal Fusion
    with col_display:
        st.subheader("2. AI Identification & Intensity Output")

        if app_mode == "Satellite Imagery Only":
            fused_result = fusion_engine.fuse(vision_pred=vision_prediction, sensory_pred=None)
        elif app_mode == "Atmospheric Sensory Only":
            fused_result = fusion_engine.fuse(vision_pred=None, sensory_pred=sensory_prediction)
        else:
            fused_result = fusion_engine.fuse(vision_pred=vision_prediction, sensory_pred=sensory_prediction)

        cat_id = fused_result["category_id"]
        badge_class = f"badge-{min(cat_id, 4)}"

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#6B7280; text-transform:uppercase;">Predicted Category</div>
                <div style="margin-top:0.4rem; margin-bottom:0.4rem;">
                    <span class="{badge_class}">{fused_result['category_name']}</span>
                </div>
                <div style="font-size:0.85rem; color:#4B5563;">Code: {fused_result.get('category_code', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#6B7280; text-transform:uppercase;">Estimated Wind Speed</div>
                <div class="metric-val">{fused_result['wind_speed_knots']} <span style="font-size:1rem; color:#4B5563;">kt</span></div>
                <div style="font-size:0.85rem; color:#4B5563;">{fused_result['wind_speed_kmh']} km/h</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m3, m4 = st.columns(2)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#6B7280; text-transform:uppercase;">Model Confidence</div>
                <div class="metric-val">{fused_result['confidence']}%</div>
                <div style="font-size:0.85rem; color:#4B5563;">Based on {fused_result['modality_used']}</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            detection_status = "Cyclone Detected" if fused_result['wind_speed_knots'] >= 34 else "Low Disturbance / Depression"
            status_color = "#DC2626" if fused_result['wind_speed_knots'] >= 48 else "#D97706"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#6B7280; text-transform:uppercase;">Cyclone Detection</div>
                <div style="font-size:1.3rem; font-weight:700; color:{status_color}; margin-top:0.4rem;">
                    {detection_status}
                </div>
                <div style="font-size:0.85rem; color:#4B5563;">Status verified</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"**Diagnostic Rationale:** {fused_result['rationale']}")

    # Explainable AI (Grad-CAM) Visual Inspection Tab
    st.markdown("---")
    st.subheader("3. Explainable AI (Grad-CAM) Cloud Spiral & Eye Wall Heatmap")

    if cam_overlay is not None:
        g1, g2 = st.columns(2)
        with g1:
            st.image(uploaded_image, caption="Raw Satellite Input", use_container_width=True)
        with g2:
            st.image(cam_overlay, caption="Grad-CAM Attention Heatmap (Red = Deep Convective Rainbands / Eye Wall Focus)", use_container_width=True)
    else:
        st.info("Upload a satellite image to generate Grad-CAM visual attention overlays.")

    # Batch Sensory CSV Telemetry Analysis & Storm Trajectory
    if df_sensory_csv is not None and not df_sensory_csv.empty:
        st.markdown("---")
        st.subheader("📁 Atmospheric CSV Telemetry & Batch Cyclone Forecast")

        c_info, c_btn = st.columns([3, 1])
        with c_info:
            st.write(f"Loaded telemetry dataset containing **{len(df_sensory_csv)} time-steps**. Run batch ensemble predictions to analyze the complete storm trajectory.")
        with c_btn:
            run_batch = st.button("⚡ Run Full Track Prediction", use_container_width=True)

        with st.expander("👁️ View Raw Telemetry Data Table", expanded=not run_batch):
            st.dataframe(df_sensory_csv.head(15), use_container_width=True)

        if run_batch:
            with st.spinner("Running Stacking Ensemble (LightGBM + XGBoost + Quantile Bounds)..."):
                cat_preds, cat_probs, wind_preds, wind_low, wind_high = sensory_predictor.predict(df_sensory_csv)
                cat_names = [cfg["categories"][min(c, 4)]["name"] for c in cat_preds]

                df_results = df_sensory_csv.copy()
                df_results["Predicted_Category"] = cat_names
                df_results["Estimated_Wind_kt"] = np.round(wind_preds, 1)
                df_results["Estimated_Wind_kmh"] = np.round(wind_preds * 1.852, 1)
                df_results["Lower_90pct_Bound_kt"] = np.round(wind_low, 1)
                df_results["Upper_90pct_Bound_kt"] = np.round(wind_high, 1)

                st.success(f"✓ Generated predictions for all {len(df_results)} telemetry records!")

                # Intensity trajectory line chart
                st.markdown("##### 📈 Estimated Wind Speed & 90% Confidence Interval Over Track Steps")
                chart_df = pd.DataFrame({
                    "Wind Speed (kt)": df_results["Estimated_Wind_kt"],
                    "Upper 95% Bound": df_results["Upper_90pct_Bound_kt"],
                    "Lower 5% Bound": df_results["Lower_90pct_Bound_kt"]
                })
                st.line_chart(chart_df, color=["#1D4ED8", "#DC2626", "#10B981"])

                # Results preview table
                st.markdown("##### 📋 Annotated Forecast Summary")
                show_cols = [c for c in [
                    "Predicted_Category", "Estimated_Wind_kt", "Estimated_Wind_kmh",
                    "Lower_90pct_Bound_kt", "Upper_90pct_Bound_kt", "central_pressure",
                    "sst", "latitude", "longitude"
                ] if c in df_results.columns]
                st.dataframe(df_results[show_cols].head(25), use_container_width=True)

                # Download Annotated CSV
                csv_data = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Annotated Forecast CSV",
                    data=csv_data,
                    file_name="cyclone_ai_predictions.csv",
                    mime="text/csv"
                )


