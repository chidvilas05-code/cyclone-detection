"""
Tropical Cyclone 3D Physical Simulation & Multi-Task Forecaster Web Dashboard.
Dedicated 3D geospatial digital twin featuring:
  - 3D Interactive Earth Map with rotating multi-armed convective vortex
  - Split Right Operational Panel: Live Sensory Observations vs AI Multi-Task Predictions
  - Predicted Cyclone Movement Path rendered as a vivid Red Dotted Line
  - Multi-basin simulation scenarios & procedural random storm generation
"""

import sys
import os
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Set page config for immersive 3D visualization
st.set_page_config(
    page_title="3D Cyclone Physical Simulation & Forecaster",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 100%;
    }
    .main-header {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #4B5563;
        margin-bottom: 0.8rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/cyclone.png", width=64)
st.sidebar.title("Operational Command")

st.sidebar.markdown("""
### 🌐 System Architecture
- **Spatiotemporal Engine**: Multi-Task Dual-Stream ConvNeXt + Temporal GRU (V2)
- **3D Geospatial Engine**: Interactive Orthographic Globe & Vortex Shader
- **Safety Objective**: Asymmetric Severe Wind Loss ($2.5\\times$ safety penalty)
- **Inference Mode**: Test-Time Augmentation (TTA) with mixed-precision
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Maritime Simulation Basins")
st.sidebar.markdown("""
Switch between active ocean basins directly inside the 3D map:
- 🌊 **Bay of Bengal**: Super Cyclone *Amphan*
- 🌊 **Arabian Sea**: Extremely Severe Cyclone *Biparjoy*
- 🌊 **Western North Pacific**: Cat 5 Super Typhoon *Haiyan*
- 🌊 **South China Sea**: Severe Typhoon *Yagi*
- 🎲 **Random Generator**: Brand-new unseen occurrences in any basin
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Standalone Full-Screen Viewer")
st.sidebar.info(
    "To launch the 3D visualization directly in a full-screen browser window, run:\n\n"
    "`python src/visualization/launch_simulation.py`"
)

# Main Title & Subtitle
st.markdown('<div class="main-header">🌀 AI Tropical Cyclone 3D Geospatial Simulation & Multi-Task Forecaster</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Interactive 3D Earth Map • Live Sensory Telemetry Stream vs AI Multi-Task Predictions • Predicted Movement Path (Red Dotted Line)</div>',
    unsafe_allow_html=True
)

# Load and Render 3D Simulation
sim_html_path = Path("src/visualization/cyclone_3d_simulation.html")

if sim_html_path.exists():
    with open(sim_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    components.html(html_content, height=920, scrolling=False)
else:
    st.error(f"3D Simulation file not found at: `{sim_html_path}`")
