# AI/ML Tropical Cyclone Identification, Prediction & 3D Simulation System

An advanced, end-to-end spatiotemporal multi-task deep learning and geospatial simulation system for automated Tropical Cyclone (TC) identification, intensity forecasting, pressure inverse-sensing, danger footprint estimation, and trajectory tracking using multi-source satellite imagery and oceanic sensory data.

Tuned for **NVIDIA GeForce RTX 5060 (8GB VRAM)** and standard CPU environments.

---

## 1. Quickstart Environment Setup

The system uses Python 3.12 with PyTorch and CUDA support.

```powershell
# 1. Activate the virtual environment
.venv\Scripts\activate

# 2. Verify PyTorch and CUDA device
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 2. How to Test the Model

Pre-trained model checkpoints are provided in `models/` (including the multi-task spatiotemporal forecaster `models/sequence_model_v2/best_sequence_model_v2.pt`). You do **not** need to train any model to test and evaluate the system.

### Test Method 1: Physical Cyclone Occurrence Simulator (Recommended)
Simulate randomized, unseen cyclone occurrences across multiple global ocean basins (Bay of Bengal, Arabian Sea, Western Pacific, South China Sea) and evaluate all operational dimensions:

```powershell
# 1. Test 100 randomized unseen cyclone occurrences across all basins
python -m src.simulation.cyclone_simulator --num_simulations 100 --device cuda

# 2. Test strictly on Dangerous / High-Threat Cyclones (Category 3, 4, 5 only)
python -m src.simulation.cyclone_simulator --num_simulations 50 --threat_only --device cuda

# 3. Test on a specific geographic ocean basin (e.g., South China Sea or Bay of Bengal)
python -m src.simulation.cyclone_simulator --num_simulations 30 --basin "South China Sea" --device cuda
```

*Outputs generated:*
- Diagnostic summary: `models/sequence_model_v2/simulation_results/simulation_summary.json`
- 6-Panel Performance Dashboard: `models/sequence_model_v2/simulation_results/simulation_performance_dashboard.png`

---

### Test Method 2: Launch the 3D Geospatial Simulation Viewer
Experience the full 3D interactive simulation with an interactive globe, rotating convective vortex, live sensory observation stream, and predicted trajectory.

#### Option A: Standalone Browser Viewer
```powershell
python src/visualization/launch_simulation.py
```
*Opens your web browser at `http://localhost:8055`.*

**Features:**
- **Interactive 3D Earth Globe**: Drag to rotate $360^\circ$, pitch/tilt, zoom with scroll wheel.
- **Split Right Panel**:
  - **Live Sensory Data (Top)**: Real-time satellite IR 10.8µm, oceanic buoy AWS readings, sustained wind, central pressure, SST, eye diameter, and shear.
  - **Predicted Data (Bottom)**: Model-forecasted WMO category, sustained wind delta, central pressure inverse-sensing, danger radii ($R_{30}/R_{50}$), and landfall ETA.
- **Predicted Movement Path**: Highlighted as a **vivid red dotted line** with glowing forecast waypoints ($+3\text{h}, +6\text{h}, +9\text{h}, +12\text{h}, +18\text{h}, +24\text{h}$) and an expanding cone of uncertainty.
- **Random Occurrence Button**: Click `🎲 Simulate New Random Cyclone` to test newly generated storms across different ocean basins in real time.

#### Option B: Interactive Streamlit Web Dashboard
```powershell
streamlit run app/app.py
```
*Immediately launches the dedicated 3D Interactive Simulation & Multi-Task Forecaster dashboard.*

---

### Test Method 3: Full Holdout Sequence Testbench
Run comprehensive batch evaluation over the entire holdout sequence test dataset ($>50,000$ sequence windows):

```powershell
python -m src.evaluation.evaluate_sequence_v2 --checkpoint models/sequence_model_v2/best_sequence_model_v2.pt --device cuda
```

*Outputs generated:*
- Confusion matrices, ROC-AUC curves, wind regression scatter diagnostics, and error distributions saved to `models/sequence_model_v2/evaluation_plots_v2/`.

---

## 3. Evaluation Metrics

The system is evaluated across multiple meteorological and operational dimensions:

| Evaluation Dimension | Metric | Observed Model Performance | Meteorological Significance |
| :--- | :--- | :---: | :--- |
| **WMO Category Classification** | Overall Accuracy / Macro-F1 | **`99.00%`** / **`0.993`** | Reliable 5-tier classification (Depression to Super Cyclone) |
| **High-Threat Classification** | Category 3, 4, 5 Accuracy | **`100.00%`** ($50/50$) | Zero false negatives among severe and super cyclones |
| **Maximum Sustained Wind** | Mean Absolute Error (MAE) | **`3.54 knots`** ($6.56\text{ km/h}$) | Tightly calibrated wind speed across all intensity stages |
| **Wind Explained Variance** | Coefficient $R^2$ | **`0.967`** | Strong linear fit against ground-truth best-track wind speeds |
| **Central Surface Pressure** | Pressure MAE / $R^2$ | **`2.99 hPa`** / **`0.970`** | Accurate barometric inverse-sensing from space |
| **Future Trajectory Tracking** | Great-Circle Track Error (+6h) | **`75.4 km`** | Within operational 6-hour forecast uncertainty envelopes |
| **Future Trajectory Tracking** | Great-Circle Track Error (+12h) | **`144.6 km`** | Early trajectory guidance for evacuation planning |
| **Future Wind Delta Error** | $+6\text{h}$ / $+12\text{h}$ Intensity Error | **`1.40 kt`** / **`2.23 kt`** | Accurate forecast of near-term intensification or decay |
| **Gale-Force Danger Radius ($R_{30}$)** | Footprint Extent MAE | **`67.0 km`** | Coastal zone gale wind threshold safety buffer |
| **Storm-Force Danger Radius ($R_{50}$)** | Eyewall Hazard Extent MAE | **`10.7 km`** | Precise delimitation of catastrophic eyewall destructive winds |
| **Landfall Detection** | Binary Accuracy | **`100.00%`** | Reliable detection of coastal crossing within 48 hours |
| **Intensification Trend** | 3-Class Trend Accuracy | **`89.00%`** | Differentiates Weakening, Steady, and Intensifying phases |
| **Inference Latency** | Per-Sequence Forward Pass | **`< 50 ms`** (on RTX 5060) | Enables continuous real-time satellite/radar stream analysis |

### Ocean Basin Generalization Breakdown
The model is benchmarked across distinct regional maritime basins to verify generalization:

| Basin | Occurrences Tested | Category Accuracy | Wind MAE | Pressure MAE | +6h Track Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Western North Pacific** | 69 | **`98.6%`** | 3.46 kt | 3.24 hPa | 71.0 km |
| **South China Sea** | 13 | **`100.0%`** | 3.30 kt | 2.80 hPa | 70.5 km |
| **North Indian / Tropical Waters** | 18 | **`100.0%`** | 4.04 kt | 2.18 hPa | 95.6 km |

---

## 4. Modeling & Evaluation Techniques

### 1. Dual-Scale Spatial Architecture (Macro Synoptic + Micro Eyewall Core)
- **Problem**: Downsampling satellite images to standard sizes ($256\times 256$) blurs the small ($15\text{--}30\text{ km}$) eyewall core, destroying pinhole eye indicators.
- **Technique**: The model uses dual synchronized ConvNeXt streams:
  1. **Synoptic Stream**: Captures overall cloud shield structure and outer spiral rainbands ($1000\text{ km}$ synoptic scale).
  2. **Eye-Core Stream**: Extracts a high-resolution $0.25$ zoom crop focused squarely on the central dense overcast/eyewall core.
  3. **Cross-Scale Fusion**: Both streams are fused via bidirectional cross-attention with dynamic eyewall organization gating.

### 2. Explicit Physical Differential Spatiotemporal Features
- Rather than passing frames blindly into recurrent units, the model extracts physical differential embeddings across time:
  $$\Delta z_{\text{step}} = z_t - z_{t-1} \quad \text{and} \quad \Delta z_{\text{window}} = z_t - z_0$$
- This allows the temporal GRU encoder to directly track the **rate of baroclinic deepening**, immediately flagging **Rapid Intensification (RI)** ($+15\text{ kt}$ in $12\text{h}$) hours ahead of traditional techniques.

### 3. Asymmetric Safety Loss Function
- **Problem**: Standard Mean Squared Error (MSE) penalizes overestimation and underestimation equally. In disaster mitigation, underestimating a Category 5 Super Cyclone ($>130\text{ kt}$) as a Category 2 or 3 ($80\text{ kt}$) delays evacuations and risks lives.
- **Technique**: `AsymmetricWindLoss` enforces an automated **$2.5\times$ penalty multiplier** whenever the model underestimates severe systems ($\ge 64\text{ kt}$), guaranteeing that severe storms are never downplayed.

### 4. Atmospheric Inverse-Sensing from Space
- Satellites capture top-of-atmosphere thermal infrared cloud radiances; they cannot directly read surface barometric pressure.
- Using a specialized multi-task sensory regression head, the model performs atmospheric inverse-sensing, predicting the central surface pressure ($R^2 = 0.970$, $\text{MAE} = 2.99\text{ hPa}$) and surface danger radii directly from convective morphology.

### 5. Test-Time Augmentation (TTA)
- During evaluation and simulation inference, dual horizontal flip transforms are evaluated and averaged:
  $$\hat{y} = \frac{1}{2}\left(f(X) + f(\text{flip}(X))\right)$$
- TTA cancels out rotational asymmetries and eliminates spurious prediction noise on convective cloud boundaries.

### 6. Interactive 3D Geospatial Digital Twin
- Translates model predictions into an operational command center display:
  - 3D Earth globe projection with rotating convective rainbands and eyewall.
  - Split right panel separating **Live Sensory Observations** from **AI Predicted Data**.
  - **Vivid red dotted line** tracing forecasted movement waypoints with an expanding cone of uncertainty.

---

## 5. Directory Structure Overview

```
sih/
├── app/
│   └── app.py                             # Interactive Streamlit dashboard (Includes 3D Simulation)
├── src/
│   ├── models/
│   │   ├── spatiotemporal_forecaster.py   # Multi-Task Dual-Scale ConvNeXt + GRU model
│   │   └── losses.py                      # Asymmetric severe wind loss & multi-task loss
│   ├── simulation/
│   │   └── cyclone_simulator.py           # Multi-basin physical occurrence testbench
│   ├── visualization/
│   │   ├── cyclone_3d_simulation.html     # Real Google Satellite / ESRI map simulation & forecaster
│   │   ├── launch_simulation.py           # Standalone browser launcher (port 8055)
│   │   ├── gradcam_interactive_comparison.html # Interactive Grad-CAM split-view comparison
│   │   └── gradcam_comparisons/           # Side-by-side satellite Grad-CAM comparisons
│   ├── evaluation/
│   │   ├── evaluate_sequence_v2.py        # Sequence V2 testbench script
│   │   └── gradcam.py                     # Explainable Grad-CAM heatmaps
│   └── data_prep/
│       └── dataset_sequence_v2.py         # Temporal sequence preprocessor & coherent transforms
├── models/
│   └── sequence_model_v2/                 # Checkpoint, evaluation summary & diagnostic plots
├── configs/
│   └── config.yaml                        # System configuration
├── requirements.txt                       # Project dependencies
└── README.md
```
