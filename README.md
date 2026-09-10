# AI/ML Tropical Cyclone Identification, Classification & Prediction System

An end-to-end deep learning and tabular intelligence system for automated Tropical Cyclone (TC) identification, category classification, and intensity prediction using multi-source satellite imagery and atmospheric sensory data.

Tuned for **NVIDIA GeForce RTX 5060 (8GB VRAM)** and a **2-day rapid execution timeline**.

---

## 1. Datasets to Download & Placement

To finish within 2 days, **do not download raw terabyte-scale satellite NetCDF files**. Instead, download the pre-cropped, benchmark datasets below and place them directly into the respective folders.

### Dataset A: Satellite Images (for Vision Transfer Learning Model)
Choose any of these active datasets:

* **Option 1 (Active Kaggle Dataset - TCIR Benchmark)**:  
  **TheCycloneImageDataset**  
  - Kaggle URL: [https://www.kaggle.com/datasets/kylegraupe/thecycloneimagedataset](https://www.kaggle.com/datasets/kylegraupe/thecycloneimagedataset)  
  - CLI Download:
    ```bash
    kaggle datasets download -d kylegraupe/thecycloneimagedataset
    ```

* **Option 2 (Alternative Active Kaggle - Tropical Cyclone Intensity Regression)**:  
  **tropical-cyclone-intensity-regression (TCIR)**  
  - Kaggle URL: [https://www.kaggle.com/datasets/ayushggarg/tropical-cyclone-intensity-regression](https://www.kaggle.com/datasets/ayushggarg/tropical-cyclone-intensity-regression)  
  - CLI Download:
    ```bash
    kaggle datasets download -d ayushggarg/tropical-cyclone-intensity-regression
    ```

* **Option 3 (Direct Zenodo Open Download - No Kaggle Account Required)**:  
  **TCIR: Tropical Cyclone Information Rendering Dataset (Chen et al.)**  
  - Direct Zenodo Link: [https://zenodo.org/records/2594740](https://zenodo.org/records/2594740)

📁 **Where to place the images**:  
Unzip the downloaded archive and place the image files and/or CSV inside:
```
sih/data/images/
```

---

### Dataset B: Atmospheric & Sensory Data (for Tabular Booster Model)
Choose either of these instant-access sources:

* **Option 1 (Official NOAA IBTrACS v4 - Direct 1-Click Download, No Login Needed)**:  
  Contains official historical cyclone track parameters (central pressure, maximum sustained wind, translation speed, basin):
  - **North Indian Ocean Basin (~15 MB)**: [https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.NI.list.v04r00.csv](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.NI.list.v04r00.csv)  
  - **Global Cyclone Dataset (~150 MB)**: [https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.ALL.list.v04r00.csv](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.ALL.list.v04r00.csv)

* **Option 2 (Active Kaggle Tabular Datasets)**:  
  - **tropical-cyclone-historical-data**: [https://www.kaggle.com/datasets/daverosenman/tropical-cyclone-historical-data](https://www.kaggle.com/datasets/daverosenman/tropical-cyclone-historical-data)  
    ```bash
    kaggle datasets download -d daverosenman/tropical-cyclone-historical-data
    ```
  - **historical-tropical-storm**: [https://www.kaggle.com/datasets/ayushggarg/historical-tropical-storm](https://www.kaggle.com/datasets/ayushggarg/historical-tropical-storm)  
    ```bash
    kaggle datasets download -d ayushggarg/historical-tropical-storm
    ```

📁 **Where to place the sensory file**:  
Place the downloaded `.csv` file into:
```
sih/data/sensory/
```
*(Rename it to `cyclone_sensory.csv` or leave the filename as is; the system auto-discovers any `.csv` in that folder).*

---

## 2. Environment Setup (RTX 5060 + CUDA)

Your system has `uv` and Python 3.12 installed. Python 3.12 is optimal for Windows with PyTorch CUDA.

Open PowerShell in the `sih` directory:

```powershell
# 1. Create a Python 3.12 virtual environment using uv
uv venv --python 3.12 .venv

# 2. Activate the virtual environment
.venv\Scripts\activate

# 3. Install PyTorch with CUDA 12.4 support (for RTX 5060)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install all remaining project dependencies
uv pip install -r requirements.txt
```

---

## 3. How to Train the Models

### Step 1: Train the Satellite Vision Model (Model 1)
Trains a transfer learning model (ConvNeXt-Tiny by default) with FP16 mixed precision on your RTX 5060. Takes **~15–20 minutes**:

```powershell
python -m src.training.train_vision
```
*Optional parameters:*
```powershell
# Change backbone or batch size:
python -m src.training.train_vision --backbone efficientnet_b2 --batch-size 32 --epochs 15
```
*Outputs saved to: `models/vision_model/best_vision_model.pt`*

---

### Step 2: Train the Atmospheric Sensory Model (Model 2)
Trains a LightGBM/XGBoost multi-class classifier and regressor on the sensory CSV. Takes **< 60 seconds**:

```powershell
python -m src.training.train_sensory
```
*Outputs saved to: `models/sensory_model/sensory_pipeline.joblib`*

---

### Step 3: Run Full Evaluation
Evaluates test accuracy, Macro-F1, wind speed RMSE/MAE, and generates confusion matrices:

```powershell
python -m src.evaluation.evaluate_all
```
*Evaluation figures saved to: `models/evaluation_results/`*

---

## 4. Launch the Visualization Dashboard

Launch the interactive Streamlit early-warning web interface:

```powershell
streamlit run app/app.py
```

### Dashboard Features:
1. **Satellite Image Upload**: Upload any cyclone IR or multispectral satellite frame.
2. **Explainable AI (Grad-CAM)**: Generates attention heatmaps highlighting storm eye walls and spiral bands.
3. **Atmospheric Sensor Sliders**: Interactively adjust central pressure, SST, wind shear, and translation speed.
4. **Multimodal Consensus**: Fuses image features and sensor readings into a unified severity category, estimated wind speed (knots & km/h), and confidence score.
