# MolPropLab — Molecular Property Prediction Platform

An end-to-end full-stack machine learning application for predicting molecular properties from SMILES.

## 1. Core Features

- **Multi-property prediction:** Predict 10 molecular properties at once (molecular weight, LogP, LogS, pKa, boiling point, melting point, refractive index, vapor pressure, density, flash point)
- **Python ML stack:** RDKit preprocessing, molecular descriptors, ECFP (1024), PyTorch Geometric PotentialNet/GIN, LightGBM/RandomForest baselines
- **Uncertainty estimation:** MC-Dropout (GNN) / small ensemble (baseline)
- **Interpretability:** SHAP (baseline), gradient saliency (GNN) → atom-level importance JSON
- **Backend:** Node.js + Express + TypeScript; invokes Python via `child_process.spawn`
- **Frontend:** React + Vite + Tailwind + shadcn-style components + 3Dmol.js
- **Batch jobs:** Simple in-memory queue & CSV/XLSX upload with model selection

Enter a SMILES string and get predictions for all properties, uncertainty estimates, and atom-importance heatmaps in one step.

## 2. Repository Structure

### Top Level

- **`ml/`:** ML core (data processing, train/eval, inference scripts, model weights and evaluation results).
- **`server/`:** Backend API (Express + TypeScript) for prediction endpoints, batch job queue, and subprocess calls to `ml/inference.py`.
- **`frontend/`:** Web UI (React + Vite + Tailwind) with single prediction, batch prediction, model browser, comparison charts, and 3D molecular visualization.
- **`scripts/`:** Helper install scripts (mainly Linux/macOS) for one-command dependency setup.
- **`requirements.txt`:** Python dependencies.
- **`environment.yml`:** Conda environment (includes RDKit and some system deps) for one-shot environment creation.

### `ml/` (Machine Learning)

- **`ml/data/`:** Per-property datasets (CSV with `smiles,target` columns).
- **`ml/saved_models/`:** Trained model weights and normalization params (train first):
  - Baseline: `baseline_<property>_v1.pkl`
  - GNN: `gnn_<property>_v1.pth` + `gnn_<property>_v1_norm.json`
- **`ml/configs/`:** Training configuration.
- **`ml/utils.py`:** Shared utilities (SMILES cleaning, ECFP/descriptor features, graph building, scaffold split, preprocessing, etc.).
- **`ml/train_baseline.py` / `ml/train_gnn.py`:** Training scripts for baseline and GNN models.
- **`ml/eval_baseline.py` / `ml/eval_gnn.py`:** Evaluation scripts (RMSE, MAE, R², MAPE, correlation, etc.).
- **`ml/compare_models.py`:** Compare baseline vs GNN; outputs `*_comparison.json` and `comparison_summary.json` for the frontend.
- **`ml/inference.py`:** Inference entry point (single SMILES or CSV/XLSX batch; JSON/CSV output).
- **`ml/tests/`:** Python unit tests.

### `server/` (Backend)

- **`server/src/index.ts`:** Main server; `/predict`, `/batch_predict`, `/models`, etc.
- **`server/tmp/`:** Temp directory for uploads and batch processing.
- **`server/test/`:** Backend tests.

### `frontend/` (Frontend)

- **`frontend/src/App.tsx`:** Routes and page logic (single/batch prediction, model browser, comparison charts).
- **`frontend/src/ui.tsx` / `frontend/src/index.css`:** UI components and global styles.

## 3. Requirements

> Note: Environment used during development and experimentation.

### System

- **OS:** Windows 11
- **CPU:** 64-bit, Intel(R) Core(TM) i7-14650HX, 16 cores
- **RAM:** 16 GB
- **GPU:** CUDA-capable NVIDIA GPU
  - **Model:** NVIDIA GeForce RTX 4050 Laptop GPU
  - **CUDA:** 12.7
  - **Driver:** 566.24

### Python

- **Version:** 3.11.14
- **Package manager:** Conda recommended for RDKit (pip on Windows is not recommended)
- **Key library versions:** See `requirements.txt`

> You can also create the Conda environment from `environment.yml` at the project root.

### Node.js

- **Version:** 20.19.5 recommended
  - Download from [nodejs.org](https://nodejs.org/)
  - Or use the archive: [Node.js v20.19.5](https://nodejs.org/dist/v20.19.5/) and add the folder containing `node.exe` to your PATH

---

## 4. Installation

Example flow for **Conda + Windows**; macOS/Linux are similar with minor path differences.

### Quick Install (recommended on Linux/macOS)

```bash
# 1. Create and activate Conda environment
conda create -n molproplab python=3.11 -y
conda activate molproplab

# 2. Run setup script (installs all dependencies)
bash scripts/setup.sh
```

The script will:

- Check the Conda environment
- Install RDKit
- Install all Python dependencies
- Install Node.js backend and frontend dependencies

> **Note:** On Windows, use Git Bash or WSL for this script, or follow the manual steps below.

---

### Manual Installation

#### 1. Create Conda environment

```bash
conda create -n molproplab python=3.11 -y
conda activate molproplab
```

#### 2. Install RDKit

```bash
conda install -c conda-forge rdkit -y
```

> **Note:** On Windows, avoid `pip install rdkit`; Conda is more reliable.

#### 3. Install remaining Python dependencies

```bash
# From project root
pip install -r requirements.txt
```

> **Important:**
> - **NumPy 1.26.4** is used for stability on Windows
> - **PyTorch:** The default install is CPU-only. For GPU support:
>   ```bash
>   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
>   ```
> - If PyTorch Geometric is hard to install, you can skip it; GNN will **fall back to baseline** automatically

#### 4. Install Node.js dependencies

```bash
cd server
npm install

cd ../frontend
npm install
```

---

## 5. Running the Services

Keep the **Python environment activated** (`conda activate molproplab`).

### Start the backend API

From the project root:

```bash
cd server
npm run dev
```

Backend default: `http://localhost:3001`

> **Note:** The server picks the interpreter in `server/src/index.ts` via `PYTHON` / `CONDA_PREFIX` / `python`. For a different Conda env:
> - Set before start: `set PYTHON=your\python\path && npm run dev` (Windows) or `export PYTHON=...` (Unix)
> - Or set the `PYTHON` system environment variable

### Start the frontend

In a separate terminal (no strict Python requirement):

```bash
cd frontend
npm run dev
```

Frontend default: `http://localhost:5173`

---

## 6. Usage

### Web UI

1. Open `http://localhost:5173` in a browser
2. Go to **Single Prediction**
3. Enter a SMILES, e.g.:
   - `CCO` (ethanol)
   - `CC(=O)O` (acetic acid)
   - `c1ccccc1` (benzene)
4. Choose model type (`baseline` or `gnn`)
5. Click **Predict**
6. You will see:
   - A **table of predictions for all properties**, including:
     - Molecular weight (MW)
     - LogP (lipophilicity)
     - LogS (aqueous solubility)
     - pKa
     - Boiling point
     - Melting point
     - Refractive index
     - Vapor pressure
     - Density
     - Flash point
   - **Predicted value** and **uncertainty (σ)** per property
   - A **3D molecular view** with atom-level heatmap (brighter = higher importance)

### Batch prediction

1. Open **Batch Prediction**
2. Select model type (`baseline` or `gnn`)
3. Upload a CSV or XLSX with a `smiles` column (see `ml/data/logp.csv`)
4. The backend queues the job
5. Poll job status
6. Download results when done (columns: `{property}_prediction` and `{property}_uncertainty`)

---

## 7. Python CLI

### Train models

```bash
cd ml

# Train baseline models for all properties
python train_baseline.py

# Train GNN models for all properties (requires PyTorch and PyTorch Geometric)
python train_gnn.py
```

> **Note:** Training scripts train all 10 properties. For first use, run `python train_baseline.py` so predictions include all properties.

### Evaluate models

```bash
# Evaluate baseline (RMSE, MAE, R2, MAPE, etc.)
python eval_baseline.py

# Evaluate GNN
python eval_gnn.py
```

> **Note:** Evaluation uses `scaffold_split`, consistent with training, for reliable metrics.

### Compare models

```bash
# Compare baseline vs GNN for one property
python compare_models.py --property logp

# Compare all properties
python compare_models.py
```

Outputs include RMSE, MAE, R², MAPE, correlation, and which model performs better per property.

### Inference

```bash
# Single SMILES (all properties)
python inference.py --smiles "CCO" --model baseline --json
python inference.py --smiles "CCO" --model gnn --json

# Batch (CSV and XLSX)
python inference.py --csv data/logp.csv --output out.csv --model baseline
python inference.py --csv data/logp.csv --output out.csv --model gnn
```

> **Note:** Single prediction returns all 10 properties. Batch output columns use `{property}_prediction` and `{property}_uncertainty`.

---

## 8. Troubleshooting

### 1. RDKit install fails

**Problem:** Cannot install or import RDKit

**Fix:**

- Use a **Conda** environment
- Run: `conda install -c conda-forge rdkit`
- Avoid `pip install rdkit` on Windows

### 2. PyTorch DLL errors (Windows)

**Problem:** Errors like `OSError: [WinError 127] Error loading "shm.dll"` or `[WinError 182]`

**Fix:**

1. Install **Visual C++ Redistributable**
   - https://aka.ms/vs/17/release/vc_redist.x64.exe
2. If it persists, reinstall PyTorch:
   ```bash
   pip uninstall -y torch torchvision torchaudio
   pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install "numpy<2"
   ```
3. The code sets `TORCH_SHM_DISABLE=1` to reduce shared-memory issues

### 3. GNN falls back to baseline

**Problem:** You select GNN but the response `model` field is `baseline`

**Causes:**

- PyTorch or PyTorch Geometric not installed correctly
- DLL load failure on Windows

**Diagnosis:**

- Check server logs
- Inspect Python stderr from the subprocess

**Fix:**

- See “PyTorch DLL errors” above
- Install Visual C++ Redistributable
- Ensure the Python environment is activated

### 4. Port conflicts

**Problem:** Port already in use

**Fix:**

- Backend: change `PORT` in `server/src/index.ts`
- Frontend: change `server.port` in `frontend/vite.config.ts`
- Defaults:
  - Backend: `http://localhost:3001`
  - Frontend: `http://localhost:5173`

### 5. Python path

**Problem:** Server cannot find the Python interpreter

**Fix:**

- Before start: `set PYTHON=your\python\path && npm run dev` (Windows)
- Resolution order: `PYTHON` env var → `CONDA_PREFIX/python.exe` → system `python`

---

## 9. Models

### Data preprocessing

Before training and evaluation:

1. **Missing values:** Rows with empty SMILES or target are removed
2. **Outliers:** IQR method removes values outside Q1 − 1.5×IQR to Q3 + 1.5×IQR (only when n > 10)
3. **Normalization:**
   - Baseline: Z-score on input features (descriptors + ECFP)
   - GNN: Standardize targets using training-set mean and std

### Train/val/test split

- **Method:** `scaffold_split` (scaffold-based; sample-count allocation)
  - Molecules with the same scaffold stay in the same split to reduce leakage
  - Large clusters may be split to approximate target ratios
  - Default: 70% train, 15% val, 15% test
  - Training: train on 70%, early stopping on 15% val
  - Evaluation: metrics on 15% test

### Multi-property prediction

The system predicts these 10 properties:

1. **Molecular weight (MW)** — g/mol
2. **LogP** — lipophilicity
3. **LogS** — aqueous solubility, log(mol/L)
4. **pKa** — acid dissociation constant
5. **Boiling point** — °C
6. **Melting point** — °C
7. **Refractive index** — dimensionless
8. **Vapor pressure** — Pa
9. **Density** — g/cm³
10. **Flash point** — °C

Each property has its own baseline and GNN model. At inference time, all property models run and results are returned together.

### Baseline model

- **Algorithm:** LightGBM (preferred) or RandomForest (fallback)
- **Features:** Molecular descriptors + ECFP (1024 bits), Z-score normalized on training data
- **Uncertainty:** Std dev across a 3-model ensemble
- **Interpretability:** SHAP mapped to atom importance
- **Regularization:** Dynamic hyperparameters by dataset size; subsample and `colsample_bytree`
- **Use when:** Fast prediction, small/medium data, no GPU
- **Data:** At least ~20 samples per property; 100+ recommended

### GNN model

- **Algorithm:** PotentialNet (default) or GIN (configurable)
- **Features:** Graph neural network on molecular graphs; targets standardized on training data
- **Uncertainty:** MC-Dropout (20 forward passes)
- **Interpretability:** Gradient saliency → atom importance
- **Regularization:** Dropout, early stopping on val RMSE, architecture scaled to dataset size
- **Use when:** Larger datasets (500+ recommended), complex patterns, GPU helpful
- **Notes:**
  - Falls back to baseline if PyTorch/PyG unavailable
  - Baseline often wins on small data (<200 samples)
  - Hyperparameters (hidden size, epochs, etc.) adapt to dataset size

### Evaluation metrics

- **RMSE** — root mean squared error
- **MAE** — mean absolute error (robust to outliers)
- **R²** — coefficient of determination
- **MAPE** — mean absolute percentage error (cross-property comparison)
- **Correlation** — linear correlation between predictions and labels

All metrics are computed on the test set with `scaffold_split`, matching training.

---

## 10. Tests

### Python

```bash
cd ml
python -m tests.test_inference
```

### Node.js

```bash
cd server
npm test
```
