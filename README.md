# Hierarchical Local-Global Framework for Real-Time Cyber-Physical Attack Detection in Industrial Control Systems

This repository contains the source code, models, and evaluation scripts for a hierarchical anomaly detection framework designed for Industrial Control Systems (ICS). The system combines lightweight edge-based local models with a centralized global model, fused via a novel topology-aware attention mechanism to detect and localize cyber-physical attacks.

---

## Table of Contents

- [Overview](#overview)
- [Core Contributions](#core-contributions)
  - [1. GenAI-Assisted Feature Reduction Framework](#1-genai-assisted-feature-reduction-framework)
  - [2. Attack-Point Localized Labeling (APLL)](#2-attack-point-localized-labeling-apll)
  - [3. Hierarchical Local-Global Architecture](#3-hierarchical-local-global-architecture)
  - [4. Topology-Constrained Attention Fusion (TCAF)](#4-topology-constrained-attention-fusion-tcaf)
  - [5. Edge-Deployability Analysis](#5-edge-deployability-analysis)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)

---

## Overview

Industrial Control Systems (ICS) operate in environments where cyber and physical processes are tightly coupled, making anomaly detection both critical and challenging. This framework addresses that challenge through a two-tier hierarchical design:

- **Local Layer**: Lightweight Markov + Dwell-Time models run directly on distributed edge nodes (one per PLC), exploiting deterministic ladder logic and physical timing boundaries.
- **Global Layer**: A centralized USAD (UnSupervised Anomaly Detection) model processes a reduced cross-PLC feature set to capture complex, multi-stage physical constraints.
- **Fusion Layer**: A Topology-Constrained Attention Fusion (TCAF) mechanism combines local and global scores using plant topology as a structural prior.

The framework is validated on the **SWaT (Secure Water Treatment)** dataset and demonstrated to be edge-deployable with sub-millisecond local inference times.

<img width="878" height="795" alt="image" src="https://github.com/user-attachments/assets/ba966678-8b1f-479d-8164-3c35d15c8f0e" />


---

## Core Contributions

### 1. GenAI-Assisted Feature Reduction Framework

- Utilizes a Large Language Model (LLM) with Human-In-The-Loop (HITL) feedback to parse Piping and Instrumentation Diagrams (P&ID) and extract inter-PLC dependent registers.
- Reduces the 51 raw SWaT features down to **23 essential cross-PLC features** while dropping 11 zero-variance constant features.
- Reduces manual review effort while completely matching the expert-curated human baseline.

### 2. Attack-Point Localized Labeling (APLL)

- Provides a systematic methodology to derive clean, per-subsystem ground truth in physically coupled ICS environments.
- Classifies attacks as single-subsystem or multi-subsystem, reserving multi-subsystem attacks exclusively for the global layer.
- Eliminates the label noise introduced by naive impact-based labeling, assigning positive labels only to the stage where the attack originates.

### 3. Hierarchical Local-Global Architecture

**Global Layer**
- Deploys a USAD model on a centralized server using the 23-feature subset.
- USAD uses an adversarial two-decoder setup to capture complex physical constraints.
- Achieves an **F1-score of 0.8377** and **99.53% precision**.

**Local Layer**
- Deploys Markov + Dwell-Time models on distributed edge nodes, one per PLC.
- Maps deterministic PLC ladder logic and strict physical timing boundaries.
- Outperforms statistical and deep learning local baselines with a **Micro-F1 of 0.6310**.

### 4. Topology-Constrained Attention Fusion (TCAF)

- A learned fusion mechanism that combines the global USAD score with the six local Markov anomaly scores.
- Restricts attention weights exclusively to physically adjacent stages using a fixed 6×6 binary adjacency matrix that encodes plant topology, including the Stage 6 → Stage 3 backwash edge.
- Improves anomaly **F1-score to 0.9063** and **Recall to 0.8295**, while reducing False Positives to **56** (compared to 132 under simple rule-based fusion).
- Maintains per-stage attack localization even for attacks that bypass individual local models, indicated via the attention distribution matrix.

### 5. Edge-Deployability Analysis

- Validates real-time operational feasibility through detailed CPU, memory, network, and inference time profiling.
- Filtering Modbus polling from 40 active baseline tags to 23 cross-stage tags reduces Modbus network bandwidth from **598.40 kbps to 343.84 kbps**.
- Markov + Dwell-Time edge models execute with **sub-millisecond inference times** (0.2111 ms average).

---

## Repository Structure

```
Team_33/
│
├── datasets/                          # Data preparation and preprocessing
│   ├── swat/                          # Raw and processed SWaT dataset files
│   ├── hai/                           # HAI dataset files
│   ├── wadi/                          # WADI dataset files
│   ├── clean_data.py                  # Data cleaning utilities
│   ├── find_min_max.py                # Feature range analysis
│   ├── plot_columns.py                # Dataset visualization
│   ├── swat-preprocessing.ipynb       # SWaT preprocessing notebook
│   └── dataset_link.txt               # Links to download raw datasets
│
├── global_models/                     # Centralized global anomaly detection
│   ├── usad/                          # USAD model (primary global model)
│   │   ├── usad.ipynb                 # Training and evaluation notebook
│   │   ├── usad_swat_model_full.pth   # Full saved model weights
│   │   ├── usad_swat_weights.pth      # Encoder/decoder weights
│   │   ├── swat_scaler.pkl            # Feature scaler
│   │   └── roc_curve_usad_swat.png    # ROC curve output
│   ├── autoencoder/                   # Autoencoder baseline
│   ├── isolation_forest/              # Isolation Forest baseline
│   ├── lstm_ae/                       # LSTM Autoencoder baseline
│   ├── vae/                           # Variational Autoencoder baseline
│   ├── attack.csv                     # Attack subset (23 features)
│   ├── normal.csv                     # Normal subset (23 features)
│   └── split_stages.py                # Script to split data by stage
│
├── local_models/                      # Edge-deployed per-PLC local models
│   ├── 01_markov_dwelltime_model.ipynb   # Markov + Dwell-Time (primary local model)
│   ├── 02_pasad_model.ipynb              # PASAD baseline
│   ├── 03_invariant_rules_model.ipynb    # Invariant Rules baseline
│   ├── 04_lstm_autoencoder_model.ipynb   # LSTM-AE baseline
│   ├── 05_cusum_ewma_model.ipynb         # CUSUM/EWMA baseline
│   ├── models/                           # Saved local model artifacts
│   ├── dataset/, dataset_new/, dataset_raw/  # Per-stage split datasets
│   ├── train_iforest.py                  # Isolation Forest training script
│   └── train_rf.ipynb                    # Random Forest training notebook
│
├── fusion_models/                     # Score fusion layer
│   ├── learned_fusion.ipynb           # TCAF learned attention fusion (primary)
│   ├── simple_fusion.ipynb            # Rule-based fusion baseline
│   ├── train_learned_fusion.py        # Training script for TCAF
│   ├── combined_output_simple.csv     # Fused outputs (simple fusion)
│   ├── global_output.csv              # Raw global model scores
│   └── actual_output.csv              # Ground truth labels
│
├── baseline_models/                   # Global-layer baselines (full feature set)
│   ├── usad/                          # USAD on full 51 features
│   ├── autoencoder/                   # Autoencoder baseline
│   ├── isolation_forest/              # Isolation Forest baseline
│   ├── lstm_ae/                       # LSTM-AE baseline
│   ├── vae/                           # VAE baseline
│   ├── random_forest/                 # Random Forest baseline
│   ├── attack.csv / normal.csv        # Full-feature dataset splits
│   └── split_stages.py                # Stage splitting utility
│
├── testbed/                           # Live inference and edge simulation
│   ├── local/                         # Per-PLC inference scripts
│   │   ├── 01_markov_inference.py     # Markov + Dwell-Time live inference
│   │   ├── 02_pasad_inference.py      # PASAD live inference
│   │   ├── 03_invariant_inference.py  # Invariant rules live inference
│   │   ├── 04_lstm_ae_inference.py    # LSTM-AE live inference
│   │   ├── 05_cusum_ewma_inference.py # CUSUM/EWMA live inference
│   │   ├── models/                    # Serialised edge model files
│   │   └── mapping_by_stage.json      # Stage-to-sensor register mapping
│   ├── global/                        # Global model live inference
│   ├── fusion/                        # Fusion layer live inference
│   ├── baseline/                      # Baseline model inference
│   ├── plc_writer.py                  # Modbus PLC data writer
│   ├── plc_ones_writer.py             # Utility for writing constant signals
│   ├── rf_live_inference.py           # Random Forest live inference
│   ├── mapping.json                   # Full sensor-to-register mapping
│   └── results/                       # Inference output logs
│
├── Reference/                         # Literature and reference materials
│   ├── papers/                        # Research papers consulted
│   └── websites/                      # Online resources
│
├── review1/                           # Review 1 deliverables
├── review2/                           # Review 2 deliverables
├── review3/                           # Review 3 deliverables
│
├── .gitignore
└── LICENSE
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (for USAD and LSTM-based models)
- scikit-learn, pandas, numpy
- Jupyter Notebook / JupyterLab

Install dependencies:

```bash
pip install torch scikit-learn pandas numpy jupyter matplotlib seaborn
```

### 1. Obtain the Dataset

The SWaT dataset is available from the iTrust research centre at SUTD. Request access and download links are provided in [`datasets/dataset_link.txt`](datasets/dataset_link.txt). Place the raw files under `datasets/swat/` and run the preprocessing notebook:

```bash
jupyter notebook datasets/swat-preprocessing.ipynb
```

### 2. Train the Global Model (USAD)

Open and run the training notebook for the primary global model:

```bash
jupyter notebook global_models/usad/usad.ipynb
```

Trained weights will be saved to `global_models/usad/usad_swat_weights.pth`.

### 3. Train the Local Models (Markov + Dwell-Time)

Per-PLC local models are trained in:

```bash
jupyter notebook local_models/01_markov_dwelltime_model.ipynb
```

Serialised model artifacts will be saved under `local_models/models/`.

### 4. Run Fusion

To evaluate the Topology-Constrained Attention Fusion (TCAF):

```bash
jupyter notebook fusion_models/learned_fusion.ipynb
```

For comparison, the simple rule-based fusion baseline is in `fusion_models/simple_fusion.ipynb`.

### 5. Live Testbed Inference

To simulate real-time inference against a Modbus-connected PLC testbed, use the scripts under `testbed/`:

```bash
# Start local (edge) Markov inference on a PLC node
python testbed/local/01_markov_inference.py

# Start global USAD inference on the central server
# (see testbed/global/ for the corresponding script)
```

> **Note:** Live testbed scripts require a Modbus-capable PLC or simulator. Refer to [`testbed/README.md`](testbed/README.md) for hardware and network configuration details.

### 6. Evaluate Baselines

All global and local baselines (Autoencoder, VAE, LSTM-AE, Isolation Forest, PASAD, Invariant Rules, CUSUM/EWMA) are available in `global_models/`, `local_models/`, and `baseline_models/`. Each subdirectory contains its own notebook for training and evaluation.
