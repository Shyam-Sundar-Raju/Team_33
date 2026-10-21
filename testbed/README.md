# Team 33 SWaT Testbed Deployment Guide

Welcome to the Team 33 SWaT anomaly detection testbed. This guide outlines the deployment architecture across 3 Raspberry Pi edge devices, instructions for running both the Decentralized (Fusion + Global + Local) and Monolithic (Baseline) models, and the profiling tasks required by the team to validate edge performance.

---

## 1. Architecture & Hardware Layout

You will deploy the system across **3 Raspberry Pis** acting as edge nodes. The dataset and Modbus simulator (`plc_writer.py`) will run on a central laptop/server and stream data to the PLCs, which the Pis will poll.

* **Pi 1 (Local Edge 1):** Runs Local Models for Stages 1, 2, and 3.
* **Pi 2 (Local Edge 2):** Runs Local Models for Stages 4, 5, and 6.
* **Pi 3 (Central Aggregator):** Runs the Global Model and Fusion Node. *Also used to run Baseline models in isolation.*

---

## 2. Provisioning the Raspberry Pis

Before beginning, ensure all 3 Pis have the runtime environment correctly established.

### 2.1 Dependencies
Install the required machine learning and networking libraries on all Pis. 
*Note: Due to memory constraints, installing `tensorflow` and `torch` on edge devices may take some time or require wheel files specific to the ARM architecture.*

```bash
pip install numpy pandas scikit-learn pymodbus
pip install torch torchvision torchaudio  # For PyTorch (USAD, VAE, Learned Fusion)
pip install tensorflow                    # For Keras (Autoencoder, LSTM-AE)
```

### 2.2 Network & Stream Validation
Before running models, verify that the Pis can reach the simulated PLCs on Port `502` and the Fusion node on Port `50000`.
Run `plc_writer.py` on your central machine to start writing CSV data to the PLCs. Use a simple Python script or Modbus client on the Pis to ensure registers are being read in real-time or near-real-time.

---

## 3. Running the Testbed

There are two modes of operation. **Do not run both modes at the same time.**

### Mode A: Decentralized Architecture (Global + Local + Fusion)

This is the primary architecture. The local and global models run concurrently and stream their predictions to the Fusion Node over UDP (`127.0.0.1:50000` by default; you will need to update the `FUSION_IP` in the scripts to the actual IP address of **Pi 3** on your local network).

**On Pi 3 (Start the Aggregator first):**
1. Start the Fusion Node: `python3 testbed/fusion/simple_fusion_node.py` (or `learned_fusion_node.py`)
2. Start the Global Model: `python3 testbed/global/01_autoencoder_inference.py` (or any of the 5 global models)

**On Pi 1:**
1. Edit the top of your chosen local model script to `STAGE = 1`, then run it in the background.
2. Edit to `STAGE = 2`, run in the background.
3. Edit to `STAGE = 3`, run in the background.
*(Example: `python3 testbed/local/01_markov_inference.py`)*

**On Pi 2:**
1. Follow the same process for `STAGE = 4`, `STAGE = 5`, and `STAGE = 6`.

You should see Pi 3's Fusion Node outputting real-time predictions as UDP packets arrive from Pi 1, Pi 2, and the local Global model.

### Mode B: Baseline Architecture (Monolithic)

The Baseline architecture serves as the traditional, single-layer benchmark. It does not use UDP IPC or Fusion. It ingests all 51 features directly.

**On Pi 3:**
1. Ensure the Fusion node is stopped.
2. Run any baseline model directly: `python3 testbed/baseline/01_autoencoder_inference.py`

---

## 4. Team Profiling & Implementation Tasks

The team must execute the following evaluation tasks to validate the testbed's viability in an edge computing environment.

### Task A: Measure Compute Costs
For each model (Global, Local, Baseline), capture:
1. **CPU Usage (%):** Use tools like `htop` or `psutil`.
2. **Memory Usage (MB):** Track RAM consumed during the inference loop.
3. **Network & Bandwidth Usages:** Measure bytes transmitted/received per cycle over the network for polling Modbus and forwarding predictions.
4. **Inference Time (ms):** Measure the duration of the `model.predict()` block per sample.
5. **Model Footprint:** Record the size (MB) and parameter count of the `.keras`, `.pth`, and `.pkl` files.

### Task B: Throughput Limits
Determine the scaling limits of the Raspberry Pi. 
* Increase the number of simulated PLCs(ie, local model), increase it step by step, record the cpu and ram usage and see where the system resourses deplete.
* Decrease the `POLL_INTERVAL` (e.g., from `0.1s` to `0.01s`) while running a single local model.
* Record the threshold at which the Pi begins dropping packets, lagging behind the stream, or exhausting CPU bounds.

--- 
Finish till this by 30/9/26
---

### Task C: End-to-End Detection Latency
Measure the true deployment latency:
1. Note the exact timestamp when `plc_writer.py` injects an anomalous attack row into the Modbus registers.
2. Note the exact timestamp when the Fusion Node (in Mode A) or Baseline script (in Mode B) outputs `ATTACK DETECTED!`.
3. Calculate the delta. This measures full pipeline delay (Modbus IO -> Inference -> UDP -> Fusion -> Alert).

### Task D: Offline vs. Deployment Parity
Compare the accuracy/F1-score generated on the Raspberry Pis against the metrics obtained in the training notebooks on the developer machine. Confirm that environmental constraints or hardware architectures have not degraded the precision of the weights.