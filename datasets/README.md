# SWaT Dataset

This directory contains the Secure Water Treatment (SWaT) dataset and various preprocessed versions.

## Original Datasets
- **`normal.csv`**: The original dataset containing normal operational data.
- **`attack.csv`**: The original dataset containing data captured during physical and cyber attacks.

## Directory Structure & Characteristics

### 1. `stage_wise/`
This folder contains the original datasets divided logically into the 6 physical stages of the SWaT testbed.
- Contains both `attack` and `normal` data separated stage by stage.
- **Characteristic Note**: In the `normal` stage-wise datasets within this folder, the `attack` parameter is absent (since all records in the normal dataset correspond to a non-attack state).

### 2. `col_removed/`
This folder contains a cleaned version of the `normal` dataset where parameters with absolutely zero variance (i.e., `max == min`) have been stripped out.
- Contains the main filtered dataset (`normal.csv` inside this folder).
- Contains the stage-wise splits of this filtered dataset (e.g., `normal_stage1.csv` through `normal_stage6.csv`).

#### Zero-Variance Columns Removed
A total of 12 columns had zero variance in the normal dataset and were removed. Below is the breakdown of the removed columns by their respective stages:

- **Stage 1**: `P102`
- **Stage 2**: `P201`, `P202`, `P204`, `P206`
- **Stage 3**: *(No columns removed)*
- **Stage 4**: `P401`, `P403`, `P404`
- **Stage 5**: `P502`
- **Stage 6**: `P601`, `P603`
- **Other**: `attack` (Removed from the normal dataset as it is always 0)
