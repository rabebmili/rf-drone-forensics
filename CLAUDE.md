# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RF Drone Forensics: a PyTorch-based pipeline for classifying drone RF signals using spectrograms and a CNN. Binary classification (drone vs. background) with optional 4-class multi-class support (Background, AR drone, Bepop drone, Phantom drone). Uses the DroneRF dataset.

## Setup & Environment

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

Key dependencies: torch, scipy (STFT), scikit-learn (metrics/splitting), pandas, numpy, matplotlib.

## Data Preparation Pipeline

Run these scripts **in order** from the project root. Each is executed as a module:

```bash
# 1. Build file-level metadata from raw CSVs
python -m src.datasets.build_dronerf_metadata

# 2. Create segment index (131072-sample windows, 50% overlap)
python -m src.datasets.build_dronerf_segments

# 3. Stratified train/val/test split (70/15/15, split at file level to avoid leakage)
python -m src.datasets.split_segments_by_file

# 4. Precompute spectrograms as .npy files for fast training
python -m src.preprocessing.precompute_spectrograms
```

## Training

```bash
# Unified training (recommended) — supports all models × tasks
python -m src.training.train_multimodel --model smallrf --task binary --epochs 20
python -m src.training.train_multimodel --model resnet --task multiclass --epochs 30
python -m src.training.train_multimodel --model transformer --task binary --epochs 20

# Traditional ML baselines (SVM + Random Forest)
python -m src.training.train_baselines

# Legacy scripts
python -m src.training.train_binary_dronerf_fast
python -m src.training.train_binary_dronerf

# Run all experiments (baselines + models + robustness + open-set + explainability)
python -m src.training.run_all_experiments --task binary
python -m src.training.run_all_experiments --task multiclass
```

Model outputs save to `outputs/{model}_{task}/` with subdirs: `models/`, `figures/`.

## Forensic Analysis

```bash
python -m src.forensics.run_forensic_analysis --file data/raw/DroneRF/AR\ drone/10100H_0.csv
python -m src.forensics.run_forensic_analysis --file path/to/signal.csv --model resnet --task multiclass
```

Generates: `forensic_report.json` + `forensic_timeline.png` in `outputs/forensic_reports/`.

## Running Tests

Tests are standalone scripts (no pytest framework):

```bash
python -m src.preprocessing.test_segmentation
python -m src.preprocessing.test_one_real_file
python -m src.datasets.test_dronerf_segment_dataset
python -m src.training.test_split_loader
```

## Architecture

### Data Flow
```
Raw CSV signals → segment_signal (131072 samples, 65536 hop)
    → compute_log_spectrogram (STFT: nperseg=512, noverlap=256)
    → SmallRFNet CNN → binary/multiclass prediction
```

### Module Structure

- **`src/datasets/`** — Data loading, metadata building, segmentation indexing, dataset classes
  - `load_signal.py`: Reads raw CSV signal files into numpy arrays
  - `build_dronerf_metadata.py`: Scans `data/raw/DroneRF/` folder hierarchy to infer labels
  - `build_dronerf_segments.py`: Creates sliding-window segment index
  - `split_segments_by_file.py`: File-level stratified split (prevents data leakage)
  - `dronerf_segment_dataset.py`: `DroneRFSegmentDataset` — computes STFT on-the-fly
  - `dronerf_precomputed_dataset.py`: `DroneRFPrecomputedDataset` — loads precomputed `.npy` spectrograms
- **`src/preprocessing/`** — Signal processing
  - `segmentation.py`: `segment_signal()` sliding window function
  - `stft_utils.py`: `compute_log_spectrogram()` — STFT with log-magnitude normalization (zero-mean, unit-variance)
  - `precompute_spectrograms.py`: Batch-processes all segments into `.npy` files
- **`src/models/`** — Neural network architectures
  - `cnn_spectrogram.py`: `SmallRFNet` — 3-layer CNN (Conv+BN+ReLU+Pool) → AdaptiveAvgPool → FC head with dropout
  - `resnet_spectrogram.py`: `RFResNet` — lightweight ResNet with residual blocks for deeper comparison
  - `transformer_spectrogram.py`: `RFTransformer` — lightweight Vision Transformer (ViT) with patch embeddings
- **`src/training/`** — Training loops
  - `train_binary_dronerf.py`: On-the-fly STFT training (batch_size=8)
  - `train_binary_dronerf_fast.py`: Precomputed spectrogram training (batch_size=16), saves training curves
  - `train_multimodel.py`: Unified training script for all models (SmallRF/ResNet/Transformer) × (binary/multiclass)
  - `train_multimodel_v2.py`: Improved training with augmentation, label smoothing, early stopping, weight decay
  - `train_baselines.py`: Traditional ML baselines (SVM + Random Forest) on handcrafted features
  - `run_all_experiments.py`: Master runner for all experiments
- **`src/evaluation/`** — Evaluation and analysis
  - `metrics.py`: Comprehensive metrics (accuracy, F1, ROC-AUC, ECE, confusion matrix, PR curves)
  - `feature_extraction.py`: Handcrafted spectrogram features for ML baselines
  - `robustness.py`: SNR robustness evaluation with synthetic noise injection
  - `openset.py`: Open-set/OOD detection (MSP, energy scoring, Mahalanobis distance)
  - `run_openset_all.py`: Open-set with all 3 drone classes held out separately
  - `cross_condition.py`: Cross-condition evaluation (AWGN, interference, fading, combined degradations)
  - `explainability.py`: Grad-CAM heatmaps on spectrograms
  - `generate_thesis_tables.py`: Generate thesis-ready comparison tables from all results
- **`src/forensics/`** — Forensic investigation tools
  - `timeline.py`: Segment-by-segment classification with confidence + anomaly detection
  - `run_forensic_analysis.py`: CLI tool to analyze a signal file and generate forensic report

### Data Directory Layout
```
data/raw/DroneRF/{class_folder}/*.csv     # Raw 1D RF signals
data/metadata/dronerf_metadata.csv         # File-level labels
data/metadata/dronerf_segments.csv         # Segment boundaries
data/metadata/dronerf_segments_split.csv   # With train/val/test assignment
data/processed/dronerf_spectrograms/       # Precomputed .npy spectrograms
```

### Key Constants
- Segment window: 131,072 samples; hop: 65,536 (50% overlap)
- STFT: nperseg=512, noverlap=256, fs=1.0
- Model input: `[batch, 1, H, W]` single-channel spectrogram
- Training: Adam lr=1e-3, CrossEntropyLoss, 5 epochs

### Class Labels
| label_multiclass | label_binary | Class Name |
|---|---|---|
| 0 | 0 | Background RF activities |
| 1 | 1 | AR drone |
| 2 | 1 | Bepop drone |
| 3 | 1 | Phantom drone |

## Conventions

- All scripts run as modules from the project root (`python -m src.module.name`)
- Dataset splits are done at file level (all segments from one file go to the same split)
- Windows paths are used in metadata CSVs (backslash separators)
- `num_workers=0` in DataLoaders for Windows compatibility
