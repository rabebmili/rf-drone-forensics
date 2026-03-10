# RF Drone Forensics

AI-based RF drone forensics pipeline for detection, attribution, open-set recognition, robustness evaluation, explainability, and forensic reporting using spectrograms and deep learning.

**Datasets:** DroneRF (4 classes, raw RF signals) + RFUAV (37 drone types, spectrogram images)

**Models:** SmallRFNet (CNN), RFResNet, RFTransformer, SVM, Random Forest

---

## Quick Start (full setup from scratch)

### 1. Clone the repository

```bash
git clone git@github.com:rabebmili/rf-drone-forensics.git
cd rf-drone-forensics
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Activate (pick your OS):
source .venv/Scripts/activate   # Windows Git Bash
source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

### 3. Download the datasets

#### DroneRF (required, ~8 GB)

The dataset is **not included** in the repo (too large).

**Option A — Google Drive (shared by the team):**
> Ask the repo owner for a shared link, download the zip, and extract it.

**Option B — Original source:**
> Download from [IEEE DataPort -- DroneRF](https://ieee-dataport.org/open-access/drone-remote-controller-rf-signal-dataset)

Place the CSV files so the folder structure looks exactly like this:

```
data/
└── raw/
    └── DroneRF/
        ├── AR drone/            # 18 CSV files
        ├── Background RF activites/  # 20 CSV files (typo is intentional)
        ├── Bepop drone/         # 21 CSV files
        └── Phantom drone/       # 21 CSV files
```

> **Note:** The folder name is `Background RF activites` (with a typo -- that's how the original dataset spells it). Do not rename it.

#### RFUAV (optional, ~5 GB, for cross-dataset experiments)

Downloads pre-generated spectrogram images (37 drone types) from [HuggingFace](https://huggingface.co/datasets/kitofrank/RFUAV):

```bash
python -m src.datasets.download_rfuav
```

This downloads only the spectrogram images (~5,679 JPGs), not the full 1.3TB raw data. Files are saved to `data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/train/`.

### 4. Run the DroneRF data preparation pipeline (in order)

```bash
# Step 1: Build file-level metadata from raw CSVs
python -m src.datasets.build_dronerf_metadata

# Step 2: Create segment index (131072-sample windows, 50% overlap)
python -m src.datasets.build_dronerf_segments

# Step 3: Stratified train/val/test split (70/15/15, split at file level)
python -m src.datasets.split_segments_by_file

# Step 4: Precompute spectrograms as .npy files for fast training
python -m src.preprocessing.precompute_spectrograms
```

After this, your `data/` folder will also contain:
```
data/
├── metadata/
│   ├── dronerf_metadata.csv
│   ├── dronerf_segments.csv
│   ├── dronerf_segments_split.csv
│   └── dronerf_precomputed.csv
└── processed/
    └── dronerf_spectrograms/   # .npy files
```

### 5. Train models

```bash
# Unified training (v1) -- all models x tasks
python -m src.training.train_multimodel --model smallrf --task binary --epochs 20
python -m src.training.train_multimodel --model resnet --task multiclass --epochs 30
python -m src.training.train_multimodel --model transformer --task binary --epochs 20

# Improved training (v2) -- augmentation, label smoothing, early stopping
python -m src.training.train_multimodel_v2 --model smallrf --task multiclass --epochs 50
python -m src.training.train_multimodel_v2 --model resnet --task multiclass --epochs 50
python -m src.training.train_multimodel_v2 --model transformer --task multiclass --epochs 50

# Traditional ML baselines (SVM + Random Forest)
python -m src.training.train_baselines
python -m src.training.train_baselines --label_col label_multiclass --output_dir outputs/baselines_multiclass
```

Model weights and training curves are saved to `outputs/{model}_{task}/`.

### 6. Run all experiments

Master runner for baselines + all models + robustness + open-set + explainability:

```bash
python -m src.training.run_all_experiments --task binary
python -m src.training.run_all_experiments --task multiclass
```

### 7. Evaluation modules

#### Robustness (SNR noise injection)

```bash
# Included in run_all_experiments, or run standalone via robustness.py
```

#### Cross-condition robustness (AWGN, interference, fading, combined)

```bash
python -m src.evaluation.cross_condition --model smallrf --task multiclass
python -m src.evaluation.cross_condition --model resnet --task multiclass
python -m src.evaluation.cross_condition --model transformer --task multiclass
```

Tests 4 degradation types at 5 severity levels each.

#### Open-set detection (all holdout classes)

```bash
python -m src.evaluation.run_openset_all --model smallrf --epochs 20
python -m src.evaluation.run_openset_all --model resnet --epochs 20
```

Trains separate models with each drone class held out, then evaluates OOD detection using MSP, Energy scoring, and Mahalanobis distance.

#### Cross-dataset evaluation (DroneRF <-> RFUAV)

Requires RFUAV to be downloaded first (see step 3).

```bash
python -m src.evaluation.cross_dataset --model resnet --epochs 20
```

Runs 3 experiments:
1. Train on DroneRF -> Test on RFUAV
2. Train on RFUAV -> Test on DroneRF
3. Combined training (DroneRF + RFUAV) -> Test on both

#### Explainability (Grad-CAM)

```bash
# Included in run_all_experiments, or via explainability.py
```

Generates Grad-CAM heatmaps on spectrograms showing which time-frequency regions drive predictions.

#### Generate thesis tables

```bash
python -m src.evaluation.generate_thesis_tables
```

Aggregates all `results.json` files into formatted comparison tables.

### 8. Forensic analysis

Analyze a single signal file and generate a forensic report:

```bash
python -m src.forensics.run_forensic_analysis --file "data/raw/DroneRF/AR drone/10100H_0.csv"
python -m src.forensics.run_forensic_analysis --file "data/raw/DroneRF/AR drone/10100H_0.csv" --model resnet --task multiclass
```

Outputs: `forensic_report.json` + `forensic_timeline.png` in `outputs/forensic_reports/`.

---

## Project Structure

```
rf-drone-forensics/
├── src/
│   ├── datasets/
│   │   ├── load_signal.py                # Read raw CSV signals
│   │   ├── build_dronerf_metadata.py     # Scan folders -> metadata CSV
│   │   ├── build_dronerf_segments.py     # Sliding window segment index
│   │   ├── split_segments_by_file.py     # File-level stratified split
│   │   ├── dronerf_segment_dataset.py    # On-the-fly STFT dataset
│   │   ├── dronerf_precomputed_dataset.py # Precomputed .npy dataset
│   │   ├── rfuav_dataset.py              # RFUAV JPG spectrogram loader
│   │   └── download_rfuav.py             # Download RFUAV from HuggingFace
│   ├── preprocessing/
│   │   ├── segmentation.py               # Sliding window function
│   │   ├── stft_utils.py                 # STFT + log-magnitude normalization
│   │   └── precompute_spectrograms.py    # Batch precompute .npy files
│   ├── models/
│   │   ├── cnn_spectrogram.py            # SmallRFNet (155K params)
│   │   ├── resnet_spectrogram.py         # RFResNet (697K params)
│   │   └── transformer_spectrogram.py    # RFTransformer (375K params)
│   ├── training/
│   │   ├── train_multimodel.py           # Unified training (v1)
│   │   ├── train_multimodel_v2.py        # Improved training (augmentation, early stopping)
│   │   ├── train_baselines.py            # SVM + Random Forest
│   │   └── run_all_experiments.py        # Master experiment runner
│   ├── evaluation/
│   │   ├── metrics.py                    # Accuracy, F1, ROC-AUC, ECE, plots
│   │   ├── feature_extraction.py         # Handcrafted features for baselines
│   │   ├── robustness.py                 # SNR noise injection
│   │   ├── cross_condition.py            # AWGN, interference, fading
│   │   ├── openset.py                    # MSP, Energy, Mahalanobis OOD
│   │   ├── run_openset_all.py            # Open-set with all holdout classes
│   │   ├── cross_dataset.py              # DroneRF <-> RFUAV evaluation
│   │   ├── explainability.py             # Grad-CAM heatmaps
│   │   └── generate_thesis_tables.py     # Summary tables
│   └── forensics/
│       ├── timeline.py                   # Per-segment classification + anomaly
│       └── run_forensic_analysis.py      # CLI forensic report generator
├── data/                  # Not in Git -- downloaded/generated locally
├── outputs/               # Not in Git -- generated by training
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Datasets

### DroneRF

| label_multiclass | label_binary | Class Name                |
|------------------|--------------|---------------------------|
| 0                | 0            | Background RF activities  |
| 1                | 1            | AR drone                  |
| 2                | 1            | Bepop drone               |
| 3                | 1            | Phantom drone             |

- 80 raw CSV files, 4 classes
- Source: [IEEE DataPort](https://ieee-dataport.org/open-access/drone-remote-controller-rf-signal-dataset)

### RFUAV

- 5,679 pre-generated spectrogram images (JPG), 37 drone/controller types
- All samples are drones (no background class)
- Source: [HuggingFace](https://huggingface.co/datasets/kitofrank/RFUAV)

## Model Architectures

| Model | Type | Parameters | Description |
|-------|------|------------|-------------|
| SmallRFNet | CNN | 155K | 3x (Conv+BN+ReLU+MaxPool) -> AdaptiveAvgPool -> FC + Dropout |
| RFResNet | ResNet | 697K | 3 residual blocks (32->64->128 channels) -> AdaptiveAvgPool -> FC |
| RFTransformer | Hybrid CNN-Transformer | 375K | CNN stem (3x stride-2 conv) -> 128 tokens -> 2 Transformer blocks (4-head attention) -> CLS token -> FC |

All models take input shape `[batch, 1, 257, 511]` (single-channel spectrogram).

## Key Technical Details

- **Segment window:** 131,072 samples, hop 65,536 (50% overlap)
- **STFT:** nperseg=512, noverlap=256, fs=1.0
- **Normalization:** log-magnitude, zero-mean, unit-variance
- **Splits:** 70% train / 15% val / 15% test (file-level stratified, no data leakage)
- **V2 training enhancements:** SpecAugment (time/freq masking), additive noise, time shift, label smoothing (0.1), AdamW (weight decay 1e-4), gradient clipping, early stopping (patience=12)
- **Open-set methods:** MSP, Energy scoring, Mahalanobis distance
- **Cross-condition degradations:** AWGN, narrowband interference, multipath fading, combined
- **Explainability:** Grad-CAM on last convolutional layer
- **Windows note:** DataLoaders use `num_workers=0` for compatibility
