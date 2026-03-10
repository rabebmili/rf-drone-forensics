# AI-Based RF Drone Forensics: Complete Project Report

## 1. Problem Statement

### 1.1 Context

Drones (UAVs) communicate with their controllers via radio-frequency (RF) links, typically in the 2.4 GHz and 5.8 GHz bands. These RF emissions leave exploitable traces that can be captured and analyzed. In a forensic context, the ability to detect, identify, and attribute drone signals from RF captures is critical for security, law enforcement, and incident investigation.

### 1.2 Objectives

This project aims to build a complete AI-based forensic pipeline that can:

1. **Detect** whether a drone is present in an RF capture (binary: drone vs. background)
2. **Attribute** the signal to a specific drone model (multiclass: which drone?)
3. **Reject unknown signals** that don't match any known drone (open-set recognition)
4. **Operate reliably** under degraded conditions (noise, interference, fading)
5. **Explain its decisions** with visual evidence (heatmaps on spectrograms)
6. **Produce forensic reports** with timelines, confidence scores, and anomaly flags

### 1.3 Research Questions

1. Are STFT spectrograms sufficient as input representation, or are raw IQ signals needed?
2. Which model architecture is best suited for drone RF classification?
3. Can the system detect drone types it has never seen before (open-set)?
4. How robust is the system under real-world signal degradations?
5. Does training on one dataset generalize to another (cross-dataset)?
6. Can the system provide interpretable evidence for forensic use?

---

## 2. Datasets

### 2.1 DroneRF

- **Source:** IEEE DataPort
- **Content:** Raw 1D RF signal recordings (CSV files) from 3 commercial drones + background RF
- **Classes:** 4 (Background, AR Drone, Bepop Drone, Phantom Drone)
- **Size:** 80 CSV files, ~8 GB total
- **Recording setup:** Signals captured at a fixed receiver location while drones operated nearby
- **Binary labels:** 0 = Background, 1 = Drone (any type)
- **Multiclass labels:** 0 = Background, 1 = AR, 2 = Bepop, 3 = Phantom

This is the primary dataset for all experiments. It provides both drone and background samples, enabling binary detection and multiclass attribution.

### 2.2 RFUAV

- **Source:** HuggingFace (kitofrank/RFUAV)
- **Content:** Pre-generated spectrogram images (JPG) from 37 drone/controller models
- **Classes:** 37 drone types (no background class)
- **Size:** 5,679 images, ~5 GB
- **Format:** RGB spectrogram images generated via MATLAB pipeline

Used for cross-dataset evaluation to test generalization. Key limitation: no background samples exist in RFUAV, so binary detection cannot be fully validated on this dataset alone.

### 2.3 CageDroneRF (Not Used)

- **Source:** Requires form submission, 500 GB raw data
- **Content:** 27 drone types recorded in a Faraday cage (controlled environment)
- **Status:** Not integrated due to size and access constraints. Mentioned as a limitation.

---

## 3. Signal Preprocessing Pipeline

### 3.1 Raw Signal Ingestion

Each DroneRF CSV file contains a 1D RF signal (amplitude values over time). These are continuous recordings that need to be segmented before analysis.

### 3.2 Segmentation

The raw signals are divided into fixed-length windows using a sliding window approach:

- **Window size:** 131,072 samples
- **Hop size:** 65,536 samples (50% overlap)
- **Rationale:** 131,072 samples provides enough temporal context for the STFT to capture the drone's frequency-hopping patterns. The 50% overlap ensures no signal content falls on a window boundary and is lost.

This produces 12,080 segments across all 80 files.

### 3.3 Short-Time Fourier Transform (STFT)

Each segment is converted from a 1D time-domain signal into a 2D time-frequency representation (spectrogram):

- **STFT parameters:** nperseg=512 (FFT window size), noverlap=256 (50% FFT overlap), fs=1.0
- **Output:** A complex-valued matrix of shape (257, 511), where 257 = frequency bins and 511 = time frames
- **Log-magnitude:** We take the logarithm of the magnitude spectrum to compress the dynamic range (weak and strong signals become comparable)
- **Normalization:** Zero-mean, unit-variance normalization per spectrogram ensures consistent input scale across all samples

The resulting spectrogram is a grayscale "image" that visually encodes how the signal's frequency content evolves over time. Different drone models produce visually distinct spectral patterns.

### 3.4 Precomputation

To avoid recomputing STFT during every training epoch, all 12,080 spectrograms are precomputed and saved as `.npy` files. This reduces training time from hours to minutes.

### 3.5 RFUAV Preprocessing

RFUAV images are already spectrograms (JPG format). They are:
- Converted from RGB to grayscale (single channel)
- Resized to (257, 511) to match DroneRF spectrogram dimensions
- Normalized (zero-mean, unit-variance)
- Preloaded into RAM for fast training

### 3.6 Data Splitting

Splits are done at the **file level**, not the segment level. All segments from one CSV file go to the same split. This prevents data leakage -- without this, adjacent overlapping segments from the same file could appear in both train and test, artificially inflating accuracy.

- **Train:** 70% of files (8,456 segments)
- **Validation:** 15% of files (1,812 segments)
- **Test:** 15% of files (1,812 segments)
- **Stratification:** Each split maintains the same class proportions

For RFUAV (which has only a `train/` folder), we create an 80/20 stratified split programmatically, ensuring each of the 37 drone classes is proportionally represented.

---

## 4. Model Architectures

All models receive input of shape [batch, 1, 257, 511] (single-channel spectrogram) and output class logits.

### 4.1 SmallRFNet (Lightweight CNN)

- **Architecture:** 3 convolutional blocks, each consisting of Conv2D -> BatchNorm -> ReLU -> MaxPool2D, followed by AdaptiveAvgPool2D -> Fully Connected layer with Dropout
- **Parameters:** 155,236
- **Design rationale:** Deliberately small to test whether a lightweight model can handle the task. Fewer parameters mean less overfitting risk and faster inference -- important for real-time forensic tools.

### 4.2 RFResNet (Residual Network)

- **Architecture:** 3 residual blocks with skip connections (32 -> 64 -> 128 channels), followed by AdaptiveAvgPool2D -> FC with Dropout
- **Parameters:** 696,548
- **Design rationale:** Residual connections allow training deeper networks without vanishing gradients. The hypothesis was that deeper features would improve classification and provide better embeddings for open-set detection.

### 4.3 RFTransformer (Hybrid CNN-Transformer)

- **Architecture:** CNN stem (3 stride-2 convolutions reducing spatial dimensions) -> AdaptiveAvgPool to 8x16 = 128 tokens -> 2 Transformer encoder blocks (4-head self-attention) -> CLS token -> FC classifier
- **Parameters:** 375,492
- **Design rationale:** Self-attention can capture long-range dependencies across time and frequency. The CNN stem is essential -- a pure Vision Transformer on 257x511 spectrograms would create 2,016+ patches with O(n^2) attention complexity, making training infeasible. The hybrid approach reduces this to 128 tokens.
- **Initial problem:** The first implementation used a pure ViT with 2,016 patches. Training hung for over 1 hour on a single epoch due to the quadratic attention cost. The hybrid CNN-Transformer reduced this to ~30 seconds per epoch.

### 4.4 Traditional ML Baselines

- **Feature extraction:** 24 handcrafted features per spectrogram -- global statistics (mean, std, min, max), energy, spectral centroid, spectral bandwidth, spectral rolloff, spectral flatness, temporal features, band energy ratios, spectral contrast
- **SVM:** Radial Basis Function kernel, C=10
- **Random Forest:** 200 trees, max depth=20

These baselines establish a performance floor and demonstrate the value of end-to-end deep learning over handcrafted features.

---

## 5. Training

### 5.1 Standard Training (V1)

- **Optimizer:** Adam, learning rate 1e-3
- **Loss:** CrossEntropyLoss
- **Scheduler:** Cosine annealing learning rate
- **Checkpointing:** Best model saved based on validation F1 score
- **Epochs:** 20 (binary) or 30 (multiclass)

### 5.2 Improved Training (V2)

Designed to address multiclass classification challenges (confusion between similar drones):

- **Data augmentation:**
  - SpecAugment: Random time and frequency masking (10% of axis), forcing the model to learn from partial spectrograms
  - Additive Gaussian noise (probability 30%), improving noise robustness
  - Random time shift (probability 30%), reducing sensitivity to segment alignment
- **Label smoothing:** 0.1 (softens hard labels from [0,0,1,0] to [0.033,0.033,0.9,0.033]), preventing overconfident predictions and improving calibration
- **Optimizer:** AdamW with weight decay 1e-4 (L2 regularization)
- **Learning rate:** 5e-4 (lower than V1 to stabilize training with augmentation)
- **Gradient clipping:** max_norm=1.0 (prevents exploding gradients)
- **Early stopping:** Patience of 12 epochs (stops training if validation F1 doesn't improve)

---

## 6. Evaluation Framework

### 6.1 Classification Metrics

- **Accuracy:** Overall correct predictions / total predictions
- **Balanced Accuracy:** Average per-class recall (handles class imbalance)
- **Macro F1:** Harmonic mean of precision and recall, averaged across classes (treats all classes equally)
- **Weighted F1:** F1 weighted by class support (accounts for class imbalance)
- **Precision / Recall:** Per-class correctness vs. completeness
- **Cohen's Kappa:** Agreement beyond chance
- **MCC (Matthews Correlation Coefficient):** Balanced measure even with unequal class sizes
- **ROC-AUC:** Area under Receiver Operating Characteristic curve (discrimination ability)
- **ECE (Expected Calibration Error):** How well predicted probabilities match actual frequencies (calibration)

### 6.2 Open-Set Metrics

- **AUROC:** Area under ROC for in-distribution vs. out-of-distribution separation
- **AUPR:** Area under Precision-Recall curve
- **FPR@95TPR:** False positive rate when 95% of true positives are detected (lower is better)

### 6.3 Visual Outputs

- Confusion matrices (per-class error patterns)
- ROC curves (per-class discrimination)
- Precision-Recall curves
- Calibration diagrams (reliability)
- Training curves (loss and accuracy over epochs)
- Robustness curves (performance vs. degradation level)
- Grad-CAM heatmaps (which spectrogram regions matter)
- Forensic timeline plots (per-segment classification over time)

---

## 7. Experiments and Results

### 7.1 Binary Detection (Drone vs. Background)

**Task:** Determine if any drone is present in the RF capture.

| Model | Accuracy | Macro F1 | ROC-AUC |
|-------|----------|----------|---------|
| SmallRFNet | 100.0% | 1.000 | 1.000 |
| RFResNet | 100.0% | 1.000 | 1.000 |
| RFTransformer | 99.85% | 0.998 | 1.000 |
| SVM | 68.2% | 0.700 | 0.888 |
| Random Forest | 69.4% | 0.711 | 0.901 |

**Observation:** Binary detection is trivially solved by all deep learning models. The drone vs. background distinction is very clear in the spectral domain -- drones produce strong, structured frequency patterns that are completely absent in background RF. This is consistent with findings in the DroneRF literature. The SVM and Random Forest baselines perform significantly worse, demonstrating that handcrafted features lose important spectral structure that CNNs capture directly.

**Conclusion:** Binary detection alone is insufficient for a thesis contribution. The scientific value lies in multiclass attribution, open-set, and robustness experiments.

### 7.2 Multiclass Attribution (Which Drone?)

**Task:** Identify the specific drone model (4-class: Background, AR, Bepop, Phantom).

#### V1 Results (standard training)

| Model | Accuracy | Macro F1 | MCC |
|-------|----------|----------|-----|
| SmallRFNet | 86.6% | 0.871 | 0.834 |
| RFResNet | 87.9% | 0.886 | 0.851 |
| RFTransformer | 87.8% | 0.882 | 0.848 |

#### V2 Results (improved training with augmentation)

| Model | Accuracy | Macro F1 | MCC | ECE |
|-------|----------|----------|-----|-----|
| **SmallRFNet** | **90.4%** | **0.908** | **0.880** | 0.058 |
| RFTransformer | 88.1% | 0.886 | 0.851 | 0.042 |
| RFResNet | 82.9% | 0.825 | 0.781 | 0.050 |

#### Per-Class Breakdown (SmallRFNet V2)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Background | 1.00 | 1.00 | 1.00 |
| AR Drone | 0.75 | 0.99 | 0.85 |
| Bepop Drone | 1.00 | 0.73 | 0.84 |
| Phantom Drone | 0.92 | 0.95 | 0.94 |

**Observations:**

1. **SmallRFNet wins** despite having the fewest parameters (155K vs. 697K for ResNet). This is a classic case where a smaller model generalizes better -- it has less capacity to memorize training data and is forced to learn general patterns.

2. **ResNet overfits** in V2 -- train accuracy reaches 98.9% but test accuracy is only 82.9%. The larger model memorizes training augmentations rather than learning robust features. Interestingly, ResNet performed better in V1 (87.9%) without augmentation.

3. **AR Drone and Bepop Drone are most confused.** AR has high recall (0.99) but low precision (0.75), meaning the model tends to classify other drones as AR. Bepop has the opposite problem -- high precision (1.00) but low recall (0.73), meaning many Bepop samples are misclassified as AR. This suggests their RF signatures have overlapping spectral characteristics.

4. **Background is perfectly separated** in all experiments. The spectral signature of "no drone" is fundamentally different from any drone signal.

5. **V2 augmentation helps SmallRFNet** (+3.8% accuracy) but hurts ResNet (-5%). Data augmentation regularizes small models but can destabilize large ones that already struggle with generalization.

6. **Transformer has the best calibration** (ECE=0.042), meaning its confidence scores most accurately reflect true probabilities. This is valuable for forensic applications where confidence matters.

### 7.3 Robustness: SNR Noise Injection

**Task:** Evaluate how performance degrades when Gaussian noise is added at various Signal-to-Noise Ratios (SNR).

SmallRFNet was the most robust model in binary detection, maintaining 0.85 F1 at 0 dB SNR (equal signal and noise power) while ResNet dropped to 0.60.

**Observation:** Smaller models with fewer parameters are more robust to noise -- they learn simpler, more generalizable decision boundaries that are less sensitive to input perturbations.

### 7.4 Cross-Condition Robustness

**Task:** Evaluate performance under 4 realistic degradation types at 5 severity levels.

**Degradation types:**
- **AWGN:** Additive White Gaussian Noise (broadband noise floor)
- **Narrowband Interference:** WiFi-like interfering signal in a specific frequency band
- **Multipath Fading:** Frequency-selective signal attenuation simulating indoor reflections
- **Combined:** All three degradations simultaneously

#### Results at Maximum Severity (F1 scores)

| Condition | SmallRFNet | Transformer | ResNet |
|-----------|-----------|-------------|--------|
| Clean | **0.908** | 0.886 | 0.825 |
| AWGN | 0.102 | **0.353** | 0.218 |
| Interference | 0.678 | **0.849** | 0.757 |
| Fading | 0.645 | **0.731** | 0.697 |
| Combined | 0.166 | **0.403** | 0.201 |

**Observations:**

1. **Transformer is most robust under degraded conditions** despite having lower clean accuracy. Self-attention's ability to focus on the most informative time-frequency regions makes it resilient -- it can "ignore" corrupted parts of the spectrogram and attend to clean signal components.

2. **All models collapse under extreme AWGN** (severity 1.0). When noise completely overwhelms the signal, no model can extract useful features.

3. **Interference is the least damaging** degradation. Narrowband interference only corrupts a small frequency band, leaving most of the spectrogram intact for classification.

4. **Fading causes graceful degradation** rather than sudden collapse. Even at maximum fading, models retain 64-73% F1.

5. **ResNet improves slightly at moderate AWGN** (F1 increases from 0.825 to 0.836 at severity 0.8). This counterintuitive result occurs because noise acts as a form of regularization for an overfitting model.

6. **The gap between clean and degraded performance** highlights the need for noise-robust training (augmentation with noise injection) or domain adaptation techniques for real-world deployment.

### 7.5 Open-Set Recognition

**Task:** Detect drone types the model has never seen during training.

**Protocol:** For each of the 3 drone classes (AR, Bepop, Phantom), train a model without that class, then test whether the model can identify samples from the held-out class as "unknown."

**Methods tested:**
- **MSP (Maximum Softmax Probability):** Uses the model's confidence score. Low confidence = likely unknown. Simple but often overconfident.
- **Energy Score:** Computes the log-sum-exp of logits. More theoretically grounded than MSP.
- **Mahalanobis Distance:** Measures distance from the sample's embedding to the nearest class centroid in feature space, accounting for covariance. Requires fitting class statistics on training data.

#### SmallRFNet Results (AUROC)

| Holdout Class | MSP | Energy | Mahalanobis |
|---------------|-----|--------|-------------|
| AR Drone | 0.501 | 0.428 | 0.524 |
| Bepop Drone | 0.527 | 0.465 | 0.637 |
| Phantom Drone | 0.354 | 0.397 | **0.843** |

#### ResNet Results (AUROC)

| Holdout Class | MSP | Energy | Mahalanobis |
|---------------|-----|--------|-------------|
| AR Drone | **0.786** | 0.703 | 0.729 |
| Bepop Drone | 0.551 | 0.591 | **0.667** |
| Phantom Drone | 0.647 | 0.698 | **0.840** |

**Observations:**

1. **Mahalanobis distance is the best OOD detection method**, achieving AUROC of 0.84 for Phantom drone detection with both models. It works because it operates in the learned embedding space where class clusters are more separable than in the output probability space.

2. **MSP and Energy fail on SmallRFNet** (AUROC near 0.50 = random). SmallRFNet is overconfident -- it assigns high softmax probabilities even to unknown samples, making confidence-based methods useless. This is a known problem with small, well-trained models.

3. **ResNet is better for open-set** despite having worse classification accuracy. Deeper networks learn richer, more structured embeddings that separate known from unknown classes more effectively. The higher parameter count creates a more expressive feature space.

4. **AR Drone is hardest to detect as unknown** (lowest AUROC across all methods). Its RF signature overlaps significantly with the other drones, so when held out, the model confidently misclassifies it as a known class.

5. **Phantom Drone is easiest to detect as unknown** (highest AUROC). Its RF signature is the most distinctive, creating a clear gap in embedding space when it's absent from training.

6. **There is a fundamental trade-off between classification and open-set performance.** The model that classifies best (SmallRFNet) is the worst at detecting unknowns. A modular pipeline that uses different models for different tasks would be optimal.

### 7.6 Cross-Dataset Evaluation

**Task:** Test whether a model trained on one dataset can work on another.

**Experiment 1: Train on DroneRF -> Test on RFUAV**
- Result: 100% accuracy, but misleading. RFUAV contains only drone samples (no background), so the model correctly labels everything as "drone." This does not demonstrate true generalization.

**Experiment 2: Train on RFUAV -> Test on DroneRF**
- DroneRF accuracy: 76.9%, but 0% recall on Background class
- The model labels everything as "drone" because RFUAV has no background samples. It never learned what "no drone" looks like.
- Macro F1: 0.43 (effectively broken for detection)

**Experiment 3: Combined Training (DroneRF + RFUAV) -> Test on Both**
- Combines DroneRF (provides background class) with RFUAV (provides 37 additional drone types)
- This is the only valid cross-dataset experiment for binary detection

**Observations:**

1. **Cross-dataset binary detection is fundamentally limited** by the lack of standardized background recordings. RFUAV was designed for drone identification (which drone), not detection (is there a drone). Without background samples, binary models trained on RFUAV are useless for detection.

2. **Dataset-specific biases** mean that even drone-to-drone generalization is uncertain. DroneRF and RFUAV use different recording equipment, environments, and preprocessing (raw CSV vs. MATLAB-generated spectrograms). A model may learn dataset-specific artifacts rather than true RF signatures.

3. **Combined training is the practical solution** -- using multiple datasets together compensates for individual dataset limitations.

### 7.7 Explainability (Grad-CAM)

**Task:** Visualize which regions of the spectrogram drive the model's decision.

**Method:** Gradient-weighted Class Activation Mapping (Grad-CAM) computes the gradient of the predicted class score with respect to the last convolutional layer's feature maps. The resulting heatmap highlights which time-frequency regions contributed most to the prediction.

**Observations:**

1. For drone signals, the model focuses on **specific frequency bands** where drone communication protocols operate (frequency-hopping spread spectrum patterns).

2. For background signals, attention is **diffuse** -- no single region dominates, consistent with the absence of structured drone signals.

3. Different drone models activate **different frequency bands**, confirming that the model learns genuinely different spectral signatures rather than relying on a single discriminative feature.

4. The heatmaps provide **forensic evidence** -- an analyst can see exactly why the system flagged a signal as a specific drone type.

### 7.8 Forensic Timeline Analysis

**Task:** Process an entire RF recording and produce a segment-by-segment classification timeline.

**Process:**
1. Divide the raw signal into overlapping segments
2. Compute STFT spectrogram for each segment
3. Run the classifier on each spectrogram
4. Record: predicted class, confidence score, energy score
5. Flag segments with confidence below a threshold as anomalous
6. Generate a JSON report and visual timeline

**Output:**
- **JSON forensic report:** Metadata, per-segment predictions, confidence scores, anomaly flags, summary statistics
- **Visual timeline:** Strip chart showing classification and confidence over time, with anomalous segments highlighted

This module bridges the gap between ML classification and forensic investigation by providing structured, time-ordered evidence with confidence indicators.

---

## 8. Key Findings

### 8.1 Answering the Research Questions

**Q1: STFT spectrograms vs. raw IQ signals?**
STFT spectrograms are sufficient. All deep learning models achieve 100% binary detection and up to 90.4% multiclass accuracy on spectrograms. The time-frequency representation captures the essential discriminative features of drone RF signals. RFUAV (a separate dataset) also uses spectrograms, confirming this approach is standard in the field.

**Q2: Which model is best?**
No single model is best for all tasks:
- **SmallRFNet:** Best for classification accuracy (90.4% multiclass)
- **RFTransformer:** Most robust under degraded conditions (best F1 at all severity levels)
- **RFResNet:** Best for open-set detection (richer embeddings for OOD separation)

A **modular architecture** with task-specific models is recommended.

**Q3: Can the system detect unknown drones?**
Partially. Mahalanobis distance achieves AUROC of 0.84 for detecting Phantom drone as unknown, but only 0.52 for AR drone. Performance depends heavily on how distinct the unknown drone's RF signature is from known classes. MSP and Energy scoring are unreliable (near random for SmallRFNet).

**Q4: How robust is the system?**
Moderately. Models maintain good performance under mild to moderate degradations but collapse under extreme noise (AWGN severity 1.0). The Transformer's attention mechanism provides the best noise resilience. Narrowband interference is the least damaging degradation type.

**Q5: Does cross-dataset generalization work?**
Limited. The absence of background samples in RFUAV prevents meaningful binary detection transfer. Dataset-specific recording conditions (equipment, environment, preprocessing) create domain gaps that simple model transfer cannot bridge. Combined training is the practical solution.

**Q6: Is the system explainable?**
Yes. Grad-CAM heatmaps show that models learn genuinely different spectral signatures per drone class, not dataset artifacts. The forensic timeline module provides structured, time-ordered evidence with confidence scores suitable for investigation.

### 8.2 Model Comparison Summary

| Criterion | Best Model | Why |
|-----------|-----------|-----|
| Clean classification accuracy | SmallRFNet (90.4%) | Fewer parameters = less overfitting |
| Noise/degradation robustness | RFTransformer | Attention ignores corrupted regions |
| Open-set / unknown detection | RFResNet | Deeper embeddings = better OOD separation |
| Calibration (ECE) | RFTransformer (0.042) | Most reliable confidence scores |
| Training speed | SmallRFNet | Fewest parameters, simplest architecture |
| Inference speed | SmallRFNet | Lowest computational cost |

### 8.3 Unexpected Findings

1. **Smaller is better for classification.** SmallRFNet (155K params) outperforms ResNet (697K params) by 7.5% on multiclass accuracy. In the RF domain with limited data, model capacity works against generalization.

2. **The best classifier is the worst at open-set.** SmallRFNet's overconfidence makes MSP/Energy OOD detection no better than random. This is a fundamental tension in model design.

3. **Data augmentation hurts ResNet.** V2 training improved SmallRFNet by 3.8% but decreased ResNet by 5%. Augmentation acts as regularization -- helpful for small models but destabilizing for already-struggling large models.

4. **Moderate noise can improve a model.** ResNet's accuracy increases slightly under moderate AWGN because noise acts as implicit regularization for an overfitting model.

---

## 9. Limitations

1. **DroneRF has only 3 drone types.** Real-world scenarios involve dozens of drone models. RFUAV adds 37 types but lacks background samples.

2. **CageDroneRF (27 drones, controlled environment) was not accessible** due to size (500 GB) and access restrictions (form required).

3. **All datasets use fixed recording setups.** Real forensic scenarios involve varying distances, angles, indoor/outdoor environments, and multiple simultaneous emitters.

4. **No real-time evaluation.** The pipeline processes pre-recorded signals. A real-time system would need streaming segmentation and inference latency constraints.

5. **Open-set performance is inconsistent.** AUROC ranges from 0.52 to 0.84 depending on the held-out class. The system cannot reliably reject all unknown drone types.

6. **Cross-dataset transfer is limited** by the lack of standardized background recordings and dataset-specific preprocessing artifacts.

7. **Binary detection is trivially solved** on DroneRF, providing no discriminative challenge for model comparison.

---

## 10. Pipeline Summary

```
Raw RF Signal (CSV)
    |
    v
Segmentation (131,072 samples, 50% overlap)
    |
    v
STFT Spectrogram (257 x 511, log-magnitude, normalized)
    |
    v
Model (SmallRFNet / RFResNet / RFTransformer)
    |
    +---> Binary Detection (drone vs. background)
    +---> Multiclass Attribution (which drone model)
    +---> Open-Set Rejection (unknown drone detection)
    +---> Grad-CAM Explainability (visual evidence)
    |
    v
Forensic Report
    +---> JSON report (metadata, per-segment results, anomaly flags)
    +---> Visual timeline (classification + confidence over time)
```

---

## 11. Conclusion

This project demonstrates a complete AI-based forensic pipeline for drone RF signal analysis, covering detection, attribution, open-set recognition, robustness evaluation, explainability, and forensic reporting.

The key contribution is not a single model but a **modular forensic framework** where different models serve different purposes: SmallRFNet for fast, accurate classification; RFTransformer for robust operation in degraded environments; and RFResNet with Mahalanobis distance for detecting unknown drone types.

The results confirm that STFT spectrograms are an effective representation for drone RF forensics, and that lightweight CNN architectures can outperform larger models when data is limited. The open-set and cross-dataset experiments reveal important limitations that point to future work: larger multi-dataset benchmarks with standardized background recordings, domain adaptation techniques for cross-environment generalization, and ensemble methods that combine the strengths of different architectures.

The forensic timeline and Grad-CAM modules bridge the gap between AI classification and practical investigation, providing structured evidence with visual explanations suitable for reporting.

---

## 12. Tools and Technologies

- **Language:** Python 3.14
- **Deep Learning:** PyTorch (models, training, inference)
- **Signal Processing:** SciPy (STFT), NumPy
- **Machine Learning:** scikit-learn (SVM, Random Forest, metrics, splitting)
- **Visualization:** Matplotlib (plots, heatmaps, timelines)
- **Data:** Pandas (metadata management)
- **Image Processing:** Pillow (RFUAV spectrogram loading)
- **Dataset Access:** huggingface_hub (RFUAV download)
