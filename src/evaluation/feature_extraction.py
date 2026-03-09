"""
Handcrafted feature extraction from spectrograms for traditional ML baselines.
Extracts statistical, spectral, and energy features from log-magnitude spectrograms.
"""

import numpy as np
from scipy.stats import kurtosis, skew


def extract_spectrogram_features(spec):
    """Extract a feature vector from a 2D log-magnitude spectrogram.

    Args:
        spec: 2D numpy array of shape (n_freq, n_time)

    Returns:
        1D numpy array of features
    """
    features = []

    # Global statistics
    features.append(spec.mean())
    features.append(spec.std())
    features.append(spec.max())
    features.append(spec.min())
    features.append(np.median(spec))
    features.append(kurtosis(spec.flatten()))
    features.append(skew(spec.flatten()))

    # Energy features
    total_energy = np.sum(spec ** 2)
    features.append(total_energy)
    features.append(total_energy / spec.size)  # mean energy

    # Spectral features (averaged over time frames)
    freq_means = spec.mean(axis=1)  # mean power per frequency bin
    n_freq = len(freq_means)
    freq_axis = np.arange(n_freq)

    # Spectral centroid
    if freq_means.sum() != 0:
        spectral_centroid = np.sum(freq_axis * np.abs(freq_means)) / np.sum(np.abs(freq_means))
    else:
        spectral_centroid = 0.0
    features.append(spectral_centroid)

    # Spectral bandwidth
    if freq_means.sum() != 0:
        spectral_bw = np.sqrt(
            np.sum(((freq_axis - spectral_centroid) ** 2) * np.abs(freq_means))
            / np.sum(np.abs(freq_means))
        )
    else:
        spectral_bw = 0.0
    features.append(spectral_bw)

    # Spectral rolloff (frequency below which 85% of energy is concentrated)
    cumsum = np.cumsum(np.abs(freq_means))
    if cumsum[-1] > 0:
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        spectral_rolloff = rolloff_idx / n_freq
    else:
        spectral_rolloff = 0.0
    features.append(spectral_rolloff)

    # Spectral flatness (geometric mean / arithmetic mean)
    abs_means = np.abs(freq_means) + 1e-10
    log_mean = np.mean(np.log(abs_means))
    spectral_flatness = np.exp(log_mean) / np.mean(abs_means)
    features.append(spectral_flatness)

    # Temporal features (averaged over frequency bins)
    time_means = spec.mean(axis=0)
    features.append(time_means.std())  # temporal variation
    features.append(kurtosis(time_means))
    features.append(skew(time_means))

    # Band energy ratios (split spectrum into 4 bands)
    band_size = n_freq // 4
    for i in range(4):
        start = i * band_size
        end = (i + 1) * band_size if i < 3 else n_freq
        band_energy = np.sum(spec[start:end, :] ** 2)
        features.append(band_energy / (total_energy + 1e-10))

    # Spectral contrast (difference between peaks and valleys per band)
    for i in range(4):
        start = i * band_size
        end = (i + 1) * band_size if i < 3 else n_freq
        band = spec[start:end, :]
        if band.size > 0:
            features.append(band.max() - band.min())
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


def extract_features_from_dataset(dataset, max_samples=None):
    """Extract features from a PyTorch dataset of spectrograms.

    Args:
        dataset: PyTorch Dataset returning (tensor[1,H,W], label)
        max_samples: limit number of samples (None = all)

    Returns:
        X: feature matrix (n_samples, n_features)
        y: label array (n_samples,)
    """
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    X_list = []
    y_list = []

    for i in range(n):
        spec_tensor, label = dataset[i]
        spec = spec_tensor.squeeze(0).numpy()  # remove channel dim
        features = extract_spectrogram_features(spec)
        X_list.append(features)
        y_list.append(label.item())

        if (i + 1) % 500 == 0:
            print(f"  Extracted features: {i+1}/{n}")

    return np.array(X_list), np.array(y_list)
