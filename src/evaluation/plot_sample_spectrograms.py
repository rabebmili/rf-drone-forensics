"""
Plot one raw spectrogram sample per class from DroneRF dataset.
Shows what the model actually sees as input.

Usage:
    python -m src.evaluation.plot_sample_spectrograms
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.datasets.load_signal import load_dronerf_csv
from src.preprocessing.stft_utils import compute_log_spectrogram


CLASSES = {
    "Fond (pas de drone)": "data/raw/DroneRF/Background RF activites/00000H_21.csv",
    "AR Drone": "data/raw/DroneRF/AR drone/10111H_0.csv",
    "Bebop Drone": "data/raw/DroneRF/Bepop drone/10010H_0.csv",
    "Phantom Drone": "data/raw/DroneRF/Phantom drone/11000H_0.csv",
}

SEGMENT_LEN = 131072


def main():
    output_dir = Path("outputs/thesis_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    for idx, (class_name, file_path) in enumerate(CLASSES.items()):
        print(f"Loading {class_name}...")
        signal = load_dronerf_csv(file_path)
        segment = signal[:SEGMENT_LEN]
        f, t, spec = compute_log_spectrogram(segment)

        ax = axes[idx]
        im = ax.pcolormesh(t, f, spec, shading="gouraud", cmap="viridis")
        ax.set_title(class_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Temps", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Fr\u00e9quence", fontsize=10)
        ax.tick_params(labelsize=8)

    fig.suptitle("Spectrogrammes STFT par classe \u2014 Dataset DroneRF",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = output_dir / "sample_spectrograms_per_class.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {save_path}")


if __name__ == "__main__":
    main()
