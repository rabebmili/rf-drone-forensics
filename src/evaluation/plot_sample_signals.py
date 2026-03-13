"""
Plot one raw RF signal sample per class from DroneRF dataset.

Usage:
    python -m src.evaluation.plot_sample_signals
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.datasets.load_signal import load_dronerf_csv


CLASSES = {
    "Fond (pas de drone)": "data/raw/DroneRF/Background RF activites/00000H_21.csv",
    "AR Drone": "data/raw/DroneRF/AR drone/10111H_0.csv",
    "Bebop Drone": "data/raw/DroneRF/Bepop drone/10010H_0.csv",
    "Phantom Drone": "data/raw/DroneRF/Phantom drone/11000H_0.csv",
}

FS = 10_000_000  # 10 MHz sampling rate (DroneRF)
DURATION = 1.0   # 1 second
DISPLAY_SAMPLES = int(FS * DURATION)
colors = ["#607D8B", "#2196F3", "#4CAF50", "#FF9800"]


def main():
    output_dir = Path("outputs/thesis_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Individual PNGs per class
    for idx, (class_name, file_path) in enumerate(CLASSES.items()):
        print(f"Loading {class_name}...")
        signal = load_dronerf_csv(file_path)
        display = signal[:DISPLAY_SAMPLES]
        t = np.arange(len(display)) / FS

        fig, ax = plt.subplots(figsize=(14, 3.5))
        ax.plot(t, display, color=colors[idx], linewidth=0.2, alpha=0.85)
        ax.set_ylabel("Amplitude", fontsize=11)
        ax.set_xlabel("Temps (secondes)", fontsize=11)
        ax.set_title(f"Signal RF brut \u2014 {class_name}",
                      fontsize=14, fontweight="bold", color=colors[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.set_xticks(np.arange(0, DURATION + 0.2, 0.2))
        ax.set_xlim(0, DURATION)

        # Safe filename
        fname = class_name.replace(" ", "_").replace("(", "").replace(")", "")
        fname = fname.replace("\u00e9", "e").replace("'", "")
        save_path = output_dir / f"signal_{fname}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # Also keep the combined figure
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    for idx, (class_name, file_path) in enumerate(CLASSES.items()):
        signal = load_dronerf_csv(file_path)
        display = signal[:DISPLAY_SAMPLES]
        t = np.arange(len(display)) / FS

        ax = axes[idx]
        ax.plot(t, display, color=colors[idx], linewidth=0.2, alpha=0.85)
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.set_title(class_name, fontsize=13, fontweight="bold", color=colors[idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.set_xticks(np.arange(0, DURATION + 0.2, 0.2))

    axes[-1].set_xlabel("Temps (secondes)", fontsize=11)
    axes[-1].set_xlim(0, DURATION)

    fig.suptitle("Signaux RF bruts par classe \u2014 Dataset DroneRF",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = output_dir / "sample_signals_per_class.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nCombined figure saved: {save_path}")


if __name__ == "__main__":
    main()
