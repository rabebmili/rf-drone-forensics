"""
Generate two thesis-ready figures:
  1. All 3 drone classes overlaid (time-domain + avg spectrogram) vs Background
  2. RFUAV dataset samples — spectrogram images from diverse drone types

Usage:
    python -m src.evaluation.plot_combined_signals
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from src.datasets.load_signal import load_dronerf_csv
from src.preprocessing.stft_utils import compute_log_spectrogram
from src.preprocessing.segmentation import segment_signal


DRONE_DIRS = {
    "AR Drone": "data/raw/DroneRF/AR drone",
    "Bebop Drone": "data/raw/DroneRF/Bepop drone",
    "Phantom Drone": "data/raw/DroneRF/Phantom drone",
}

BG_DIR = "data/raw/DroneRF/Background RF activites"
RFUAV_DIR = "data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/train"

SEGMENT_LEN = 131072
HOP = 65536


def load_all_signals(class_dir):
    """Load all CSV files from a class directory."""
    folder = Path(class_dir)
    files = sorted(folder.glob("*.csv"))
    signals = []
    for f in files:
        try:
            signals.append(load_dronerf_csv(f))
        except Exception:
            pass
    return signals


def avg_spectrogram(signals):
    """Compute average spectrogram across all segments from all files."""
    specs = []
    for sig in signals:
        segs = segment_signal(sig, window_size=SEGMENT_LEN, hop_size=HOP)
        for seg in segs:
            _, _, S = compute_log_spectrogram(seg)
            specs.append(S)
    f, t, _ = compute_log_spectrogram(segs[0])
    return f, t, np.mean(specs, axis=0), len(specs)


def main():
    output_dir = Path("outputs/thesis_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    # =============================================================
    # FIGURE 1: All drones overlaid vs Background
    # =============================================================
    print("=" * 60)
    print("  FIGURE 1: DroneRF — All Drone Classes Combined")
    print("=" * 60)

    drone_colors = {"AR Drone": "#2196F3", "Bebop Drone": "#4CAF50", "Phantom Drone": "#FF9800"}

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # --- Top-left: Background time-domain ---
    print("\nLoading Background...")
    bg_signals = load_all_signals(BG_DIR)
    bg_full = np.concatenate(bg_signals)
    step = max(1, len(bg_full) // 50000)
    ax = axes[0, 0]
    ax.plot(np.arange(0, len(bg_full), step), bg_full[::step],
            color="#607D8B", linewidth=0.2, alpha=0.8)
    ax.set_title(f"Background — {len(bg_signals)} Files ({len(bg_full):,} samples)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.set_xlabel("Sample Index", fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Top-right: All 3 drones overlaid time-domain ---
    ax = axes[0, 1]
    for drone_name, drone_dir in DRONE_DIRS.items():
        print(f"Loading {drone_name}...")
        signals = load_all_signals(drone_dir)
        full = np.concatenate(signals)
        step = max(1, len(full) // 30000)
        ax.plot(np.arange(0, len(full), step), full[::step],
                color=drone_colors[drone_name], linewidth=0.15, alpha=0.5,
                label=drone_name)

    ax.set_title("All Drone Classes — Overlaid Time-Domain Signals",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.set_xlabel("Sample Index", fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Bottom-left: Background avg spectrogram ---
    print("\nComputing Background avg spectrogram...")
    f_ax, t_ax, bg_spec, bg_n = avg_spectrogram(bg_signals)
    ax = axes[1, 0]
    ax.pcolormesh(t_ax, f_ax, bg_spec, shading="gouraud", cmap="viridis")
    ax.set_title(f"Background — Avg Spectrogram ({bg_n} segments)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_xlabel("Time Frame", fontsize=10)

    # --- Bottom-right: All 3 drones combined avg spectrogram ---
    print("Computing combined drone avg spectrogram...")
    all_drone_specs = []
    total_segs = 0
    for drone_name, drone_dir in DRONE_DIRS.items():
        signals = load_all_signals(drone_dir)
        for sig in signals:
            segs = segment_signal(sig, window_size=SEGMENT_LEN, hop_size=HOP)
            for seg in segs:
                _, _, S = compute_log_spectrogram(seg)
                all_drone_specs.append(S)
                total_segs += 1

    f_ax, t_ax, _ = compute_log_spectrogram(segs[0])
    combined_spec = np.mean(all_drone_specs, axis=0)

    ax = axes[1, 1]
    ax.pcolormesh(t_ax, f_ax, combined_spec, shading="gouraud", cmap="viridis")
    ax.set_title(f"All Drones Combined — Avg Spectrogram ({total_segs} segments)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_xlabel("Time Frame", fontsize=10)

    fig.suptitle("DroneRF Dataset — Background vs All Drone Classes",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path1 = output_dir / "dronerf_combined_classes.png"
    plt.savefig(path1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nFigure 1 saved: {path1}")

    # =============================================================
    # FIGURE 2: RFUAV Dataset — Spectrogram Samples
    # =============================================================
    print("\n" + "=" * 60)
    print("  FIGURE 2: RFUAV Dataset — Spectrogram Samples")
    print("=" * 60)

    rfuav_root = Path(RFUAV_DIR)
    drone_folders = sorted([d for d in rfuav_root.iterdir() if d.is_dir()])

    # Pick 12 diverse drone types evenly spaced
    n_show = 12
    indices = np.linspace(0, len(drone_folders) - 1, n_show, dtype=int)
    selected = [drone_folders[i] for i in indices]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes_flat = axes.flatten()

    for idx, folder in enumerate(selected):
        drone_name = folder.name
        # Pick first image
        imgs = sorted(folder.glob("*.jpg"))
        if not imgs:
            imgs = sorted(folder.glob("*.png"))
        if not imgs:
            continue

        img = Image.open(imgs[0]).convert("L")
        arr = np.array(img, dtype=np.float32)

        ax = axes_flat[idx]
        ax.imshow(arr, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(drone_name, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        if idx >= 8:
            ax.set_xlabel("Time", fontsize=9)
        if idx % 4 == 0:
            ax.set_ylabel("Frequency", fontsize=9)

    fig.suptitle(f"RFUAV Dataset — Spectrogram Samples ({len(drone_folders)} Drone Types, Images Only)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path2 = output_dir / "rfuav_spectrogram_samples.png"
    plt.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nFigure 2 saved: {path2}")


if __name__ == "__main__":
    main()
