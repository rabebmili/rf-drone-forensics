"""
Plot raw RF signals and their spectrograms for ALL files in the DroneRF dataset.
For each class: concatenates all files, shows the full time-domain waveform,
and computes an average spectrogram across all segments.

Usage:
    python -m src.evaluation.plot_raw_signals
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.datasets.load_signal import load_dronerf_csv
from src.preprocessing.stft_utils import compute_log_spectrogram
from src.preprocessing.segmentation import segment_signal


CLASS_DIRS = {
    "Background": "data/raw/DroneRF/Background RF activites",
    "AR Drone": "data/raw/DroneRF/AR drone",
    "Bebop Drone": "data/raw/DroneRF/Bepop drone",
    "Phantom Drone": "data/raw/DroneRF/Phantom drone",
}

SEGMENT_LEN = 131072
HOP = 65536


def load_all_signals(class_dir):
    """Load and concatenate all CSV files from a class directory."""
    folder = Path(class_dir)
    files = sorted(folder.glob("*.csv"))
    signals = []
    for f in files:
        try:
            sig = load_dronerf_csv(f)
            signals.append(sig)
        except Exception as e:
            print(f"  Skipping {f.name}: {e}")
    return signals, len(files)


def compute_avg_spectrogram(signals):
    """Compute average spectrogram across all segments from all files."""
    specs = []
    for sig in signals:
        segments = segment_signal(sig, window_size=SEGMENT_LEN, hop_size=HOP)
        for seg in segments:
            _, _, S = compute_log_spectrogram(seg)
            specs.append(S)
    if not specs:
        return None, None, None, 0
    # Get frequency/time axes from one segment
    f, t, _ = compute_log_spectrogram(segments[0])
    avg_spec = np.mean(specs, axis=0)
    return f, t, avg_spec, len(specs)


def main():
    output_dir = Path("outputs/thesis_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(18, 14))
    colors = ["#607D8B", "#2196F3", "#4CAF50", "#FF9800"]

    for row, (class_name, class_dir) in enumerate(CLASS_DIRS.items()):
        print(f"\nLoading {class_name} (all files)...")
        signals, n_files = load_all_signals(class_dir)

        # Concatenate all signals for time-domain plot
        full_signal = np.concatenate(signals)
        total_samples = len(full_signal)
        print(f"  {n_files} files, {total_samples:,} total samples")

        # --- Left: Full concatenated time-domain waveform ---
        ax_time = axes[row, 0]
        # Downsample for plotting (every Nth sample to keep ~50k points)
        step = max(1, total_samples // 50000)
        display = full_signal[::step]
        t_axis = np.arange(len(display)) * step
        ax_time.plot(t_axis, display, color=colors[row], linewidth=0.2, alpha=0.8)
        ax_time.set_ylabel("Amplitude", fontsize=10)
        ax_time.set_title(
            f"{class_name} — All {n_files} Files ({total_samples:,} samples)",
            fontsize=11, fontweight="bold"
        )
        ax_time.spines["top"].set_visible(False)
        ax_time.spines["right"].set_visible(False)
        ax_time.tick_params(labelsize=9)
        # Format x-axis in millions
        ax_time.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
        )
        if row == 3:
            ax_time.set_xlabel("Sample Index", fontsize=10)

        # --- Right: Average spectrogram across ALL segments ---
        ax_spec = axes[row, 1]
        print(f"  Computing average spectrogram...")
        f_axis, t_axis_spec, avg_spec, n_segments = compute_avg_spectrogram(signals)
        if avg_spec is not None:
            im = ax_spec.pcolormesh(t_axis_spec, f_axis, avg_spec,
                                     shading="gouraud", cmap="viridis")
            ax_spec.set_ylabel("Frequency", fontsize=10)
            ax_spec.set_title(
                f"{class_name} — Avg Spectrogram ({n_segments} segments)",
                fontsize=11, fontweight="bold"
            )
        ax_spec.tick_params(labelsize=9)
        if row == 3:
            ax_spec.set_xlabel("Time Frame", fontsize=10)

    fig.suptitle("DroneRF Dataset — Complete Raw Signals & Average Spectrograms",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = output_dir / "raw_signals_and_spectrograms.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {save_path}")


if __name__ == "__main__":
    main()
