import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


def compute_log_spectrogram(signal, fs=1.0, nperseg=512, noverlap=256, eps=1e-10):
    """
    Calcule un spectrogramme log-magnitude à partir d'un signal 1D.
    """
    f, t, Zxx = stft(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None
    )

    S = np.abs(Zxx)
    S_log = np.log10(S + eps)
    S_log = (S_log - S_log.mean()) / (S_log.std() + eps)

    return f, t, S_log.astype(np.float32)


def plot_spectrogram(f, t, S_log, title="Spectrogramme"):
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, S_log, shading="gouraud")
    plt.title(title)
    plt.xlabel("Temps")
    plt.ylabel("Fréquence")
    plt.colorbar(label="Amplitude log")
    plt.tight_layout()
    plt.show()


def save_spectrogram(f, t, S_log, output_path, title="Spectrogramme"):
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, S_log, shading="gouraud")
    plt.title(title)
    plt.xlabel("Temps")
    plt.ylabel("Fréquence")
    plt.colorbar(label="Amplitude log")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()