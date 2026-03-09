import numpy as np


def segment_signal(signal, window_size=131072, hop_size=65536):
    """
    Découpe un signal 1D en segments glissants.
    """
    segments = []

    n = len(signal)
    for start in range(0, n - window_size + 1, hop_size):
        end = start + window_size
        seg = signal[start:end]
        segments.append(seg)

    return segments


if __name__ == "__main__":
    x = np.arange(20)
    segs = segment_signal(x, window_size=8, hop_size=4)
    for i, s in enumerate(segs):
        print(f"Segment {i}: {s}")