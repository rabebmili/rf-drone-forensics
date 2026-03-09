import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.load_signal import load_dronerf_csv
from src.preprocessing.stft_utils import compute_log_spectrogram


class DroneRFSegmentDataset(Dataset):
    def __init__(
        self,
        segments_csv="data/metadata/dronerf_segments.csv",
        label_col="label_binary",
        fs=1.0,
        nperseg=512,
        noverlap=256
    ):
        self.df = pd.read_csv(segments_csv)
        self.label_col = label_col
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        file_path = row["file_path"]
        start = int(row["start"])
        end = int(row["end"])

        signal = load_dronerf_csv(file_path)
        segment = signal[start:end]

        _, _, S_log = compute_log_spectrogram(
            segment,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap
        )

        x = torch.tensor(S_log, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        y = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        return x, y