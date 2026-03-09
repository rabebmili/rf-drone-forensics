"""
RFUAV dataset loader for pre-generated spectrogram images (JPG format).

The RFUAV dataset contains RF spectrograms from 37 drone/controller models.
Source: https://huggingface.co/datasets/kitofrank/RFUAV

Actual directory structure after download:
    data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/
    └── train/
        ├── DJI AVATA2/
        │   ├── DJI AVTA20.jpg
        │   └── ...
        ├── DJI FPV COMBO/
        │   └── ...
        └── ... (37 drone folders)

Usage:
    from src.datasets.rfuav_dataset import RFUAVDataset
    ds = RFUAVDataset("data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/train")
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split


class RFUAVDataset(Dataset):
    """Loads RFUAV spectrogram JPG images for classification.

    Converts RGB JPGs to single-channel grayscale spectrograms
    and normalizes to match our model input format (1, H, W).
    """

    def __init__(self, root_dir, target_size=(257, 512),
                 label_mode="binary", indices=None):
        """
        Args:
            root_dir: path containing drone class folders (e.g. .../train/)
            target_size: resize spectrograms to (H, W) for model compatibility
            label_mode: "binary" (drone=1) or "multiclass" (per-model label)
            indices: optional list of indices to use (for train/val splitting)
        """
        self.root = Path(root_dir)
        self.target_size = target_size
        self.label_mode = label_mode

        if not self.root.exists():
            raise FileNotFoundError(
                f"RFUAV directory not found: {self.root}\n"
                f"Download: python -m src.datasets.download_rfuav"
            )

        # Discover drone classes (subfolders)
        self.classes = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes) if label_mode == "multiclass" else 2

        # Collect all image paths + labels (JPG or PNG)
        self._all_samples = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            # Look for images in class folder or imgs/ subfolder
            img_dir = cls_dir / "imgs" if (cls_dir / "imgs").exists() else cls_dir
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in sorted(img_dir.glob(ext)):
                    if label_mode == "binary":
                        label = 1  # all RFUAV samples are drones
                    else:
                        label = self.class_to_idx[cls_name]
                    self._all_samples.append((str(img_path), label))

        # Apply index filter (for train/val splits)
        if indices is not None:
            self.samples = [self._all_samples[i] for i in indices]
        else:
            self.samples = self._all_samples

        print(f"RFUAV: {len(self.samples)} samples, {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image and convert to grayscale
        img = Image.open(img_path).convert("L")

        # Resize to match our spectrogram dimensions
        if self.target_size is not None:
            img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)

        # Convert to tensor and normalize (zero-mean, unit-variance)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - arr.mean()) / (arr.std() + 1e-10)

        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        y = torch.tensor(label, dtype=torch.long)

        return x, y

    def get_class_names(self):
        return self.classes.copy()

    def get_total_samples(self):
        return len(self._all_samples)


def create_rfuav_splits(root_dir, val_ratio=0.2, random_state=42, **kwargs):
    """Create train/val splits from a single RFUAV folder.

    The RFUAV download only has one folder (train/), no separate validation set.
    This function creates an 80/20 stratified split from that single folder,
    ensuring each drone class is proportionally represented in both splits.

    Args:
        root_dir: path to the folder containing drone class subfolders
        val_ratio: fraction of data to use for validation (default 20%)
        random_state: seed for reproducibility
        **kwargs: passed to RFUAVDataset (e.g. label_mode, target_size)

    Returns:
        train_dataset, val_dataset
    """

    # Step 1: Load the entire dataset (all 5,559 images across 37 classes)
    # This scans all subfolders and collects (image_path, label) pairs
    full = RFUAVDataset(root_dir, **kwargs)

    # Get total number of samples before any filtering
    n = full.get_total_samples()

    # Step 2: Extract the label for each sample
    # We need these labels to do STRATIFIED splitting — meaning each class
    # keeps the same proportion in train and val (e.g., if "DJI FPV" is 5%
    # of total data, it will be ~5% of train AND ~5% of val)
    all_labels = [full._all_samples[i][1] for i in range(n)]

    # Step 3: Split sample INDICES (not the data itself) into train/val
    # We split indices [0, 1, 2, ..., n-1] rather than copying files,
    # then pass these indices to RFUAVDataset to filter which samples it uses.
    # stratify=all_labels ensures proportional class representation in both sets
    train_idx, val_idx = train_test_split(
        list(range(n)),       # indices into _all_samples
        test_size=val_ratio,  # 20% goes to validation
        random_state=random_state,  # reproducible split
        stratify=all_labels   # keep class balance in both sets
    )

    # Step 4: Create two separate dataset objects, each only seeing its indices
    # The RFUAVDataset constructor re-scans all files, then filters to only
    # keep samples at the given indices — so train_ds and val_ds share no data
    train_ds = RFUAVDataset(root_dir, indices=train_idx, **kwargs)
    val_ds = RFUAVDataset(root_dir, indices=val_idx, **kwargs)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds
