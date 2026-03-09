"""
Download RFUAV spectrogram images from Hugging Face.

Downloads ONLY the pre-generated spectrogram images (not the 1.3TB raw data).
This is the "ImageSet-AllDrones-MatlabPipeline" subset (~19K PNG images).

Usage:
    python -m src.datasets.download_rfuav
    python -m src.datasets.download_rfuav --output_dir data/raw/RFUAV

Requirements:
    pip install huggingface_hub
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download RFUAV spectrogram images")
    parser.add_argument("--output_dir", default="data/raw/RFUAV",
                        help="Where to save the dataset")
    parser.add_argument("--subset", default="spectrograms",
                        choices=["spectrograms", "validation"],
                        help="Which subset to download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("huggingface_hub not installed. Installing...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download, hf_hub_download

    repo_id = "kitofrank/RFUAV"

    if args.subset == "spectrograms":
        print("Downloading RFUAV spectrogram images...")
        print("  Source: https://huggingface.co/datasets/kitofrank/RFUAV")
        print(f"  Target: {output_dir}")
        print()
        print("  This downloads the ImageSet-AllDrones-MatlabPipeline folder.")
        print("  Size: ~19,085 spectrogram PNG images")
        print()

        # Download only the spectrogram folders
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(output_dir),
                allow_patterns=["ImageSet-AllDrones-MatlabPipeline/**"],
            )
            print(f"\nDownload complete!")
            print(f"Spectrograms saved to: {output_dir / 'ImageSet-AllDrones-MatlabPipeline'}")
            print()
            print("Next step: Organize into train/valid splits:")
            print(f"  python -m src.datasets.prepare_rfuav --input_dir {output_dir / 'ImageSet-AllDrones-MatlabPipeline'}")
        except Exception as e:
            print(f"Error downloading: {e}")
            print()
            print("Alternative: Download manually from:")
            print("  https://huggingface.co/datasets/kitofrank/RFUAV/tree/main")
            print(f"  Extract ImageSet-AllDrones-MatlabPipeline to: {output_dir}")

    elif args.subset == "validation":
        print("Downloading RFUAV validation set (5 drones)...")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(output_dir),
                allow_patterns=["ValidationSet_5Drones/**"],
            )
            print(f"\nDownload complete!")
            print(f"Validation set saved to: {output_dir / 'ValidationSet_5Drones'}")
        except Exception as e:
            print(f"Error: {e}")
            print("Download manually from: https://huggingface.co/datasets/kitofrank/RFUAV/tree/main")


if __name__ == "__main__":
    main()
