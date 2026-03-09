"""
Cross-dataset evaluation: train on one dataset, test on another.

Experiments:
1. Train on DroneRF -> Test on RFUAV (binary: can model generalize to unseen drone types?)
2. Train on RFUAV -> Test on DroneRF (does a richer training set help?)
3. Combined training -> Test on both

Usage:
    python -m src.evaluation.cross_dataset --model resnet
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt

from src.datasets.dronerf_precomputed_dataset import DroneRFPrecomputedDataset
from src.datasets.rfuav_dataset import RFUAVDataset, create_rfuav_splits
from src.models.cnn_spectrogram import SmallRFNet
from src.models.resnet_spectrogram import RFResNet
from src.models.transformer_spectrogram import RFTransformer
from src.evaluation.metrics import (
    collect_predictions, compute_classification_metrics, print_metrics_summary
)


MODEL_REGISTRY = {
    "smallrf": SmallRFNet,
    "resnet": RFResNet,
    "transformer": RFTransformer,
}


def train_model(model, train_loader, val_loader, device, epochs=20, lr=5e-4):
    """Quick training loop returning best model state."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc


def evaluate_on_dataset(model, loader, device, dataset_name, class_names=None):
    """Evaluate model on a dataset and print results."""
    y_true, y_pred, y_prob = collect_predictions(model, loader, device)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, class_names)
    print_metrics_summary(metrics, f"Tested on {dataset_name}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument("--model", default="resnet", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dronerf_csv", default="data/metadata/dronerf_precomputed.csv")
    parser.add_argument("--rfuav_root", default="data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output_dir", default="outputs/cross_dataset")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if RFUAV is available
    rfuav_available = Path(args.rfuav_root).exists()
    if not rfuav_available:
        print(f"\nRFUAV dataset not found at: {args.rfuav_root}")
        print("To enable cross-dataset experiments:")
        print("  1. python -m src.datasets.download_rfuav")
        print("  2. Images should be at: data/raw/RFUAV/ImageSet-AllDrones-MatlabPipeline/train/")
        print("\nSkipping cross-dataset evaluation.")
        return

    # Load datasets
    class_names_binary = ["Background", "Drone"]

    # DroneRF (binary)
    dronerf_train = DroneRFPrecomputedDataset(args.dronerf_csv, split="train", label_col="label_binary")
    dronerf_val = DroneRFPrecomputedDataset(args.dronerf_csv, split="val", label_col="label_binary")
    dronerf_test = DroneRFPrecomputedDataset(args.dronerf_csv, split="test", label_col="label_binary")

    # RFUAV (binary — all samples are drones, split from single folder)
    rfuav_train, rfuav_valid = create_rfuav_splits(
        args.rfuav_root, val_ratio=0.2, label_mode="binary"
    )

    dronerf_train_loader = DataLoader(dronerf_train, batch_size=16, shuffle=True, num_workers=0)
    dronerf_val_loader = DataLoader(dronerf_val, batch_size=16, shuffle=False, num_workers=0)
    dronerf_test_loader = DataLoader(dronerf_test, batch_size=16, shuffle=False, num_workers=0)
    rfuav_train_loader = DataLoader(rfuav_train, batch_size=16, shuffle=True, num_workers=0)
    rfuav_valid_loader = DataLoader(rfuav_valid, batch_size=16, shuffle=False, num_workers=0)

    all_results = {}
    ModelClass = MODEL_REGISTRY[args.model]

    # =========================================
    # Experiment 1: Train DroneRF -> Test RFUAV
    # =========================================
    print(f"\n{'='*60}")
    print("  EXPERIMENT 1: Train on DroneRF -> Test on RFUAV")
    print(f"{'='*60}")

    # Load pre-trained DroneRF model
    weights_path = f"outputs/{args.model}_binary_v2/models/best_model.pt"
    if not Path(weights_path).exists():
        weights_path = f"outputs/{args.model}_binary/models/best_model.pt"

    if Path(weights_path).exists():
        model = ModelClass(num_classes=2).to(device)
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
        print(f"  Loaded pre-trained: {weights_path}")

        # Test on DroneRF (sanity check)
        m1_dronerf = evaluate_on_dataset(model, dronerf_test_loader, device,
                                          "DroneRF", class_names_binary)

        # Test on RFUAV (cross-dataset)
        m1_rfuav = evaluate_on_dataset(model, rfuav_valid_loader, device,
                                        "RFUAV", class_names_binary)

        all_results["dronerf_to_rfuav"] = {
            "same_dataset_acc": m1_dronerf["accuracy"],
            "cross_dataset_acc": m1_rfuav["accuracy"],
            "same_dataset_f1": m1_dronerf["macro_f1"],
            "cross_dataset_f1": m1_rfuav["macro_f1"],
        }
    else:
        print(f"  No pre-trained model found at {weights_path}")

    # =========================================
    # Experiment 2: Train RFUAV -> Test DroneRF
    # =========================================
    print(f"\n{'='*60}")
    print("  EXPERIMENT 2: Train on RFUAV -> Test on DroneRF")
    print(f"{'='*60}")

    model2 = ModelClass(num_classes=2).to(device)
    print(f"  Training {args.model} on RFUAV...")
    model2, best_val = train_model(
        model2, rfuav_train_loader, rfuav_valid_loader, device, epochs=args.epochs
    )

    # Test on RFUAV (sanity check)
    m2_rfuav = evaluate_on_dataset(model2, rfuav_valid_loader, device,
                                    "RFUAV", class_names_binary)

    # Test on DroneRF (cross-dataset)
    m2_dronerf = evaluate_on_dataset(model2, dronerf_test_loader, device,
                                      "DroneRF", class_names_binary)

    all_results["rfuav_to_dronerf"] = {
        "same_dataset_acc": m2_rfuav["accuracy"],
        "cross_dataset_acc": m2_dronerf["accuracy"],
        "same_dataset_f1": m2_rfuav["macro_f1"],
        "cross_dataset_f1": m2_dronerf["macro_f1"],
    }

    # =========================================
    # Experiment 3: Combined (DroneRF + RFUAV) -> Test on both
    # =========================================
    print(f"\n{'='*60}")
    print("  EXPERIMENT 3: Combined training (DroneRF + RFUAV) -> Test on both")
    print(f"{'='*60}")

    # Combine training sets: DroneRF (has background + drones) + RFUAV (drones only)
    combined_train = ConcatDataset([dronerf_train, rfuav_train])
    combined_val = ConcatDataset([dronerf_val, rfuav_valid])
    combined_train_loader = DataLoader(combined_train, batch_size=16, shuffle=True, num_workers=0)
    combined_val_loader = DataLoader(combined_val, batch_size=16, shuffle=False, num_workers=0)

    print(f"  Combined train: {len(combined_train)} (DroneRF: {len(dronerf_train)} + RFUAV: {len(rfuav_train)})")
    print(f"  Combined val: {len(combined_val)} (DroneRF: {len(dronerf_val)} + RFUAV: {len(rfuav_valid)})")

    model3 = ModelClass(num_classes=2).to(device)
    print(f"  Training {args.model} on combined dataset...")
    model3, best_val = train_model(
        model3, combined_train_loader, combined_val_loader, device, epochs=args.epochs
    )

    # Test on DroneRF
    m3_dronerf = evaluate_on_dataset(model3, dronerf_test_loader, device,
                                      "DroneRF", class_names_binary)

    # Test on RFUAV
    m3_rfuav = evaluate_on_dataset(model3, rfuav_valid_loader, device,
                                    "RFUAV", class_names_binary)

    all_results["combined"] = {
        "dronerf_acc": m3_dronerf["accuracy"],
        "dronerf_f1": m3_dronerf["macro_f1"],
        "rfuav_acc": m3_rfuav["accuracy"],
        "rfuav_f1": m3_rfuav["macro_f1"],
        "train_size": len(combined_train),
    }

    # =========================================
    # Summary
    # =========================================
    print(f"\n{'='*60}")
    print(f"  CROSS-DATASET SUMMARY — {args.model}")
    print(f"{'='*60}")
    print(f"  {'Experiment':<30} {'Same-DS Acc':>12} {'Cross-DS Acc':>12} {'Gap':>8}")
    print(f"  {'-'*62}")
    for exp_name, r in all_results.items():
        gap = r["same_dataset_acc"] - r["cross_dataset_acc"]
        print(f"  {exp_name:<30} {r['same_dataset_acc']:>12.4f} {r['cross_dataset_acc']:>12.4f} {gap:>8.4f}")

    # Save results
    with open(output_dir / f"cross_dataset_{args.model}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {output_dir / f'cross_dataset_{args.model}.json'}")


if __name__ == "__main__":
    main()
