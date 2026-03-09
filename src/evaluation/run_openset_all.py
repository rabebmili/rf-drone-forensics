"""
Run open-set evaluation with each drone class held out as unknown.
Tests all 3 drone classes (AR, Bepop, Phantom) as holdout.

Usage:
    python -m src.evaluation.run_openset_all
    python -m src.evaluation.run_openset_all --model resnet
    python -m src.evaluation.run_openset_all --model smallrf
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch import nn, optim

from src.datasets.dronerf_precomputed_dataset import DroneRFPrecomputedDataset
from src.models.cnn_spectrogram import SmallRFNet
from src.models.resnet_spectrogram import RFResNet
from src.models.transformer_spectrogram import RFTransformer
from src.evaluation.openset import (
    compute_msp_scores, compute_energy_scores,
    compute_mahalanobis_scores, fit_mahalanobis,
    evaluate_ood_detection, _plot_ood_distributions,
)


MODEL_REGISTRY = {
    "smallrf": SmallRFNet,
    "resnet": RFResNet,
    "transformer": RFTransformer,
}

CLASS_NAMES = ["Background", "AR Drone", "Bepop Drone", "Phantom Drone"]
DRONE_CLASSES = {1: "AR Drone", 2: "Bepop Drone", 3: "Phantom Drone"}


def train_openset_model(model_name, holdout_class, csv_path, device, epochs=20, lr=5e-4):
    """Train a model excluding one class, returning model + loaders."""
    # Load full datasets
    full_train = DroneRFPrecomputedDataset(csv_path, split="train", label_col="label_multiclass")
    full_val = DroneRFPrecomputedDataset(csv_path, split="val", label_col="label_multiclass")

    # Filter out holdout class from train and val
    train_idx = [i for i in range(len(full_train))
                 if full_train.df.iloc[i]["label_multiclass"] != holdout_class]
    val_idx = [i for i in range(len(full_val))
               if full_val.df.iloc[i]["label_multiclass"] != holdout_class]

    # Remap labels: remove the holdout class from label space
    # Known classes are the remaining ones
    known_classes = sorted([c for c in range(4) if c != holdout_class])
    label_remap = {old: new for new, old in enumerate(known_classes)}
    num_known = len(known_classes)

    class RemappedSubset(torch.utils.data.Dataset):
        def __init__(self, base_ds, indices, remap):
            self.base = base_ds
            self.indices = indices
            self.remap = remap

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            x, y = self.base[self.indices[idx]]
            return x, torch.tensor(self.remap[y.item()], dtype=torch.long)

    train_ds = RemappedSubset(full_train, train_idx, label_remap)
    val_ds = RemappedSubset(full_val, val_idx, label_remap)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # Train model
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(num_classes=num_known).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_f1 = 0.0
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

        # Quick val check
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        if val_acc > best_f1:
            best_f1 = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return model, train_loader, num_known, label_remap, known_classes


def run_ood_for_holdout(model, model_name, holdout_class, csv_path, device,
                         train_loader, num_known, output_dir):
    """Evaluate OOD detection for a specific holdout class."""
    full_test = DroneRFPrecomputedDataset(csv_path, split="test", label_col="label_multiclass")

    # Split test set into known (in-distribution) and unknown (OOD)
    known_idx = []
    unknown_idx = []
    for i in range(len(full_test)):
        label = full_test.df.iloc[i]["label_multiclass"]
        if label == holdout_class:
            unknown_idx.append(i)
        else:
            known_idx.append(i)

    known_ds = Subset(full_test, known_idx)
    unknown_ds = Subset(full_test, unknown_idx)

    known_loader = DataLoader(known_ds, batch_size=16, shuffle=False, num_workers=0)
    unknown_loader = DataLoader(unknown_ds, batch_size=16, shuffle=False, num_workers=0)

    print(f"  Known: {len(known_idx)} | Unknown ({DRONE_CLASSES[holdout_class]}): {len(unknown_idx)}")

    results = {}

    # MSP
    in_msp, _ = compute_msp_scores(model, known_loader, device)
    ood_msp, _ = compute_msp_scores(model, unknown_loader, device)
    results["MSP"] = evaluate_ood_detection(in_msp, ood_msp, "MSP")

    # Energy
    in_energy, _ = compute_energy_scores(model, known_loader, device)
    ood_energy, _ = compute_energy_scores(model, unknown_loader, device)
    results["Energy"] = evaluate_ood_detection(in_energy, ood_energy, "Energy")

    # Mahalanobis (if model has get_embedding)
    if hasattr(model, "get_embedding"):
        class_means, cov_inv = fit_mahalanobis(model, train_loader, device, num_known)
        in_maha, _ = compute_mahalanobis_scores(model, known_loader, device, class_means, cov_inv)
        ood_maha, _ = compute_mahalanobis_scores(model, unknown_loader, device, class_means, cov_inv)
        results["Mahalanobis"] = evaluate_ood_detection(in_maha, ood_maha, "Mahalanobis")

    # Save plots
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    holdout_name = DRONE_CLASSES[holdout_class].replace(" ", "_")
    _plot_ood_distributions(in_msp, ood_msp, "MSP Score", str(out),
                            f"msp_holdout_{holdout_name}.png")
    _plot_ood_distributions(in_energy, ood_energy, "Energy Score", str(out),
                            f"energy_holdout_{holdout_name}.png")

    return results


def main():
    parser = argparse.ArgumentParser(description="Open-set evaluation with all holdout classes")
    parser.add_argument("--model", default="resnet", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--csv_path", default="data/metadata/dronerf_precomputed.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"outputs/{args.model}_openset_full"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    all_results = {}

    for holdout_class, holdout_name in DRONE_CLASSES.items():
        print(f"\n{'='*60}")
        print(f"  OPEN-SET: Holdout = {holdout_name} (class {holdout_class})")
        print(f"{'='*60}")

        # Train model without this class
        print(f"  Training {args.model} without {holdout_name}...")
        model, train_loader, num_known, label_remap, known_classes = train_openset_model(
            args.model, holdout_class, args.csv_path, device, epochs=args.epochs
        )

        known_names = [CLASS_NAMES[c] for c in known_classes]
        print(f"  Known classes: {known_names}")

        # Evaluate OOD detection
        holdout_dir = f"{args.output_dir}/holdout_{holdout_name.replace(' ', '_')}"
        results = run_ood_for_holdout(
            model, args.model, holdout_class, args.csv_path, device,
            train_loader, num_known, holdout_dir
        )
        all_results[holdout_name] = results

    # Summary table
    print(f"\n{'='*70}")
    print(f"  OPEN-SET SUMMARY — {args.model}")
    print(f"{'='*70}")
    print(f"  {'Holdout':<15} {'Method':<15} {'AUROC':>8} {'AUPR':>8} {'FPR@95':>8}")
    print(f"  {'-'*55}")
    for holdout_name, methods in all_results.items():
        for method_name, m in methods.items():
            print(f"  {holdout_name:<15} {method_name:<15} {m['auroc']:>8.4f} "
                  f"{m['aupr']:>8.4f} {m['fpr_at_95tpr']:>8.4f}")

    # Save summary
    out_path = Path(args.output_dir) / "openset_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for holdout, methods in all_results.items():
        serializable[holdout] = {}
        for method, metrics in methods.items():
            serializable[holdout][method] = {
                k: float(v) for k, v in metrics.items()
            }

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSummary saved: {out_path}")


if __name__ == "__main__":
    main()
