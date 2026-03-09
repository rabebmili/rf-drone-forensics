"""
Cross-condition evaluation: simulates different RF environments by applying
realistic signal degradations. Since CageDroneRF and RFUAV datasets are not
available, this module tests generalization by evaluating trained models under:

1. Different SNR levels (Gaussian noise)
2. WiFi-like interference (narrowband jamming in specific frequency bands)
3. Multipath fading simulation (frequency-selective attenuation)
4. Combined degradations

This provides evidence of robustness comparable to cross-dataset evaluation.

Usage:
    python -m src.evaluation.cross_condition --model resnet --task multiclass
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from src.datasets.dronerf_precomputed_dataset import DroneRFPrecomputedDataset
from src.models.cnn_spectrogram import SmallRFNet
from src.models.resnet_spectrogram import RFResNet
from src.models.transformer_spectrogram import RFTransformer
from src.evaluation.metrics import collect_predictions, compute_classification_metrics


MODEL_REGISTRY = {
    "smallrf": SmallRFNet,
    "resnet": RFResNet,
    "transformer": RFTransformer,
}


class DegradedDataset(Dataset):
    """Applies realistic RF degradations to spectrograms."""

    def __init__(self, base_dataset, degradation="none", severity=1.0):
        """
        Args:
            base_dataset: PyTorch Dataset returning (tensor[1,H,W], label)
            degradation: one of "none", "awgn", "interference", "fading", "combined"
            severity: 0.0 (no effect) to 1.0 (maximum degradation)
        """
        self.base = base_dataset
        self.degradation = degradation
        self.severity = severity

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]

        if self.degradation == "none":
            return x, y

        if self.degradation == "awgn":
            x = self._add_awgn(x)
        elif self.degradation == "interference":
            x = self._add_interference(x)
        elif self.degradation == "fading":
            x = self._add_fading(x)
        elif self.degradation == "combined":
            x = self._add_awgn(x)
            x = self._add_interference(x)
            x = self._add_fading(x)

        return x, y

    def _add_awgn(self, x):
        """Additive White Gaussian Noise at controlled power."""
        signal_power = torch.mean(x ** 2)
        # severity maps to SNR: 0.0 → 30dB, 1.0 → -5dB
        snr_db = 30 - 35 * self.severity
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(x) * torch.sqrt(noise_power.clamp(min=1e-10))
        return x + noise

    def _add_interference(self, x):
        """Narrowband interference — simulates WiFi/Bluetooth in specific freq bands."""
        H, W = x.shape[1], x.shape[2]

        # Number and width of interference bands scale with severity
        n_bands = max(1, int(3 * self.severity))
        band_width = max(1, int(H * 0.05 * self.severity))

        interference = torch.zeros_like(x)
        for _ in range(n_bands):
            center = torch.randint(band_width, H - band_width, (1,)).item()
            start = max(0, center - band_width // 2)
            end = min(H, center + band_width // 2)

            # Strong narrowband energy
            power = 2.0 * self.severity * torch.std(x).item()
            interference[:, start:end, :] = power * torch.randn(1, end - start, W)

        return x + interference

    def _add_fading(self, x):
        """Frequency-selective fading — attenuates random frequency bands."""
        H = x.shape[1]

        # Create fading profile (smooth random attenuation across frequency)
        n_control_points = 8
        control = 1.0 - self.severity * 0.8 * torch.rand(n_control_points)
        # Interpolate to full frequency resolution
        indices = torch.linspace(0, n_control_points - 1, H)
        fading_profile = torch.zeros(H)
        for i in range(H):
            idx = indices[i].item()
            low = int(idx)
            high = min(low + 1, n_control_points - 1)
            frac = idx - low
            fading_profile[i] = control[low] * (1 - frac) + control[high] * frac

        # Apply to spectrogram: multiply each frequency bin
        return x * fading_profile.unsqueeze(0).unsqueeze(2)


def evaluate_condition(model, test_dataset, device, degradation, severity,
                       batch_size=16, class_names=None):
    """Evaluate model under a specific degradation condition."""
    degraded = DegradedDataset(test_dataset, degradation=degradation, severity=severity)
    loader = DataLoader(degraded, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true, y_pred, y_prob = collect_predictions(model, loader, device)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, class_names)
    return metrics


def run_cross_condition_evaluation(model, test_dataset, device, class_names=None,
                                     output_dir="outputs/cross_condition",
                                     model_name="Model"):
    """Run comprehensive cross-condition evaluation."""
    print(f"\n{'='*60}")
    print(f"  CROSS-CONDITION EVALUATION: {model_name}")
    print(f"{'='*60}")

    degradations = ["none", "awgn", "interference", "fading", "combined"]
    severities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    results = {}

    for deg in degradations:
        results[deg] = {}
        if deg == "none":
            metrics = evaluate_condition(model, test_dataset, device, "none", 0.0,
                                          class_names=class_names)
            results[deg][0.0] = {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
            print(f"  {deg:<15} sev=0.0 | Acc: {metrics['accuracy']:.4f} | F1: {metrics['macro_f1']:.4f}")
            continue

        for sev in severities:
            if sev == 0.0:
                continue  # skip, same as "none"
            metrics = evaluate_condition(model, test_dataset, device, deg, sev,
                                          class_names=class_names)
            results[deg][sev] = {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
            print(f"  {deg:<15} sev={sev:.1f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['macro_f1']:.4f}")

    # Plot results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for deg in ["awgn", "interference", "fading", "combined"]:
        sevs = sorted([s for s in results[deg].keys()])
        accs = [results[deg][s]["accuracy"] for s in sevs]
        f1s = [results[deg][s]["macro_f1"] for s in sevs]

        axes[0].plot(sevs, accs, "o-", label=deg, linewidth=2, markersize=5)
        axes[1].plot(sevs, f1s, "o-", label=deg, linewidth=2, markersize=5)

    # Add clean baseline
    clean_acc = results["none"][0.0]["accuracy"]
    clean_f1 = results["none"][0.0]["macro_f1"]
    for ax in axes:
        ax.axhline(y=clean_acc if ax == axes[0] else clean_f1,
                    color="black", linestyle="--", alpha=0.5, label="Clean baseline")

    axes[0].set_xlabel("Degradation Severity")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"Accuracy vs. Degradation — {model_name}")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Degradation Severity")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title(f"Macro F1 vs. Degradation — {model_name}")
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "cross_condition_curves.png", dpi=150)
    plt.close()
    print(f"  Cross-condition plot saved: {out_dir / 'cross_condition_curves.png'}")

    # Save results JSON
    serializable = {}
    for deg, sevs in results.items():
        serializable[deg] = {str(k): v for k, v in sevs.items()}
    with open(out_dir / "cross_condition_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-condition robustness evaluation")
    parser.add_argument("--model", default="resnet", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--task", default="multiclass", choices=["binary", "multiclass"])
    parser.add_argument("--weights_dir", default=None,
                        help="Directory containing models/ subdir. Auto-detected if not set.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = 2 if args.task == "binary" else 4
    label_col = "label_binary" if args.task == "binary" else "label_multiclass"
    class_names = (
        ["Background", "Drone"] if args.task == "binary"
        else ["Background", "AR Drone", "Bepop Drone", "Phantom Drone"]
    )

    # Try v2 weights first, fall back to v1
    if args.weights_dir is None:
        v2_path = f"outputs/{args.model}_{args.task}_v2/models/best_model.pt"
        v1_path = f"outputs/{args.model}_{args.task}/models/best_model.pt"
        if Path(v2_path).exists():
            weights_path = v2_path
        elif Path(v1_path).exists():
            weights_path = v1_path
        else:
            print(f"No weights found for {args.model} {args.task}")
            return
    else:
        weights_path = f"{args.weights_dir}/models/best_model.pt"

    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    model.eval()
    print(f"Loaded: {weights_path}")

    test_ds = DroneRFPrecomputedDataset(
        "data/metadata/dronerf_precomputed.csv", split="test", label_col=label_col
    )

    output_dir = f"outputs/{args.model}_{args.task}_cross_condition"
    run_cross_condition_evaluation(
        model, test_ds, device,
        class_names=class_names,
        output_dir=output_dir,
        model_name=f"{args.model} ({args.task})"
    )


if __name__ == "__main__":
    main()
