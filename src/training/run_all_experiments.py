"""
Master experiment runner — trains all models, runs baselines, evaluates robustness,
open-set detection, and generates explainability outputs.

Usage:
    python -m src.training.run_all_experiments
    python -m src.training.run_all_experiments --skip_baselines --skip_openset
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.dronerf_precomputed_dataset import DroneRFPrecomputedDataset
from src.models.cnn_spectrogram import SmallRFNet
from src.models.resnet_spectrogram import RFResNet
from src.models.transformer_spectrogram import RFTransformer
from src.evaluation.metrics import full_evaluation
from src.evaluation.robustness import run_robustness_evaluation
from src.evaluation.explainability import generate_gradcam_examples


def load_model(model_name, num_classes, weights_path, device):
    registry = {
        "smallrf": SmallRFNet,
        "resnet": RFResNet,
        "transformer": RFTransformer,
    }
    model = registry[model_name](num_classes=num_classes).to(device)
    if Path(weights_path).exists():
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
        print(f"Loaded weights: {weights_path}")
    else:
        print(f"WARNING: weights not found at {weights_path}, using random weights")
    return model


def run_comparison(csv_path, task, device, output_base="outputs"):
    """Run evaluation for all trained models and produce a comparison table."""
    label_col = "label_binary" if task == "binary" else "label_multiclass"
    num_classes = 2 if task == "binary" else 4
    class_names = (
        ["Background", "Drone"] if task == "binary"
        else ["Background", "AR Drone", "Bepop Drone", "Phantom Drone"]
    )

    test_ds = DroneRFPrecomputedDataset(csv_path, split="test", label_col=label_col)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    model_configs = [
        ("smallrf", f"{output_base}/smallrf_{task}/models/best_model.pt"),
        ("resnet", f"{output_base}/resnet_{task}/models/best_model.pt"),
        ("transformer", f"{output_base}/transformer_{task}/models/best_model.pt"),
    ]

    all_results = {}

    for model_name, weights_path in model_configs:
        if not Path(weights_path).exists():
            print(f"\nSkipping {model_name} — no trained weights at {weights_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name} ({task})")
        print(f"{'='*60}")

        model = load_model(model_name, num_classes, weights_path, device)
        out_dir = f"{output_base}/{model_name}_{task}/figures"

        metrics, _, _, _ = full_evaluation(
            model, test_loader, device,
            class_names=class_names,
            output_dir=out_dir,
            model_name=model_name
        )

        all_results[model_name] = {
            k: v for k, v in metrics.items()
            if k != "classification_report" and v is not None and not isinstance(v, str)
        }

    # Print comparison table
    if all_results:
        print(f"\n{'='*70}")
        print(f"  MODEL COMPARISON — {task.upper()}")
        print(f"{'='*70}")
        header = f"  {'Model':<15} {'Accuracy':>10} {'Bal.Acc':>10} {'MacroF1':>10} {'MCC':>10} {'ECE':>10}"
        print(header)
        print(f"  {'-'*65}")
        for name, m in all_results.items():
            print(f"  {name:<15} {m.get('accuracy',0):>10.4f} {m.get('balanced_accuracy',0):>10.4f} "
                  f"{m.get('macro_f1',0):>10.4f} {m.get('mcc',0):>10.4f} {m.get('ece',0):>10.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/metadata/dronerf_precomputed.csv")
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_robustness", action="store_true")
    parser.add_argument("--skip_openset", action="store_true")
    parser.add_argument("--skip_explainability", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    label_col = "label_binary" if args.task == "binary" else "label_multiclass"
    num_classes = 2 if args.task == "binary" else 4
    class_names = (
        ["Background", "Drone"] if args.task == "binary"
        else ["Background", "AR Drone", "Bepop Drone", "Phantom Drone"]
    )

    # =====================================================
    # 1. BASELINES (SVM + Random Forest)
    # =====================================================
    if not args.skip_baselines:
        print("\n" + "="*60)
        print("  PHASE 2: TRAINING BASELINES (SVM + RF)")
        print("="*60)
        from src.training.train_baselines import main as train_baselines
        train_baselines(csv_path=args.csv_path, label_col=label_col)

    # =====================================================
    # 2. COMPARE ALL MODELS
    # =====================================================
    print("\n" + "="*60)
    print("  MODEL COMPARISON")
    print("="*60)
    all_results = run_comparison(args.csv_path, args.task, device)

    # =====================================================
    # 3. ROBUSTNESS (for each trained model)
    # =====================================================
    if not args.skip_robustness:
        test_ds = DroneRFPrecomputedDataset(args.csv_path, split="test", label_col=label_col)

        for model_name in ["smallrf", "resnet", "transformer"]:
            weights_path = f"outputs/{model_name}_{args.task}/models/best_model.pt"
            if not Path(weights_path).exists():
                continue

            model = load_model(model_name, num_classes, weights_path, device)
            out_dir = f"outputs/{model_name}_{args.task}/robustness"

            run_robustness_evaluation(
                model, test_ds, device, out_dir,
                model_name=model_name, class_names=class_names
            )

    # =====================================================
    # 4. OPEN-SET (hold out one class, only for multiclass)
    # =====================================================
    if not args.skip_openset and args.task == "multiclass":
        from src.evaluation.openset import run_openset_evaluation

        test_ds = DroneRFPrecomputedDataset(args.csv_path, split="test", label_col=label_col)

        for model_name in ["resnet"]:  # ResNet has get_embedding
            weights_path = f"outputs/{model_name}_{args.task}/models/best_model.pt"
            if not Path(weights_path).exists():
                continue

            # Hold out Phantom drone (class 3) as unknown
            model = load_model(model_name, num_classes, weights_path, device)
            train_ds = DroneRFPrecomputedDataset(args.csv_path, split="train", label_col=label_col)
            train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)

            out_dir = f"outputs/{model_name}_{args.task}/openset"
            run_openset_evaluation(
                model, test_ds, device,
                holdout_class=3,  # Phantom drone
                train_loader=train_loader,
                num_known_classes=num_classes,
                output_dir=out_dir
            )

    # =====================================================
    # 5. EXPLAINABILITY (Grad-CAM)
    # =====================================================
    if not args.skip_explainability:
        test_ds = DroneRFPrecomputedDataset(args.csv_path, split="test", label_col=label_col)

        for model_name in ["smallrf", "resnet"]:
            weights_path = f"outputs/{model_name}_{args.task}/models/best_model.pt"
            if not Path(weights_path).exists():
                continue

            model = load_model(model_name, num_classes, weights_path, device)
            out_dir = f"outputs/{model_name}_{args.task}/explainability"

            try:
                generate_gradcam_examples(
                    model, test_ds, device,
                    model_name=model_name,
                    class_names=class_names,
                    output_dir=out_dir,
                    n_per_class=3
                )
            except Exception as e:
                print(f"Grad-CAM failed for {model_name}: {e}")

    print("\n" + "="*60)
    print("  ALL EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
