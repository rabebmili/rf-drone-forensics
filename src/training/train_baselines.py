"""
Traditional ML baselines: SVM and Random Forest on handcrafted spectrogram features.
Provides comparison against CNN approaches.
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
import joblib

from src.datasets.dronerf_precomputed_dataset import DroneRFPrecomputedDataset
from src.evaluation.feature_extraction import extract_features_from_dataset


def train_and_evaluate_baseline(clf, clf_name, X_train, y_train, X_val, y_val,
                                X_test, y_test, scaler, output_dir, class_names):
    """Train a sklearn classifier and evaluate on val/test sets."""
    print(f"\n{'='*60}")
    print(f"  Training: {clf_name}")
    print(f"{'='*60}")

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")

        num_classes = len(np.unique(y_train))
        roc = None
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X)
            if num_classes == 2:
                roc = roc_auc_score(y, y_prob[:, 1])
            else:
                try:
                    roc = roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
                except ValueError:
                    pass

        results[split_name] = {
            "accuracy": acc,
            "macro_f1": f1,
            "roc_auc": roc,
        }

        print(f"\n  {split_name.upper()} Results:")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Macro F1:  {f1:.4f}")
        if roc is not None:
            print(f"    ROC-AUC:   {roc:.4f}")
        print(f"\n{classification_report(y, y_pred, target_names=class_names, zero_division=0)}")

    results["train_time_seconds"] = train_time

    # Save model and results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / f"{clf_name.lower().replace(' ', '_')}_model.joblib"
    scaler_path = out / f"{clf_name.lower().replace(' ', '_')}_scaler.joblib"
    results_path = out / f"{clf_name.lower().replace(' ', '_')}_results.json"

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)

    # Convert numpy types to Python native for JSON serialization
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: float(vv) if vv is not None else None for kk, vv in v.items()}
        else:
            serializable[k] = float(v) if isinstance(v, (np.floating, float)) else v
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"  Model saved: {model_path}")
    print(f"  Results saved: {results_path}")

    return results


def main(
    csv_path="data/metadata/dronerf_precomputed.csv",
    label_col="label_binary",
    output_dir="outputs/baselines"
):
    num_classes = 4 if label_col == "label_multiclass" else 2
    class_names = (
        ["Background", "Drone"] if num_classes == 2
        else ["Background", "AR Drone", "Bepop Drone", "Phantom Drone"]
    )

    print("Loading datasets and extracting features...")

    train_ds = DroneRFPrecomputedDataset(csv_path, split="train", label_col=label_col)
    val_ds = DroneRFPrecomputedDataset(csv_path, split="val", label_col=label_col)
    test_ds = DroneRFPrecomputedDataset(csv_path, split="test", label_col=label_col)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    X_train, y_train = extract_features_from_dataset(train_ds)
    X_val, y_val = extract_features_from_dataset(val_ds)
    X_test, y_test = extract_features_from_dataset(test_ds)

    print(f"  Feature vector size: {X_train.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    all_results = {}

    # SVM with RBF kernel
    svm_clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    all_results["SVM"] = train_and_evaluate_baseline(
        svm_clf, "SVM", X_train, y_train, X_val, y_val,
        X_test, y_test, scaler, output_dir, class_names
    )

    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    )
    all_results["Random Forest"] = train_and_evaluate_baseline(
        rf_clf, "Random Forest", X_train, y_train, X_val, y_val,
        X_test, y_test, scaler, output_dir, class_names
    )

    # Comparison summary
    print(f"\n{'='*60}")
    print("  BASELINE COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Test Acc':>10} {'Test F1':>10} {'Test AUC':>10}")
    print(f"  {'-'*50}")
    for name, res in all_results.items():
        test_res = res["test"]
        auc_str = f"{test_res['roc_auc']:.4f}" if test_res['roc_auc'] else "N/A"
        print(f"  {name:<20} {test_res['accuracy']:>10.4f} {test_res['macro_f1']:>10.4f} {auc_str:>10}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train baseline classifiers (SVM + RF)")
    parser.add_argument("--csv_path", default="data/metadata/dronerf_precomputed.csv")
    parser.add_argument("--label_col", default="label_binary",
                        choices=["label_binary", "label_multiclass"])
    parser.add_argument("--output_dir", default="outputs/baselines")
    args = parser.parse_args()
    main(csv_path=args.csv_path, label_col=args.label_col, output_dir=args.output_dir)
