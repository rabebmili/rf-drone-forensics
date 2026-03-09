"""
Comprehensive evaluation metrics for RF drone forensics.
Covers: classification metrics, ROC/PR curves, confusion matrix,
calibration (ECE), and per-class analysis.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score
)


CLASS_NAMES_BINARY = ["Background", "Drone"]
CLASS_NAMES_MULTI = ["Background", "AR Drone", "Bepop Drone", "Phantom Drone"]


def collect_predictions(model, loader, device, return_probs=True):
    """Run model on a DataLoader and collect predictions, labels, and probabilities."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            if return_probs:
                all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs) if return_probs else None

    return y_true, y_pred, y_prob


def compute_classification_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """Compute comprehensive classification metrics."""
    num_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = CLASS_NAMES_BINARY if num_classes <= 2 else CLASS_NAMES_MULTI

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names[:num_classes], zero_division=0
        ),
    }

    if y_prob is not None and num_classes == 2:
        results["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
    elif y_prob is not None and num_classes > 2:
        try:
            results["roc_auc_ovr"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except ValueError:
            results["roc_auc_ovr"] = None

    if y_prob is not None:
        results["ece"] = compute_ece(y_true, y_pred, y_prob)

    return results


def compute_ece(y_true, y_pred, y_prob, n_bins=15):
    """Expected Calibration Error — measures how well predicted probabilities
    match actual accuracy."""
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)

    return ece


def plot_confusion_matrix(y_true, y_pred, class_names=None, output_path=None,
                          title="Confusion Matrix", normalize=None):
    """Plot and optionally save confusion matrix."""
    num_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    if class_names is None:
        class_names = CLASS_NAMES_BINARY if num_classes <= 2 else CLASS_NAMES_MULTI

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names[:num_classes])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Confusion matrix saved: {output_path}")
    plt.close()


def plot_roc_curves(y_true, y_prob, class_names=None, output_path=None):
    """Plot ROC curves — binary or one-vs-rest for multiclass."""
    num_classes = y_prob.shape[1]
    if class_names is None:
        class_names = CLASS_NAMES_BINARY if num_classes <= 2 else CLASS_NAMES_MULTI

    fig, ax = plt.subplots(figsize=(8, 6))

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"Drone (AUC = {roc_auc:.3f})")
    else:
        for i in range(num_classes):
            y_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"ROC curves saved: {output_path}")
    plt.close()


def plot_precision_recall_curves(y_true, y_prob, class_names=None, output_path=None):
    """Plot Precision-Recall curves."""
    num_classes = y_prob.shape[1]
    if class_names is None:
        class_names = CLASS_NAMES_BINARY if num_classes <= 2 else CLASS_NAMES_MULTI

    fig, ax = plt.subplots(figsize=(8, 6))

    if num_classes == 2:
        prec, rec, _ = precision_recall_curve(y_true, y_prob[:, 1])
        ap = average_precision_score(y_true, y_prob[:, 1])
        ax.plot(rec, prec, label=f"Drone (AP = {ap:.3f})")
    else:
        for i in range(num_classes):
            y_binary = (y_true == i).astype(int)
            prec, rec, _ = precision_recall_curve(y_binary, y_prob[:, i])
            ap = average_precision_score(y_binary, y_prob[:, i])
            ax.plot(rec, prec, label=f"{class_names[i]} (AP = {ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"PR curves saved: {output_path}")
    plt.close()


def plot_calibration_diagram(y_true, y_prob, n_bins=10, output_path=None):
    """Reliability diagram for calibration analysis."""
    confidences = np.max(y_prob, axis=1)
    y_pred = np.argmax(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(bin_confs, bin_accs, width=1.0 / n_bins, alpha=0.7, edgecolor="black", label="Model")
    ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_title("Calibration Diagram")
    ax.legend()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Calibration diagram saved: {output_path}")
    plt.close()


def print_metrics_summary(metrics, model_name="Model"):
    """Print a formatted summary of classification metrics."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:           {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:        {metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision:    {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:       {metrics['macro_recall']:.4f}")
    print(f"  Cohen's Kappa:      {metrics['cohen_kappa']:.4f}")
    print(f"  MCC:                {metrics['mcc']:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    if "roc_auc_ovr" in metrics and metrics["roc_auc_ovr"] is not None:
        print(f"  ROC-AUC (OvR):      {metrics['roc_auc_ovr']:.4f}")
    if "ece" in metrics:
        print(f"  ECE:                {metrics['ece']:.4f}")
    print(f"{'='*60}")
    print(f"\n{metrics['classification_report']}")


def full_evaluation(model, loader, device, class_names=None, output_dir=None,
                    model_name="Model"):
    """Run complete evaluation: metrics + all plots."""
    y_true, y_pred, y_prob = collect_predictions(model, loader, device)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, class_names)
    print_metrics_summary(metrics, model_name)

    if output_dir:
        out = Path(output_dir)
        plot_confusion_matrix(y_true, y_pred, class_names,
                              output_path=out / "confusion_matrix.png")
        plot_roc_curves(y_true, y_prob, class_names,
                        output_path=out / "roc_curves.png")
        plot_precision_recall_curves(y_true, y_prob, class_names,
                                     output_path=out / "pr_curves.png")
        plot_calibration_diagram(y_true, y_prob,
                                 output_path=out / "calibration_diagram.png")

    return metrics, y_true, y_pred, y_prob
