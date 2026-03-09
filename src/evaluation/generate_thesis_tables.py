"""
Generate thesis-ready comparison tables and summary figures from all experiment results.

Usage:
    python -m src.evaluation.generate_thesis_tables
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def print_latex_table(headers, rows, caption=""):
    """Print a LaTeX-formatted table."""
    n_cols = len(headers)
    col_fmt = "l" + "c" * (n_cols - 1)

    print(f"\n% {caption}")
    print(f"\\begin{{tabular}}{{{col_fmt}}}")
    print("\\hline")
    print(" & ".join(headers) + " \\\\")
    print("\\hline")
    for row in rows:
        print(" & ".join(str(v) for v in row) + " \\\\")
    print("\\hline")
    print("\\end{tabular}")


def main():
    output_dir = Path("outputs/thesis_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================
    # Table 1: Binary Detection Comparison
    # =========================================
    print("\n" + "="*70)
    print("  TABLE 1: BINARY DETECTION — ALL METHODS")
    print("="*70)

    binary_models = []
    for model_name in ["smallrf", "resnet", "transformer"]:
        for suffix in ["_binary_v2", "_binary"]:
            r = load_json(f"outputs/{model_name}{suffix}/results.json")
            if r:
                binary_models.append({
                    "name": f"{model_name}{'*' if 'v2' in suffix else ''}",
                    "accuracy": r.get("accuracy", 0),
                    "macro_f1": r.get("macro_f1", 0),
                    "mcc": r.get("mcc", 0),
                    "roc_auc": r.get("roc_auc", r.get("roc_auc_ovr", 0)),
                    "ece": r.get("ece", 0),
                    "params": r.get("param_count", "?"),
                })
                break

    # Add baselines
    for name in ["svm", "random_forest"]:
        r = load_json(f"outputs/baselines/{name}_results.json")
        if r and "test" in r:
            binary_models.append({
                "name": name.replace("_", " ").title(),
                "accuracy": r["test"]["accuracy"],
                "macro_f1": r["test"]["macro_f1"],
                "mcc": "-",
                "roc_auc": r["test"]["roc_auc"] or "-",
                "ece": "-",
                "params": "-",
            })

    print(f"\n  {'Model':<18} {'Accuracy':>10} {'Macro F1':>10} {'MCC':>10} "
          f"{'ROC-AUC':>10} {'ECE':>10} {'Params':>10}")
    print(f"  {'-'*78}")
    for m in binary_models:
        acc = f"{m['accuracy']:.4f}" if isinstance(m['accuracy'], float) else m['accuracy']
        f1 = f"{m['macro_f1']:.4f}" if isinstance(m['macro_f1'], float) else m['macro_f1']
        mcc = f"{m['mcc']:.4f}" if isinstance(m['mcc'], float) else m['mcc']
        auc = f"{m['roc_auc']:.4f}" if isinstance(m['roc_auc'], float) else m['roc_auc']
        ece = f"{m['ece']:.4f}" if isinstance(m['ece'], float) else m['ece']
        params = f"{m['params']:,}" if isinstance(m['params'], int) else m['params']
        print(f"  {m['name']:<18} {acc:>10} {f1:>10} {mcc:>10} {auc:>10} {ece:>10} {params:>10}")

    # =========================================
    # Table 2: Multi-class Attribution
    # =========================================
    print("\n" + "="*70)
    print("  TABLE 2: MULTI-CLASS ATTRIBUTION — ALL METHODS")
    print("="*70)

    multi_models = []
    for model_name in ["smallrf", "resnet", "transformer"]:
        # Prefer v2 results
        for suffix in ["_multiclass_v2", "_multiclass"]:
            r = load_json(f"outputs/{model_name}{suffix}/results.json")
            if r:
                multi_models.append({
                    "name": f"{model_name}{'*' if 'v2' in suffix else ''}",
                    "version": "v2" if "v2" in suffix else "v1",
                    "accuracy": r.get("accuracy", 0),
                    "bal_accuracy": r.get("balanced_accuracy", 0),
                    "macro_f1": r.get("macro_f1", 0),
                    "mcc": r.get("mcc", 0),
                    "ece": r.get("ece", 0),
                    "params": r.get("param_count", "?"),
                })
                break

        # Also show v1 if v2 exists
        for suffix in ["_multiclass_v2", "_multiclass"]:
            r = load_json(f"outputs/{model_name}{suffix}/results.json")
            if r and suffix == "_multiclass":
                already = any(m["name"] == model_name and m["version"] == "v1" for m in multi_models)
                if not already:
                    multi_models.append({
                        "name": model_name,
                        "version": "v1",
                        "accuracy": r.get("accuracy", 0),
                        "bal_accuracy": r.get("balanced_accuracy", 0),
                        "macro_f1": r.get("macro_f1", 0),
                        "mcc": r.get("mcc", 0),
                        "ece": r.get("ece", 0),
                        "params": r.get("param_count", "?"),
                    })

    # Add baselines
    for name in ["svm", "random_forest"]:
        r = load_json(f"outputs/baselines_multiclass/{name}_results.json")
        if r and "test" in r:
            multi_models.append({
                "name": name.replace("_", " ").title(),
                "version": "-",
                "accuracy": r["test"]["accuracy"],
                "bal_accuracy": "-",
                "macro_f1": r["test"]["macro_f1"],
                "mcc": "-",
                "ece": "-",
                "params": "-",
            })

    print(f"\n  {'Model':<18} {'Version':>8} {'Accuracy':>10} {'Bal.Acc':>10} "
          f"{'Macro F1':>10} {'MCC':>10} {'ECE':>10}")
    print(f"  {'-'*78}")
    for m in multi_models:
        acc = f"{m['accuracy']:.4f}" if isinstance(m['accuracy'], float) else m['accuracy']
        bal = f"{m['bal_accuracy']:.4f}" if isinstance(m['bal_accuracy'], float) else m['bal_accuracy']
        f1 = f"{m['macro_f1']:.4f}" if isinstance(m['macro_f1'], float) else m['macro_f1']
        mcc = f"{m['mcc']:.4f}" if isinstance(m['mcc'], float) else m['mcc']
        ece = f"{m['ece']:.4f}" if isinstance(m['ece'], float) else m['ece']
        print(f"  {m['name']:<18} {m['version']:>8} {acc:>10} {bal:>10} "
              f"{f1:>10} {mcc:>10} {ece:>10}")

    # =========================================
    # Table 3: Robustness Summary
    # =========================================
    print("\n" + "="*70)
    print("  TABLE 3: ROBUSTNESS (BINARY) — F1 @ DIFFERENT SNR LEVELS")
    print("="*70)

    snr_levels = ["clean", "snr_30dB", "snr_20dB", "snr_10dB", "snr_5dB", "snr_0dB", "snr_-5dB", "snr_-10dB"]
    snr_display = ["Clean", "30dB", "20dB", "10dB", "5dB", "0dB", "-5dB", "-10dB"]

    print(f"\n  {'Model':<15}", end="")
    for s in snr_display:
        print(f" {s:>7}", end="")
    print()
    print(f"  {'-'*80}")

    # (robustness data from the run_all_experiments output — stored in plots, not JSON)
    # We'll note this for the user to check the plots

    print("  (See robustness plots in outputs/*/robustness/robustness_vs_snr.png)")

    # =========================================
    # Table 4: Open-Set Summary
    # =========================================
    print("\n" + "="*70)
    print("  TABLE 4: OPEN-SET DETECTION — OOD METRICS BY HOLDOUT CLASS")
    print("="*70)

    for model_name in ["resnet", "smallrf", "transformer"]:
        summary = load_json(f"outputs/{model_name}_openset_full/openset_summary.json")
        if not summary:
            continue

        print(f"\n  Model: {model_name}")
        print(f"  {'Holdout':<15} {'Method':<15} {'AUROC':>8} {'AUPR':>8} {'FPR@95':>8}")
        print(f"  {'-'*55}")
        for holdout_name, methods in summary.items():
            for method, m in methods.items():
                print(f"  {holdout_name:<15} {method:<15} {m['auroc']:>8.4f} "
                      f"{m['aupr']:>8.4f} {m['fpr_at_95tpr']:>8.4f}")

    # =========================================
    # Generate combined comparison bar chart
    # =========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Binary comparison
    b_names = [m["name"] for m in binary_models]
    b_f1s = [m["macro_f1"] if isinstance(m["macro_f1"], float) else 0 for m in binary_models]
    colors_b = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    axes[0].barh(b_names, b_f1s, color=colors_b[:len(b_names)])
    axes[0].set_xlim(0.95, 1.005)
    axes[0].set_xlabel("Macro F1")
    axes[0].set_title("Binary Detection — Macro F1")
    for i, v in enumerate(b_f1s):
        axes[0].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

    # Multiclass comparison
    m_names = [f"{m['name']} ({m['version']})" for m in multi_models]
    m_f1s = [m["macro_f1"] if isinstance(m["macro_f1"], float) else 0 for m in multi_models]
    colors_m = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#795548", "#607D8B"]
    axes[1].barh(m_names, m_f1s, color=colors_m[:len(m_names)])
    axes[1].set_xlim(0.5, 1.0)
    axes[1].set_xlabel("Macro F1")
    axes[1].set_title("Multi-class Attribution — Macro F1")
    for i, v in enumerate(m_f1s):
        axes[1].text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison chart saved: {output_dir / 'model_comparison_bar.png'}")

    print(f"\n{'='*70}")
    print("  ALL THESIS TABLES GENERATED")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
