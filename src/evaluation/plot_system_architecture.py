"""
Generate a comprehensive system architecture diagram for the thesis.
Shows the full RF Drone Forensics framework: data, preprocessing,
models, evaluation axes, and forensic output — all in French.

Usage:
    python -m src.evaluation.plot_system_architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ── Drawing helpers ──────────────────────────────────────────────────

def box(ax, x, y, w, h, label, fc, ec=None, fontsize=10,
        tc="white", fw="bold", zorder=2, ls="-"):
    """Draw a rounded rectangle with centered text."""
    if ec is None:
        ec = fc
    patch = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        facecolor=fc, edgecolor=ec, linewidth=1.8,
        zorder=zorder, linestyle=ls)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fw, color=tc, zorder=zorder + 1)


def bg(ax, x, y, w, h, color, alpha=0.08):
    """Draw a section background."""
    patch = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.03",
        facecolor=color, edgecolor=color, linewidth=1.5,
        alpha=alpha, zorder=0, linestyle="--")
    ax.add_patch(patch)


def harrow(ax, x1, x2, y, color, lw=2.0):
    """Horizontal arrow from (x1,y) to (x2,y)."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=3)


def varrow(ax, x, y1, y2, color, lw=2.0):
    """Vertical arrow from (x,y1) down to (x,y2)."""
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=3)


def hline(ax, x1, x2, y, color, lw=2.0):
    """Horizontal line segment (no arrowhead)."""
    ax.plot([x1, x2], [y, y], color=color, lw=lw,
            solid_capstyle="round", zorder=2)


def vline(ax, x, y1, y2, color, lw=2.0):
    """Vertical line segment (no arrowhead)."""
    ax.plot([x, x], [y1, y2], color=color, lw=lw,
            solid_capstyle="round", zorder=2)


def dot(ax, x, y, color, size=5):
    """Small circle at a junction point."""
    ax.plot(x, y, 'o', color=color, markersize=size, zorder=4)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    output_dir = Path("outputs/thesis_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_xlim(-0.5, 21.5)
    ax.set_ylim(-1.0, 15.5)
    ax.axis("off")

    # ── Title ──
    ax.text(10.5, 15.0,
            "Architecture du Système de Forensique RF de Drones",
            ha="center", fontsize=22, fontweight="bold", color="#1a1a1a")

    # ══════════════════════════════════════════════════════════════════
    # 1. ACQUISITION DES DONNÉES  (top-left)
    # ══════════════════════════════════════════════════════════════════
    s1_x, s1_y, s1_w, s1_h = 0, 11.2, 6.5, 2.8
    bg(ax, s1_x, s1_y, s1_w, s1_h, "#1565C0")
    ax.text(3.25, 13.7, "1. Acquisition des données",
            ha="center", fontsize=14, fontweight="bold", color="#0D47A1")

    box(ax, 0.3, 12.6, 2.8, 0.8,
        "Dataset DroneRF\n(4 classes, 10 MHz)", "#1565C0")
    box(ax, 3.5, 12.6, 2.8, 0.8,
        "Signaux RF bruts\n(.csv, 1D)", "#1976D2")
    harrow(ax, 3.1, 3.5, 13.0, "#0D47A1")

    classes = ["Fond", "AR Drone", "Bebop", "Phantom"]
    cls_colors = ["#607D8B", "#2196F3", "#4CAF50", "#FF9800"]
    for i, (cls, cc) in enumerate(zip(classes, cls_colors)):
        box(ax, 0.3 + i * 1.55, 11.5, 1.4, 0.55, cls, cc, fontsize=8.5)

    # Internal: Signaux RF bruts → classes (from center of right box down to class row)
    varrow(ax, 4.9, 12.6, 12.1, "#0D47A1", lw=1.3)

    # ══════════════════════════════════════════════════════════════════
    # 2. PRÉTRAITEMENT  (mid-left)
    # ══════════════════════════════════════════════════════════════════
    s2_x, s2_y, s2_w, s2_h = 0, 8.3, 6.5, 2.6
    bg(ax, s2_x, s2_y, s2_w, s2_h, "#00838F")
    ax.text(3.25, 10.6, "2. Prétraitement",
            ha="center", fontsize=14, fontweight="bold", color="#006064")

    box(ax, 0.3, 9.5, 1.8, 0.7,
        "Segmentation\ndu signal", "#00838F", fontsize=9)
    box(ax, 2.4, 9.5, 1.8, 0.7,
        "Transformée\nSTFT", "#00897B", fontsize=9)
    box(ax, 4.5, 9.5, 1.8, 0.7,
        "Spectrogramme\nlog-magnitude\nnormalisé", "#00695C", fontsize=9)
    harrow(ax, 2.1, 2.4, 9.85, "#006064", lw=1.5)
    harrow(ax, 4.2, 4.5, 9.85, "#006064", lw=1.5)

    box(ax, 0.3, 8.6, 5.9, 0.55,
        "Découpage : 70% train / 15% val / 15% test  "
        "(par fichier, sans fuite)",
        "#E0F2F1", ec="#00897B", fontsize=9, tc="#004D40", fw="medium")

    # ══════════════════════════════════════════════════════════════════
    # 3. MODÈLES DE CLASSIFICATION  (center)
    # ══════════════════════════════════════════════════════════════════
    s3_x, s3_y, s3_w, s3_h = 7.5, 5.8, 6.5, 5.1
    bg(ax, s3_x, s3_y, s3_w, s3_h, "#2E7D32")
    ax.text(10.75, 10.6, "3. Modèles de classification",
            ha="center", fontsize=14, fontweight="bold", color="#1B5E20")

    models = [
        ("SmallRFNet\n(CNN)", "#1565C0"),
        ("RFResNet\n(Résiduel)", "#2E7D32"),
        ("RFTransformer\n(Attention)", "#6A1B9A"),
    ]
    for i, (name, color) in enumerate(models):
        mx = 7.8 + i * 2.1
        box(ax, mx, 9.2, 1.9, 1.0, name, color, fontsize=10)

    box(ax, 7.8, 7.8, 5.9, 0.8,
        "Baselines ML : SVM + Random Forest\n"
        "(features handcrafted sur spectrogrammes)",
        "#795548", fontsize=10)

    box(ax, 7.8, 6.2, 2.8, 0.9,
        "Tâche binaire\nDrone vs Fond",
        "#E8F5E9", ec="#2E7D32", fontsize=10, tc="#1B5E20")
    box(ax, 10.9, 6.2, 2.8, 0.9,
        "Tâche multiclasse\n4 classes",
        "#E8F5E9", ec="#2E7D32", fontsize=10, tc="#1B5E20")

    # Internal arrows in Section 3
    # Models row → Baselines (vertical drop)
    varrow(ax, 10.75, 9.2, 8.6 + 0.05, "#555555", lw=1.3)
    # Baselines → Tasks (vertical drop, splitting into two)
    vline(ax, 10.75, 7.8, 7.4, "#555555", lw=1.3)
    hline(ax, 9.2, 12.3, 7.4, "#555555", lw=1.3)
    varrow(ax, 9.2, 7.4, 7.1 + 0.05, "#555555", lw=1.3)
    varrow(ax, 12.3, 7.4, 7.1 + 0.05, "#555555", lw=1.3)
    dot(ax, 10.75, 7.4, "#555555", size=4)

    # ══════════════════════════════════════════════════════════════════
    # 4. AXES D'ÉVALUATION  (right column, stacked)
    # ══════════════════════════════════════════════════════════════════
    ex = 15.0  # left edge of eval sections

    # 4a. Robustesse
    bg(ax, ex, 10.2, 6.0, 2.0, "#E65100")
    ax.text(ex + 3.0, 11.9, "4a. Robustesse",
            ha="center", fontsize=13, fontweight="bold", color="#BF360C")
    box(ax, ex + 0.2, 10.5, 1.7, 0.85,
        "Bruit AWGN\n(SNR variable)", "#E65100", fontsize=9)
    box(ax, ex + 2.1, 10.5, 1.7, 0.85,
        "Interférence\nRF", "#F57C00", fontsize=9)
    box(ax, ex + 4.0, 10.5, 1.7, 0.85,
        "Fading\nmultitrajet", "#FF9800", fontsize=9)
    # Internal arrows 4a
    harrow(ax, ex + 1.9, ex + 2.1, 10.92, "#FFF3E0", lw=1.3)
    harrow(ax, ex + 3.8, ex + 4.0, 10.92, "#FFF3E0", lw=1.3)

    # 4b. Open-set
    bg(ax, ex, 7.8, 6.0, 2.1, "#AD1457")
    ax.text(ex + 3.0, 9.6, "4b. Détection Open-Set",
            ha="center", fontsize=13, fontweight="bold", color="#880E4F")
    box(ax, ex + 0.2, 8.1, 1.5, 0.85,
        "MSP\nScoring", "#AD1457", fontsize=9)
    box(ax, ex + 1.9, 8.1, 1.5, 0.85,
        "Énergie\nScoring", "#C2185B", fontsize=9)
    box(ax, ex + 3.6, 8.1, 2.2, 0.85,
        "Distance de\nMahalanobis", "#D81B60", fontsize=9)
    # Internal arrows 4b
    harrow(ax, ex + 1.7, ex + 1.9, 8.52, "#FFD6E0", lw=1.3)
    harrow(ax, ex + 3.4, ex + 3.6, 8.52, "#FFD6E0", lw=1.3)

    # 4c. Explicabilité
    bg(ax, ex, 5.4, 6.0, 2.1, "#4A148C")
    ax.text(ex + 3.0, 7.2, "4c. Explicabilité",
            ha="center", fontsize=13, fontweight="bold", color="#4A148C")
    box(ax, ex + 0.2, 5.7, 1.8, 0.85,
        "Grad-CAM\n(heatmaps)", "#6A1B9A", fontsize=9.5)
    box(ax, ex + 2.2, 5.7, 1.8, 0.85,
        "Signatures\nRF clés", "#7B1FA2", fontsize=9.5)
    box(ax, ex + 4.2, 5.7, 1.6, 0.85,
        "Analyse\ndes décisions", "#8E24AA", fontsize=9.5)
    # Internal arrows 4c
    harrow(ax, ex + 2.0, ex + 2.2, 6.12, "#E1BEE7", lw=1.3)
    harrow(ax, ex + 4.0, ex + 4.2, 6.12, "#E1BEE7", lw=1.3)

    # ══════════════════════════════════════════════════════════════════
    # 5. MÉTRIQUES  (bottom-center)
    # ══════════════════════════════════════════════════════════════════
    bg(ax, 3.5, 2.8, 8.0, 2.6, "#37474F")
    ax.text(7.5, 5.1, "5. Métriques d'évaluation",
            ha="center", fontsize=13, fontweight="bold", color="#263238")

    metrics = [
        ("Accuracy\nF1-Score", "#455A64"),
        ("Matrice de\nconfusion", "#546E7A"),
        ("ROC-AUC\nPR-AUC", "#607D8B"),
        ("ECE\n(calibration)", "#78909C"),
        ("AUROC\nFPR@95", "#90A4AE"),
    ]
    for i, (label, color) in enumerate(metrics):
        box(ax, 3.8 + i * 1.5, 3.2, 1.35, 0.85, label, color, fontsize=8.5)

    # ══════════════════════════════════════════════════════════════════
    # 6. MODULE FORENSIQUE  (bottom-right)
    # ══════════════════════════════════════════════════════════════════
    bg(ax, 12.5, 0.5, 8.5, 4.8, "#B71C1C")
    ax.text(16.75, 5.0, "6. Module forensique",
            ha="center", fontsize=14, fontweight="bold", color="#B71C1C")

    # Row A — two wide boxes
    box(ax, 12.8, 3.8, 3.8, 0.8,
        "Analyse segment par segment\n+ score de confiance", "#C62828")
    box(ax, 17.0, 3.8, 3.8, 0.8,
        "Détection d'anomalies\n+ transitions temporelles", "#D32F2F")
    # Internal arrow Row A
    harrow(ax, 16.6, 17.0, 4.2, "#FFCDD2", lw=1.3)

    # Row B — three output boxes
    box(ax, 12.8, 2.5, 2.5, 0.8,
        "Chronologie\ndes événements", "#E53935")
    box(ax, 15.6, 2.5, 2.5, 0.8,
        "Rapport JSON\nstructuré", "#EF5350")
    box(ax, 18.4, 2.5, 2.4, 0.8,
        "Visualisations\n(timeline)", "#F44336")

    # Row C — final output
    box(ax, 13.5, 0.8, 5.8, 0.85,
        "forensic_report.json  +  forensic_timeline.png",
        "#FFEBEE", ec="#C62828", fontsize=11, tc="#B71C1C", fw="bold")

    # ══════════════════════════════════════════════════════════════════
    # ARROWS — clean orthogonal routing
    # ══════════════════════════════════════════════════════════════════

    # --- 1 → 2 : simple vertical drop ---
    varrow(ax, 3.25, s1_y, s2_y + s2_h + 0.1, "#0D47A1", lw=2.5)

    # --- 2 → 3 : horizontal right ---
    harrow(ax, s2_x + s2_w, s3_x, 9.85, "#006064", lw=2.5)

    # --- 3 → 4a/4b/4c : vertical bus on right side of models ---
    bus_x = s3_x + s3_w + 0.3       # 14.3
    bus_top = 10.95                  # center of 4a
    bus_mid = 8.55                   # center of 4b
    bus_bot = 6.15                   # center of 4c

    # Stub from models section to bus
    hline(ax, s3_x + s3_w, bus_x, 9.7, "#555555", lw=2.2)
    # Vertical bus
    vline(ax, bus_x, bus_bot, bus_top, "#555555", lw=2.2)
    # Junction dot where stub meets bus
    dot(ax, bus_x, 9.7, "#555555")

    # Branch → 4a Robustesse
    dot(ax, bus_x, bus_top, "#E65100", size=6)
    harrow(ax, bus_x, ex, bus_top, "#E65100", lw=2)

    # Branch → 4b Open-Set
    dot(ax, bus_x, bus_mid, "#AD1457", size=6)
    harrow(ax, bus_x, ex, bus_mid, "#AD1457", lw=2)

    # Branch → 4c Explicabilité
    dot(ax, bus_x, bus_bot, "#6A1B9A", size=6)
    harrow(ax, bus_x, ex, bus_bot, "#6A1B9A", lw=2)

    # --- 3 → 5 : L-shape down then left ---
    m5_jx = 9.2   # x junction (within models section)
    m5_jy = 5.45  # y junction (just below models)
    vline(ax, m5_jx, s3_y, m5_jy, "#37474F", lw=2.2)
    dot(ax, m5_jx, m5_jy, "#37474F")
    # Horizontal line going LEFT to metrics section right edge
    hline(ax, m5_jx, 7.5, m5_jy, "#37474F", lw=2.2)
    # Arrow pointing down into the metrics section
    varrow(ax, 7.5, m5_jy, 5.1 + 0.15, "#37474F", lw=2.2)

    # --- 3 → 6 : L-shape down then right ---
    m6_jx = 12.0  # x junction (right side of models)
    m6_jy = 5.3   # y junction (just below models, offset from m5)
    vline(ax, m6_jx, s3_y, m6_jy, "#B71C1C", lw=2.2)
    hline(ax, m6_jx, 16.75, m6_jy, "#B71C1C", lw=2.2)
    dot(ax, m6_jx, m6_jy, "#B71C1C")
    varrow(ax, 16.75, m6_jy, 5.0 + 0.15, "#B71C1C", lw=2.2)

    # --- Forensic internal: Row A → Row B (straight drops) ---
    fc = "#B71C1C"
    # Left pair: center of left-A → center of left-B
    varrow(ax, 14.05, 3.8, 3.35, fc, lw=1.5)
    # Right pair: need to span — right-A center → mid-B
    varrow(ax, 16.85, 3.8, 3.35, fc, lw=1.5)
    # Far right
    varrow(ax, 19.6, 3.8, 3.35, fc, lw=1.5)

    # Row B → Row C: merge into single arrow
    merge_y = 2.1
    for bx in [14.05, 16.85, 19.6]:
        vline(ax, bx, 2.5, merge_y, fc, lw=1.3)
    hline(ax, 14.05, 19.6, merge_y, fc, lw=1.3)
    varrow(ax, 16.4, merge_y, 0.8 + 0.85 + 0.1, fc, lw=1.8)

    # ══════════════════════════════════════════════════════════════════
    # LEGEND  (bottom strip)
    # ══════════════════════════════════════════════════════════════════
    ax.text(0.5, 0.2, "Légende :", fontsize=11,
            fontweight="bold", color="#333333")
    legend_items = [
        ("#1565C0", "Données"),
        ("#00838F", "Prétraitement"),
        ("#2E7D32", "Modèles DL"),
        ("#795548", "Baselines ML"),
        ("#E65100", "Robustesse"),
        ("#AD1457", "Open-Set"),
        ("#6A1B9A", "Explicabilité"),
        ("#B71C1C", "Forensique"),
    ]
    for i, (color, label) in enumerate(legend_items):
        lx = 2.0 + i * 2.35
        box(ax, lx, 0.05, 0.3, 0.3, "", color, fontsize=1, zorder=5)
        ax.text(lx + 0.4, 0.2, label, fontsize=8.5, va="center",
                color="#333333", zorder=5)

    # ── Dataset note (dashed) ──
    box(ax, 0, 2.8, 3.2, 2.6,
        "Datasets\nutilisés :\n\n• DroneRF\n• RFUAV\n\nEn cours :\n• CageDroneRF",
        "#ECEFF1", ec="#90A4AE", fontsize=9.5, tc="#37474F",
        fw="medium", ls="--")

    plt.tight_layout()
    save_path = output_dir / "system_architecture_fr.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Architecture diagram saved: {save_path}")


if __name__ == "__main__":
    main()
