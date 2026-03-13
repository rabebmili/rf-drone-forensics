"""
Generate two additional architecture figures:
  1. High-level model comparison (branching diagram)
  2. Per-model detailed block diagrams

Usage:
    python -m src.evaluation.plot_model_overview
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def rbox(ax, x, y, w, h, label, color, tc="white", fs=11, fw="bold", ec=None):
    """Draw a rounded box."""
    if ec is None:
        ec = color
    box = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015",
        facecolor=color, edgecolor=ec, linewidth=1.5, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, fontweight=fw, color=tc, zorder=3)


def arr(ax, x1, y1, x2, y2, color="#555555", lw=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=1)


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: High-level branching overview
# ═══════════════════════════════════════════════════════════════
def figure1():
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(5, 9.65, "Classification des spectrogrammes RF",
            ha="center", fontsize=17, fontweight="bold", color="#1a1a1a")

    # Input
    rbox(ax, 3.2, 8.8, 3.6, 0.55, "Spectrogramme RF\n[1 x 257 x 511]",
         "#455A64", fs=12)

    # Three branches
    cx_left, cx_mid, cx_right = 1.0, 3.7, 6.4
    bw, bh = 2.6, 0.7

    # Arrows from input to 3 models
    arr(ax, 5.0, 8.8, 2.3, 8.0)
    arr(ax, 5.0, 8.8, 5.0, 8.0)
    arr(ax, 5.0, 8.8, 7.7, 8.0)

    # Model boxes
    rbox(ax, cx_left, 7.3, bw, bh, "SmallRFNet\n(CNN compact)", "#1565C0", fs=12)
    rbox(ax, cx_mid, 7.3, bw, bh, "RFResNet\n(CNN r\u00e9siduel)", "#2E7D32", fs=12)
    rbox(ax, cx_right, 7.3, bw, bh, "RFTransformer\n(Attention)", "#6A1B9A", fs=12)

    # Arrows down
    for cx in [cx_left, cx_mid, cx_right]:
        arr(ax, cx + bw / 2, 7.3, cx + bw / 2, 6.8)

    # Feature extraction descriptions
    desc_h = 0.85
    rbox(ax, cx_left, 5.95, bw, desc_h,
         "Extraction de\nmotifs locaux\ntemps\u2013fr\u00e9quence",
         "#E3F2FD", tc="#0D47A1", fs=10, fw="normal", ec="#1565C0")
    rbox(ax, cx_mid, 5.95, bw, desc_h,
         "Extraction de\ncaract\u00e9ristiques\nprofondes",
         "#E8F5E9", tc="#1B5E20", fs=10, fw="normal", ec="#2E7D32")
    rbox(ax, cx_right, 5.95, bw, desc_h,
         "Extraction locale\n+ mod\u00e9lisation\nglobale",
         "#F3E5F5", tc="#4A148C", fs=10, fw="normal", ec="#6A1B9A")

    # Arrows converge
    for cx in [cx_left, cx_mid, cx_right]:
        arr(ax, cx + bw / 2, 5.95, 5.0, 5.35)

    # Classification layers
    rbox(ax, 2.8, 4.6, 4.4, 0.6, "Couches de classification\n(Fully Connected + Softmax)",
         "#FF6F00", fs=12)

    arr(ax, 5.0, 4.6, 5.0, 4.05)

    # Final prediction
    rbox(ax, 2.2, 3.3, 5.6, 0.6,
         "Pr\u00e9diction finale\n(drone / non-drone  ou  classe de drone)",
         "#C62828", fs=12)

    return fig


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Per-model detailed blocks
# ═══════════════════════════════════════════════════════════════
def figure2():
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    models = [
        {
            "title": "SmallRFNet",
            "subtitle": "CNN compact \u2014 155K param\u00e8tres",
            "color": "#1565C0",
            "light": "#E3F2FD",
            "tc": "#0D47A1",
            "layers": [
                "Conv 3x3 \u2192 ReLU \u2192 Pooling",
                "Conv 3x3 \u2192 ReLU \u2192 Pooling",
                "Conv 3x3 \u2192 ReLU \u2192 AvgPool",
                "Flatten \u2192 Dense \u2192 Sortie",
            ],
            "description": "Extraction rapide de motifs\nlocaux dans le spectrogramme",
        },
        {
            "title": "RFResNet",
            "subtitle": "CNN r\u00e9siduel \u2014 697K param\u00e8tres",
            "color": "#2E7D32",
            "light": "#E8F5E9",
            "tc": "#1B5E20",
            "layers": [
                "Conv initiale 7x7",
                "Bloc r\u00e9siduel 1 (32 ch) x2",
                "Bloc r\u00e9siduel 2 (64 ch) x2",
                "Bloc r\u00e9siduel 3 (128 ch) x2",
                "Global Pooling \u2192 Dense",
            ],
            "description": "Extraction de repr\u00e9sentations\nplus profondes et plus stables",
        },
        {
            "title": "RFTransformer",
            "subtitle": "Hybride CNN + Attention \u2014 375K param\u00e8tres",
            "color": "#6A1B9A",
            "light": "#F3E5F5",
            "tc": "#4A148C",
            "layers": [
                "CNN Stem (3 conv, s=2)",
                "Embedding + [CLS] token",
                "Bloc d'attention x2",
                "Agr\u00e9gation \u2192 Dense",
            ],
            "description": "Mod\u00e9lisation des d\u00e9pendances\nglobales dans le spectrogramme",
        },
    ]

    for idx, (ax, m) in enumerate(zip(axes, models)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Title
        ax.text(5, 9.5, m["title"], ha="center", fontsize=16,
                fontweight="bold", color=m["color"])
        ax.text(5, 9.1, m["subtitle"], ha="center", fontsize=10, color="#777777")

        # Input
        rbox(ax, 2.5, 8.2, 5.0, 0.55, "Entr\u00e9e : spectrogramme RF",
             "#455A64", fs=11)

        arr(ax, 5.0, 8.2, 5.0, 7.85)

        # Model box with layers inside
        n_layers = len(m["layers"])
        box_h = 0.55 * n_layers + 0.6
        box_y = 7.7 - box_h

        # Outer model box
        outer = mpatches.FancyBboxPatch(
            (1.5, box_y), 7.0, box_h,
            boxstyle="round,pad=0.02",
            facecolor=m["light"], edgecolor=m["color"], linewidth=2, zorder=1)
        ax.add_patch(outer)

        # Model name inside box (top)
        ax.text(5.0, box_y + box_h - 0.25, m["title"],
                ha="center", fontsize=12, fontweight="bold", color=m["color"])

        # Layer rows
        ly = box_y + box_h - 0.65
        for layer_text in m["layers"]:
            rbox(ax, 2.0, ly, 6.0, 0.4, layer_text, m["color"], fs=11, fw="normal")
            ly -= 0.55

        # Arrow down from model box
        arr(ax, 5.0, box_y, 5.0, box_y - 0.35)

        # Description box
        desc_y = box_y - 1.1
        rbox(ax, 1.5, desc_y, 7.0, 0.7, m["description"],
             "white", tc=m["tc"], fs=11, fw="normal", ec=m["color"])

    fig.suptitle("Architecture d\u00e9taill\u00e9e des mod\u00e8les de classification",
                 fontsize=17, fontweight="bold", y=0.98)

    return fig


def main():
    output_dir = Path("outputs/thesis_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1 = figure1()
    fig1.tight_layout()
    p1 = output_dir / "model_overview_branching.png"
    fig1.savefig(p1, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"Figure 1 saved: {p1}")

    fig2 = figure2()
    fig2.tight_layout()
    p2 = output_dir / "model_overview_detailed.png"
    fig2.savefig(p2, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Figure 2 saved: {p2}")


if __name__ == "__main__":
    main()
