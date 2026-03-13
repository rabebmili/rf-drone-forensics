"""
Generate a thesis-ready figure showing the 3 model architectures
used for RF spectrogram classification. Clean style, no dimension numbers.

Usage:
    python -m src.evaluation.plot_architectures
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def draw_block(ax, x, y, w, h, label, color, edge_color=None, fontsize=9,
               text_color="white", fontweight="bold", alpha=1.0):
    if edge_color is None:
        edge_color = color
    box = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015",
        facecolor=color, edgecolor=edge_color, linewidth=1.2,
        alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color="#666666"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=1)


def draw_skip_arrow(ax, x1, y1, x2, y2, color="#999999"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                connectionstyle="arc3,rad=-0.3",
                                linestyle="dashed"), zorder=1)


def draw_model_a(ax, x_off, y_off):
    """SmallRFNet — CNN compact."""
    bw, bh = 1.6, 0.38
    gap = 0.18
    cx = x_off + 0.2

    ax.text(cx + bw / 2, y_off + 7.0, "SmallRFNet (CNN)", ha="center",
            fontsize=14, fontweight="bold", color="#0D47A1")
    ax.text(cx + bw / 2, y_off + 6.7, "155K param\u00e8tres", ha="center",
            fontsize=10, color="#666666")

    layers = [
        ("Entr\u00e9e\nSpectrogramme RF", "#ECEFF1", "#455A64"),
        ("Conv2d + BN + ReLU", "#1565C0", "white"),
        ("MaxPool", "#1976D2", "white"),
        ("Conv2d + BN + ReLU", "#1565C0", "white"),
        ("MaxPool", "#1976D2", "white"),
        ("Conv2d + BN + ReLU", "#1565C0", "white"),
        ("AdaptiveAvgPool", "#1976D2", "white"),
        ("Aplatissement", "#78909C", "white"),
        ("Dense + ReLU\n+ Dropout", "#0D47A1", "white"),
        ("Sortie (C classes)", "#B71C1C", "white"),
    ]

    y = y_off + 6.2
    for i, (label, color, tc) in enumerate(layers):
        draw_block(ax, cx, y, bw, bh, label, color, fontsize=8.5, text_color=tc)
        if i < len(layers) - 1:
            draw_arrow(ax, cx + bw / 2, y, cx + bw / 2, y - gap)
        y -= (bh + gap)


def draw_model_b(ax, x_off, y_off):
    """RFResNet — CNN r\u00e9siduel."""
    bw, bh = 1.6, 0.38
    gap = 0.18
    cx = x_off + 0.2

    ax.text(cx + bw / 2, y_off + 7.0, "RFResNet", ha="center",
            fontsize=14, fontweight="bold", color="#1B5E20")
    ax.text(cx + bw / 2, y_off + 6.7, "697K param\u00e8tres", ha="center",
            fontsize=10, color="#666666")

    layers = [
        ("Entr\u00e9e\nSpectrogramme RF", "#ECEFF1", "#455A64", False),
        ("Conv initiale\n+ BN + ReLU + MaxPool", "#2E7D32", "white", False),
        ("Bloc r\u00e9siduel x2", "#388E3C", "white", True),
        ("Bloc r\u00e9siduel x2", "#388E3C", "white", True),
        ("Bloc r\u00e9siduel x2", "#388E3C", "white", True),
        ("AdaptiveAvgPool", "#43A047", "white", False),
        ("Aplatissement", "#78909C", "white", False),
        ("Dropout\n+ Dense (C classes)", "#B71C1C", "white", False),
    ]

    y = y_off + 6.2
    for i, (label, color, tc, has_skip) in enumerate(layers):
        draw_block(ax, cx, y, bw, bh, label, color, fontsize=8.5, text_color=tc)
        if i < len(layers) - 1:
            draw_arrow(ax, cx + bw / 2, y, cx + bw / 2, y - gap)
        if has_skip:
            skip_x = cx + bw + 0.05
            draw_skip_arrow(ax, skip_x, y + bh * 0.8, skip_x, y + bh * 0.2, "#81C784")
            ax.text(skip_x + 0.12, y + bh / 2, "+", fontsize=11,
                    fontweight="bold", color="#81C784", ha="center", va="center")
        y -= (bh + gap)


def draw_model_c(ax, x_off, y_off):
    """RFTransformer — Hybride CNN + Attention."""
    bw, bh = 1.6, 0.38
    gap = 0.18
    cx = x_off + 0.2

    ax.text(cx + bw / 2, y_off + 7.0, "RFTransformer", ha="center",
            fontsize=14, fontweight="bold", color="#4A148C")
    ax.text(cx + bw / 2, y_off + 6.7, "375K param\u00e8tres", ha="center",
            fontsize=10, color="#666666")

    stem_layers = [
        ("Entr\u00e9e\nSpectrogramme RF", "#ECEFF1", "#455A64"),
        ("Conv + BN + ReLU", "#6A1B9A", "white"),
        ("Conv + BN + ReLU", "#6A1B9A", "white"),
        ("Conv + BN + ReLU", "#6A1B9A", "white"),
        ("AvgPool\n\u2192 s\u00e9quence de tokens", "#7B1FA2", "white"),
    ]

    transformer_layers = [
        ("Token [CLS]\n+ Emb. positionnelle", "#E65100", "white"),
        ("Bloc Transformer x2\n(attention multi-t\u00eate)", "#F57C00", "white"),
        ("LayerNorm\ntoken [CLS]", "#FF9800", "#333333"),
        ("Dropout\n+ Dense (C classes)", "#B71C1C", "white"),
    ]

    y = y_off + 6.2

    ax.text(cx - 0.15, y + 0.1, "CNN\nStem", fontsize=8, color="#6A1B9A",
            fontweight="bold", ha="center", va="center", rotation=90)

    for i, (label, color, tc) in enumerate(stem_layers):
        draw_block(ax, cx, y, bw, bh, label, color, fontsize=8.5, text_color=tc)
        if i < len(stem_layers) - 1:
            draw_arrow(ax, cx + bw / 2, y, cx + bw / 2, y - gap)
        y -= (bh + gap)

    draw_arrow(ax, cx + bw / 2, y + bh + gap, cx + bw / 2, y + bh)

    ax.text(cx - 0.15, y - 0.5, "Trans-\nformer", fontsize=8, color="#E65100",
            fontweight="bold", ha="center", va="center", rotation=90)

    for i, (label, color, tc) in enumerate(transformer_layers):
        draw_block(ax, cx, y, bw, bh, label, color, fontsize=8.5, text_color=tc)
        if i < len(transformer_layers) - 1:
            draw_arrow(ax, cx + bw / 2, y, cx + bw / 2, y - gap)
        y -= (bh + gap)


def main():
    output_dir = Path("outputs/thesis_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1.5, 8.0)
    ax.axis("off")

    ax.text(4.0, 7.7, "Architectures des mod\u00e8les pour la classification de spectrogrammes RF",
            ha="center", fontsize=16, fontweight="bold", color="#1a1a1a")

    for sep_x in [2.35, 5.35]:
        ax.plot([sep_x, sep_x], [-1.2, 7.1], color="#E0E0E0", linewidth=1.5,
                linestyle="--", zorder=0)

    draw_model_a(ax, -0.3, 0.0)
    draw_model_b(ax, 2.7, 0.0)
    draw_model_c(ax, 5.7, 0.0)

    # Legend
    legend_y = -1.1
    legend_items = [
        ("#1565C0", "Convolution + BN + ReLU"),
        ("#388E3C", "Bloc r\u00e9siduel"),
        ("#6A1B9A", "CNN Stem (Transformer)"),
        ("#F57C00", "Auto-attention"),
        ("#B71C1C", "T\u00eate de classification"),
    ]
    total_w = len(legend_items) * 1.7
    start_x = 4.0 - total_w / 2
    for i, (color, label) in enumerate(legend_items):
        lx = start_x + i * 1.7
        draw_block(ax, lx, legend_y, 0.25, 0.2, "", color, fontsize=1)
        ax.text(lx + 0.32, legend_y + 0.1, label, fontsize=8.5, va="center",
                color="#333333")

    plt.tight_layout()
    save_path = output_dir / "model_architectures.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Architecture diagram saved: {save_path}")


if __name__ == "__main__":
    main()
