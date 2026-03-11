"""
Generate a thesis-ready pipeline diagram showing the 5 phases of the
RF Drone Forensics system.

Usage:
    python -m src.evaluation.plot_pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ── Layout constants ──────────────────────────────────────────────
PHASE_X = 1.0          # left edge of phase boxes
PHASE_W = 8.0          # width of every phase box
PHASE_H = 1.35         # height of each phase box
PHASE_GAP = 0.55       # vertical gap between phases (for arrow)
SUB_H = 0.42           # height of sub-step boxes
SUB_PAD = 0.45         # left/right inner padding inside phase box
SUB_BOT = 0.18         # bottom padding for sub-boxes inside phase
SUB_GAP = 0.15         # horizontal gap between sub-boxes
NUM_BADGE_X = 0.32     # x-offset of the circled number from phase left edge


def draw_phase(ax, y, label, color_main, color_dark, color_light, color_text,
               sub_labels, number):
    """Draw one phase box with auto-laid-out sub-boxes inside."""
    # Phase background
    box = mpatches.FancyBboxPatch(
        (PHASE_X, y), PHASE_W, PHASE_H,
        boxstyle="round,pad=0.02",
        facecolor=color_main, edgecolor=color_dark, linewidth=2,
        zorder=2,
    )
    ax.add_patch(box)

    # Phase number badge
    ax.text(PHASE_X + NUM_BADGE_X, y + PHASE_H / 2, str(number),
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=color_light,
            bbox=dict(boxstyle="circle,pad=0.3", facecolor=color_dark,
                      edgecolor="none"),
            zorder=4)

    # Phase title (top area of box)
    ax.text(PHASE_X + PHASE_W / 2, y + PHASE_H - 0.22, label,
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="white", zorder=4)

    # ── Auto-layout sub-boxes ──
    n = len(sub_labels)
    inner_left = PHASE_X + SUB_PAD
    inner_right = PHASE_X + PHASE_W - SUB_PAD + 0.1
    usable_w = inner_right - inner_left
    total_gap = SUB_GAP * (n - 1)
    sw = (usable_w - total_gap) / n  # equal width per sub-box

    arrow_color = color_light
    sy = y + SUB_BOT

    for i, txt in enumerate(sub_labels):
        sx = inner_left + i * (sw + SUB_GAP)
        # Sub-box
        sb = mpatches.FancyBboxPatch(
            (sx, sy), sw, SUB_H,
            boxstyle="round,pad=0.012",
            facecolor=color_light, edgecolor=color_dark,
            linewidth=1.0, zorder=3,
        )
        ax.add_patch(sb)
        ax.text(sx + sw / 2, sy + SUB_H / 2, txt,
                ha="center", va="center", fontsize=8.5,
                color=color_text, fontweight="medium", zorder=4)

        # Arrow between sub-boxes
        if i < n - 1:
            ax.annotate(
                "", xy=(sx + sw + SUB_GAP * 0.15, sy + SUB_H / 2),
                xytext=(sx + sw + SUB_GAP * 0.85, sy + SUB_H / 2),
                arrowprops=dict(arrowstyle="<|-", color=arrow_color, lw=1.8),
                zorder=3,
            )


def draw_down_arrow(ax, y_from, y_to, color):
    """Draw a thick downward arrow between two phases."""
    mid_x = PHASE_X + PHASE_W / 2
    ax.annotate(
        "", xy=(mid_x, y_to + PHASE_H), xytext=(mid_x, y_from),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=2.5,
            connectionstyle="arc3,rad=0",
        ),
        zorder=1,
    )


def main(**kwargs):
    output_dir = Path("outputs/thesis_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_h = 5 * PHASE_H + 4 * PHASE_GAP + 1.5  # phases + gaps + title
    fig, ax = plt.subplots(1, 1, figsize=(16, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    # ── Language toggle ──
    lang = kwargs.get("lang", "en")

    if lang == "fr":
        title = "Pipeline de Forensique RF de Drones"
        subtitle = "Des signaux RF bruts aux rapports forensiques"
        phases = [
            {
                "label": "Pr\u00e9traitement des donn\u00e9es RF",
                "colors": ("#1565C0", "#0D47A1", "#E3F2FD", "#0D47A1"),
                "subs": ["Captures\nRF brutes", "Segmentation\n(131K \u00e9ch.)", "Transform\u00e9e\nSTFT", "Spectrogrammes\nlog-magnitude"],
            },
            {
                "label": "D\u00e9tection et Classification",
                "colors": ("#2E7D32", "#1B5E20", "#E8F5E9", "#1B5E20"),
                "subs": ["D\u00e9tection binaire\n(Drone vs Fond)", "Attribution\nmulticlasse", "SmallRF / ResNet\n/ Transformer"],
            },
            {
                "label": "D\u00e9tection Open-Set",
                "colors": ("#E65100", "#BF360C", "#FFF3E0", "#BF360C"),
                "subs": ["Scoring\nMSP", "Scoring par\n\u00c9nergie", "Distance de\nMahalanobis", "Classe\ninconnue ?"],
            },
            {
                "label": "Explicabilit\u00e9",
                "colors": ("#6A1B9A", "#4A148C", "#F3E5F5", "#4A148C"),
                "subs": ["Cartes Grad-CAM\n(heatmaps)", "Signatures RF\ndiscriminantes", "Visualisation\ndes d\u00e9cisions"],
            },
            {
                "label": "G\u00e9n\u00e9ration du rapport forensique",
                "colors": ("#B71C1C", "#880E0E", "#FFEBEE", "#880E0E"),
                "subs": ["Chronologie\ndes \u00e9v\u00e9nements", "Scores de\nconfiance", "Rapport\nJSON", "Sorties\nvisuelles"],
            },
        ]
    else:
        title = "RF Drone Forensics Pipeline"
        subtitle = "From Raw RF Signals to Forensic Reports"
        phases = [
            {
                "label": "RF Data Preprocessing",
                "colors": ("#1565C0", "#0D47A1", "#E3F2FD", "#0D47A1"),
                "subs": ["Raw RF\nCaptures", "Segmentation\n(131K window)", "STFT\nTransform", "Log-Mag\nSpectrograms"],
            },
            {
                "label": "Detection & Classification",
                "colors": ("#2E7D32", "#1B5E20", "#E8F5E9", "#1B5E20"),
                "subs": ["Binary Detection\n(Drone vs BG)", "Multi-class\nAttribution", "SmallRF / ResNet\n/ Transformer"],
            },
            {
                "label": "Open-Set Detection",
                "colors": ("#E65100", "#BF360C", "#FFF3E0", "#BF360C"),
                "subs": ["MSP\nScoring", "Energy\nScoring", "Mahalanobis\nDistance", "Unknown\nClass?"],
            },
            {
                "label": "Explainability",
                "colors": ("#6A1B9A", "#4A148C", "#F3E5F5", "#4A148C"),
                "subs": ["Grad-CAM\nHeatmaps", "RF Signature\nHighlighting", "Decision\nVisualization"],
            },
            {
                "label": "Forensic Report Generation",
                "colors": ("#B71C1C", "#880E0E", "#FFEBEE", "#880E0E"),
                "subs": ["Event\nTimeline", "Confidence\nScores", "JSON\nReport", "Visual\nOutputs"],
            },
        ]

    # Title
    ax.text(PHASE_X + PHASE_W / 2, total_h - 0.3, title,
            ha="center", va="center", fontsize=20, fontweight="bold",
            color="#1a1a1a")
    ax.text(PHASE_X + PHASE_W / 2, total_h - 0.7, subtitle,
            ha="center", va="center", fontsize=12, color="#666666",
            style="italic")

    # Phase Y positions (top to bottom)
    y_start = total_h - 1.2 - PHASE_H
    ys = [y_start - i * (PHASE_H + PHASE_GAP) for i in range(5)]

    for i, (y, phase) in enumerate(zip(ys, phases)):
        c_main, c_dark, c_light, c_text = phase["colors"]
        draw_phase(ax, y, phase["label"], c_main, c_dark, c_light, c_text,
                   phase["subs"], i + 1)

        # Arrow to next phase
        if i < 4:
            draw_down_arrow(ax, y, ys[i + 1], c_dark)

    plt.tight_layout()
    suffix = f"_{lang}" if lang != "en" else ""
    save_path = output_dir / f"pipeline_diagram{suffix}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Pipeline diagram saved: {save_path}")


if __name__ == "__main__":
    main(lang="en")
    main(lang="fr")
