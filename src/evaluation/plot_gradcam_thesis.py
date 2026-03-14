"""
Generate a thesis-ready Grad-CAM figure showing activation maps
for each drone class from the ResNet model.

Usage:
    python -m src.evaluation.plot_gradcam_thesis
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


CLASSES = ["Background", "AR_Drone", "Bepop_Drone", "Phantom_Drone"]
CLASS_LABELS_FR = ["Fond (pas de drone)", "AR Drone", "Bebop Drone", "Phantom Drone"]
MODEL = "resnet_multiclass"


def main():
    output_dir = Path("outputs/thesis_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))

    col_titles = ["Spectrogramme original", "Carte Grad-CAM", "Superposition"]

    for row, (cls, label_fr) in enumerate(zip(CLASSES, CLASS_LABELS_FR)):
        img_path = Path(f"outputs/{MODEL}/explainability/gradcam_{cls}_sample0.png")
        if not img_path.exists():
            print(f"  Missing: {img_path}")
            continue

        # Load the combined image and split into 3 panels
        full_img = mpimg.imread(str(img_path))
        h, w = full_img.shape[:2]
        third = w // 3

        panels = [
            full_img[:, :third],
            full_img[:, third:2*third],
            full_img[:, 2*third:],
        ]

        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.axis("off")

            if row == 0:
                ax.set_title(col_titles[col], fontsize=14, fontweight="bold", pad=10)

        # Row label on the left
        axes[row, 0].text(-0.08, 0.5, label_fr, transform=axes[row, 0].transAxes,
                          fontsize=13, fontweight="bold", va="center", ha="right",
                          rotation=90)

    fig.suptitle("Cartes d'activation Grad-CAM appliqu\u00e9es aux spectrogrammes RF\n"
                 "(Mod\u00e8le RFResNet \u2014 classification multiclasse)",
                 fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    save_path = output_dir / "gradcam_thesis_figure.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Figure saved: {save_path}")


if __name__ == "__main__":
    main()
