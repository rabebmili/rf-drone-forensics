"""
Explainability module: Grad-CAM heatmaps on spectrograms.
Provides visual explanations of which time-frequency regions drive model decisions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """Grad-CAM for CNN models operating on spectrograms.

    Highlights the regions of the spectrogram most important for the prediction.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: nn.Module — the convolutional layer to extract activations from
                          (typically the last conv layer before the classifier)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for a single input.

        Args:
            input_tensor: [1, 1, H, W] input spectrogram tensor
            target_class: class index to explain (None = predicted class)

        Returns:
            heatmap: [H, W] numpy array normalized to [0, 1]
            predicted_class: int
            confidence: float
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backpropagate for target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Compute Grad-CAM
        gradients = self.gradients[0]  # [C, h, w]
        activations = self.activations[0]  # [C, h, w]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input size
        cam_np = cam.cpu().numpy()
        from scipy.ndimage import zoom
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        zoom_h = input_h / cam_np.shape[0]
        zoom_w = input_w / cam_np.shape[1]
        heatmap = zoom(cam_np, (zoom_h, zoom_w), order=1)

        return heatmap, target_class, confidence


def get_target_layer(model, model_name="smallrf"):
    """Get the appropriate target layer for Grad-CAM based on model type."""
    if model_name == "smallrf":
        # Last conv layer in features sequential
        return model.features[8]  # 3rd Conv2d
    elif model_name == "resnet":
        # Last residual block's last conv
        return model.layer3[-1].conv2
    else:
        raise ValueError(f"Grad-CAM target layer not defined for model: {model_name}")


def plot_gradcam(spectrogram, heatmap, predicted_class, confidence,
                 class_names=None, output_path=None, title=None):
    """Plot spectrogram with Grad-CAM overlay.

    Args:
        spectrogram: 2D numpy array [H, W]
        heatmap: 2D numpy array [H, W] in [0, 1]
        predicted_class: int
        confidence: float
        class_names: list of class name strings
        output_path: save path (optional)
        title: custom title (optional)
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(10)]

    class_label = class_names[predicted_class] if predicted_class < len(class_names) else str(predicted_class)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original spectrogram
    axes[0].imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Original Spectrogram")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Frequency")

    # Grad-CAM heatmap
    axes[1].imshow(heatmap, aspect="auto", origin="lower", cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Frequency")

    # Overlay
    axes[2].imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
    axes[2].imshow(heatmap, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
    axes[2].set_title(f"Prediction: {class_label} ({confidence:.2%})")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Frequency")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM saved: {output_path}")
    plt.close()


def generate_gradcam_examples(model, dataset, device, model_name="smallrf",
                               class_names=None, output_dir="outputs/explainability",
                               n_per_class=3):
    """Generate Grad-CAM visualizations for sample spectrograms from each class."""
    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect indices per class
    class_indices = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        c = label.item()
        if c not in class_indices:
            class_indices[c] = []
        if len(class_indices[c]) < n_per_class:
            class_indices[c].append(i)

    for cls, indices in class_indices.items():
        for j, idx in enumerate(indices):
            x, y = dataset[idx]
            x_input = x.unsqueeze(0).to(device)

            heatmap, pred_cls, conf = gradcam.generate(x_input)
            spectrogram = x.squeeze(0).numpy()

            cls_name = class_names[cls] if class_names else f"class_{cls}"
            fname = f"gradcam_{cls_name.replace(' ', '_')}_sample{j}.png"
            plot_gradcam(spectrogram, heatmap, pred_cls, conf,
                         class_names=class_names,
                         output_path=str(out_dir / fname))
