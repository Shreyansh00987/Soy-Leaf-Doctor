"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module.
Provides visual explainability overlays highlighting foliar lesions for farmers and judges.
"""
import io
import base64
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Computes class activation heatmaps for MobileNetV3-Large convolutional features.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.model.eval()

        # MobileNetV3-Large last convolutional feature layer is model.backbone.features[-1]
        if target_layer is None:
            self.target_layer = self.model.backbone.features[-1]
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self, input_tensor: torch.Tensor, target_class_idx: int = None
    ) -> np.ndarray:
        """
        Generates a 2D normalized heatmap [0, 1] for input_tensor [1, 3, H, W].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(output, dim=-1))

        score = output[0, target_class_idx]
        score.backward()

        # Global average pooling of gradients: [1, C, 1, 1]
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3], keepdim=True)

        # Weighted combination of forward activation maps
        weighted_activations = self.activations * pooled_gradients
        cam = torch.sum(weighted_activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Clean up hooks and memory immediately to prevent OOM
        self.gradients = None
        self.activations = None
        self.model.zero_grad(set_to_none=True)

        # Upsample to match input tensor resolution
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def apply_heatmap_to_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    max_dim: int = 512,
) -> Image.Image:
    """
    Overlays normalized heatmap [0, 1] onto original PIL image using a custom colormap.
    Safely downscales large inputs to max_dim to maintain low memory profile (<10 MB).
    """
    if original_image.mode != "RGB":
        original_image = original_image.convert("RGB")

    # Downscale large screenshots/photos to max_dim to eliminate OOM spikes
    w, h = original_image.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        target_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        target_image = original_image.resize(target_size, Image.Resampling.BILINEAR)
    else:
        target_image = original_image

    width, height = target_image.size
    heatmap_img = Image.fromarray((heatmap * 255.0).astype(np.uint8)).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    h_arr = np.array(heatmap_img, dtype=np.float32) / 255.0

    # Custom Jet-like color mapping: Blue -> Cyan -> Yellow -> Red
    r = np.clip(1.5 - np.abs(2.0 * h_arr - 1.0) * 2.0, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * h_arr - 0.5) * 2.0, 0.0, 1.0)
    b = np.clip(1.5 - np.abs(2.0 * h_arr - 0.0) * 2.0, 0.0, 1.0)
    color_cam = np.stack([r, g, b], axis=-1) * 255.0

    orig_arr = np.array(target_image, dtype=np.float32)
    blended = (1.0 - alpha) * orig_arr + alpha * color_cam
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def image_to_base64(image: Image.Image) -> str:
    """Encodes PIL image to Base64 JPEG string for ultra-fast, memory-safe web rendering."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85, optimize=True)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"
