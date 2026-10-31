"""
Neural network model architecture and inference wrappers for Soy Leaf Doctor.
"""
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

CLASSES: List[str] = [
    "Caterpillar_Semilooper_Pest",
    "Frog_Leaf_Eye",
    "Healthy",
    "Mosaic",
    "Rust",
    "Spectoria_Brown_Spot",
]

NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}


class SoyLeafClassifier(nn.Module):
    """
    MobileNetV3-Large backbone adapted for 6-class soybean foliar disease classification.
    Lightweight footprint (~21 MB), fast CPU inference (<50 ms), suitable for edge deployment.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)

        # Replace classification head
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.25, inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits [B, num_classes]"""
        return self.backbone(x)

    def predict_probs(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Returns calibrated probabilities on the probability simplex [B, num_classes].
        """
        logits = self.forward(x)
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        return F.softmax(logits, dim=-1)

    def predict_tta(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        4-View Dihedral Test-Time Augmentation (TTA):
          1. Original
          2. Horizontal flip
          3. Vertical flip
          4. Horizontal + Vertical flip (180 deg rotation)
        Averages softmax probabilities across all 4 dihedral views.
        """
        v1 = x
        v2 = torch.flip(x, dims=[-1])
        v3 = torch.flip(x, dims=[-2])
        v4 = torch.flip(x, dims=[-2, -1])

        p1 = self.predict_probs(v1, temperature)
        p2 = self.predict_probs(v2, temperature)
        p3 = self.predict_probs(v3, temperature)
        p4 = self.predict_probs(v4, temperature)

        return (p1 + p2 + p3 + p4) / 4.0


def build_model(pretrained: bool = False, checkpoint_path: str = None) -> SoyLeafClassifier:
    """Instantiate and optionally load trained weights."""
    model = SoyLeafClassifier(num_classes=NUM_CLASSES, pretrained=pretrained)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    model.eval()
    return model
