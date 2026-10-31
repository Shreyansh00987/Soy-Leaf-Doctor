"""
Multi-backbone architecture registry for Soy Leaf Doctor (HARITVA).
Supports MobileNetV3-Large, EfficientNet-B0, and RegNetY-400MF.
All backbones are lightweight (<= 25 MB) and support 4-view dihedral TTA.
"""
from typing import List, Optional, Tuple
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


class GenericSoyClassifier(nn.Module):
    """
    Standardized wrapper for torchvision backbones with custom 2-layer classifier head.
    """

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_large",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        if backbone_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_large(weights=weights)
            in_features = base.classifier[0].in_features
            base.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(512, num_classes),
            )
            self.model = base
            self.feature_extractor = base.features

        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, 512),
                nn.SiLU(inplace=True),
                nn.Linear(512, num_classes),
            )
            self.model = base
            self.feature_extractor = base.features

        elif backbone_name == "regnet_y_400mf":
            weights = models.RegNet_Y_400MF_Weights.DEFAULT if pretrained else None
            base = models.regnet_y_400mf(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(512, num_classes),
            )
            self.model = base
            self.feature_extractor = base.trunk_output

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict_probs(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = self.forward(x)
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        return F.softmax(logits, dim=-1)

    def predict_tta(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """4-View Dihedral Test-Time Augmentation."""
        v1 = x
        v2 = torch.flip(x, dims=[-1])
        v3 = torch.flip(x, dims=[-2])
        v4 = torch.flip(x, dims=[-2, -1])

        p1 = self.predict_probs(v1, temperature)
        p2 = self.predict_probs(v2, temperature)
        p3 = self.predict_probs(v3, temperature)
        p4 = self.predict_probs(v4, temperature)

        return (p1 + p2 + p3 + p4) / 4.0


def save_fp16_weights(model: nn.Module, save_path: str):
    """Saves weights in FP16 to halve disk size while retaining full inference accuracy."""
    state_dict = model.state_dict()
    fp16_state_dict = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
    torch.save(fp16_state_dict, save_path)


def load_weights_safe(model: nn.Module, checkpoint_path: str, device: torch.device = torch.device("cpu")):
    """Loads state dict safely, converting FP16 weights back to model precision."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()
    clean_dict = {}
    for k, v in state_dict.items():
        if k in model_state and model_state[k].dtype != v.dtype:
            clean_dict[k] = v.to(model_state[k].dtype)
        else:
            clean_dict[k] = v
    model.load_state_dict(clean_dict)
    return model
