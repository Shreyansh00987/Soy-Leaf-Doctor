"""
Dataset loaders and augmentation pipelines for Soy Leaf Doctor.
Includes robust error handling and domain-shift simulating transforms.
"""
import os
from typing import Callable, Optional, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from .model import CLASS_TO_IDX

# Standard ImageNet normalization parameters
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """
    Simulates field-day distribution shifts: varying sunlight angles,
    overcast lighting, camera shake, and phone sensors.
    """
    return T.Compose(
        [
            T.Resize((256, 256)),
            T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> T.Compose:
    """Deterministic evaluation preprocessing."""
    return T.Compose(
        [
            T.Resize((256, 256)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )


class SoyLeafDataset(Dataset):
    """
    Loads soybean leaf images with fail-soft fallback on corrupt files.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[Callable] = None,
        is_test: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform or get_eval_transforms()
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image_id = str(row["image_id"]).replace(".jpg", "").replace(".jpeg", "")

        # Search for file with .jpg, .jpeg, or raw
        candidates = [
            os.path.join(self.image_dir, f"{image_id}.jpg"),
            os.path.join(self.image_dir, f"{image_id}.jpeg"),
            os.path.join(self.image_dir, image_id),
        ]

        img_path = None
        for c in candidates:
            if os.path.exists(c):
                img_path = c
                break

        # Fail-soft: if file missing or corrupt, generate neutral RGB canvas
        if img_path is None:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        tensor = self.transform(image)

        if self.is_test:
            return tensor, -1, image_id

        target_class = row["label"]
        target_idx = CLASS_TO_IDX.get(target_class, 0)
        return tensor, target_idx, image_id
