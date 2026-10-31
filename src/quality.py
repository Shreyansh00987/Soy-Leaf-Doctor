"""
Pre-inference Optical Quality Gate for Soy Leaf Doctor.
Validates blur, exposure, and image fidelity before passing inputs to deep learning models.
Implemented using pure NumPy and Pillow for maximum portability across headless sandboxes.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from PIL import Image


@dataclass
class QualityAssessment:
    passed: bool
    blur_score: float
    brightness: float
    issue: str
    recommendation: str


def check_image_quality(
    image: Image.Image,
    blur_threshold: float = 65.0,
    min_brightness: float = 0.15,
    max_brightness: float = 0.85,
) -> QualityAssessment:
    """
    Evaluates whether the photograph meets diagnostic clarity standards.
    Returns QualityAssessment with actionable coaching guidance if failed.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to standard analysis size for consistent variance scale
    sample = image.resize((256, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(sample, dtype=np.float32) / 255.0

    # Luminance calculation (Rec. 601 coefficients)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    mean_brightness = float(np.mean(gray))

    # Discrete 3x3 Laplacian kernel approximation for edge sharpness
    # Kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * gray
    )
    blur_score = float(np.var(laplacian) * 10000.0)

    # Rule evaluation
    if mean_brightness < min_brightness:
        return QualityAssessment(
            passed=False,
            blur_score=round(blur_score, 2),
            brightness=round(mean_brightness, 2),
            issue="Too Dark / Underexposed",
            recommendation="Hold leaf in diffused ambient light or avoid heavy shadow.",
        )

    if mean_brightness > max_brightness:
        return QualityAssessment(
            passed=False,
            blur_score=round(blur_score, 2),
            brightness=round(mean_brightness, 2),
            issue="Too Bright / Overexposed",
            recommendation="Avoid direct harsh flash or excessive solar glare reflecting off the leaf.",
        )

    if blur_score < blur_threshold:
        return QualityAssessment(
            passed=False,
            blur_score=round(blur_score, 2),
            brightness=round(mean_brightness, 2),
            issue="Blurry / Out of Focus",
            recommendation="Hold smartphone camera steady approximately 15-20 cm from the diseased leaf surface.",
        )

    return QualityAssessment(
        passed=True,
        blur_score=round(blur_score, 2),
        brightness=round(mean_brightness, 2),
        issue="None",
        recommendation="Image quality meets agricultural diagnostic criteria.",
    )
