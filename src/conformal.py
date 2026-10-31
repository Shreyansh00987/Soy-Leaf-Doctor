"""
Inductive Split-Conformal Prediction Module.
Provides mathematically guaranteed coverage sets (|C| = 1, 2, >= 3) to triage cases.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class ConformalTriageResult:
    label: str
    prediction_set: List[str]
    set_size: int
    triage_state: str  # 'CONFIDENT', 'AMBIGUOUS', 'ESCALATE'
    confidence_level: str  # 'HIGH', 'MEDIUM', 'LOW'
    probabilities: Dict[str, float]


class ConformalCalibrator:
    """
    Computes and applies conformal thresholds on calibrated prediction probabilities.
    Guarantees that the true label is in C(X) with probability >= 1 - alpha (e.g. 90%).
    """

    def __init__(self, alpha: float = 0.10, q_hat: Optional[float] = None):
        self.alpha = alpha
        self.q_hat = q_hat if q_hat is not None else 0.85

    def calibrate(self, probs: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculates q_hat on holdout calibration split.
        probs: [N, num_classes] array of probabilities
        targets: [N] array of integer class indices
        """
        n = len(targets)
        # Non-conformity score: 1 - prob(true_class)
        true_probs = probs[np.arange(n), targets]
        scores = 1.0 - true_probs

        # Quantile index at (n + 1)(1 - alpha) / n
        level = np.clip(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 0.0, 1.0)
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        return self.q_hat

    def predict(
        self, probs: np.ndarray, class_names: List[str]
    ) -> ConformalTriageResult:
        """
        Produces prediction set for a single 1D probability distribution.
        """
        argmax_idx = int(np.argmax(probs))
        top_label = class_names[argmax_idx]

        # Conformal set: all classes where p_c >= 1 - q_hat
        threshold = max(0.0, 1.0 - self.q_hat)
        pred_set = [
            class_names[i] for i in range(len(class_names)) if probs[i] >= threshold
        ]

        # If set is empty, include argmax
        if not pred_set:
            pred_set = [top_label]

        # Sort prediction set by probability descending
        pred_set.sort(key=lambda c: probs[class_names.index(c)], reverse=True)
        set_size = len(pred_set)

        if set_size == 1:
            state = "CONFIDENT"
            confidence = "HIGH"
        elif set_size == 2:
            state = "AMBIGUOUS"
            confidence = "MEDIUM"
        else:
            state = "ESCALATE"
            confidence = "LOW"

        prob_dict = {
            class_names[i]: float(round(probs[i], 4)) for i in range(len(class_names))
        }

        return ConformalTriageResult(
            label=top_label,
            prediction_set=pred_set,
            set_size=set_size,
            triage_state=state,
            confidence_level=confidence,
            probabilities=prob_dict,
        )
