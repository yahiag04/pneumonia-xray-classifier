from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_binary_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    accuracy = _safe_div(tp + tn, len(y_true))
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision_pneumonia = _safe_div(tp, tp + fp)
    precision_normal = _safe_div(tn, tn + fn)
    f1_pneumonia = _f1(precision_pneumonia, sensitivity)
    f1_normal = _f1(precision_normal, specificity)

    roc_auc = None
    pr_auc = None
    if len(set(y_true.tolist())) == 2:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "precision_normal": precision_normal,
        "precision_pneumonia": precision_pneumonia,
        "f1_normal": f1_normal,
        "f1_pneumonia": f1_pneumonia,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "support_normal": int((y_true == 0).sum()),
        "support_pneumonia": int((y_true == 1).sum()),
    }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)

