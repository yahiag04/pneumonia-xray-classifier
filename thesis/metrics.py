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


def compute_multiclass_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    class_names: Sequence[str],
) -> dict[str, float | int | list[list[int]]]:
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    y_pred = np.argmax(y_prob, axis=1)
    class_indices = list(range(len(class_names)))

    matrix = [
        [
            int(((y_true == true_index) & (y_pred == pred_index)).sum())
            for pred_index in class_indices
        ]
        for true_index in class_indices
    ]

    recalls = []
    precisions = []
    f1_scores = []
    result: dict[str, float | int | list[list[int]]] = {
        "num_samples": int(len(y_true)),
        "accuracy": _safe_div(int((y_true == y_pred).sum()), len(y_true)),
        "confusion_matrix": matrix,
    }
    for index, class_name in enumerate(class_names):
        tp = int(((y_true == index) & (y_pred == index)).sum())
        fp = int(((y_true != index) & (y_pred == index)).sum())
        fn = int(((y_true == index) & (y_pred != index)).sum())
        support = int((y_true == index).sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _f1(precision, recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        result[f"support_{class_name}"] = support
        result[f"precision_{class_name}"] = precision
        result[f"recall_{class_name}"] = recall
        result[f"f1_{class_name}"] = f1

    result["balanced_accuracy"] = float(np.mean(recalls)) if recalls else 0.0
    result["precision_macro"] = float(np.mean(precisions)) if precisions else 0.0
    result["recall_macro"] = result["balanced_accuracy"]
    result["f1_macro"] = float(np.mean(f1_scores)) if f1_scores else 0.0
    return result
