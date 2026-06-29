from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def accuracy(y_true, y_score, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    return float((y_true == y_pred).mean())


def balanced_accuracy(y_true, y_score, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    sensitivity = _safe_mean(y_pred[y_true == 1] == 1)
    specificity = _safe_mean(y_pred[y_true == 0] == 0)
    return float((sensitivity + specificity) / 2)


def roc_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("ROC-AUC requires both classes")

    ranks = _midranks(np.concatenate([positives, negatives]))
    positive_ranks = ranks[: len(positives)]
    return float((positive_ranks.sum() - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives)))


def paired_correctness(y_true, pred_a, pred_b) -> dict[str, int]:
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    return {
        "both_correct": int((correct_a & correct_b).sum()),
        "a_correct_b_wrong": int((correct_a & ~correct_b).sum()),
        "a_wrong_b_correct": int((~correct_a & correct_b).sum()),
        "both_wrong": int((~correct_a & ~correct_b).sum()),
    }


def mcnemar_exact(a_correct_b_wrong: int, a_wrong_b_correct: int) -> dict[str, float | int]:
    b = int(a_correct_b_wrong)
    c = int(a_wrong_b_correct)
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(b, c)
        p_value = min(1.0, 2.0 * sum(math.comb(discordant, i) * 0.5**discordant for i in range(tail + 1)))
    return {
        "a_correct_b_wrong": b,
        "a_wrong_b_correct": c,
        "discordant": discordant,
        "p_value": float(p_value),
    }


def delong_roc_test(y_true, scores_a, scores_b) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    order = np.argsort(-y_true)
    m = int(y_true.sum())
    n = len(y_true) - m
    if m == 0 or n == 0:
        raise ValueError("DeLong test requires both classes")

    predictions = np.vstack([scores_a, scores_b])[:, order]
    aucs, covariance = _fast_delong(predictions, m)
    diff = float(aucs[0] - aucs[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    if variance <= 0:
        p_value = 1.0 if diff == 0 else 0.0
        z_score = math.inf if diff > 0 else -math.inf if diff < 0 else 0.0
    else:
        z_score = diff / math.sqrt(variance)
        p_value = 2.0 * _normal_sf(abs(z_score))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_difference": diff,
        "z_score": float(z_score),
        "p_value": float(max(0.0, min(1.0, p_value))),
    }


def bootstrap_metric_ci(
    y_true,
    y_score,
    metric_fn: Callable,
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    negative_indices = np.flatnonzero(y_true == 0)
    positive_indices = np.flatnonzero(y_true == 1)
    if len(negative_indices) == 0 or len(positive_indices) == 0:
        raise ValueError("Bootstrap CI requires both classes")
    values = []
    for _ in range(n_bootstraps):
        indices = np.concatenate(
            [
                rng.choice(negative_indices, size=len(negative_indices), replace=True),
                rng.choice(positive_indices, size=len(positive_indices), replace=True),
            ]
        )
        values.append(float(metric_fn(y_true[indices], y_score[indices])))
    if not values:
        raise ValueError("No valid bootstrap samples")
    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return {
        "point": float(metric_fn(y_true, y_score)),
        "ci_low": lower,
        "ci_high": upper,
        "n_bootstraps": len(values),
    }


def bootstrap_mean_difference_ci(
    before,
    after,
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.shape != after.shape:
        raise ValueError("before and after must have the same shape")
    if before.ndim != 1 or len(before) == 0:
        raise ValueError("before and after must be non-empty one-dimensional arrays")
    differences = after - before
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(differences), size=len(differences))
        values.append(float(differences[indices].mean()))
    return {
        "point": float(differences.mean()),
        "ci_low": float(np.percentile(values, 100 * alpha / 2)),
        "ci_high": float(np.percentile(values, 100 * (1 - alpha / 2))),
        "n_bootstraps": len(values),
    }


def holm_bonferroni(p_values) -> list[float]:
    p_values = [float(value) for value in p_values]
    adjusted = [0.0] * len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    running_max = 0.0
    total = len(indexed)
    for rank, (original_index, p_value) in enumerate(indexed):
        corrected = min(1.0, (total - rank) * p_value)
        running_max = max(running_max, corrected)
        adjusted[original_index] = running_max
    return adjusted


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for classifier_index in range(k):
        tx[classifier_index] = _midranks(positive_examples[classifier_index])
        ty[classifier_index] = _midranks(negative_examples[classifier_index])
        tz[classifier_index] = _midranks(predictions_sorted_transposed[classifier_index])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(covariance)


def _midranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1
        index = end
    return ranks


def _safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))
