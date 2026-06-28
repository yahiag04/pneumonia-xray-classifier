from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

from thesis.metrics import compute_binary_metrics


CSV_FIELDS = [
    "model_name",
    "checkpoint",
    "threshold",
    "tn",
    "fp",
    "fn",
    "tp",
    "accuracy",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "precision_normal",
    "precision_pneumonia",
    "f1_normal",
    "f1_pneumonia",
    "roc_auc",
    "pr_auc",
    "support_normal",
    "support_pneumonia",
    "loss",
    "seconds_per_image",
]


def compute_threshold_rows(
    model_name: str,
    checkpoint: str,
    labels: Sequence[int],
    probabilities: Sequence[float],
    thresholds: Sequence[float],
    loss: float,
    seconds_per_image: float,
) -> list[dict]:
    normalized_thresholds = _normalize_thresholds(thresholds)
    rows = []
    for threshold in normalized_thresholds:
        metrics = compute_binary_metrics(labels, probabilities, threshold=threshold)
        rows.append(
            {
                "model_name": model_name,
                "checkpoint": checkpoint,
                **metrics,
                "loss": float(loss),
                "seconds_per_image": float(seconds_per_image),
            }
        )
    return rows


def select_best_rows(rows: Sequence[dict]) -> dict[str, dict]:
    best_by_model: dict[str, dict] = {}
    for row in rows:
        model_name = str(row["model_name"])
        current = best_by_model.get(model_name)
        if current is None or _best_row_key(row) > _best_row_key(current):
            best_by_model[model_name] = dict(row)
    return best_by_model


def select_threshold(
    rows: Sequence[dict],
    metric: str = "balanced_accuracy",
    min_sensitivity: float | None = None,
) -> dict:
    candidates = [dict(row) for row in rows]
    if min_sensitivity is not None:
        candidates = [
            row
            for row in candidates
            if float(row.get("sensitivity", 0.0)) >= float(min_sensitivity)
        ]
    if not candidates:
        raise ValueError("No threshold rows satisfy the selection constraints.")
    if any(metric not in row for row in candidates):
        raise ValueError(f"Metric '{metric}' is not present in every threshold row.")

    return max(
        candidates,
        key=lambda row: (float(row[metric]), -float(row["threshold"])),
    )


def write_sweep_outputs(
    rows: Sequence[dict],
    json_path: str | Path,
    csv_path: str | Path,
    metadata: dict,
) -> None:
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_rows = sorted(
        (dict(row) for row in rows),
        key=lambda row: (str(row["model_name"]), float(row["threshold"])),
    )
    payload = {
        "metadata": metadata,
        "best_by_model": select_best_rows(ordered_rows),
        "rows": ordered_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(ordered_rows)


def _normalize_thresholds(thresholds: Sequence[float]) -> list[float]:
    normalized = sorted({float(threshold) for threshold in thresholds})
    if not normalized:
        raise ValueError("At least one threshold is required.")
    if any(threshold < 0.0 or threshold > 1.0 for threshold in normalized):
        raise ValueError("Thresholds must be within [0, 1].")
    return normalized


def _best_row_key(row: dict) -> tuple[float, float]:
    return float(row["balanced_accuracy"]), -float(row["threshold"])
