#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.model_complexity import build_complexity_row, profile_model
from thesis.model_registry import build_model, expected_channels


DEFAULT_MODELS = [
    "pneumonia_net",
    "resnet18",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "densenet121",
]

PERFORMANCE_COLUMNS = [
    "rsna_balanced_accuracy",
    "chittagong_balanced_accuracy",
    "kermany_balanced_accuracy",
    "rsna_roc_auc",
    "chittagong_roc_auc",
    "kermany_roc_auc",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile model parameter counts and inference GMAC for thesis comparison models."
    )
    parser.add_argument("--model", action="append", choices=DEFAULT_MODELS)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", default="outputs/model_complexity")
    parser.add_argument(
        "--performance-csv",
        default="outputs/evaluations/rsna_adult/adult_branch_summary_all_models.csv",
        help="Optional adult-branch summary CSV to join balanced accuracy and ROC-AUC metrics.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    performance = load_performance(Path(args.performance_csv))

    rows = []
    for model_name in args.model or DEFAULT_MODELS:
        input_shape = (expected_channels(model_name), args.image_size, args.image_size)
        model = build_model(model_name, pretrained=False)
        model_performance = performance.get(model_name)
        rows.append(
            build_complexity_row(
                model_name,
                profile_model(model, input_shape=input_shape),
                input_shape=input_shape,
                performance=model_performance,
            )
        )

    write_csv(rows, output_dir / "model_complexity.csv")
    (output_dir / "model_complexity.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


def load_performance(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        output: dict[str, dict[str, float]] = {}
        for row in reader:
            model_name = row["model_name"]
            output[model_name] = {
                column: float(row[column])
                for column in PERFORMANCE_COLUMNS
                if column in row and row[column] != ""
            }
        return output


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "model_name",
        "input_shape",
        "parameters",
        "trainable_parameters",
        "macs",
        "gmac",
        "rsna_balanced_accuracy",
        "rsna_balanced_accuracy_per_gmac",
        "chittagong_balanced_accuracy",
        "chittagong_balanced_accuracy_per_gmac",
        "kermany_balanced_accuracy",
        "kermany_balanced_accuracy_per_gmac",
        "rsna_roc_auc",
        "chittagong_roc_auc",
        "kermany_roc_auc",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
