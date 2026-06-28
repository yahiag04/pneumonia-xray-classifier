#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "outputs/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_LABELS = {
    "pneumonia_net": "PneumoniaNet",
    "resnet18": "ResNet18",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
    "densenet121": "DenseNet121",
}

MODEL_ORDER = [
    "pneumonia_net",
    "resnet18",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "densenet121",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate thesis plots from saved metric JSON files.")
    parser.add_argument("--runs-dir", default="outputs/runs_fair")
    parser.add_argument("--evaluations-dir", default="outputs/evaluations")
    parser.add_argument("--output-dir", default="outputs/plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    internal = load_internal_metrics(Path(args.runs_dir))
    nih = load_nih_metrics(Path(args.evaluations_dir))

    if not internal:
        print("No internal metrics found.", file=sys.stderr)
        return 1
    if not nih:
        print("No NIH metrics found.", file=sys.stderr)
        return 1

    write_metrics_csv(output_dir / "metrics_for_plots.csv", internal, nih)
    plot_dataset_confusion_matrices(internal, output_dir / "internal" / "confusion_matrices", "Internal test")
    plot_dataset_confusion_matrices(nih, output_dir / "nih" / "confusion_matrices", "NIH 224x224")
    plot_auc_bars(internal, output_dir / "internal" / "roc_auc_internal.png", "ROC-AUC - Internal test")
    plot_auc_bars(nih, output_dir / "nih" / "roc_auc_nih_224.png", "ROC-AUC - NIH 224x224")
    plot_auc_comparison(internal, nih, output_dir / "comparison" / "roc_auc_internal_vs_nih.png")

    print(f"Plots written to {output_dir}")
    return 0


def load_internal_metrics(runs_dir: Path) -> dict[str, dict]:
    metrics = {}
    for model_name in MODEL_ORDER:
        path = runs_dir / model_name / "training_summary.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        test_metrics = payload.get("test_metrics")
        if test_metrics:
            metrics[model_name] = test_metrics
    return metrics


def load_nih_metrics(evaluations_dir: Path) -> dict[str, dict]:
    result = {}
    for path in sorted(evaluations_dir.glob("*_nih_224.json")):
        payload = json.loads(path.read_text())
        model_name = payload.get("model_name")
        if model_name in MODEL_LABELS:
            result[model_name] = payload
    return result


def write_metrics_csv(output_path: Path, internal: dict[str, dict], nih: dict[str, dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "model",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "f1_pneumonia",
        "roc_auc",
        "pr_auc",
        "seconds_per_image",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, metrics_by_model in [("internal", internal), ("nih_224", nih)]:
            for model_name in ordered_models(metrics_by_model):
                row = {"dataset": dataset, "model": model_name}
                row.update({field: metrics_by_model[model_name].get(field) for field in fields if field not in row})
                writer.writerow(row)


def plot_dataset_confusion_matrices(metrics_by_model: dict[str, dict], output_dir: Path, dataset_label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name in ordered_models(metrics_by_model):
        plot_confusion_matrix(
            metrics_by_model[model_name],
            output_dir / f"{model_name}_confusion_matrix.png",
            f"{MODEL_LABELS[model_name]} - {dataset_label}",
        )


def plot_confusion_matrix(metrics: dict, output_path: Path, title: str) -> None:
    matrix = [
        [int(metrics["tn"]), int(metrics["fp"])],
        [int(metrics["fn"]), int(metrics["tp"])],
    ]
    row_totals = [max(sum(row), 1) for row in matrix]

    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=160)
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["Normal", "Pneumonia"])
    ax.set_yticks([0, 1], labels=["Normal", "Pneumonia"])

    threshold = max(max(row) for row in matrix) * 0.55
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            pct = value / row_totals[row_idx] * 100
            color = "white" if value > threshold else "#1f2937"
            ax.text(
                col_idx,
                row_idx,
                f"{value:,}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_auc_bars(metrics_by_model: dict[str, dict], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = ordered_models(metrics_by_model)
    labels = [MODEL_LABELS[model] for model in models]
    values = [float(metrics_by_model[model]["roc_auc"]) for model in models]

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
    bars = ax.bar(labels, values, color=["#2f6f9f", "#3a8f6b", "#c97b32", "#8d5a9e", "#6f7f8f"][: len(labels)])
    ax.set_title(title, pad=12)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_auc_comparison(internal: dict[str, dict], nih: dict[str, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = [model for model in MODEL_ORDER if model in internal and model in nih]
    labels = [MODEL_LABELS[model] for model in models]
    x_positions = list(range(len(models)))
    width = 0.38

    internal_values = [float(internal[model]["roc_auc"]) for model in models]
    nih_values = [float(nih[model]["roc_auc"]) for model in models]

    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=160)
    bars_internal = ax.bar([x - width / 2 for x in x_positions], internal_values, width, label="Internal", color="#2f6f9f")
    bars_nih = ax.bar([x + width / 2 for x in x_positions], nih_values, width, label="NIH 224x224", color="#c97b32")

    ax.set_title("ROC-AUC - Internal vs NIH 224x224", pad=12)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x_positions, labels=labels, rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for bars in [bars_internal, bars_nih]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def ordered_models(metrics_by_model: dict[str, dict]) -> list[str]:
    return [model for model in MODEL_ORDER if model in metrics_by_model]


if __name__ == "__main__":
    raise SystemExit(main())
