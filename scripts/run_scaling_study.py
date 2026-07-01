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

from thesis.model_complexity import profile_model
from thesis.model_registry import build_model, expected_channels
from thesis.train import TrainConfig, evaluate_checkpoint, train_model


PNEUMONIA_WIDTHS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
EFFICIENTNET_VARIANTS = [
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
]
METRICS = ["balanced_accuracy", "sensitivity", "specificity", "roc_auc"]


def parse_manifest_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Manifest must be formatted as name=/path/to/manifest.csv")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("Manifest name cannot be empty")
    return name, Path(path)


def scaling_run_id(spec: dict[str, Any]) -> str:
    if spec["family"] == "pneumonia_net":
        width = str(spec["width"]).replace(".", "_")
        return f"pneumonia_net_width_{width}"
    return str(spec["variant"])


def metric_columns(dataset_names: list[str]) -> list[str]:
    return [f"{dataset}_{metric}" for dataset in dataset_names for metric in METRICS]


def build_scaling_specs(
    pneumonia_widths: list[float] | None = None,
    efficientnet_variants: list[str] | None = None,
) -> list[dict[str, Any]]:
    specs = [
        {
            "family": "pneumonia_net",
            "variant": f"width_{width:g}",
            "model_name": "pneumonia_net",
            "width": width,
        }
        for width in (pneumonia_widths or PNEUMONIA_WIDTHS)
    ]
    specs.extend(
        {
            "family": "efficientnet",
            "variant": variant,
            "model_name": variant,
            "width": 1.0,
        }
        for variant in (efficientnet_variants or EFFICIENTNET_VARIANTS)
    )
    return specs


def run_scaling_study(args: argparse.Namespace) -> list[dict[str, Any]]:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    eval_dir = output_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    manifests = list(args.manifest or [])
    dataset_names = ["rsna"] + [name for name, _ in manifests]
    rows = []

    for spec in build_scaling_specs(args.pneumonia_width, args.efficientnet_variant):
        run_id = scaling_run_id(spec)
        summary = train_model(
            TrainConfig(
                data_root=data_root,
                model_name=spec["model_name"],
                run_name=run_id,
                output_dir=output_dir / "runs",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                image_size=args.image_size,
                val_fraction=args.val_fraction,
                pretrained=not args.no_pretrained,
                freeze_backbone=args.freeze_backbone,
                model_width=spec["width"],
                train_size=args.train_size,
                num_workers=args.num_workers,
                seed=args.seed,
                device=args.device,
            )
        )
        checkpoint = Path(summary["checkpoint"])
        row = build_base_row(spec, image_size=args.image_size)

        evaluations = {
            "rsna": evaluate_checkpoint(
                checkpoint,
                data_root=data_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
            )
        }
        for dataset_name, manifest_csv in manifests:
            evaluations[dataset_name] = evaluate_checkpoint(
                checkpoint,
                manifest_csv=manifest_csv,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
            )
        for dataset_name, result in evaluations.items():
            (eval_dir / f"{run_id}_{dataset_name}.json").write_text(
                json.dumps(result, indent=2)
            )
            for metric in METRICS:
                row[f"{dataset_name}_{metric}"] = result.get(metric)
        rows.append(row)

    csv_path = output_dir / "scaling_study.csv"
    write_scaling_csv(rows, csv_path, dataset_names)
    plot_scaling_curves(rows, dataset_names, output_dir)
    return rows


def build_base_row(spec: dict[str, Any], image_size: int) -> dict[str, Any]:
    input_shape = (expected_channels(spec["model_name"]), image_size, image_size)
    model = build_model(
        spec["model_name"],
        pretrained=False,
        width=spec["width"],
    )
    profile = profile_model(model, input_shape=input_shape)
    return {
        "family": spec["family"],
        "variant": spec["variant"],
        "model_name": spec["model_name"],
        "width": spec["width"] if spec["family"] == "pneumonia_net" else "",
        "params": profile.parameters,
        "gmac": profile.gmac,
    }


def write_scaling_csv(
    rows: list[dict[str, Any]],
    path: Path,
    dataset_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "variant",
        "model_name",
        "width",
        "params",
        "gmac",
        *metric_columns(dataset_names),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_saturation_knee(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    ordered = sorted(points, key=lambda row: row["parameters"])
    for index, point in enumerate(ordered[1:], start=1):
        baseline = _largest_point_at_most_half_params(ordered[:index], point["parameters"])
        if baseline is None:
            continue
        gain = point["balanced_accuracy"] - baseline["balanced_accuracy"]
        if gain < 0.01:
            return point
    return None


def _largest_point_at_most_half_params(
    previous_points: list[dict[str, Any]],
    current_params: float,
) -> dict[str, Any] | None:
    candidates = [
        point for point in previous_points if point["parameters"] <= current_params / 2
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["parameters"])


def plot_scaling_curves(
    rows: list[dict[str, Any]],
    dataset_names: list[str],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    for x_key, filename, x_label, log_x in [
        ("params", "balanced_accuracy_vs_params.png", "Parameters", True),
        ("gmac", "balanced_accuracy_vs_gmac.png", "GMAC", False),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for dataset_name in dataset_names:
            metric_key = f"{dataset_name}_balanced_accuracy"
            for family, marker in [("pneumonia_net", "o"), ("efficientnet", "s")]:
                family_rows = [
                    row for row in rows if row["family"] == family and row.get(metric_key) is not None
                ]
                family_rows.sort(key=lambda row: row[x_key])
                if not family_rows:
                    continue
                ax.plot(
                    [row[x_key] for row in family_rows],
                    [row[metric_key] for row in family_rows],
                    marker=marker,
                    label=f"{dataset_name} - {family}",
                )
                _mark_width_one(ax, family_rows, x_key, metric_key)
                _mark_knee(ax, family_rows, x_key, metric_key)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Balanced accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def _mark_width_one(ax: Any, rows: list[dict[str, Any]], x_key: str, metric_key: str) -> None:
    for row in rows:
        if row["family"] == "pneumonia_net" and row["width"] == 1.0:
            ax.scatter([row[x_key]], [row[metric_key]], marker="*", s=180, color="black", zorder=5)
            ax.annotate("PN width=1.0", (row[x_key], row[metric_key]), fontsize=8)


def _mark_knee(ax: Any, rows: list[dict[str, Any]], x_key: str, metric_key: str) -> None:
    points = [
        {
            "parameters": row["params"],
            "balanced_accuracy": row[metric_key],
            "row": row,
        }
        for row in rows
    ]
    knee = find_saturation_knee(points)
    if knee is None:
        return
    row = knee["row"]
    ax.scatter([row[x_key]], [row[metric_key]], marker="X", s=120, color="red", zorder=6)
    ax.annotate("knee", (row[x_key], row[metric_key]), fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled scaling curves for PneumoniaNet width and EfficientNet B0-B3."
    )
    parser.add_argument("--data-root", default="data/rsna_binary_size_matched")
    parser.add_argument("--output-dir", default="outputs/scaling_study")
    parser.add_argument("--manifest", action="append", type=parse_manifest_arg, default=[])
    parser.add_argument("--pneumonia-width", action="append", type=float, default=None)
    parser.add_argument(
        "--efficientnet-variant",
        action="append",
        choices=EFFICIENTNET_VARIANTS,
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Optional balanced training subset size. Default uses the full current train split.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the classifier head for EfficientNet variants.",
    )
    args = parser.parse_args()

    rows = run_scaling_study(args)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
