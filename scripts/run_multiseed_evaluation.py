#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.model_complexity import profile_model
from thesis.model_registry import build_model, expected_channels
from thesis.reproducibility import set_global_seed
from thesis.train import TrainConfig, evaluate_checkpoint, train_model


DEFAULT_MODELS = [
    "pneumonia_net",
    "resnet18",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "densenet121",
]
DEFAULT_SEEDS = [42]
METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "roc_auc",
    "pr_auc",
    "f1_normal",
    "f1_pneumonia",
    "precision_normal",
    "precision_pneumonia",
    "tn",
    "fp",
    "fn",
    "tp",
    "loss",
    "seconds_per_image",
]
PER_SEED_FIELDNAMES = [
    "seed",
    "model_name",
    "dataset",
    "checkpoint",
    "params",
    "gmac",
    "best_epoch",
    "epochs_run",
    "best_val_loss",
    *METRIC_KEYS,
]
AGGREGATE_FIELDNAMES = [
    "model_name",
    "dataset",
    "metric",
    "n",
    "mean",
    "std",
    "ci95_low",
    "ci95_high",
]


def parse_manifest_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Manifest must be formatted as name=/path/to/manifest.csv")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("Manifest name cannot be empty")
    return name, Path(path)


def parse_nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("seed must be non-negative")
    return parsed


def seed_run_id(seed: int) -> str:
    return f"seed_{seed}"


def build_model_list(models: list[str] | None) -> list[str]:
    return models or list(DEFAULT_MODELS)


def build_seed_list(seeds: list[int] | None) -> list[int]:
    result = seeds or list(DEFAULT_SEEDS)
    if len(set(result)) != len(result):
        raise ValueError("Duplicate seeds would overwrite per-seed outputs")
    return result


def run_multiseed_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    evaluations_root = output_dir / "evaluations"
    runs_root = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = list(args.manifest or [])
    models = build_model_list(args.model)
    seeds = build_seed_list(args.seed)
    per_seed_rows: list[dict[str, Any]] = []
    seed_reports = []

    for seed in seeds:
        seed_report = set_global_seed(seed, deterministic=args.deterministic)
        seed_reports.append(seed_report.__dict__)
        seed_run_root = runs_root / seed_run_id(seed)
        seed_eval_root = evaluations_root / seed_run_id(seed)
        seed_eval_root.mkdir(parents=True, exist_ok=True)

        for model_name in models:
            summary = train_model(
                TrainConfig(
                    data_root=data_root,
                    model_name=model_name,
                    output_dir=seed_run_root,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    patience=args.patience,
                    image_size=args.image_size,
                    val_fraction=args.val_fraction,
                    train_size=args.train_size,
                    num_workers=args.num_workers,
                    seed=seed,
                    device=args.device,
                )
            )
            checkpoint = Path(summary["checkpoint"])
            complexity = model_complexity_row(model_name, args.image_size)
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
                (seed_eval_root / f"{model_name}_{dataset_name}.json").write_text(
                    json.dumps(result, indent=2)
                )
                per_seed_rows.append(
                    build_per_seed_row(
                        seed=seed,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        checkpoint=checkpoint,
                        summary=summary,
                        complexity=complexity,
                        metrics=result,
                    )
                )

    aggregate_rows = aggregate_metric_rows(per_seed_rows, METRIC_KEYS)
    write_csv(output_dir / "per_seed.csv", per_seed_rows, PER_SEED_FIELDNAMES)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows, AGGREGATE_FIELDNAMES)
    metadata = {
        "seeds": seeds,
        "models": models,
        "datasets": ["rsna", *[name for name, _ in manifests]],
        "deterministic": args.deterministic,
        "seed_reports": seed_reports,
        "ci95_method": "normal_approximation_mean_plus_minus_1.96_standard_error",
        "notes": [
            "This script wraps the existing train_model/evaluate_checkpoint functions and does not replace single-seed outputs.",
            "Bitwise reproducibility is not guaranteed across hardware and backend kernels.",
        ],
    }
    (output_dir / "per_seed.json").write_text(
        json.dumps({"metadata": metadata, "rows": per_seed_rows}, indent=2)
    )
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps({"metadata": metadata, "rows": aggregate_rows}, indent=2)
    )
    return {"metadata": metadata, "per_seed": per_seed_rows, "aggregate": aggregate_rows}


def model_complexity_row(model_name: str, image_size: int) -> dict[str, Any]:
    input_shape = (expected_channels(model_name), image_size, image_size)
    model = build_model(model_name, pretrained=False)
    profile = profile_model(model, input_shape=input_shape)
    return {"params": profile.parameters, "gmac": profile.gmac}


def build_per_seed_row(
    seed: int,
    model_name: str,
    dataset_name: str,
    checkpoint: Path,
    summary: dict[str, Any],
    complexity: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "seed": seed,
        "model_name": model_name,
        "dataset": dataset_name,
        "checkpoint": str(checkpoint),
        "params": complexity["params"],
        "gmac": complexity["gmac"],
        "best_epoch": summary.get("best_epoch"),
        "epochs_run": len(summary.get("history", [])),
        "best_val_loss": summary.get("best_val_loss"),
    }
    for metric in METRIC_KEYS:
        row[metric] = metrics.get(metric)
    return row


def aggregate_metric_rows(
    per_seed_rows: Iterable[dict[str, Any]],
    metric_keys: Iterable[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in per_seed_rows:
        for metric in metric_keys:
            value = row.get(metric)
            if value is None:
                continue
            grouped[(str(row["model_name"]), str(row["dataset"]), metric)].append(float(value))

    aggregate_rows = []
    for (model_name, dataset, metric), values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        std = sample_std(values)
        margin = 1.96 * std / math.sqrt(len(values)) if values else 0.0
        aggregate_rows.append(
            {
                "model_name": model_name,
                "dataset": dataset,
                "metric": metric,
                "n": len(values),
                "mean": mean,
                "std": std,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
            }
        )
    return aggregate_rows


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run binary CXR training/evaluation across multiple seeds and aggregate metrics."
    )
    parser.add_argument("--data-root", default="data/rsna_binary_size_matched")
    parser.add_argument("--output-dir", default="outputs/multiseed")
    parser.add_argument("--model", action="append", choices=DEFAULT_MODELS)
    parser.add_argument("--manifest", action="append", type=parse_manifest_arg, default=[])
    parser.add_argument("--seed", action="append", type=parse_nonnegative_int, default=None)
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
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic PyTorch backend flags where supported. This can reduce speed and may still warn.",
    )
    args = parser.parse_args()

    try:
        result = run_multiseed_evaluation(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
