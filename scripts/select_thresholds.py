#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.data import ManifestImageDataset, build_internal_splits, build_transforms
from thesis.model_registry import available_models, build_model
from thesis.threshold_sweep import (
    CSV_FIELDS,
    compute_threshold_rows,
    select_best_rows,
    select_threshold,
)
from thesis.metrics import compute_binary_metrics
from thesis.train import choose_device, collect_predictions


DEFAULT_THRESHOLDS = [round(value / 100, 2) for value in range(5, 96, 5)]
ALLOWED_SELECTION_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "f1_pneumonia",
    "sensitivity",
    "specificity",
}


def summarize_threshold_selection(
    model_name: str,
    checkpoint: str,
    labels: Sequence[int],
    probabilities: Sequence[float],
    thresholds: Sequence[float],
    loss: float,
    seconds_per_image: float,
    metric: str,
    min_sensitivity: float | None,
) -> dict:
    rows = compute_threshold_rows(
        model_name=model_name,
        checkpoint=checkpoint,
        labels=labels,
        probabilities=probabilities,
        thresholds=thresholds,
        loss=loss,
        seconds_per_image=seconds_per_image,
    )
    selected = select_threshold(
        rows,
        metric=metric,
        min_sensitivity=min_sensitivity,
    )
    return {"selected": selected, "rows": rows}


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if bool(args.val_manifest) == bool(args.val_data_root):
        parser.error(
            "provide exactly one validation source: --val-manifest or --val-data-root"
        )
    if args.test_manifest and args.test_data_root:
        parser.error(
            "provide at most one test source: --test-manifest or --test-data-root"
        )
    if not args.thresholds:
        parser.error("at least one threshold is required")
    if any(threshold < 0.0 or threshold > 1.0 for threshold in args.thresholds):
        parser.error("thresholds must be within [0, 1]")
    if args.min_sensitivity is not None and not 0.0 <= args.min_sensitivity <= 1.0:
        parser.error("--min-sensitivity must be within [0, 1]")
    if args.metric not in ALLOWED_SELECTION_METRICS:
        parser.error(
            "--metric must be one of: "
            + ", ".join(sorted(ALLOWED_SELECTION_METRICS))
        )
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")
    if args.num_workers < 0:
        parser.error("--num-workers must be greater than or equal to 0")


def build_selection_payload(
    metadata: dict,
    summary: dict,
    test_metrics: dict | None = None,
) -> dict:
    rows = [dict(row) for row in summary["rows"]]
    payload = {
        "metadata": dict(metadata),
        "best_by_model": select_best_rows(rows),
        "selected": dict(summary["selected"]),
        "rows": rows,
    }
    if test_metrics is not None:
        payload["test_metrics_at_selected_threshold"] = dict(test_metrics)
    return payload


def write_selection_outputs(
    summary: dict,
    metadata: dict,
    output_json: str | Path,
    output_csv: str | Path,
    test_metrics: dict | None = None,
) -> dict:
    payload = build_selection_payload(metadata, summary, test_metrics=test_metrics)

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(
        (dict(row) for row in summary["rows"]),
        key=lambda row: (str(row["model_name"]), float(row["threshold"])),
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(ordered_rows)
    return payload


def evaluate_selected_threshold(
    model: nn.Module,
    dataset,
    criterion: nn.Module,
    device: torch.device,
    checkpoint: str,
    model_name: str,
    threshold: float,
    batch_size: int,
    num_workers: int,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    predictions = collect_predictions(model, loader, criterion, device)
    metrics = compute_binary_metrics(
        predictions["labels"],
        predictions["probabilities"],
        threshold=threshold,
    )
    num_samples = int(predictions["num_samples"])
    metrics["loss"] = predictions["loss"]
    metrics["seconds_per_image"] = float(predictions["elapsed_seconds"]) / max(
        num_samples,
        1,
    )
    metrics["checkpoint"] = checkpoint
    metrics["model_name"] = model_name
    metrics["num_samples"] = num_samples
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select validation thresholds and evaluate selected test thresholds."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model",
        choices=available_models(),
        help="Required only for legacy checkpoints without metadata.",
    )
    parser.add_argument("--val-manifest")
    parser.add_argument("--val-data-root")
    parser.add_argument("--test-manifest")
    parser.add_argument("--test-data-root")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--metric", default="balanced_accuracy")
    parser.add_argument("--min-sensitivity", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    validate_args(parser, args)

    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    model, model_name, image_size, checkpoint_config = _load_model(
        checkpoint_path,
        args.model,
        device,
    )
    criterion = nn.BCEWithLogitsLoss()
    val_dataset = _build_dataset(
        model_name=model_name,
        image_size=image_size,
        manifest=args.val_manifest,
        data_root=args.val_data_root,
        split="val",
        checkpoint_config=checkpoint_config,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    predictions = collect_predictions(model, val_loader, criterion, device)
    seconds_per_image = float(predictions["elapsed_seconds"]) / max(
        int(predictions["num_samples"]),
        1,
    )
    summary = summarize_threshold_selection(
        model_name=model_name,
        checkpoint=str(checkpoint_path),
        labels=predictions["labels"],
        probabilities=predictions["probabilities"],
        thresholds=args.thresholds,
        loss=predictions["loss"],
        seconds_per_image=seconds_per_image,
        metric=args.metric,
        min_sensitivity=args.min_sensitivity,
    )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "validation_source": args.val_manifest or args.val_data_root,
        "test_source": args.test_manifest or args.test_data_root,
        "thresholds": sorted({float(value) for value in args.thresholds}),
        "selection_metric": args.metric,
        "min_sensitivity": args.min_sensitivity,
        "device": str(device),
    }

    test_metrics = None
    if args.test_manifest or args.test_data_root:
        selected_threshold = float(summary["selected"]["threshold"])
        test_dataset = _build_dataset(
            model_name=model_name,
            image_size=image_size,
            manifest=args.test_manifest,
            data_root=args.test_data_root,
            split="test",
            checkpoint_config=checkpoint_config,
        )
        test_metrics = evaluate_selected_threshold(
            model=model,
            dataset=test_dataset,
            criterion=criterion,
            device=device,
            checkpoint=str(checkpoint_path),
            model_name=model_name,
            threshold=selected_threshold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    payload = write_selection_outputs(
        summary,
        metadata=metadata,
        output_json=args.output_json,
        output_csv=args.output_csv,
        test_metrics=test_metrics,
    )
    print(json.dumps(payload["selected"], indent=2))
    return 0


def _load_model(
    checkpoint_path: Path,
    model_name: str | None,
    device: torch.device,
) -> tuple[nn.Module, str, int, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_meta = (
        checkpoint if isinstance(checkpoint, dict) and "model_state" in checkpoint else {}
    )
    state_dict = checkpoint_meta.get("model_state", checkpoint)
    resolved_model_name = model_name or checkpoint_meta.get("model_name")
    if resolved_model_name is None:
        raise ValueError(
            "Model name is required for legacy checkpoints without metadata."
        )
    image_size = int(checkpoint_meta.get("image_size", 224))
    model = build_model(resolved_model_name, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    checkpoint_config = checkpoint_meta.get("config", {})
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = {}
    return model, resolved_model_name, image_size, checkpoint_config


def _build_dataset(
    model_name: str,
    image_size: int,
    manifest: str | None,
    data_root: str | None,
    split: str,
    checkpoint_config: dict | None = None,
):
    if manifest:
        transform = build_transforms(model_name, image_size=image_size, train=False)
        return ManifestImageDataset(manifest, transform=transform)
    checkpoint_config = checkpoint_config or {}
    splits = build_internal_splits(
        data_root,
        model_name,
        image_size=image_size,
        val_fraction=float(checkpoint_config.get("val_fraction", 0.1)),
        seed=int(checkpoint_config.get("seed", 42)),
    )
    if split == "val":
        return splits.val
    if splits.test is None:
        raise ValueError("No test split found for internal evaluation.")
    return splits.test


if __name__ == "__main__":
    raise SystemExit(main())
