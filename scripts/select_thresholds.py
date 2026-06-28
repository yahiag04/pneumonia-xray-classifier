#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    compute_threshold_rows,
    select_threshold,
    write_sweep_outputs,
)
from thesis.train import choose_device, collect_predictions, evaluate_checkpoint


DEFAULT_THRESHOLDS = [round(value / 100, 2) for value in range(5, 96, 5)]


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

    if bool(args.val_manifest) == bool(args.val_data_root):
        raise ValueError(
            "Provide exactly one validation source: --val-manifest or --val-data-root."
        )
    if args.test_manifest and args.test_data_root:
        raise ValueError(
            "Provide at most one test source: --test-manifest or --test-data-root."
        )

    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    model, model_name, image_size = _load_model(checkpoint_path, args.model, device)
    criterion = nn.BCEWithLogitsLoss()
    val_dataset = _build_dataset(
        model_name=model_name,
        image_size=image_size,
        manifest=args.val_manifest,
        data_root=args.val_data_root,
        split="val",
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
    payload = {"metadata": metadata, **summary}

    if args.test_manifest or args.test_data_root:
        selected_threshold = float(summary["selected"]["threshold"])
        test_kwargs = {
            "checkpoint_path": checkpoint_path,
            "model_name": model_name,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "threshold": selected_threshold,
            "device": str(device),
        }
        if args.test_manifest:
            test_kwargs["manifest_csv"] = args.test_manifest
        else:
            test_kwargs["data_root"] = args.test_data_root
        payload["test_metrics_at_selected_threshold"] = evaluate_checkpoint(
            **test_kwargs
        )

    write_sweep_outputs(
        summary["rows"],
        json_path=args.output_json,
        csv_path=args.output_csv,
        metadata=metadata,
    )
    json_path = Path(args.output_json)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["selected"], indent=2))
    return 0


def _load_model(
    checkpoint_path: Path,
    model_name: str | None,
    device: torch.device,
) -> tuple[nn.Module, str, int]:
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
    return model, resolved_model_name, image_size


def _build_dataset(
    model_name: str,
    image_size: int,
    manifest: str | None,
    data_root: str | None,
    split: str,
):
    transform = build_transforms(model_name, image_size=image_size, train=False)
    if manifest:
        return ManifestImageDataset(manifest, transform=transform)
    splits = build_internal_splits(data_root, model_name, image_size=image_size)
    if split == "val":
        return splits.val
    if splits.test is None:
        raise ValueError("No test split found for internal evaluation.")
    return splits.test


if __name__ == "__main__":
    raise SystemExit(main())
