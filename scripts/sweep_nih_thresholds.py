#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.data import ManifestImageDataset, build_transforms
from thesis.model_registry import available_models, build_model
from thesis.threshold_sweep import compute_threshold_rows, write_sweep_outputs
from thesis.train import choose_device, collect_predictions


DEFAULT_MODELS = [
    "pneumonia_net",
    "resnet18",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "densenet121",
]
DEFAULT_THRESHOLDS = [0.50, 0.60, 0.65, 0.70]


def build_model_rows(
    model_name: str,
    checkpoint: str,
    thresholds: Sequence[float],
    prediction_collector: Callable[[], dict],
) -> list[dict]:
    predictions = prediction_collector()
    num_samples = int(predictions["num_samples"])
    seconds_per_image = float(predictions["elapsed_seconds"]) / max(num_samples, 1)
    return compute_threshold_rows(
        model_name=model_name,
        checkpoint=checkpoint,
        labels=predictions["labels"],
        probabilities=predictions["probabilities"],
        thresholds=thresholds,
        loss=predictions["loss"],
        seconds_per_image=seconds_per_image,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned checkpoints once and sweep NIH decision thresholds."
    )
    parser.add_argument(
        "--manifest",
        default="outputs/nih/nih_224_binary_manifest.csv",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/runs_third_finetune",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/evaluations/nih_threshold_sweep_after_third_ft.json",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/evaluations/nih_threshold_sweep_after_third_ft.csv",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=available_models(),
        default=DEFAULT_MODELS,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = choose_device(args.device)
    rows = []
    completed_models = []
    manifest = Path(args.manifest)
    checkpoint_dir = Path(args.checkpoint_dir)

    for model_name in args.models:
        checkpoint_path = checkpoint_dir / model_name / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for {model_name}: {checkpoint_path}"
            )

        print(
            f"evaluating model={model_name} checkpoint={checkpoint_path} device={device}",
            flush=True,
        )
        model, image_size = _load_model(checkpoint_path, model_name, device)
        transform = build_transforms(model_name, image_size=image_size, train=False)
        dataset = ManifestImageDataset(manifest, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        criterion = nn.BCEWithLogitsLoss()

        model_rows = build_model_rows(
            model_name=model_name,
            checkpoint=str(checkpoint_path),
            thresholds=args.thresholds,
            prediction_collector=lambda: collect_predictions(
                model,
                loader,
                criterion,
                device,
            ),
        )
        rows.extend(model_rows)
        completed_models.append(model_name)
        metadata = {
            "manifest": str(manifest),
            "checkpoint_dir": str(checkpoint_dir),
            "thresholds": sorted({float(value) for value in args.thresholds}),
            "requested_models": list(args.models),
            "completed_models": completed_models,
            "num_samples": len(dataset),
            "device": str(device),
        }
        write_sweep_outputs(
            rows,
            json_path=args.output_json,
            csv_path=args.output_csv,
            metadata=metadata,
        )
        for row in model_rows:
            print(
                f"model={model_name} threshold={row['threshold']:.2f} "
                f"bal_acc={row['balanced_accuracy']:.4f} "
                f"sens={row['sensitivity']:.4f} "
                f"spec={row['specificity']:.4f}",
                flush=True,
            )

    print(
        f"saved rows={len(rows)} json={args.output_json} csv={args.output_csv}",
        flush=True,
    )
    return 0


def _load_model(
    checkpoint_path: Path,
    model_name: str,
    device: torch.device,
) -> tuple[nn.Module, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_meta = (
        checkpoint if isinstance(checkpoint, dict) and "model_state" in checkpoint else {}
    )
    state_dict = checkpoint_meta.get("model_state", checkpoint)
    checkpoint_model_name = checkpoint_meta.get("model_name")
    if checkpoint_model_name and checkpoint_model_name != model_name:
        raise ValueError(
            f"Checkpoint model mismatch: expected {model_name}, "
            f"found {checkpoint_model_name}."
        )

    model = build_model(model_name, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, int(checkpoint_meta.get("image_size", 224))


if __name__ == "__main__":
    raise SystemExit(main())
