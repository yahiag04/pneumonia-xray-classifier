#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.train import TrainConfig, evaluate_checkpoint, train_model


DEFAULT_MODELS = [
    "pneumonia_net",
    "resnet18",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "densenet121",
]


def build_model_list(models: list[str] | None) -> list[str]:
    return models or list(DEFAULT_MODELS)


def evaluation_output_name(model_name: str, dataset_name: str) -> str:
    return f"{model_name}_{dataset_name}.json"


def parse_manifest_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Manifest must be formatted as name=/path/to/manifest.csv")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("Manifest name cannot be empty")
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser(
        description="Train the adult RSNA branch for all comparison models and optionally evaluate external manifests."
    )
    parser.add_argument("--data-root", default="data/rsna_binary_size_matched")
    parser.add_argument("--output-dir", default="outputs/runs_rsna_adult")
    parser.add_argument("--eval-output-dir", default="outputs/evaluations/rsna_adult")
    parser.add_argument("--model", action="append", choices=DEFAULT_MODELS)
    parser.add_argument("--manifest", action="append", type=parse_manifest_arg, default=[])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    eval_output_dir = Path(args.eval_output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_name in build_model_list(args.model):
        summary = train_model(
            TrainConfig(
                data_root=data_root,
                model_name=model_name,
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                image_size=args.image_size,
                num_workers=args.num_workers,
                seed=args.seed,
                device=args.device,
            )
        )
        checkpoint = output_dir / model_name / "best.pt"
        evaluations = {}
        for dataset_name, manifest_csv in args.manifest:
            result = evaluate_checkpoint(
                checkpoint,
                manifest_csv=manifest_csv,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
            )
            output_path = eval_output_dir / evaluation_output_name(model_name, dataset_name)
            output_path.write_text(json.dumps(result, indent=2))
            evaluations[dataset_name] = result
        summaries.append({"model_name": model_name, "training": summary, "evaluations": evaluations})

    summary_path = eval_output_dir / "adult_branch_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
