#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.model_registry import available_models
from thesis.train_multiclass import MulticlassTrainConfig, train_multiclass_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model on the 3-class RSNA task.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--output-dir", default="outputs/runs_rsna_multiclass")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--init-checkpoint",
        help="Optional binary checkpoint used to initialize all shape-compatible layers.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="For pretrained torchvision models, train only the final classifier head.",
    )
    args = parser.parse_args()

    config = MulticlassTrainConfig(
        data_root=Path(args.data_root),
        model_name=args.model,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        init_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else None,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )
    summary = train_multiclass_model(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
