#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.model_registry import available_models
from thesis.train import evaluate_checkpoint


DEFAULT_MODELS = [
    "pneumonia_net",
    "resnet18",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "densenet121",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate third-dataset fine-tuned checkpoints on NIH.")
    parser.add_argument("--manifest", default="outputs/nih/nih_224_binary_manifest.csv")
    parser.add_argument("--checkpoint-dir", default="outputs/runs_third_finetune")
    parser.add_argument("--output-dir", default="outputs/evaluations")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--models", nargs="+", choices=available_models(), default=DEFAULT_MODELS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        checkpoint = Path(args.checkpoint_dir) / model_name / "best.pt"
        output = output_dir / f"{model_name}_nih_224_after_third_ft.json"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {checkpoint}")

        print(f"evaluating model={model_name} checkpoint={checkpoint}", flush=True)
        started = time.perf_counter()
        result = evaluate_checkpoint(
            checkpoint,
            manifest_csv=args.manifest,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
        output.write_text(json.dumps(result, indent=2))
        elapsed = time.perf_counter() - started
        print(
            f"saved {output} "
            f"bal_acc={result['balanced_accuracy']:.4f} "
            f"sens={result['sensitivity']:.4f} "
            f"spec={result['specificity']:.4f} "
            f"roc_auc={result['roc_auc']:.4f} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
