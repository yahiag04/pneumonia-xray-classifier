#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.model_registry import available_models
from thesis.train import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint internally or on NIH manifest.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model",
        choices=available_models(),
        help="Required only when evaluating a legacy plain state_dict checkpoint.",
    )
    parser.add_argument("--data-root", help="Internal chest_xray root for test split evaluation.")
    parser.add_argument("--manifest", help="External manifest CSV, for example NIH binary manifest.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    result = evaluate_checkpoint(
        args.checkpoint,
        data_root=args.data_root,
        manifest_csv=args.manifest,
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        device=args.device,
    )
    text = json.dumps(result, indent=2)
    print(text)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)


if __name__ == "__main__":
    main()
