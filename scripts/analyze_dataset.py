#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.dataset_analysis import scan_imagefolder, summarize_records, write_analysis_outputs


def main():
    parser = argparse.ArgumentParser(description="Analyze the Tolga Dincer/Kermany chest_xray dataset.")
    parser.add_argument("--data-root", required=True, help="Path to chest_xray root containing train/test[/val].")
    parser.add_argument("--output-dir", default="outputs/dataset_analysis", help="Directory for CSV/PNG outputs.")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV outputs only, without Matplotlib plots.")
    args = parser.parse_args()

    records = scan_imagefolder(args.data_root)
    summary = summarize_records(records)
    paths = write_analysis_outputs(records, args.output_dir, make_plots=not args.no_plots)

    print(f"Total images: {summary['total_images']}")
    for split, counts in sorted(summary["splits"].items()):
        print(f"{split}: normal={counts.get('normal', 0)} pneumonia={counts.get('pneumonia', 0)}")
    print("Written:")
    for path in paths.values():
        print(f"- {path}")


if __name__ == "__main__":
    main()
