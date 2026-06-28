#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.nih import build_nih_manifest, write_manifest_csv


def main():
    parser = argparse.ArgumentParser(description="Create a NIH ChestX-ray14 Normal/Pneumonia manifest.")
    parser.add_argument("--csv", required=True, help="Path to Data_Entry_2017.csv.")
    parser.add_argument("--image-root", required=True, help="Root containing NIH image files or image folders.")
    parser.add_argument("--output", default="outputs/nih/nih_binary_manifest.csv")
    parser.add_argument(
        "--exclusive-pneumonia",
        action="store_true",
        help="Keep only NIH rows whose only pathology label is Pneumonia.",
    )
    args = parser.parse_args()

    records = build_nih_manifest(
        args.csv,
        args.image_root,
        exclusive_pneumonia=args.exclusive_pneumonia,
    )
    output = write_manifest_csv(records, args.output)
    counts = Counter(record.label for record in records)

    print(f"Manifest: {output}")
    print(f"normal={counts.get('normal', 0)} pneumonia={counts.get('pneumonia', 0)}")


if __name__ == "__main__":
    main()

