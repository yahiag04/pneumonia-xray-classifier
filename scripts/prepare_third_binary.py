#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.dataset_analysis import IMAGE_EXTENSIONS


PATHOLOGY_FOLDERS = {"COVID19", "PNEUMONIA", "TURBERCULOSIS", "TUBERCULOSIS"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create balanced Normal/Pneumonia manifests for the third dataset.")
    parser.add_argument("--data-root", required=True, help="Root containing train/val/test folders.")
    parser.add_argument("--output-dir", default="outputs/third_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    train_val_records = collect_split(data_root, ["train", "val"])
    test_records = collect_split(data_root, ["test"])

    balanced_train_val = balance_records(train_val_records, rng)
    train_records, val_records = stratified_split(
        balanced_train_val,
        val_fraction=args.val_fraction,
        rng=rng,
    )
    balanced_test = balance_records(test_records, rng)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(train_records, output_dir / "third_train_balanced.csv")
    write_manifest(val_records, output_dir / "third_val_balanced.csv")
    write_manifest(balanced_test, output_dir / "third_test_balanced.csv")
    write_manifest(train_records + val_records, output_dir / "third_trainval_balanced.csv")

    write_counts(
        output_dir / "third_dataset_counts.csv",
        {
            "raw_train_val": train_val_records,
            "raw_test": test_records,
            "balanced_train": train_records,
            "balanced_val": val_records,
            "balanced_test": balanced_test,
        },
    )

    for name, records in [
        ("raw_train_val", train_val_records),
        ("raw_test", test_records),
        ("balanced_train", train_records),
        ("balanced_val", val_records),
        ("balanced_test", balanced_test),
    ]:
        counts = Counter(record["label"] for record in records)
        print(f"{name}: normal={counts.get('normal', 0)} pneumonia={counts.get('pneumonia', 0)} total={len(records)}")

    return 0


def collect_split(data_root: Path, split_names: list[str]) -> list[dict[str, str]]:
    records = []
    for split_name in split_names:
        split_dir = data_root / split_name
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = map_folder_to_binary(path.parent.name)
            if label is None:
                continue
            records.append(
                {
                    "path": str(path),
                    "label": label,
                    "source_split": split_name,
                    "source_class": path.parent.name,
                }
            )
    if not records:
        raise ValueError(f"No binary images found in {data_root} for splits {split_names}")
    return records


def map_folder_to_binary(folder_name: str) -> str | None:
    normalized = folder_name.strip().upper()
    if normalized == "NORMAL":
        return "normal"
    if normalized in PATHOLOGY_FOLDERS:
        return "pneumonia"
    return None


def balance_records(records: list[dict[str, str]], rng: random.Random) -> list[dict[str, str]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    if not by_label["normal"] or not by_label["pneumonia"]:
        raise ValueError("Need both normal and pneumonia records to balance the dataset")

    target = min(len(by_label["normal"]), len(by_label["pneumonia"]))
    balanced = []
    for label in ["normal", "pneumonia"]:
        candidates = list(by_label[label])
        rng.shuffle(candidates)
        balanced.extend(candidates[:target])
    rng.shuffle(balanced)
    return balanced


def stratified_split(records: list[dict[str, str]], val_fraction: float, rng: random.Random):
    train_records = []
    val_records = []
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    for label in ["normal", "pneumonia"]:
        items = list(by_label[label])
        rng.shuffle(items)
        val_size = max(1, int(round(len(items) * val_fraction)))
        val_records.extend(items[:val_size])
        train_records.extend(items[val_size:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def write_manifest(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "source_split", "source_class"])
        writer.writeheader()
        writer.writerows(records)


def write_counts(output_path: Path, datasets: dict[str, list[dict[str, str]]]) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "label", "count"])
        writer.writeheader()
        for dataset_name, records in datasets.items():
            counts = Counter(record["label"] for record in records)
            for label in ["normal", "pneumonia"]:
                writer.writerow({"dataset": dataset_name, "label": label, "count": counts.get(label, 0)})


if __name__ == "__main__":
    raise SystemExit(main())
