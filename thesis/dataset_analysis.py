from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NORMAL_TOKENS = {"NORMAL"}
PNEUMONIA_TOKENS = {"PNEUMONIA", "BACTERIA", "VIRUS"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    split: str
    label: str
    width: int | None = None
    height: int | None = None


def infer_binary_label(path: Path) -> str | None:
    """Infer Normal/Pneumonia label from folder names or Tolga/Kermany filenames."""
    parts = [part.upper() for part in path.parts]
    if any(part in NORMAL_TOKENS for part in parts):
        return "normal"
    if any(part in PNEUMONIA_TOKENS for part in parts):
        return "pneumonia"

    prefix = path.stem.upper().replace("_", "-").split("-")[0]
    if prefix in NORMAL_TOKENS:
        return "normal"
    if prefix in PNEUMONIA_TOKENS:
        return "pneumonia"
    return None


def scan_imagefolder(
    root: str | Path,
    splits: Iterable[str] | None = None,
    include_sizes: bool = True,
) -> list[ImageRecord]:
    root = Path(root)
    if splits is None:
        preferred = [name for name in ("train", "val", "test") if (root / name).is_dir()]
        splits = preferred or [child.name for child in root.iterdir() if child.is_dir()]

    records: list[ImageRecord] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue

        for image_path in sorted(_iter_images(split_dir)):
            label = infer_binary_label(image_path)
            if label is None:
                continue
            width, height = _image_size(image_path) if include_sizes else (None, None)
            records.append(
                ImageRecord(
                    path=image_path,
                    split=split,
                    label=label,
                    width=width,
                    height=height,
                )
            )
    return records


def summarize_records(records: Iterable[ImageRecord]) -> dict:
    split_counts: dict[str, Counter] = defaultdict(Counter)
    image_sizes: Counter = Counter()
    total = 0

    for record in records:
        total += 1
        split_counts[record.split][record.label] += 1
        if record.width is not None and record.height is not None:
            image_sizes[(record.width, record.height)] += 1

    return {
        "total_images": total,
        "splits": {split: dict(counts) for split, counts in split_counts.items()},
        "image_sizes": image_sizes,
    }


def write_analysis_outputs(
    records: list[ImageRecord],
    output_dir: str | Path,
    make_plots: bool = True,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "records": output_dir / "dataset_records.csv",
        "split_class_counts": output_dir / "split_class_counts.csv",
        "dataset_summary": output_dir / "dataset_summary.csv",
        "image_sizes": output_dir / "image_sizes.csv",
    }

    _write_records_csv(records, paths["records"])
    _write_counts_csv(records, paths["split_class_counts"])
    _write_summary_csv(records, paths["dataset_summary"])
    _write_image_sizes_csv(records, paths["image_sizes"])
    if make_plots:
        _write_plots(records, output_dir)
    return paths


def _iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _write_records_csv(records: list[ImageRecord], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "split", "label", "width", "height"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "path": str(record.path),
                    "split": record.split,
                    "label": record.label,
                    "width": record.width or "",
                    "height": record.height or "",
                }
            )


def _write_counts_csv(records: list[ImageRecord], path: Path) -> None:
    summary = summarize_records(records)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "label", "count"])
        writer.writeheader()
        for split, counts in sorted(summary["splits"].items()):
            for label in ("normal", "pneumonia"):
                writer.writerow({"split": split, "label": label, "count": counts.get(label, 0)})


def _write_summary_csv(records: list[ImageRecord], path: Path) -> None:
    summary = summarize_records(records)
    totals = Counter()
    for counts in summary["splits"].values():
        totals.update(counts)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "total_images", "value": summary["total_images"]})
        writer.writerow({"metric": "normal_images", "value": totals.get("normal", 0)})
        writer.writerow({"metric": "pneumonia_images", "value": totals.get("pneumonia", 0)})


def _write_image_sizes_csv(records: list[ImageRecord], path: Path) -> None:
    sizes = summarize_records(records)["image_sizes"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["width", "height", "count"])
        writer.writeheader()
        for (width, height), count in sorted(sizes.items()):
            writer.writerow({"width": width, "height": height, "count": count})


def _write_plots(records: list[ImageRecord], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    summary = summarize_records(records)
    labels = ["normal", "pneumonia"]
    totals = Counter()
    for counts in summary["splits"].values():
        totals.update(counts)

    plt.figure(figsize=(6, 4))
    plt.bar(labels, [totals.get(label, 0) for label in labels], color=["#4c78a8", "#f58518"])
    plt.title("Class distribution")
    plt.xlabel("Class")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=200)
    plt.close()

    split_names = sorted(summary["splits"])
    x = range(len(split_names))
    normal_counts = [summary["splits"][split].get("normal", 0) for split in split_names]
    pneumonia_counts = [summary["splits"][split].get("pneumonia", 0) for split in split_names]

    plt.figure(figsize=(7, 4))
    plt.bar([i - 0.2 for i in x], normal_counts, width=0.4, label="normal", color="#4c78a8")
    plt.bar([i + 0.2 for i in x], pneumonia_counts, width=0.4, label="pneumonia", color="#f58518")
    plt.xticks(list(x), split_names)
    plt.title("Split x class distribution")
    plt.xlabel("Split")
    plt.ylabel("Images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "split_class_distribution.png", dpi=200)
    plt.close()
