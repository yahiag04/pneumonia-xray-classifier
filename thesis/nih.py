from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NihRecord:
    path: Path
    label: str
    source_labels: str


def parse_nih_labels(labels: str) -> set[str]:
    return {label.strip() for label in labels.split("|") if label.strip()}


def map_nih_binary_label(labels: str, exclusive_pneumonia: bool = False) -> str | None:
    label_set = parse_nih_labels(labels)
    if label_set == {"No Finding"}:
        return "normal"
    if "Pneumonia" in label_set:
        if exclusive_pneumonia and label_set != {"Pneumonia"}:
            return None
        return "pneumonia"
    return None


def build_nih_manifest(
    csv_path: str | Path,
    image_root: str | Path,
    exclusive_pneumonia: bool = False,
) -> list[NihRecord]:
    csv_path = Path(csv_path)
    image_root = Path(image_root)
    image_index = _build_image_index(image_root)
    records: list[NihRecord] = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = map_nih_binary_label(
                row["Finding Labels"],
                exclusive_pneumonia=exclusive_pneumonia,
            )
            if label is None:
                continue

            image_name = row["Image Index"]
            image_path = image_index.get(image_name)
            if image_path is None:
                continue

            records.append(
                NihRecord(
                    path=image_path,
                    label=label,
                    source_labels=row["Finding Labels"],
                )
            )
    return records


def write_manifest_csv(records: list[NihRecord], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "source_labels"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "path": str(record.path),
                    "label": record.label,
                    "source_labels": record.source_labels,
                }
            )
    return output_path


def _build_image_index(image_root: Path) -> dict[str, Path]:
    return {path.name: path for path in image_root.rglob("*") if path.is_file()}

