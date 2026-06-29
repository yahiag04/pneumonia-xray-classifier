#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


RSNA_TO_BINARY = {
    "Normal": "normal",
    "Lung Opacity": "pneumonia",
}


@dataclass(frozen=True)
class RsnaRecord:
    patient_id: str
    label: str
    dicom_path: Path
    rsna_class: str
    patient_age: str = ""
    patient_sex: str = ""
    view_position: str = ""
    source_format: str = "dicom"


def map_rsna_class(rsna_class: str) -> str | None:
    return RSNA_TO_BINARY.get(rsna_class.strip())


def collect_binary_records(
    detailed_class_csv: str | Path,
    image_dir: str | Path,
) -> list[RsnaRecord]:
    detailed_class_csv = Path(detailed_class_csv)
    image_dir = Path(image_dir)
    by_patient: dict[str, RsnaRecord] = {}

    with detailed_class_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_id = row["patientId"].strip()
            rsna_class = row["class"].strip()
            label = map_rsna_class(rsna_class)
            if label is None or patient_id in by_patient:
                continue
            dicom_path = image_dir / f"{patient_id}.dcm"
            if not dicom_path.is_file():
                continue
            by_patient[patient_id] = RsnaRecord(
                patient_id=patient_id,
                label=label,
                dicom_path=dicom_path,
                rsna_class=rsna_class,
            )

    return sorted(by_patient.values(), key=lambda record: record.patient_id)


def collect_processed_png_records(
    metadata_csv: str | Path,
    image_dir: str | Path,
) -> list[RsnaRecord]:
    metadata_csv = Path(metadata_csv)
    image_dir = Path(image_dir)
    by_patient: dict[str, RsnaRecord] = {}

    with metadata_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_id = row["patientId"].strip()
            rsna_class = row["class"].strip()
            label = map_rsna_class(rsna_class)
            if label is None or patient_id in by_patient:
                continue
            png_path = image_dir / f"{patient_id}.png"
            if not png_path.is_file():
                continue
            by_patient[patient_id] = RsnaRecord(
                patient_id=patient_id,
                label=label,
                dicom_path=png_path,
                rsna_class=rsna_class,
                patient_age=row.get("age", "").strip(),
                patient_sex=row.get("sex", "").strip(),
                view_position=row.get("position", "").strip(),
                source_format="png",
            )

    return sorted(by_patient.values(), key=lambda record: record.patient_id)


def split_balanced_records(
    records: list[RsnaRecord],
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    seed: int = 42,
) -> dict[str, list[RsnaRecord]]:
    required = train_per_class + val_per_class + test_per_class
    rng = random.Random(seed)
    by_label = {
        "normal": [record for record in records if record.label == "normal"],
        "pneumonia": [record for record in records if record.label == "pneumonia"],
    }

    for label, label_records in by_label.items():
        if len(label_records) < required:
            raise ValueError(
                f"Need {required} {label} records, found {len(label_records)}"
            )
        rng.shuffle(label_records)

    splits = {"train": [], "val": [], "test": []}
    for label_records in by_label.values():
        train_end = train_per_class
        val_end = train_end + val_per_class
        test_end = val_end + test_per_class
        splits["train"].extend(label_records[:train_end])
        splits["val"].extend(label_records[train_end:val_end])
        splits["test"].extend(label_records[val_end:test_end])

    for split_records in splits.values():
        rng.shuffle(split_records)
    return splits


def build_output_image_path(output_root: str | Path, split: str, record: RsnaRecord) -> Path:
    return Path(output_root) / split / record.label / f"{record.patient_id}.png"


def convert_dicom_to_png(dicom_path: str | Path, output_path: str | Path) -> dict[str, str]:
    import pydicom

    dicom = pydicom.dcmread(str(dicom_path))
    image = dicom.pixel_array.astype(np.float32)
    slope = float(getattr(dicom, "RescaleSlope", 1.0))
    intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
    image = image * slope + intercept

    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    image -= image.min()
    max_value = image.max()
    if max_value > 0:
        image /= max_value
    image = (image * 255).clip(0, 255).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(output_path)

    return {
        "patient_age": str(getattr(dicom, "PatientAge", "")),
        "patient_sex": str(getattr(dicom, "PatientSex", "")),
        "view_position": str(getattr(dicom, "ViewPosition", "")),
    }


def materialize_png(
    source_path: str | Path,
    output_path: str | Path,
    link_mode: str = "auto",
) -> None:
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if link_mode == "copy":
        shutil.copy2(source_path, output_path)
        return
    if link_mode == "symlink":
        output_path.symlink_to(source_path.resolve())
        return
    if link_mode == "hardlink":
        output_path.hardlink_to(source_path)
        return
    if link_mode != "auto":
        raise ValueError(f"Unsupported link mode: {link_mode}")

    try:
        output_path.hardlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, output_path)


def write_split(
    output_root: str | Path,
    split: str,
    records: list[RsnaRecord],
    link_mode: str = "auto",
) -> list[dict[str, str]]:
    rows = []
    for record in records:
        output_path = build_output_image_path(output_root, split, record)
        if record.source_format == "dicom":
            metadata = convert_dicom_to_png(record.dicom_path, output_path)
        elif record.source_format == "png":
            materialize_png(record.dicom_path, output_path, link_mode=link_mode)
            metadata = {
                "patient_age": record.patient_age,
                "patient_sex": record.patient_sex,
                "view_position": record.view_position,
            }
        else:
            raise ValueError(f"Unsupported source format: {record.source_format}")
        rows.append(
            {
                "split": split,
                "patient_id": record.patient_id,
                "label": record.label,
                "rsna_class": record.rsna_class,
                "source_path": str(record.dicom_path),
                "image_path": str(output_path),
                "source_format": record.source_format,
                **metadata,
            }
        )
    return rows


def write_metadata(output_root: str | Path, rows: list[dict[str, str]]) -> Path:
    output_path = Path(output_root) / "metadata.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "patient_id",
        "label",
        "rsna_class",
        "source_path",
        "image_path",
        "source_format",
        "patient_age",
        "patient_sex",
        "view_position",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def prepare_rsna_binary(
    rsna_root: str | Path,
    output_root: str | Path,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    seed: int = 42,
    overwrite: bool = False,
    link_mode: str = "auto",
) -> dict:
    rsna_root = Path(rsna_root)
    output_root = Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}")
        shutil.rmtree(output_root)

    detailed_csv = rsna_root / "stage_2_detailed_class_info.csv"
    image_dir = rsna_root / "stage_2_train_images"
    processed_csv = rsna_root / "stage2_train_metadata.csv"
    processed_image_dir = rsna_root / "Training" / "Images"
    if detailed_csv.is_file() and image_dir.is_dir():
        source_layout = "dicom"
        records = collect_binary_records(detailed_csv, image_dir)
    elif processed_csv.is_file() and processed_image_dir.is_dir():
        source_layout = "processed_png"
        records = collect_processed_png_records(processed_csv, processed_image_dir)
    else:
        raise FileNotFoundError(
            "Missing supported RSNA layout. Expected either "
            f"{detailed_csv} with {image_dir}, or {processed_csv} with "
            f"{processed_image_dir}."
        )

    splits = split_balanced_records(
        records,
        train_per_class=train_per_class,
        val_per_class=val_per_class,
        test_per_class=test_per_class,
        seed=seed,
    )

    metadata_rows = []
    for split, split_records in splits.items():
        metadata_rows.extend(
            write_split(output_root, split, split_records, link_mode=link_mode)
        )
    metadata_path = write_metadata(output_root, metadata_rows)

    summary = {
        "output_root": str(output_root),
        "metadata_csv": str(metadata_path),
        "source_layout": source_layout,
        "total_binary_records_available": len(records),
        "splits": {
            split: {
                "total": len(split_records),
                "normal": sum(1 for record in split_records if record.label == "normal"),
                "pneumonia": sum(1 for record in split_records if record.label == "pneumonia"),
            }
            for split, split_records in splits.items()
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a balanced Normal/Pneumonia ImageFolder from RSNA Pneumonia Detection Challenge."
    )
    parser.add_argument("--rsna-root", required=True)
    parser.add_argument("--output-root", default="data/rsna_binary_size_matched")
    parser.add_argument("--train-per-class", type=int, default=2500)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--link-mode",
        choices=["auto", "hardlink", "symlink", "copy"],
        default="auto",
        help="How to materialize processed PNG inputs. DICOM inputs are always converted.",
    )
    args = parser.parse_args()

    summary = prepare_rsna_binary(
        rsna_root=args.rsna_root,
        output_root=args.output_root,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
        overwrite=args.overwrite,
        link_mode=args.link_mode,
    )
    print(summary)


if __name__ == "__main__":
    main()
