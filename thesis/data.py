from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, Subset

from thesis.dataset_analysis import IMAGE_EXTENSIONS, infer_binary_label
from thesis.model_registry import expected_channels


LABEL_TO_INDEX = {"normal": 0, "pneumonia": 1}
INDEX_TO_LABEL = {0: "normal", 1: "pneumonia"}
MULTICLASS_LABEL_TO_INDEX = {
    "normal": 0,
    "lung_opacity": 1,
    "not_normal_no_lung_opacity": 2,
}
MULTICLASS_INDEX_TO_LABEL = {
    index: label for label, index in MULTICLASS_LABEL_TO_INDEX.items()
}


@dataclass(frozen=True)
class DatasetSplits:
    train: Dataset
    val: Dataset
    test: Dataset | None


class BinaryImageDataset(Dataset):
    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = self._collect_samples(self.root)
        if not self.samples:
            raise ValueError(f"No Normal/Pneumonia images found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def _collect_samples(self, root: Path):
        samples = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = infer_binary_label(path)
            if label is None:
                continue
            samples.append((path, LABEL_TO_INDEX[label]))
        return samples


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, transform=None):
        self.manifest_csv = Path(manifest_csv)
        self.transform = transform
        self.samples = []

        with self.manifest_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = row["label"].strip().lower()
                self.samples.append((Path(row["path"]), LABEL_TO_INDEX[label]))

        if not self.samples:
            raise ValueError(f"No samples found in manifest {self.manifest_csv}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class MultiClassImageDataset(Dataset):
    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = self._collect_samples(self.root)
        if not self.samples:
            raise ValueError(f"No multiclass RSNA images found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def _collect_samples(self, root: Path):
        samples = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_name = _infer_multiclass_label(path)
            if label_name is None:
                continue
            samples.append((path, MULTICLASS_LABEL_TO_INDEX[label_name]))
        return samples


class MultiClassManifestImageDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, transform=None):
        self.manifest_csv = Path(manifest_csv)
        self.transform = transform
        self.samples = []

        with self.manifest_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = row["label"].strip().lower()
                path_value = row.get("path") or row.get("image_path")
                if not path_value:
                    raise ValueError(
                        f"Manifest {self.manifest_csv} must contain path or image_path"
                    )
                self.samples.append((Path(path_value), MULTICLASS_LABEL_TO_INDEX[label]))

        if not self.samples:
            raise ValueError(f"No samples found in manifest {self.manifest_csv}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(model_name: str, image_size: int = 224, train: bool = False):
    transforms = _torchvision_transforms()
    channels = expected_channels(model_name)
    channel_transform = transforms.Grayscale(num_output_channels=channels)
    mean = [0.5] if channels == 1 else [0.485, 0.456, 0.406]
    std = [0.5] if channels == 1 else [0.229, 0.224, 0.225]

    steps = [channel_transform, transforms.Resize((image_size, image_size))]
    if train:
        steps.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(7)])
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


def _torchvision_transforms():
    try:
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for training/evaluation transforms. Install it before "
            "running scripts/train_model.py or scripts/evaluate_model.py."
        ) from exc
    return transforms


def build_internal_splits(
    data_root: str | Path,
    model_name: str,
    image_size: int = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> DatasetSplits:
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    train_tf = build_transforms(model_name, image_size=image_size, train=True)
    eval_tf = build_transforms(model_name, image_size=image_size, train=False)

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")

    if val_dir.is_dir():
        train_ds = BinaryImageDataset(train_dir, transform=train_tf)
        val_ds = BinaryImageDataset(val_dir, transform=eval_tf)
    else:
        train_full = BinaryImageDataset(train_dir, transform=train_tf)
        val_full = BinaryImageDataset(train_dir, transform=eval_tf)
        train_indices, val_indices = _split_indices(len(train_full), val_fraction, seed)
        train_ds = Subset(train_full, train_indices)
        val_ds = Subset(val_full, val_indices)

    test_ds = BinaryImageDataset(test_dir, transform=eval_tf) if test_dir.is_dir() else None
    return DatasetSplits(train=train_ds, val=val_ds, test=test_ds)


def build_multiclass_internal_splits(
    data_root: str | Path,
    model_name: str,
    image_size: int = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> DatasetSplits:
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    train_tf = build_transforms(model_name, image_size=image_size, train=True)
    eval_tf = build_transforms(model_name, image_size=image_size, train=False)

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")

    if val_dir.is_dir():
        train_ds = MultiClassImageDataset(train_dir, transform=train_tf)
        val_ds = MultiClassImageDataset(val_dir, transform=eval_tf)
    else:
        train_full = MultiClassImageDataset(train_dir, transform=train_tf)
        val_full = MultiClassImageDataset(train_dir, transform=eval_tf)
        train_indices, val_indices = _split_indices(len(train_full), val_fraction, seed)
        train_ds = Subset(train_full, train_indices)
        val_ds = Subset(val_full, val_indices)

    test_ds = MultiClassImageDataset(test_dir, transform=eval_tf) if test_dir.is_dir() else None
    return DatasetSplits(train=train_ds, val=val_ds, test=test_ds)


def _infer_multiclass_label(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    for label in MULTICLASS_LABEL_TO_INDEX:
        if label in parts:
            return label
    return None


def _split_indices(length: int, val_fraction: float, seed: int):
    if length < 2:
        raise ValueError("Need at least two training images to create a validation split")
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(round(length * val_fraction)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        train_indices, val_indices = indices[:-1], indices[-1:]
    return train_indices, val_indices
