#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.data import LABEL_TO_INDEX
from thesis.metrics import compute_binary_metrics
from thesis.train import choose_device


DEFAULT_WEIGHTS = [
    "densenet121-res224-rsna",
    "densenet121-res224-nih",
    "densenet121-res224-all",
]


class TorchXRayVisionManifestDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, transform):
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
        image = load_xrv_image(path)
        image = self.transform(image)
        return torch.from_numpy(image).float(), label


def load_xrv_image(path: str | Path) -> np.ndarray:
    import torchxrayvision as xrv

    image = Image.open(path).convert("L")
    array = np.asarray(image).astype(np.float32)
    array = xrv.datasets.normalize(array, 255)
    return array[None, ...]


def build_xrv_transform(image_size: int = 224):
    import torchvision
    import torchxrayvision as xrv

    return torchvision.transforms.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(image_size),
        ]
    )


def find_pneumonia_index(pathologies: list[str] | tuple[str, ...]) -> int:
    try:
        return list(pathologies).index("Pneumonia")
    except ValueError as exc:
        raise ValueError("TorchXRayVision model does not expose a 'Pneumonia' output") from exc


def build_result_payload(
    model_name: str,
    manifest_csv: str | Path,
    labels: list[int],
    probabilities: list[float],
    threshold: float,
    elapsed_seconds: float,
) -> dict:
    metrics = compute_binary_metrics(labels, probabilities, threshold=threshold)
    metrics["model_name"] = model_name
    metrics["manifest"] = str(manifest_csv)
    metrics["num_samples"] = len(labels)
    metrics["seconds_per_image"] = elapsed_seconds / max(len(labels), 1)
    return metrics


def evaluate_xrv_weights(
    weights: str,
    manifest_csv: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    threshold: float = 0.5,
    device: str | None = None,
    cache_dir: str | Path | None = None,
) -> dict:
    import torchxrayvision as xrv

    torch_device = choose_device(device)
    model = xrv.models.DenseNet(
        weights=weights,
        apply_sigmoid=True,
        cache_dir=cache_dir,
    ).to(torch_device)
    model.eval()
    pneumonia_index = find_pneumonia_index(model.pathologies)

    dataset = TorchXRayVisionManifestDataset(manifest_csv, transform=build_xrv_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    labels = []
    probabilities = []
    start = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(torch_device)
            output = model(x)
            probs = output[:, pneumonia_index].detach().cpu().numpy()
            labels.extend(y.numpy().astype(int).tolist())
            probabilities.extend(probs.astype(float).tolist())

    return build_result_payload(
        model_name=weights,
        manifest_csv=manifest_csv,
        labels=labels,
        probabilities=probabilities,
        threshold=threshold,
        elapsed_seconds=time.perf_counter() - start,
    )


def output_path_for(output_dir: str | Path, weights: str) -> Path:
    safe_name = weights.replace("/", "_")
    return Path(output_dir) / f"{safe_name}.json"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TorchXRayVision pretrained DenseNet models on a binary manifest."
    )
    parser.add_argument("--manifest", required=True, help="CSV with path,label columns.")
    parser.add_argument(
        "--weights",
        action="append",
        choices=DEFAULT_WEIGHTS,
        help="TorchXRayVision weights to evaluate. Repeat to evaluate multiple models.",
    )
    parser.add_argument("--output-dir", default="outputs/evaluations/torchxrayvision_zero_shot")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", default="outputs/torchxrayvision_cache")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_list = args.weights or DEFAULT_WEIGHTS

    results = []
    for weights in weights_list:
        result = evaluate_xrv_weights(
            weights,
            manifest_csv=args.manifest,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold=args.threshold,
            device=args.device,
            cache_dir=args.cache_dir,
        )
        output_path = output_path_for(output_dir, weights)
        output_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        results.append(result)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
