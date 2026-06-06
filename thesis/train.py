from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from thesis.data import ManifestImageDataset, build_internal_splits, build_transforms
from thesis.metrics import compute_binary_metrics
from thesis.model_registry import build_model, freeze_backbone


@dataclass(frozen=True)
class TrainConfig:
    data_root: Path
    model_name: str
    output_dir: Path
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    patience: int = 5
    image_size: int = 224
    val_fraction: float = 0.1
    pretrained: bool = True
    freeze_backbone: bool = False
    num_workers: int = 0
    seed: int = 42
    device: str | None = None


def choose_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(config: TrainConfig) -> dict:
    torch.manual_seed(config.seed)
    device = choose_device(config.device)
    run_dir = config.output_dir / config.model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = build_internal_splits(
        config.data_root,
        config.model_name,
        image_size=config.image_size,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    train_loader = DataLoader(
        splits.train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        splits.val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_model(config.model_name, pretrained=config.pretrained).to(device)
    if config.freeze_backbone:
        model = freeze_backbone(model, config.model_name)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.lr,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    checkpoint_path = run_dir / "best.pt"
    history = []

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate_loader(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_result.items()}}
        history.append(row)

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": config.model_name,
                    "pretrained": config.pretrained,
                    "freeze_backbone": config.freeze_backbone,
                    "image_size": config.image_size,
                    "threshold": 0.5,
                    "config": _jsonable_config(config),
                },
                checkpoint_path,
            )
        else:
            no_improve += 1
            if no_improve >= config.patience:
                break

    summary = {
        "model_name": config.model_name,
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    if splits.test is not None:
        test_metrics = evaluate_checkpoint(
            checkpoint_path,
            data_root=config.data_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=str(device),
        )
        summary["test_metrics"] = test_metrics
        (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    return summary


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_root: str | Path | None = None,
    manifest_csv: str | Path | None = None,
    model_name: str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    threshold: float | None = None,
    device: str | None = None,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_meta = checkpoint if isinstance(checkpoint, dict) and "model_state" in checkpoint else {}
    state_dict = checkpoint_meta.get("model_state", checkpoint)
    model_name = model_name or checkpoint_meta.get("model_name", "pneumonia_net")
    image_size = int(checkpoint_meta.get("image_size", 224))
    threshold = float(threshold if threshold is not None else checkpoint_meta.get("threshold", 0.5))
    torch_device = choose_device(device)

    model = build_model(model_name, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(torch_device)
    model.eval()

    transform = build_transforms(model_name, image_size=image_size, train=False)
    if manifest_csv is not None:
        dataset = ManifestImageDataset(manifest_csv, transform=transform)
    elif data_root is not None:
        splits = build_internal_splits(data_root, model_name, image_size=image_size)
        if splits.test is None:
            raise ValueError("No test split found for internal evaluation")
        dataset = splits.test
    else:
        raise ValueError("Provide either data_root or manifest_csv")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    criterion = nn.BCEWithLogitsLoss()
    result = evaluate_loader(model, loader, criterion, torch_device, threshold=threshold)
    result["checkpoint"] = str(checkpoint_path)
    result["model_name"] = model_name
    result["num_samples"] = len(dataset)
    return result


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    predictions = collect_predictions(model, loader, criterion, device)
    metrics = compute_binary_metrics(
        predictions["labels"],
        predictions["probabilities"],
        threshold=threshold,
    )
    total = predictions["num_samples"]
    metrics["loss"] = predictions["loss"]
    metrics["seconds_per_image"] = predictions["elapsed_seconds"] / max(total, 1)
    return metrics


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    total_loss = 0.0
    total = 0
    labels = []
    probabilities = []

    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y_float = y.float().to(device)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y_float)
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            labels.extend(y.cpu().numpy().astype(int).tolist())
            probabilities.extend(probs.cpu().numpy().astype(float).tolist())

    return {
        "labels": labels,
        "probabilities": probabilities,
        "loss": total_loss / max(total, 1),
        "num_samples": total,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.float().to(device)
        optimizer.zero_grad()
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / max(total, 1)


def _jsonable_config(config: TrainConfig) -> dict:
    result = asdict(config)
    result["data_root"] = str(config.data_root)
    result["output_dir"] = str(config.output_dir)
    return result
