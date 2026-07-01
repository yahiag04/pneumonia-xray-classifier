from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from thesis.data import (
    MULTICLASS_INDEX_TO_LABEL,
    MultiClassManifestImageDataset,
    build_multiclass_internal_splits,
    build_transforms,
)
from thesis.metrics import compute_multiclass_metrics
from thesis.model_registry import build_model, freeze_backbone
from thesis.train import choose_device, keep_frozen_modules_eval


NUM_CLASSES = len(MULTICLASS_INDEX_TO_LABEL)
CLASS_NAMES = [MULTICLASS_INDEX_TO_LABEL[index] for index in range(NUM_CLASSES)]


@dataclass(frozen=True)
class MulticlassTrainConfig:
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
    init_checkpoint: Path | None = None
    num_workers: int = 0
    seed: int = 42
    device: str | None = None


def train_multiclass_model(config: MulticlassTrainConfig) -> dict:
    torch.manual_seed(config.seed)
    device = choose_device(config.device)
    run_dir = config.output_dir / config.model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = build_multiclass_internal_splits(
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

    model = build_model(
        config.model_name,
        pretrained=config.pretrained,
        num_classes=NUM_CLASSES,
    ).to(device)
    init_summary = None
    if config.init_checkpoint is not None:
        checkpoint = _load_local_checkpoint(config.init_checkpoint)
        init_summary = load_matching_checkpoint_weights(model, checkpoint)
    if config.freeze_backbone:
        model = freeze_backbone(model, config.model_name)
    criterion = nn.CrossEntropyLoss()
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
        train_loss = _train_multiclass_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate_multiclass_loader(model, val_loader, criterion, device)
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
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "task": "rsna_multiclass",
                    "init_checkpoint": str(config.init_checkpoint) if config.init_checkpoint else None,
                    "init_summary": init_summary,
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
        "class_names": CLASS_NAMES,
        "init_checkpoint": str(config.init_checkpoint) if config.init_checkpoint else None,
        "init_summary": init_summary,
        "history": history,
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    if splits.test is not None:
        test_metrics = evaluate_multiclass_checkpoint(
            checkpoint_path,
            data_root=config.data_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=str(device),
        )
        summary["test_metrics"] = test_metrics
        (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    return summary


def evaluate_multiclass_checkpoint(
    checkpoint_path: str | Path,
    data_root: str | Path | None = None,
    manifest_csv: str | Path | None = None,
    model_name: str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | None = None,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _load_local_checkpoint(checkpoint_path)
    checkpoint_meta = checkpoint if isinstance(checkpoint, dict) and "model_state" in checkpoint else {}
    state_dict = checkpoint_meta.get("model_state", checkpoint)
    model_name = model_name or checkpoint_meta.get("model_name", "pneumonia_net")
    image_size = int(checkpoint_meta.get("image_size", 224))
    num_classes = int(checkpoint_meta.get("num_classes", NUM_CLASSES))
    class_names = checkpoint_meta.get("class_names", CLASS_NAMES)
    torch_device = choose_device(device)

    model = build_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(torch_device)
    model.eval()

    transform = build_transforms(model_name, image_size=image_size, train=False)
    if manifest_csv is not None:
        dataset = MultiClassManifestImageDataset(manifest_csv, transform=transform)
    elif data_root is not None:
        splits = build_multiclass_internal_splits(data_root, model_name, image_size=image_size)
        if splits.test is None:
            raise ValueError("No test split found for internal evaluation")
        dataset = splits.test
    else:
        raise ValueError("Provide either data_root or manifest_csv")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss()
    result = evaluate_multiclass_loader(
        model,
        loader,
        criterion,
        torch_device,
        class_names=class_names,
    )
    result["checkpoint"] = str(checkpoint_path)
    result["model_name"] = model_name
    result["num_samples"] = len(dataset)
    result["class_names"] = class_names
    return result


def evaluate_multiclass_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict:
    class_names = class_names or CLASS_NAMES
    predictions = collect_multiclass_predictions(model, loader, criterion, device)
    metrics = compute_multiclass_metrics(
        predictions["labels"],
        predictions["probabilities"],
        class_names=class_names,
    )
    total = predictions["num_samples"]
    metrics["loss"] = predictions["loss"]
    metrics["seconds_per_image"] = predictions["elapsed_seconds"] / max(total, 1)
    return metrics


def collect_multiclass_predictions(
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
            y = y.long().to(device)
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)

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


def load_matching_checkpoint_weights(model: nn.Module, checkpoint: dict) -> dict[str, list[str]]:
    source_state = checkpoint.get("model_state", checkpoint)
    target_state = model.state_dict()
    loadable = {}
    skipped_shape_mismatch = []
    skipped_missing = []

    for name, tensor in source_state.items():
        if name not in target_state:
            skipped_missing.append(name)
            continue
        if target_state[name].shape != tensor.shape:
            skipped_shape_mismatch.append(name)
            continue
        loadable[name] = tensor

    target_state.update(loadable)
    model.load_state_dict(target_state)
    return {
        "loaded": sorted(loadable),
        "skipped_shape_mismatch": sorted(skipped_shape_mismatch),
        "skipped_missing": sorted(skipped_missing),
    }


def _load_local_checkpoint(checkpoint_path: str | Path):
    return torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)


def _train_multiclass_epoch(model, loader, criterion, optimizer, device):
    model.train()
    keep_frozen_modules_eval(model)
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.long().to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / max(total, 1)


def _jsonable_config(config: MulticlassTrainConfig) -> dict:
    result = asdict(config)
    result["data_root"] = str(config.data_root)
    result["output_dir"] = str(config.output_dir)
    return result
