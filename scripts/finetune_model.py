#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.data import ManifestImageDataset, build_transforms
from thesis.model_registry import available_models, build_model, configure_trainable_layers
from thesis.train import choose_device, evaluate_checkpoint, evaluate_loader


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune an existing checkpoint on manifest CSV datasets.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest")
    parser.add_argument("--model", choices=available_models(), help="Required for legacy state_dict checkpoints.")
    parser.add_argument("--output-dir", default="outputs/runs_third_finetune")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--trainable-mode",
        choices=("all", "head", "last_block"),
        default="all",
        help="Which layers remain trainable during fine-tuning.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_meta = checkpoint if isinstance(checkpoint, dict) and "model_state" in checkpoint else {}
    state_dict = checkpoint_meta.get("model_state", checkpoint)
    model_name = args.model or checkpoint_meta.get("model_name")
    if model_name is None:
        raise ValueError("Model name is required for legacy checkpoints without metadata.")
    image_size = int(checkpoint_meta.get("image_size", 224))

    run_dir = Path(build_run_dir(args.output_dir, model_name, args.trainable_mode))
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"

    train_transform = build_transforms(model_name, image_size=image_size, train=True)
    eval_transform = build_transforms(model_name, image_size=image_size, train=False)
    train_dataset = ManifestImageDataset(args.train_manifest, transform=train_transform)
    val_dataset = ManifestImageDataset(args.val_manifest, transform=eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(model_name, pretrained=False)
    model.load_state_dict(state_dict)
    model = configure_trainable_layers(model, model_name, args.trainable_mode)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, args.lr)
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate_loader(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_seconds": elapsed,
            **{f"val_{key}": value for key, value in val_result.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_result['loss']:.6f} val_bal_acc={val_result['balanced_accuracy']:.4f} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "pretrained": False,
                    "fine_tuned_from": str(checkpoint_path),
                    "image_size": image_size,
                    "trainable_mode": args.trainable_mode,
                    "threshold": 0.5,
                    "config": vars(args),
                },
                best_path,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    summary = {
        "model_name": model_name,
        "checkpoint": str(best_path),
        "fine_tuned_from": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }

    if args.test_manifest:
        summary["test_metrics"] = evaluate_checkpoint(
            best_path,
            manifest_csv=args.test_manifest,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=str(device),
        )

    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def build_optimizer(model, lr: float):
    params = trainable_parameters(model)
    if not params:
        raise ValueError("No trainable parameters found. Check --trainable-mode and model configuration.")
    return torch.optim.Adam(params, lr=lr)


def build_run_dir(output_dir: str | Path, model_name: str, trainable_mode: str) -> str:
    output_dir = Path(output_dir)
    run_name = f"{model_name}_{trainable_mode}"
    if output_dir.name == run_name:
        return str(output_dir)
    return str(output_dir / run_name)


def keep_frozen_modules_eval(model):
    for module in model.modules():
        if module is model:
            continue
        params = list(module.parameters(recurse=True))
        if params and not any(parameter.requires_grad for parameter in params):
            module.eval()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    keep_frozen_modules_eval(model)
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


if __name__ == "__main__":
    raise SystemExit(main())
