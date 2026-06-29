from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from thesis.data import ManifestImageDataset, build_transforms
from thesis.model_registry import build_model
from thesis.stat_tests import (
    accuracy,
    balanced_accuracy,
    bootstrap_metric_ci,
    delong_roc_test,
    holm_bonferroni,
    mcnemar_exact,
    paired_correctness,
    roc_auc,
)
from thesis.train import choose_device, collect_predictions


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected name=/path format")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Name cannot be empty")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired statistical comparisons between trained CXR models.")
    parser.add_argument("--checkpoint", action="append", type=parse_named_path, required=True)
    parser.add_argument("--manifest", action="append", type=parse_named_path, required=True)
    parser.add_argument("--output-dir", default="outputs/statistical_tests")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_meta = checkpoint if isinstance(checkpoint, dict) and "model_state" in checkpoint else {}
    state_dict = checkpoint_meta.get("model_state", checkpoint)
    model_name = checkpoint_meta.get("model_name", "pneumonia_net")
    image_size = int(checkpoint_meta.get("image_size", 224))
    threshold = float(checkpoint_meta.get("threshold", 0.5))

    model = build_model(model_name, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_name, image_size, threshold


def predict_checkpoint(
    checkpoint_path: Path,
    manifest_csv: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict:
    model, model_name, image_size, threshold = load_checkpoint(checkpoint_path, device)
    dataset = ManifestImageDataset(manifest_csv, transform=build_transforms(model_name, image_size=image_size, train=False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    predictions = collect_predictions(model, loader, nn.BCEWithLogitsLoss(), device)
    predictions["model_name"] = model_name
    predictions["threshold"] = threshold
    return predictions


def model_summary(dataset_name: str, model_alias: str, predictions: dict, n_bootstraps: int, seed: int) -> dict:
    y_true = np.asarray(predictions["labels"], dtype=int)
    y_score = np.asarray(predictions["probabilities"], dtype=float)
    threshold = float(predictions["threshold"])
    y_pred = (y_score >= threshold).astype(int)
    row = {
        "dataset": dataset_name,
        "model": model_alias,
        "model_name": predictions["model_name"],
        "num_samples": len(y_true),
        "threshold": threshold,
        "accuracy": accuracy(y_true, y_score, threshold),
        "balanced_accuracy": balanced_accuracy(y_true, y_score, threshold),
        "roc_auc": roc_auc(y_true, y_score),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }
    metric_fns = {
        "accuracy": lambda labels, scores: accuracy(labels, scores, threshold),
        "balanced_accuracy": lambda labels, scores: balanced_accuracy(labels, scores, threshold),
        "roc_auc": roc_auc,
    }
    for metric_name, metric_fn in metric_fns.items():
        ci = bootstrap_metric_ci(y_true, y_score, metric_fn, n_bootstraps=n_bootstraps, seed=seed)
        row[f"{metric_name}_ci_low"] = ci["ci_low"]
        row[f"{metric_name}_ci_high"] = ci["ci_high"]
    return row


def compare_against_reference(dataset_name: str, reference: str, comparison: str, predictions: dict) -> dict:
    ref = predictions[reference]
    other = predictions[comparison]
    y_true = np.asarray(ref["labels"], dtype=int)
    ref_scores = np.asarray(ref["probabilities"], dtype=float)
    other_scores = np.asarray(other["probabilities"], dtype=float)
    ref_pred = (ref_scores >= float(ref["threshold"])).astype(int)
    other_pred = (other_scores >= float(other["threshold"])).astype(int)

    counts = paired_correctness(y_true, ref_pred, other_pred)
    mcnemar = mcnemar_exact(counts["a_correct_b_wrong"], counts["a_wrong_b_correct"])
    delong = delong_roc_test(y_true, ref_scores, other_scores)
    return {
        "dataset": dataset_name,
        "reference_model": reference,
        "comparison_model": comparison,
        **counts,
        "mcnemar_discordant": mcnemar["discordant"],
        "mcnemar_p_value": mcnemar["p_value"],
        "reference_auc": delong["auc_a"],
        "comparison_auc": delong["auc_b"],
        "auc_difference": delong["auc_difference"],
        "delong_z_score": delong["z_score"],
        "delong_p_value": delong["p_value"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_holm_adjusted_p_values(rows: list[dict]) -> list[dict]:
    for dataset_name in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
        mcnemar_adjusted = holm_bonferroni(row["mcnemar_p_value"] for row in dataset_rows)
        delong_adjusted = holm_bonferroni(row["delong_p_value"] for row in dataset_rows)
        for row, mcnemar_p, delong_p in zip(dataset_rows, mcnemar_adjusted, delong_adjusted):
            row["mcnemar_holm_p_value"] = mcnemar_p
            row["delong_holm_p_value"] = delong_p
    return rows


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    comparisons = []
    for dataset_name, manifest_csv in args.manifest:
        dataset_predictions = {}
        for model_alias, checkpoint_path in args.checkpoint:
            print(f"predicting dataset={dataset_name} model={model_alias}", flush=True)
            dataset_predictions[model_alias] = predict_checkpoint(
                checkpoint_path,
                manifest_csv,
                args.batch_size,
                args.num_workers,
                device,
            )

        dataset_summaries = [
            model_summary(dataset_name, model_alias, predictions, args.n_bootstraps, args.seed)
            for model_alias, predictions in dataset_predictions.items()
        ]
        reference = max(dataset_summaries, key=lambda row: row["balanced_accuracy"])["model"]
        summaries.extend(dataset_summaries)

        for model_alias in dataset_predictions:
            if model_alias == reference:
                continue
            comparisons.append(compare_against_reference(dataset_name, reference, model_alias, dataset_predictions))

    add_holm_adjusted_p_values(comparisons)
    write_csv(output_dir / "model_metric_bootstrap_ci.csv", summaries)
    write_csv(output_dir / "pairwise_best_model_tests.csv", comparisons)
    (output_dir / "model_metric_bootstrap_ci.json").write_text(json.dumps(summaries, indent=2))
    (output_dir / "pairwise_best_model_tests.json").write_text(json.dumps(comparisons, indent=2))
    print(json.dumps({"output_dir": str(output_dir), "models": len(args.checkpoint), "datasets": len(args.manifest)}, indent=2))


if __name__ == "__main__":
    main()
