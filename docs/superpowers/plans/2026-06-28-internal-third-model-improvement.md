# Internal Third Model Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible threshold tuning and controlled fine-tuning experiments for improving internal and third-dataset model performance.

**Architecture:** Reuse the existing training/evaluation stack in `thesis.train`, `thesis.threshold_sweep`, and manifest datasets. Add a generic threshold-selection script that evaluates validation probabilities once, selects thresholds by explicit criteria, then applies selected thresholds to test datasets. Extend fine-tuning with controlled trainable modes (`all`, `head`, `last_block`) while keeping outputs separate from existing baseline runs.

**Tech Stack:** Python, PyTorch, torchvision, scikit-learn metrics, `unittest`, existing repository scripts.

---

## File Structure

- Modify `thesis/threshold_sweep.py`: add threshold-selection helpers that choose best rows by metric and optional sensitivity constraint.
- Create `scripts/select_thresholds.py`: CLI for collecting validation predictions, sweeping thresholds, writing selected thresholds, and optionally evaluating test manifests with those thresholds.
- Modify `thesis/model_registry.py`: add `configure_trainable_layers(model, model_name, mode)` to support `all`, `head`, and `last_block`.
- Modify `scripts/finetune_model.py`: add `--trainable-mode`, use controlled trainable layers, and save metadata.
- Create or modify tests:
  - `tests/test_threshold_sweep.py`
  - `tests/test_model_registry.py`
  - `tests/test_finetune_config.py`
- Create `docs/thesis/model-improvement-results.md`: command log and result table template.

---

### Task 1: Add Threshold Selection Helpers

**Files:**
- Modify: `thesis/threshold_sweep.py`
- Test: `tests/test_threshold_sweep.py`

- [ ] **Step 1: Write failing tests for constrained threshold selection**

Add these tests to `tests/test_threshold_sweep.py`:

```python
from thesis.threshold_sweep import select_threshold


class ThresholdSelectionTest(unittest.TestCase):
    def test_select_threshold_maximizes_metric(self):
        rows = [
            {"model_name": "resnet18", "threshold": 0.3, "balanced_accuracy": 0.70, "sensitivity": 0.98},
            {"model_name": "resnet18", "threshold": 0.5, "balanced_accuracy": 0.78, "sensitivity": 0.94},
            {"model_name": "resnet18", "threshold": 0.7, "balanced_accuracy": 0.76, "sensitivity": 0.90},
        ]

        selected = select_threshold(rows, metric="balanced_accuracy")

        self.assertEqual(selected["threshold"], 0.5)
        self.assertAlmostEqual(selected["balanced_accuracy"], 0.78)

    def test_select_threshold_respects_minimum_sensitivity(self):
        rows = [
            {"model_name": "resnet18", "threshold": 0.3, "balanced_accuracy": 0.70, "sensitivity": 0.98},
            {"model_name": "resnet18", "threshold": 0.5, "balanced_accuracy": 0.78, "sensitivity": 0.94},
            {"model_name": "resnet18", "threshold": 0.7, "balanced_accuracy": 0.76, "sensitivity": 0.90},
        ]

        selected = select_threshold(rows, metric="balanced_accuracy", min_sensitivity=0.95)

        self.assertEqual(selected["threshold"], 0.3)
        self.assertAlmostEqual(selected["sensitivity"], 0.98)

    def test_select_threshold_rejects_unknown_metric(self):
        rows = [
            {"model_name": "resnet18", "threshold": 0.5, "balanced_accuracy": 0.78, "sensitivity": 0.94},
        ]

        with self.assertRaises(ValueError):
            select_threshold(rows, metric="missing_metric")
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_threshold_sweep -v
```

Expected: failure importing `select_threshold` from `thesis.threshold_sweep`.

- [ ] **Step 3: Implement threshold selection**

Add this function to `thesis/threshold_sweep.py` below `select_best_rows`:

```python
def select_threshold(
    rows: Sequence[dict],
    metric: str = "balanced_accuracy",
    min_sensitivity: float | None = None,
) -> dict:
    candidates = [dict(row) for row in rows]
    if min_sensitivity is not None:
        candidates = [
            row for row in candidates
            if float(row.get("sensitivity", 0.0)) >= float(min_sensitivity)
        ]
    if not candidates:
        raise ValueError("No threshold rows satisfy the selection constraints.")
    if any(metric not in row for row in candidates):
        raise ValueError(f"Metric '{metric}' is not present in every threshold row.")

    return max(
        candidates,
        key=lambda row: (float(row[metric]), -float(row["threshold"])),
    )
```

- [ ] **Step 4: Run threshold tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_threshold_sweep -v
```

Expected: all tests in `tests.test_threshold_sweep` pass.

- [ ] **Step 5: Commit**

```bash
git add thesis/threshold_sweep.py tests/test_threshold_sweep.py
git commit -m "Add threshold selection helper"
```

---

### Task 2: Add Generic Threshold Selection CLI

**Files:**
- Create: `scripts/select_thresholds.py`
- Test: `tests/test_threshold_sweep.py`

- [ ] **Step 1: Write failing unit test for reusable model-row generation**

Add this test to `tests/test_threshold_sweep.py`:

```python
from scripts.select_thresholds import summarize_threshold_selection


class ThresholdSelectionSummaryTest(unittest.TestCase):
    def test_summarize_threshold_selection_returns_selected_row_and_rows(self):
        summary = summarize_threshold_selection(
            model_name="efficientnet_b0",
            checkpoint="outputs/runs_fair/efficientnet_b0/best.pt",
            labels=[0, 0, 1, 1],
            probabilities=[0.1, 0.6, 0.7, 0.9],
            thresholds=[0.5, 0.65],
            loss=0.2,
            seconds_per_image=0.01,
            metric="balanced_accuracy",
            min_sensitivity=None,
        )

        self.assertEqual(summary["selected"]["model_name"], "efficientnet_b0")
        self.assertEqual(summary["selected"]["threshold"], 0.65)
        self.assertEqual(len(summary["rows"]), 2)
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_threshold_sweep -v
```

Expected: failure because `scripts.select_thresholds` does not exist.

- [ ] **Step 3: Create `scripts/select_thresholds.py`**

Create the file with this implementation:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis.data import ManifestImageDataset, build_internal_splits, build_transforms
from thesis.model_registry import available_models, build_model
from thesis.threshold_sweep import compute_threshold_rows, select_threshold, write_sweep_outputs
from thesis.train import choose_device, collect_predictions, evaluate_checkpoint


DEFAULT_THRESHOLDS = [round(value / 100, 2) for value in range(5, 96, 5)]


def summarize_threshold_selection(
    model_name: str,
    checkpoint: str,
    labels: Sequence[int],
    probabilities: Sequence[float],
    thresholds: Sequence[float],
    loss: float,
    seconds_per_image: float,
    metric: str,
    min_sensitivity: float | None,
) -> dict:
    rows = compute_threshold_rows(
        model_name=model_name,
        checkpoint=checkpoint,
        labels=labels,
        probabilities=probabilities,
        thresholds=thresholds,
        loss=loss,
        seconds_per_image=seconds_per_image,
    )
    selected = select_threshold(
        rows,
        metric=metric,
        min_sensitivity=min_sensitivity,
    )
    return {"selected": selected, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Select validation thresholds and evaluate selected test thresholds.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=available_models(), help="Required only for legacy checkpoints.")
    parser.add_argument("--val-manifest")
    parser.add_argument("--val-data-root")
    parser.add_argument("--test-manifest")
    parser.add_argument("--test-data-root")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--metric", default="balanced_accuracy")
    parser.add_argument("--min-sensitivity", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if bool(args.val_manifest) == bool(args.val_data_root):
        raise ValueError("Provide exactly one validation source: --val-manifest or --val-data-root.")
    if args.test_manifest and args.test_data_root:
        raise ValueError("Provide at most one test source: --test-manifest or --test-data-root.")

    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    model, model_name, image_size = _load_model(checkpoint_path, args.model, device)
    criterion = nn.BCEWithLogitsLoss()
    val_dataset = _build_dataset(
        model_name=model_name,
        image_size=image_size,
        manifest=args.val_manifest,
        data_root=args.val_data_root,
        split="val",
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    predictions = collect_predictions(model, val_loader, criterion, device)
    seconds_per_image = float(predictions["elapsed_seconds"]) / max(int(predictions["num_samples"]), 1)
    summary = summarize_threshold_selection(
        model_name=model_name,
        checkpoint=str(checkpoint_path),
        labels=predictions["labels"],
        probabilities=predictions["probabilities"],
        thresholds=args.thresholds,
        loss=predictions["loss"],
        seconds_per_image=seconds_per_image,
        metric=args.metric,
        min_sensitivity=args.min_sensitivity,
    )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "validation_source": args.val_manifest or args.val_data_root,
        "test_source": args.test_manifest or args.test_data_root,
        "thresholds": sorted({float(value) for value in args.thresholds}),
        "selection_metric": args.metric,
        "min_sensitivity": args.min_sensitivity,
        "device": str(device),
    }
    payload = {"metadata": metadata, **summary}

    if args.test_manifest or args.test_data_root:
        selected_threshold = float(summary["selected"]["threshold"])
        test_kwargs = {
            "checkpoint_path": checkpoint_path,
            "model_name": model_name,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "threshold": selected_threshold,
            "device": str(device),
        }
        if args.test_manifest:
            test_kwargs["manifest_csv"] = args.test_manifest
        else:
            test_kwargs["data_root"] = args.test_data_root
        payload["test_metrics_at_selected_threshold"] = evaluate_checkpoint(**test_kwargs)

    write_sweep_outputs(
        summary["rows"],
        json_path=args.output_json,
        csv_path=args.output_csv,
        metadata=metadata,
    )
    json_path = Path(args.output_json)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["selected"], indent=2))
    return 0


def _load_model(checkpoint_path: Path, model_name: str | None, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_meta = checkpoint if isinstance(checkpoint, dict) and "model_state" in checkpoint else {}
    state_dict = checkpoint_meta.get("model_state", checkpoint)
    resolved_model_name = model_name or checkpoint_meta.get("model_name")
    if resolved_model_name is None:
        raise ValueError("Model name is required for legacy checkpoints without metadata.")
    image_size = int(checkpoint_meta.get("image_size", 224))
    model = build_model(resolved_model_name, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, resolved_model_name, image_size


def _build_dataset(
    model_name: str,
    image_size: int,
    manifest: str | None,
    data_root: str | None,
    split: str,
):
    transform = build_transforms(model_name, image_size=image_size, train=False)
    if manifest:
        return ManifestImageDataset(manifest, transform=transform)
    splits = build_internal_splits(data_root, model_name, image_size=image_size)
    if split == "val":
        return splits.val
    if splits.test is None:
        raise ValueError("No test split found for internal evaluation.")
    return splits.test


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run threshold tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_threshold_sweep -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/select_thresholds.py tests/test_threshold_sweep.py
git commit -m "Add validation threshold selection CLI"
```

---

### Task 3: Add Controlled Trainable Layer Modes

**Files:**
- Modify: `thesis/model_registry.py`
- Test: `tests/test_model_registry.py`

- [ ] **Step 1: Write failing tests for trainable modes**

Add these tests to `tests/test_model_registry.py`:

```python
import unittest
import torch.nn as nn

from thesis.model_registry import configure_trainable_layers


class DummyEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(2, 1))


class TrainableModeTest(unittest.TestCase):
    def test_configure_trainable_layers_head_mode_only_unfreezes_classifier(self):
        model = DummyEfficientNet()

        configure_trainable_layers(model, "efficientnet_b0", "head")

        self.assertFalse(any(parameter.requires_grad for parameter in model.features.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))

    def test_configure_trainable_layers_last_block_unfreezes_last_feature_block_and_classifier(self):
        model = DummyEfficientNet()

        configure_trainable_layers(model, "efficientnet_b0", "last_block")

        self.assertFalse(any(parameter.requires_grad for parameter in model.features[0].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.features[-1].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))

    def test_configure_trainable_layers_all_mode_unfreezes_everything(self):
        model = DummyEfficientNet()

        configure_trainable_layers(model, "efficientnet_b0", "all")

        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
```

- [ ] **Step 2: Run model registry tests and confirm failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_model_registry -v
```

Expected: failure importing `configure_trainable_layers`.

- [ ] **Step 3: Implement trainable layer configuration**

Add this function to `thesis/model_registry.py` and update `freeze_backbone` to call it:

```python
def configure_trainable_layers(model: nn.Module, model_name: str, mode: str) -> nn.Module:
    model_name = model_name.lower()
    mode = mode.lower()
    if mode not in {"all", "head", "last_block"}:
        raise ValueError("Trainable mode must be one of: all, head, last_block")

    for parameter in model.parameters():
        parameter.requires_grad = mode == "all"
    if mode == "all" or model_name == "pneumonia_net":
        return model

    if model_name in {"resnet18", "resnet50"}:
        if mode == "last_block":
            _unfreeze_module(model.layer4)
        _unfreeze_module(model.fc)
    elif model_name == "densenet121":
        if mode == "last_block":
            _unfreeze_module(model.features.denseblock4)
            _unfreeze_module(model.features.norm5)
        _unfreeze_module(model.classifier)
    elif model_name in {"efficientnet_b0", "mobilenet_v3_large"}:
        if mode == "last_block":
            _unfreeze_module(model.features[-1])
        _unfreeze_module(model.classifier)
    else:
        raise ValueError(f"Cannot configure unknown model '{model_name}'")
    return model


def freeze_backbone(model: nn.Module, model_name: str) -> nn.Module:
    return configure_trainable_layers(model, model_name, "head")
```

- [ ] **Step 4: Run model registry tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_model_registry -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add thesis/model_registry.py tests/test_model_registry.py
git commit -m "Add controlled trainable layer modes"
```

---

### Task 4: Wire Trainable Modes Into Fine-Tuning

**Files:**
- Modify: `scripts/finetune_model.py`
- Test: `tests/test_finetune_config.py`

- [ ] **Step 1: Write tests for optimizer parameter filtering**

Create `tests/test_finetune_config.py`:

```python
import unittest

import torch.nn as nn

from scripts.finetune_model import trainable_parameters


class FineTuneConfigTest(unittest.TestCase):
    def test_trainable_parameters_returns_only_requires_grad_parameters(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        for parameter in model[0].parameters():
            parameter.requires_grad = False

        params = list(trainable_parameters(model))

        self.assertEqual(params, list(model[1].parameters()))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_finetune_config -v
```

Expected: failure importing `trainable_parameters`.

- [ ] **Step 3: Update `scripts/finetune_model.py` imports and arguments**

Change the import:

```python
from thesis.model_registry import available_models, build_model, configure_trainable_layers
```

Add this parser argument:

```python
parser.add_argument(
    "--trainable-mode",
    choices=("all", "head", "last_block"),
    default="all",
    help="Which layers remain trainable during fine-tuning.",
)
```

- [ ] **Step 4: Apply trainable mode and filtered optimizer**

After loading the checkpoint state into the model, replace the model setup and optimizer with:

```python
model = build_model(model_name, pretrained=False)
model.load_state_dict(state_dict)
model = configure_trainable_layers(model, model_name, args.trainable_mode)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(trainable_parameters(model), lr=args.lr)
```

Add this function near `train_epoch`:

```python
def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
```

In the checkpoint payload, add:

```python
"trainable_mode": args.trainable_mode,
```

- [ ] **Step 5: Run fine-tune config test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_finetune_config -v
```

Expected: test passes.

- [ ] **Step 6: Run related tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_model_registry tests.test_finetune_config -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/finetune_model.py tests/test_finetune_config.py
git commit -m "Support controlled fine-tuning modes"
```

---

### Task 5: Add Improvement Results Document

**Files:**
- Create: `docs/thesis/model-improvement-results.md`

- [ ] **Step 1: Create results document**

Create `docs/thesis/model-improvement-results.md`:

```markdown
# Model improvement results

Data: 2026-06-28

## Obiettivo

Questa fase misura miglioramenti su dataset interno e terzo dataset tramite:

- soglia decisionale scelta su validation;
- fine-tuning controllato con backbone congelato o ultimo blocco sbloccato.

NIH resta una valutazione separata di domain shift e non viene usato come target
di ottimizzazione in questa fase.

## Esperimenti pianificati

### Threshold tuning

Modelli:

- `efficientnet_b0`
- `resnet18`

Output:

```text
outputs/threshold_sweeps/<model>_internal_validation.json
outputs/threshold_sweeps/<model>_internal_validation.csv
outputs/threshold_sweeps/<model>_third_validation.json
outputs/threshold_sweeps/<model>_third_validation.csv
```

### Fine-tuning controllato

Modalita:

- `head`
- `last_block`

Output:

```text
outputs/runs_improved/<model>_<mode>/best.pt
outputs/evaluations_improved/<model>_<mode>_third_test.json
```

## Tabella risultati

| Esperimento | Modello | Dataset test | Soglia | TN | FP | FN | TP | Accuracy | Sensitivity | Specificity | Balanced acc. | F1 Pneumonia | ROC-AUC | PR-AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | efficientnet_b0 | Interno | 0.50 | 174 | 60 | 1 | 389 | 0.9022 | 0.9974 | 0.7436 | 0.8705 | 0.9273 | 0.9859 | n/a |
| Baseline | resnet18 | Terzo balanced | 0.50 | 143 | 91 | 1 | 233 | 0.8034 | 0.9957 | 0.6111 | 0.8034 | 0.8351 | 0.9620 | 0.9565 |

## Comandi

Comandi da compilare dopo l'implementazione:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/select_thresholds.py --help
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/finetune_model.py --help
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/thesis/model-improvement-results.md
git commit -m "Document model improvement result template"
```

---

### Task 6: Run Verification Suite

**Files:**
- No source edits.

- [ ] **Step 1: Run focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_threshold_sweep tests.test_model_registry tests.test_finetune_config -v
```

Expected: all focused tests pass.

- [ ] **Step 2: Run full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
```

Expected: all repository tests pass.

- [ ] **Step 3: Commit any missed fixes**

If a test failure required a fix in `thesis/model_registry.py` and `scripts/finetune_model.py`, commit only those touched files:

```bash
git add thesis/model_registry.py scripts/finetune_model.py
git commit -m "Fix model improvement test failures"
```

---

### Task 7: Run First Experiment Commands

**Files:**
- Output only under `outputs/threshold_sweeps`, `outputs/runs_improved`, and `outputs/evaluations_improved`.

- [ ] **Step 1: Select internal validation threshold for EfficientNet-B0**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/select_thresholds.py \
  --checkpoint outputs/runs_fair/efficientnet_b0/best.pt \
  --val-data-root /Users/yahiaghallale/Downloads/chest_xray \
  --test-data-root /Users/yahiaghallale/Downloads/chest_xray \
  --output-json outputs/threshold_sweeps/efficientnet_b0_internal_validation.json \
  --output-csv outputs/threshold_sweeps/efficientnet_b0_internal_validation.csv \
  --metric balanced_accuracy \
  --min-sensitivity 0.95
```

Expected: JSON contains `selected` and `test_metrics_at_selected_threshold`.

- [ ] **Step 2: Select third validation threshold for ResNet18 fine-tuned checkpoint**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/select_thresholds.py \
  --checkpoint outputs/runs_third_finetune/resnet18/best.pt \
  --val-manifest outputs/third_dataset/third_val_balanced.csv \
  --test-manifest outputs/third_dataset/third_test_balanced.csv \
  --output-json outputs/threshold_sweeps/resnet18_third_validation.json \
  --output-csv outputs/threshold_sweeps/resnet18_third_validation.csv \
  --metric balanced_accuracy \
  --min-sensitivity 0.95
```

Expected: JSON contains selected threshold and third test metrics.

- [ ] **Step 3: Fine-tune EfficientNet-B0 head-only**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/finetune_model.py \
  --checkpoint outputs/runs_fair/efficientnet_b0/best.pt \
  --train-manifest outputs/third_dataset/third_train_balanced.csv \
  --val-manifest outputs/third_dataset/third_val_balanced.csv \
  --test-manifest outputs/third_dataset/third_test_balanced.csv \
  --output-dir outputs/runs_improved/efficientnet_b0_head \
  --trainable-mode head \
  --epochs 10 \
  --patience 3 \
  --lr 5e-6
```

Expected: `outputs/runs_improved/efficientnet_b0_head/efficientnet_b0/best.pt` and `training_summary.json` are created.

- [ ] **Step 4: Fine-tune ResNet18 last-block**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/finetune_model.py \
  --checkpoint outputs/runs_fair/resnet18/best.pt \
  --train-manifest outputs/third_dataset/third_train_balanced.csv \
  --val-manifest outputs/third_dataset/third_val_balanced.csv \
  --test-manifest outputs/third_dataset/third_test_balanced.csv \
  --output-dir outputs/runs_improved/resnet18_last_block \
  --trainable-mode last_block \
  --epochs 10 \
  --patience 3 \
  --lr 5e-6
```

Expected: `outputs/runs_improved/resnet18_last_block/resnet18/best.pt` and `training_summary.json` are created.

- [ ] **Step 5: Update results document manually from JSON summaries**

Add rows to `docs/thesis/model-improvement-results.md` from:

```text
outputs/threshold_sweeps/efficientnet_b0_internal_validation.json
outputs/threshold_sweeps/resnet18_third_validation.json
outputs/runs_improved/efficientnet_b0_head/efficientnet_b0/training_summary.json
outputs/runs_improved/resnet18_last_block/resnet18/training_summary.json
```

- [ ] **Step 6: Commit results document only**

```bash
git add docs/thesis/model-improvement-results.md
git commit -m "Record model improvement experiment results"
```

Do not commit large checkpoint files unless the repository owner explicitly wants them tracked.
