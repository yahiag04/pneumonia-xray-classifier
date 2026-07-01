# Pneumonia X-ray Classifier

Repository for an academic thesis project on binary chest X-ray classification:
`normal` vs `pneumonia`.

The project compares a custom CNN (`PneumoniaNet`) with common convolutional
architectures and evaluates their behavior across internal and external
datasets, including zero-shot tests and post-training fine-tuning experiments.


## What is included

- `thesis/`: reusable training, evaluation, metrics, NIH parsing and threshold
  utilities.
- `models/`: custom `PneumoniaNet` architecture.
- `scripts/`: command-line scripts for training, evaluation, fine-tuning,
  manifest preparation and plotting.
- `tests/`: unit tests for the reusable thesis code.

Generated datasets, model checkpoints and experiment outputs are intentionally
excluded from Git. They are large and/or local-machine specific.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

## Main scripts

Train a model on an ImageFolder dataset:

```bash
python scripts/train_model.py --help
```

Evaluate a checkpoint on a CSV manifest:

```bash
python scripts/evaluate_model.py --help
```

Fine-tune an existing checkpoint:

```bash
python scripts/finetune_model.py --help
```

Profile model parameters, GMAC and performance-per-GMAC summaries:

```bash
python scripts/profile_model_complexity.py --help
```

Prepare the adult RSNA branch after downloading the Kaggle competition files:

```bash
python scripts/prepare_rsna_binary.py --help
```

Prepare the 3-class RSNA branch after downloading the Kaggle competition files:

```bash
python scripts/prepare_rsna_multiclass.py --help
```

Train all adult-branch models and evaluate external manifests:

```bash
python scripts/run_adult_branch.py --help
```

Train or evaluate one model on the 3-class RSNA task:

```bash
python scripts/train_multiclass_model.py --help
python scripts/evaluate_multiclass_model.py --help
```

Select a decision threshold on validation predictions:

```bash
python scripts/select_thresholds.py --help
```

## Thesis follow-ups

- Computational profile: report parameters and GMAC for each comparison model.
- Performance/cost study: compare balanced accuracy against GMAC.
- Multiclass extension: add the third RSNA class (`No Lung Opacity / Not Normal`)
  and evaluate the task as multi-class classification.

## Reproducibility notes

The repository does not include:

- raw X-ray datasets;
- generated manifests under `outputs/`;
- trained checkpoints (`.pt`, `.pth`, `.ckpt`);
- local cache files or Python bytecode.
