# NIH Threshold Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate all five third-dataset fine-tuned checkpoints on NIH once per model and calculate metrics at thresholds 0.50, 0.60, 0.65, and 0.70.

**Architecture:** Extract reusable label/probability collection from the existing evaluation loop, then add a focused threshold-sweep module that converts one prediction set into metric rows and writes JSON/CSV outputs. A CLI loads each checkpoint once, runs one NIH inference, applies all thresholds offline, and prints a concise comparison.

**Tech Stack:** Python 3, PyTorch, NumPy, scikit-learn, `unittest`, JSON, CSV.

---

### Task 1: Reusable Prediction Collection

**Files:**
- Modify: `thesis/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing prediction-collection test**

Create `tests/test_train.py` with a deterministic model and loader. Assert that
`collect_predictions` returns the expected labels, sigmoid probabilities, mean
loss, sample count, and non-negative elapsed time.

```python
class IdentityLogitModel(nn.Module):
    def forward(self, x):
        return x[:, :1]


loader = DataLoader(
    TensorDataset(
        torch.tensor([[-2.0], [0.0], [2.0]]),
        torch.tensor([0, 1, 1]),
    ),
    batch_size=2,
)
result = collect_predictions(
    IdentityLogitModel(),
    loader,
    nn.BCEWithLogitsLoss(),
    torch.device("cpu"),
)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_train -v
```

Expected: import failure because `collect_predictions` does not exist.

- [ ] **Step 3: Implement collection and preserve evaluation behavior**

Add `collect_predictions(model, loader, criterion, device)` to
`thesis/train.py`. It must run the current inference loop and return:

```python
{
    "labels": list[int],
    "probabilities": list[float],
    "loss": float,
    "num_samples": int,
    "elapsed_seconds": float,
}
```

Refactor `evaluate_loader` to call this function, pass labels and probabilities
to `compute_binary_metrics`, and calculate `seconds_per_image` from the returned
elapsed time. Existing result keys and values must remain compatible.

- [ ] **Step 4: Run focused and regression tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_train tests.test_metrics -v
```

Expected: all tests pass.

### Task 2: Threshold Sweep Logic and Serialization

**Files:**
- Create: `thesis/threshold_sweep.py`
- Create: `tests/test_threshold_sweep.py`

- [ ] **Step 1: Write failing sweep tests**

Test `compute_threshold_rows` with known labels and probabilities at thresholds
`[0.5, 0.6, 0.7]`. Assert:

- one row per threshold;
- model metadata is present;
- predicted positives `tp + fp` never increase as threshold rises;
- the best threshold is selected by maximum balanced accuracy, breaking ties
  toward the lower threshold.

Test `write_sweep_outputs` in a temporary directory and assert that JSON and CSV
contain all supplied model-threshold rows and expected metric columns.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_threshold_sweep -v
```

Expected: import failure because `thesis.threshold_sweep` does not exist.

- [ ] **Step 3: Implement the sweep module**

Implement:

```python
def compute_threshold_rows(
    model_name: str,
    checkpoint: str,
    labels: Sequence[int],
    probabilities: Sequence[float],
    thresholds: Sequence[float],
    loss: float,
    seconds_per_image: float,
) -> list[dict]:
    ...


def select_best_rows(rows: Sequence[dict]) -> dict[str, dict]:
    ...


def write_sweep_outputs(
    rows: Sequence[dict],
    json_path: str | Path,
    csv_path: str | Path,
    metadata: dict,
) -> None:
    ...
```

Validate thresholds are within `[0, 1]`, sort them numerically, reject an empty
threshold list, and write deterministic JSON/CSV field ordering.

- [ ] **Step 4: Run focused and complete unit tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_threshold_sweep -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
```

Expected: all tests pass.

### Task 3: Five-Model NIH CLI

**Files:**
- Create: `scripts/sweep_nih_thresholds.py`
- Test: `tests/test_threshold_sweep.py`

- [ ] **Step 1: Add a failing orchestration test**

Test a helper that receives a model name, mocked prediction collector output,
and thresholds, then returns four rows without invoking prediction collection
more than once. Use a counting callable rather than patching PyTorch internals.

- [ ] **Step 2: Run the orchestration test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_threshold_sweep -v
```

Expected: failure because the orchestration helper does not exist.

- [ ] **Step 3: Implement the CLI**

The CLI must support:

```text
--manifest
--checkpoint-dir
--output-json
--output-csv
--thresholds
--models
--batch-size
--num-workers
--device
```

Defaults:

```text
manifest=outputs/nih/nih_224_binary_manifest.csv
checkpoint-dir=outputs/runs_third_finetune
output-json=outputs/evaluations/nih_threshold_sweep_after_third_ft.json
output-csv=outputs/evaluations/nih_threshold_sweep_after_third_ft.csv
thresholds=0.50 0.60 0.65 0.70
models=all five supported thesis models
```

For each model, load the checkpoint and NIH dataset once, call
`collect_predictions` once, compute all threshold rows, save final outputs after
each completed model for recoverability, and print balanced accuracy,
sensitivity, and specificity for each threshold.

- [ ] **Step 4: Run tests and CLI help**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/sweep_nih_thresholds.py --help
```

Expected: tests pass and help lists all arguments.

### Task 4: Execute NIH Sweep and Document Results

**Files:**
- Create: `outputs/evaluations/nih_threshold_sweep_after_third_ft.json`
- Create: `outputs/evaluations/nih_threshold_sweep_after_third_ft.csv`
- Modify: `docs/thesis/third-dataset-results.md`

- [ ] **Step 1: Run all five models**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/sweep_nih_thresholds.py \
  --batch-size 32 \
  --num-workers 0
```

Expected: 20 rows, one for each of five models and four thresholds.

- [ ] **Step 2: Validate generated artifacts**

Run:

```bash
.venv/bin/python -c "import csv,json; from pathlib import Path; j=json.loads(Path('outputs/evaluations/nih_threshold_sweep_after_third_ft.json').read_text()); r=list(csv.DictReader(Path('outputs/evaluations/nih_threshold_sweep_after_third_ft.csv').open())); assert len(j['rows']) == 20; assert len(r) == 20; assert len({x['model_name'] for x in r}) == 5; assert {float(x['threshold']) for x in r} == {0.5,0.6,0.65,0.7}; print('validated 20 rows')"
```

Expected: `validated 20 rows`.

- [ ] **Step 3: Document the exploratory comparison**

Append a threshold-sweep section to `docs/thesis/third-dataset-results.md` with
the generated metrics table, per-model best balanced-accuracy threshold, and the
methodological warning that selecting a threshold on the NIH test manifest is
exploratory and not a leakage-free final estimate.

- [ ] **Step 4: Run final verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
git diff --check
```

Expected: all tests pass and `git diff --check` exits successfully.
