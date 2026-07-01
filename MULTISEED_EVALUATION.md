# Multi-Seed Evaluation

Run repeated binary CXR training/evaluation without replacing the existing
single-seed outputs:

```bash
.venv/bin/python scripts/run_multiseed_evaluation.py \
  --data-root outputs/rsna_adult_binary \
  --output-dir outputs/multiseed \
  --seed 42 \
  --seed 43 \
  --seed 44 \
  --manifest kermany=outputs/kermany/kermany_test_manifest.csv \
  --manifest chittagong=outputs/chittagong/chittagong_testing_manifest.csv
```

The script writes one checkpoint tree per seed under `runs/seed_<seed>/`, one
evaluation directory per seed under `evaluations/seed_<seed>/`, a per-seed table
in `per_seed.csv`/`per_seed.json`, and aggregate `mean`, sample `std`, and
normal-approximation `ci95` rows in `aggregate_metrics.csv`/`aggregate_metrics.json`.

Seeding covers Python `random`, NumPy, PyTorch CPU RNG, and CUDA RNGs when CUDA
is available. Passing `--deterministic` also requests deterministic PyTorch
backend flags where supported.

This does not guarantee bitwise reproducibility across all machines or backends.
CUDA/cuDNN kernels, MPS kernels, unsupported deterministic operations, driver
versions, floating-point reduction order, and data-loader worker scheduling can
still affect exact trajectories. The multi-seed outputs are intended to measure
run-to-run variability, not to prove bitwise determinism.
