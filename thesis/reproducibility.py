from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedReport:
    seed: int
    deterministic: bool
    cuda_available: bool
    notes: list[str]


def set_global_seed(seed: int, deterministic: bool = False) -> SeedReport:
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    notes = [
        "random, numpy, torch, and CUDA RNGs are seeded when available.",
        "Bitwise reproducibility is not guaranteed across devices, drivers, or nondeterministic backend kernels.",
    ]
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        notes.append(
            "Deterministic backend flags were requested; unsupported operations may still warn or vary by backend."
        )
    else:
        notes.append("Deterministic backend algorithms were not forced.")

    return SeedReport(
        seed=seed,
        deterministic=deterministic,
        cuda_available=cuda_available,
        notes=notes,
    )
