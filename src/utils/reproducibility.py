"""
Reproducibility Utilities
=========================

Deterministic training: seed fixing, deterministic CUDA ops,
and reproducible DataLoader workers.

Usage::

    from src.utils.reproducibility import set_seed, seed_worker, get_generator

    set_seed(42)
    loader = DataLoader(
        dataset,
        worker_init_fn=seed_worker,
        generator=get_generator(42),
    )
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import get_worker_info


def set_seed(seed: int) -> None:
    """
    Fix all random seeds for reproducible experiments.

    Sets seeds for: ``random``, ``numpy``, ``torch`` (CPU & CUDA),
    and enables deterministic CuDNN behavior.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch >= 1.8 deterministic algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # Older PyTorch versions


def seed_dataset_transform(dataset: object, seed: int) -> bool:
    """Seed the first Albumentations-style transform owned by a dataset tree."""
    current = dataset
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        transform = getattr(current, "transform", None)
        if transform is not None and hasattr(transform, "set_random_seed"):
            transform.set_random_seed(int(seed))
            return True
        current = getattr(current, "dataset", None)
    return False


def worker_seed_from_torch_initial_seed(initial_seed: int) -> int:
    """Map PyTorch's run/fold/worker seed to the 32-bit augmentation seed."""
    return int(initial_seed % 2**32)


def fold_seed(experiment_seed: int, fold_index: int) -> int:
    """Derive the zero-based fold stream seed used by model and loader RNGs."""
    if fold_index < 0:
        raise ValueError("fold_index must be non-negative")
    return int(experiment_seed) + int(fold_index)


def seed_worker(worker_id: int) -> None:
    """
    Worker init function for deterministic ``DataLoader`` multiprocessing.

    Pass this as ``worker_init_fn`` to ``DataLoader``.

    Args:
        worker_id: Worker index (provided by DataLoader).
    """
    # ``torch.initial_seed`` is deterministically derived from the DataLoader
    # generator seed and worker id.  In OOF that generator is seeded with
    # experiment_seed + fold_index, so the complete owner chain is
    # run/fold -> DataLoader -> worker -> Albumentations.
    worker_seed = worker_seed_from_torch_initial_seed(torch.initial_seed())
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    worker_info = get_worker_info()
    if worker_info is not None:
        seed_dataset_transform(worker_info.dataset, worker_seed)


def get_generator(seed: int) -> torch.Generator:
    """
    Create a seeded :class:`torch.Generator` for ``DataLoader`` shuffling.

    Args:
        seed: Integer seed value.

    Returns:
        A seeded :class:`torch.Generator`.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
