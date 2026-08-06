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


def seed_worker(worker_id: int) -> None:
    """
    Worker init function for deterministic ``DataLoader`` multiprocessing.

    Pass this as ``worker_init_fn`` to ``DataLoader``.

    Args:
        worker_id: Worker index (provided by DataLoader).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
