"""
Train Entry Point
=================

Simple one-liner interface for training a single model.

CLI::

    python train.py configs/resnet50.yaml

Notebook::

    from train import train
    train("configs/resnet50.yaml")
"""

from __future__ import annotations

import sys
from typing import Any, Dict

from src.engine.trainer import Trainer
from src.utils.config import load_config


def train(config_path: str) -> Dict[str, Any]:
    """
    Train a model using the specified YAML configuration.

    This is the primary entry point for both CLI and notebook usage.
    After training, automatically evaluates on the test set (if available)
    and saves all artifacts (checkpoints, history, predictions, plots).

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary with training results (best accuracy, timing, etc.).
    """
    config = load_config(config_path)
    trainer = Trainer(config)

    # Train
    results = trainer.train()

    # Evaluate with best model
    trainer.load_best_model()
    metrics = trainer.evaluate()

    # Save probability cache for ensemble
    trainer.save_predictions()

    results["eval_metrics"] = metrics
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_path>")
        print("Example: python train.py configs/resnet50.yaml")
        sys.exit(1)

    train(sys.argv[1])
