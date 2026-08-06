"""
Train Script
============

Single model training script.

CLI::

    python scripts/train.py configs/resnet50.yaml
    python main.py train configs/resnet50.yaml
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.engine.trainer import Trainer
from src.utils.config import load_config


def train(config_path: str, resume: bool = False) -> Dict[str, Any]:
    """
    Train a model using the specified YAML configuration.

    Args:
        config_path: Path to the YAML configuration file.
        resume: If True, resumes training from last_model.pth if found.

    Returns:
        Dictionary with training results (best accuracy, timing, etc.).
    """
    config = load_config(config_path)
    trainer = Trainer(config, resume=resume)

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
    import argparse

    parser = argparse.ArgumentParser(description="Train a single model")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last_model.pth if available",
    )

    args = parser.parse_args()
    train(args.config, resume=args.resume)
