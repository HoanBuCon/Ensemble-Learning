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
from src.utils.provenance import verify_dataset_snapshot, write_experiment_manifest


def train(config_path: str, resume: bool = False) -> Dict[str, Any]:
    """
    Train a model using the specified YAML configuration.

    Args:
        config_path: Path to the YAML configuration file.
        resume: If True, resumes training from last_model.pth if found.

    Returns:
        Dictionary with training results (best accuracy, timing, etc.).
    """
    verify_dataset_snapshot()
    trainer = Trainer(config_path, resume=resume)
    config = trainer.config

    # Train
    results = trainer.train()

    # Evaluate with best model
    trainer.load_best_model()
    val_metrics = trainer.evaluate(dataloader=trainer.val_loader)
    if trainer.test_loader is None:
        raise FileNotFoundError("Frozen dataset test split is required for final evaluation")
    test_metrics = trainer.evaluate(dataloader=trainer.test_loader)

    results["val_metrics"] = val_metrics
    results["eval_metrics"] = test_metrics
    results["save_dir"] = trainer.save_dir
    write_experiment_manifest(
        os.path.join(trainer.save_dir, "experiment_manifest.json"),
        protocol="single_split",
        model=config.model.name,
        config_path=config_path,
        seed=config.seed,
        checkpoint_path=os.path.join(trainer.save_dir, "best_model.pth"),
        prediction_path=os.path.join(trainer.save_dir, "test_predictions.npz"),
        arguments={"resume": resume},
    )
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
