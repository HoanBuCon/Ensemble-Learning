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
from src.utils.provenance import require_git_commit, verify_dataset_snapshot, write_experiment_manifest
from src.utils.run_identity import model_run_root, validate_run_id


def train(
    config_path: str,
    resume: bool = False,
    *,
    run_id: str,
    results_root: str = "RESULTS",
) -> Dict[str, Any]:
    """
    Train a model using the specified YAML configuration.

    Args:
        config_path: Path to the YAML configuration file.
        resume: If True, resumes training from last_model.pth if found.

    Returns:
        Dictionary with training results (best accuracy, timing, etc.).
    """
    require_git_commit()
    verify_dataset_snapshot()
    validated_run_id = validate_run_id(run_id)
    from src.utils.config import load_config
    config_preview = load_config(config_path)
    exact_output = model_run_root(
        results_root, validated_run_id, "single_split", config_preview.model.name
    )
    trainer = Trainer(
        config_path,
        resume=resume,
        run_id=validated_run_id,
        exact_save_dir=str(exact_output),
    )
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
        run_id=validated_run_id,
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", default="RESULTS")

    args = parser.parse_args()
    train(
        args.config,
        resume=args.resume,
        run_id=args.run_id,
        results_root=args.results_root,
    )
