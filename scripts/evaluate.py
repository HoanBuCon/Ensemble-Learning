"""
Evaluate Script
===============

Standalone evaluation script for a trained model checkpoint.

CLI::

    python scripts/evaluate.py configs/resnet50.yaml
    python main.py evaluate configs/resnet50.yaml --split test
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from src.datasets.dataset import create_dataloaders
from src.engine.checkpoint import CheckpointManager
from src.engine.evaluator import evaluate_model
from src.models.factory import create_model
from src.utils.config import load_config
from src.utils.reproducibility import set_seed
from src.utils.provenance import load_dataset_manifest_sha256, require_git_commit, sha256_file
from src.utils.run_identity import model_run_root, validate_run_id


def select_evaluation_loader(split: str, val_loader, test_loader):
    """Select exactly the requested split; cross-split fallback is forbidden."""
    if split == "test":
        if test_loader is None:
            raise FileNotFoundError("Test split requested but no test loader exists")
        return test_loader
    if split == "val":
        if val_loader is None:
            raise FileNotFoundError("Validation split requested but no validation loader exists")
        return val_loader
    raise ValueError("split must be exactly 'val' or 'test'")


def evaluate(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    split: str = "test",
    *,
    run_id: str,
    results_root: str = "RESULTS",
) -> Dict[str, Any]:
    """
    Evaluate a trained model and generate all reports.

    Args:
        config_path: Path to the YAML configuration file.
        checkpoint_path: Path to a specific checkpoint. If ``None``,
                         loads the best checkpoint from the config's save_dir.
        split: Dataset split to evaluate on (``'test'`` or ``'val'``).

    Returns:
        Metrics dictionary.
    """
    require_git_commit()
    config = load_config(config_path)
    run_id = validate_run_id(run_id)
    output_dir = str(model_run_root(results_root, run_id, "single_split", config.model.name))
    set_seed(config.seed)

    device = torch.device(config.device)

    # Create model
    model = create_model(
        model_name=config.model.name,
        pretrained=False,  # We'll load weights from checkpoint
        num_classes=config.model.num_classes,
    )

    # Load checkpoint
    if checkpoint_path is None:
        ckpt_manager = CheckpointManager(save_dir=output_dir)
        ckpt = ckpt_manager.load_best(device=str(device))
        checkpoint_path = os.path.join(output_dir, "best_model.pth")
    else:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    print(
        f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
        f"(best {ckpt.get('monitor', 'metric')}: {ckpt.get('best_value', '?')})"
    )

    # Create dataloaders
    _, val_loader, test_loader, class_names = create_dataloaders(config)

    dataloader = select_evaluation_loader(split, val_loader, test_loader)

    # Evaluate
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        class_names=class_names,
        output_dir=output_dir,
        device=str(device),
        split=split,
        protocol="single_split",
        method=config.model.name,
        run_id=run_id,
        dataset_manifest_sha256=load_dataset_manifest_sha256(),
        config_sha256=sha256_file(config_path),
        source_commit=require_git_commit(),
        artifact_hashes={"checkpoint": sha256_file(checkpoint_path)},
        aggregation_semantics="single_checkpoint_inference",
    )

    print(f"\n{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"{'='*50}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", default="RESULTS")

    args = parser.parse_args()
    evaluate(
        args.config, checkpoint_path=args.checkpoint, split=args.split,
        run_id=args.run_id, results_root=args.results_root,
    )
