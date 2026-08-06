"""
Evaluate Entry Point
====================

Standalone evaluation for a trained model.

CLI::

    python evaluate.py configs/resnet50.yaml
    python evaluate.py configs/resnet50.yaml --checkpoint outputs/resnet50/best_model.pth

Notebook::

    from evaluate import evaluate
    metrics = evaluate("configs/resnet50.yaml")
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional

import torch

from src.datasets.dataset import create_dataloaders
from src.engine.checkpoint import CheckpointManager
from src.engine.evaluator import evaluate_model
from src.models.factory import create_model
from src.utils.config import load_config
from src.utils.reproducibility import set_seed


def evaluate(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    split: str = "test",
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
    config = load_config(config_path)
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
        ckpt_manager = CheckpointManager(save_dir=config.checkpoint.save_dir)
        ckpt = ckpt_manager.load_best(device=str(device))
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

    if split == "test" and test_loader is not None:
        dataloader = test_loader
    else:
        dataloader = val_loader

    # Evaluate
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        class_names=class_names,
        output_dir=config.checkpoint.save_dir,
        device=str(device),
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

    args = parser.parse_args()
    evaluate(args.config, checkpoint_path=args.checkpoint, split=args.split)
