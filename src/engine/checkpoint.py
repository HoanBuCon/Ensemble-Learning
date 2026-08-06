"""
Checkpoint Manager
==================

Saves and loads model checkpoints, optimizer/scheduler state,
training history, and metric snapshots.

Usage::

    from src.engine.checkpoint import CheckpointManager

    ckpt = CheckpointManager(save_dir="./outputs/resnet50", monitor="val_accuracy", mode="max")
    ckpt.save_if_best(state, current_metric=92.5)
    ckpt.save_last(state)
    restored = ckpt.load_best()
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """
    Manages model checkpoint persistence.

    Tracks the best monitored metric and saves/loads full training state
    including model weights, optimizer, scheduler, epoch, and metrics.

    Args:
        save_dir: Directory to store checkpoints.
        monitor: Metric name to track for "best" saving.
        mode: ``'max'`` if higher is better, ``'min'`` if lower.
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_accuracy",
        mode: str = "max",
    ) -> None:
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode

        os.makedirs(save_dir, exist_ok=True)

        if mode == "max":
            self.best_value: float = -float("inf")
        else:
            self.best_value = float("inf")

        self._is_best_fn = (
            (lambda new, old: new > old) if mode == "max"
            else (lambda new, old: new < old)
        )

    def _build_state(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        history: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a serializable state dictionary."""
        state: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_value": self.best_value,
            "monitor": self.monitor,
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if metrics is not None:
            state["metrics"] = metrics
        if history is not None:
            state["history"] = history
        return state

    def save_if_best(
        self,
        model: torch.nn.Module,
        current_metric: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        history: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save checkpoint if ``current_metric`` improves on the best so far.

        Returns:
            ``True`` if a new best was saved.
        """
        if self._is_best_fn(current_metric, self.best_value):
            self.best_value = current_metric
            state = self._build_state(
                model, optimizer, scheduler, epoch, metrics, history
            )
            path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(state, path)
            return True
        return False

    def save_last(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        history: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save the most recent checkpoint (overwritten every epoch)."""
        state = self._build_state(
            model, optimizer, scheduler, epoch, metrics, history
        )
        path = os.path.join(self.save_dir, "last_model.pth")
        torch.save(state, path)

    def save_history(self, history: Dict[str, Any]) -> None:
        """Save training history as JSON."""
        path = os.path.join(self.save_dir, "training_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save evaluation metrics as JSON."""
        path = os.path.join(self.save_dir, "metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

    def load_best(self, device: str = "cpu") -> Dict[str, Any]:
        """Load the best checkpoint."""
        return self._load(os.path.join(self.save_dir, "best_model.pth"), device)

    def load_last(self, device: str = "cpu") -> Dict[str, Any]:
        """Load the last checkpoint."""
        return self._load(os.path.join(self.save_dir, "last_model.pth"), device)

    @staticmethod
    def _load(path: str, device: str = "cpu") -> Dict[str, Any]:
        """Load a checkpoint from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location=device, weights_only=False)
