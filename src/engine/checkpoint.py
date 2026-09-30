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
from typing import Any, Dict, Mapping, Optional

import torch


CHECKPOINT_PROVENANCE_FIELDS = (
    "run_id",
    "protocol",
    "backbone",
    "dataset_manifest_sha256",
    "source_commit",
    "effective_config_sha256",
    "fold_identity",
)


def scientific_checkpoint_provenance(
    *,
    run_id: str,
    protocol: str,
    backbone: str,
    dataset_manifest_sha256: str,
    source_commit: str,
    effective_config_sha256: str,
    fold_identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the immutable origin shared by best/last scientific checkpoints."""
    return {
        "run_id": str(run_id),
        "protocol": str(protocol),
        "backbone": str(backbone),
        "dataset_manifest_sha256": str(dataset_manifest_sha256),
        "source_commit": str(source_commit),
        "effective_config_sha256": str(effective_config_sha256),
        "fold_identity": dict(fold_identity) if fold_identity is not None else None,
    }


def validate_checkpoint_provenance(
    checkpoint: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    expected_role: str,
) -> Dict[str, Any]:
    """Fail closed unless a checkpoint has the exact scientific origin requested."""
    actual = checkpoint.get("checkpoint_provenance")
    if not isinstance(actual, dict):
        raise ValueError("Scientific checkpoint is missing required checkpoint_provenance")
    for field in CHECKPOINT_PROVENANCE_FIELDS:
        if field not in actual:
            raise ValueError(f"Checkpoint provenance missing field: {field}")
        if field not in expected:
            raise ValueError(f"Expected checkpoint provenance missing field: {field}")
        if actual[field] != expected[field]:
            raise ValueError(
                f"Checkpoint provenance mismatch at {field}: "
                f"actual={actual[field]!r}, expected={expected[field]!r}"
            )
    if actual.get("checkpoint_role") != expected_role:
        raise ValueError(
            "Checkpoint provenance mismatch at checkpoint_role: "
            f"actual={actual.get('checkpoint_role')!r}, expected={expected_role!r}"
        )
    return dict(actual)


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
        loss_gate_tolerance: float = 0.05,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.loss_gate_tolerance = loss_gate_tolerance
        self.provenance = dict(provenance) if provenance is not None else None
        if self.provenance is not None:
            missing = [
                field for field in CHECKPOINT_PROVENANCE_FIELDS
                if field not in self.provenance
            ]
            if missing:
                raise ValueError(
                    f"Checkpoint provenance missing required fields: {missing}"
                )
        self.min_val_loss: float = float("inf")
        self.best_val_loss: float = float("inf")
        self.best_epoch: Optional[int] = None

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
        scaler: Optional[Any] = None,
        early_stop_counter: int = 0,
        checkpoint_role: str = "last",
    ) -> Dict[str, Any]:
        """Build a serializable state dictionary."""
        state: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_metric": self.best_value,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "monitor": self.monitor,
            "mode": self.mode,
            "loss_gate_tolerance": self.loss_gate_tolerance,
            "min_observed_val_loss": self.min_val_loss,
            "early_stop_counter": int(early_stop_counter),
            "resume_semantics": "logical_state_continuation_rng_not_persisted",
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()
        if metrics is not None:
            state["metrics"] = metrics
        if history is not None:
            state["history"] = history
        if self.provenance is not None:
            state["checkpoint_provenance"] = {
                **self.provenance,
                "checkpoint_role": checkpoint_role,
            }
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
        scaler: Optional[Any] = None,
        early_stop_counter: int = 0,
    ) -> bool:
        """
        Save checkpoint if ``current_metric`` improves on the best so far,
        guarded by the Dual-Metric Loss Gate to prevent saving overfitted checkpoints.

        Returns:
            ``True`` if a new best was saved, ``False`` if rejected or not improved.
        """
        # Track minimum validation loss across all epochs
        if metrics is not None and "val_loss" in metrics:
            val_loss = float(metrics["val_loss"])
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss

        if self._is_best_fn(current_metric, self.best_value):
            # Dual-Metric Safeguard: If monitoring Accuracy, ensure Val Loss hasn't diverged
            if self.mode == "max" and metrics is not None and "val_loss" in metrics:
                val_loss = float(metrics["val_loss"])
                if self.loss_gate_tolerance > 0 and self.min_val_loss < float("inf"):
                    max_allowed_loss = (1.0 + self.loss_gate_tolerance) * self.min_val_loss
                    if val_loss > max_allowed_loss:
                        # Rejected by Loss-Gate Safeguard
                        return False

            self.best_value = current_metric
            self.best_epoch = epoch
            if metrics is not None and "val_loss" in metrics:
                self.best_val_loss = float(metrics["val_loss"])
            state = self._build_state(
                model, optimizer, scheduler, epoch, metrics, history,
                scaler, early_stop_counter, "best",
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
        scaler: Optional[Any] = None,
        early_stop_counter: int = 0,
    ) -> None:
        """Save the most recent checkpoint (overwritten every epoch)."""
        state = self._build_state(
            model, optimizer, scheduler, epoch, metrics, history,
            scaler, early_stop_counter, "last",
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
        """Load the exact requested best checkpoint or fail explicitly."""
        primary_path = os.path.join(self.save_dir, "best_model.pth")
        state = self._load(primary_path, device)
        if self.provenance is not None:
            validate_checkpoint_provenance(
                state, self.provenance, expected_role="best"
            )
        return state

    def load_last(self, device: str = "cpu") -> Dict[str, Any]:
        """Load the exact requested last checkpoint or fail explicitly."""
        primary_path = os.path.join(self.save_dir, "last_model.pth")
        state = self._load(primary_path, device)
        if self.provenance is not None:
            validate_checkpoint_provenance(
                state, self.provenance, expected_role="last"
            )
        return state

    @staticmethod
    def load_scientific(
        path: str,
        expected_provenance: Mapping[str, Any],
        *,
        expected_role: str,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Load an explicitly selected scientific checkpoint and verify origin."""
        state = CheckpointManager._load(path, device)
        validate_checkpoint_provenance(
            state, expected_provenance, expected_role=expected_role
        )
        return state

    def restore_tracking(self, state: Dict[str, Any]) -> None:
        """Restore checkpoint-selection state from a persisted checkpoint."""
        self.best_value = float(
            state.get("best_metric", state.get("best_value", self.best_value))
        )
        best_epoch = state.get("best_epoch", state.get("epoch"))
        self.best_epoch = int(best_epoch) if best_epoch is not None else None
        self.best_val_loss = float(state.get("best_val_loss", self.best_val_loss))
        self.min_val_loss = float(
            state.get("min_observed_val_loss", self.min_val_loss)
        )

    @staticmethod
    def _load(path: str, device: str = "cpu") -> Dict[str, Any]:
        """Load a checkpoint from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location=device, weights_only=False)
