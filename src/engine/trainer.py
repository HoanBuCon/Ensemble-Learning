"""
Generic Trainer
===============

One Trainer to rule them all. Works for every backbone registered in
the ModelFactory — ResNet, DenseNet, EfficientNet, Swin, or any future
model. The Trainer never inspects what model it is training.

Features:
    - Training loop with progress bars
    - Validation after each epoch
    - Mixed precision (AMP)
    - Gradient clipping
    - Gradient accumulation
    - Learning rate scheduling with warmup
    - Early stopping
    - Checkpoint saving (best + last)
    - Console logging (Rich tables)
    - CSV logging
    - TensorBoard logging
    - Training history persistence

Usage::

    from src.engine.trainer import Trainer
    from src.utils.config import load_config

    config = load_config("configs/resnet50.yaml")
    trainer = Trainer(config)
    results = trainer.train()
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.dataset import create_dataloaders
from src.engine.checkpoint import CheckpointManager
from src.engine.evaluator import evaluate_model, run_inference, save_predictions
from src.models.factory import create_model
from src.utils.config import ExperimentConfig, load_config
from src.utils.logger import (
    CSVLogger,
    log_epoch,
    log_training_end,
    log_training_start,
    setup_logger,
)
from src.utils.reproducibility import set_seed
from src.utils.visualization import plot_training_curves


class Trainer:
    """
    Generic, config-driven trainer for any registered backbone.

    The Trainer is fully configured by an :class:`ExperimentConfig`.
    It creates the model, dataloaders, optimizer, scheduler, and loss
    function internally — callers only provide a config path or object.

    Args:
        config: Either an :class:`ExperimentConfig` or a path to a YAML file.
    """

    def __init__(self, config: ExperimentConfig | str) -> None:
        if isinstance(config, str):
            config = load_config(config)

        self.config = config
        self.device = torch.device(config.device)

        # Reproducibility
        set_seed(config.seed)

        # Logger
        self.logger = setup_logger(
            config.experiment_name,
            log_dir=config.checkpoint.save_dir if config.logging.console else None,
        )

        # Model
        self.model = create_model(
            model_name=config.model.name,
            pretrained=config.model.pretrained,
            num_classes=config.model.num_classes,
            drop_rate=config.model.drop_rate,
        ).to(self.device)

        # Data
        self.train_loader, self.val_loader, self.test_loader, self.class_names = (
            create_dataloaders(config)
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.train.label_smoothing
        )

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Mixed precision
        self.use_amp = config.train.mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Checkpoint manager
        self.ckpt_manager = CheckpointManager(
            save_dir=config.checkpoint.save_dir,
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
        )

        # CSV logger
        self.csv_logger: Optional[CSVLogger] = None
        if config.logging.csv:
            self.csv_logger = CSVLogger(
                filepath=os.path.join(config.checkpoint.save_dir, "history.csv"),
                fieldnames=[
                    "epoch", "train_loss", "train_accuracy",
                    "val_loss", "val_accuracy", "lr", "elapsed",
                ],
            )

        # TensorBoard
        self.tb_writer = None
        if config.logging.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(
                log_dir=os.path.join(config.checkpoint.save_dir, "tensorboard")
            )

        # History
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "lr": [],
        }

        # Early stopping state
        self._early_stop_counter = 0
        self._best_metric = (
            -float("inf") if config.checkpoint.mode == "max" else float("inf")
        )

        # Model info
        self.num_params = sum(p.numel() for p in self.model.parameters())

    # ================================================================
    # Optimizer & Scheduler Factories
    # ================================================================

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        cfg = self.config.train
        params = self.model.parameters()

        optimizers = {
            "adam": lambda: torch.optim.Adam(
                params, lr=cfg.lr, weight_decay=cfg.weight_decay
            ),
            "adamw": lambda: torch.optim.AdamW(
                params, lr=cfg.lr, weight_decay=cfg.weight_decay
            ),
            "sgd": lambda: torch.optim.SGD(
                params, lr=cfg.lr, momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            ),
        }

        name = cfg.optimizer.lower()
        if name not in optimizers:
            raise ValueError(
                f"Unknown optimizer '{name}'. "
                f"Available: {list(optimizers.keys())}"
            )
        return optimizers[name]()

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Build learning rate scheduler from config."""
        cfg = self.config.train

        schedulers = {
            "cosine": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-7,
            ),
            "step": lambda: torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=cfg.step_size, gamma=cfg.step_gamma,
            ),
            "plateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=self.config.checkpoint.mode,
                factor=cfg.plateau_factor, patience=cfg.plateau_patience,
            ),
            "onecycle": lambda: torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=cfg.lr,
                steps_per_epoch=len(self.train_loader),
                epochs=cfg.epochs,
            ),
            "none": lambda: None,
        }

        name = cfg.scheduler.lower()
        if name not in schedulers:
            raise ValueError(
                f"Unknown scheduler '{name}'. "
                f"Available: {list(schedulers.keys())}"
            )
        return schedulers[name]()

    # ================================================================
    # Training Loop
    # ================================================================

    def train(self) -> Dict[str, Any]:
        """
        Execute the full training loop.

        Returns:
            Dictionary containing final metrics, history, and timing info.
        """
        cfg = self.config
        total_epochs = cfg.train.epochs

        log_training_start(
            experiment_name=cfg.experiment_name,
            model_name=cfg.model.name,
            num_params=self.num_params,
            device=str(self.device),
            epochs=total_epochs,
            batch_size=cfg.data.batch_size,
        )

        global_start = time.time()
        best_epoch = 0

        for epoch in range(1, total_epochs + 1):
            epoch_start = time.time()

            # --- Learning rate warmup ---
            if epoch <= cfg.train.warmup_epochs:
                warmup_lr = cfg.train.lr * (epoch / cfg.train.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            # --- Train one epoch ---
            train_loss, train_acc = self._train_one_epoch(epoch, total_epochs)

            # --- Validate ---
            val_loss, val_acc = self._validate()

            # --- Current LR ---
            current_lr = self.optimizer.param_groups[0]["lr"]

            # --- Record history ---
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["lr"].append(current_lr)

            # --- Scheduler step ---
            if self.scheduler is not None and epoch > cfg.train.warmup_epochs:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    monitor_val = (
                        val_acc if cfg.checkpoint.monitor == "val_accuracy"
                        else val_loss
                    )
                    self.scheduler.step(monitor_val)
                elif not isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.OneCycleLR,
                ):
                    self.scheduler.step()

            # --- Timing ---
            epoch_time = time.time() - epoch_start
            elapsed_total = time.time() - global_start
            eta = epoch_time * (total_epochs - epoch)

            elapsed_str = self._format_time(elapsed_total)
            eta_str = self._format_time(eta)

            # --- Console logging ---
            if cfg.logging.console:
                log_epoch(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    lr=current_lr,
                    elapsed=elapsed_str,
                    eta=eta_str,
                )

            # --- CSV logging ---
            if self.csv_logger is not None:
                self.csv_logger.log({
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_accuracy": f"{train_acc:.4f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_accuracy": f"{val_acc:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "elapsed": elapsed_str,
                })

            # --- TensorBoard ---
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("Loss/train", train_loss, epoch)
                self.tb_writer.add_scalar("Loss/val", val_loss, epoch)
                self.tb_writer.add_scalar("Accuracy/train", train_acc, epoch)
                self.tb_writer.add_scalar("Accuracy/val", val_acc, epoch)
                self.tb_writer.add_scalar("LR", current_lr, epoch)

            # --- Checkpointing ---
            current_metric = (
                val_acc if cfg.checkpoint.monitor == "val_accuracy" else val_loss
            )
            metrics_snapshot = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }

            if cfg.checkpoint.save_best:
                is_best = self.ckpt_manager.save_if_best(
                    model=self.model,
                    current_metric=current_metric,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=metrics_snapshot,
                    history=self.history,
                )
                if is_best:
                    best_epoch = epoch
                    self.logger.info(
                        f"New best {cfg.checkpoint.monitor}: {current_metric:.4f} "
                        f"(epoch {epoch})"
                    )

            if cfg.checkpoint.save_last:
                self.ckpt_manager.save_last(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=metrics_snapshot,
                    history=self.history,
                )

            # --- Early stopping ---
            if cfg.checkpoint.mode == "max":
                improved = current_metric > self._best_metric
            else:
                improved = current_metric < self._best_metric

            if improved:
                self._best_metric = current_metric
                self._early_stop_counter = 0
            else:
                self._early_stop_counter += 1
                if self._early_stop_counter >= cfg.train.early_stopping_patience:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(patience={cfg.train.early_stopping_patience})"
                    )
                    break

        # --- Post-training ---
        total_time = time.time() - global_start
        total_time_str = self._format_time(total_time)

        # Save history
        self.ckpt_manager.save_history(self.history)
        plot_training_curves(self.history, cfg.checkpoint.save_dir)

        # Close TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.close()

        best_val_acc = max(self.history["val_accuracy"])
        log_training_end(
            experiment_name=cfg.experiment_name,
            best_accuracy=best_val_acc,
            best_epoch=best_epoch,
            total_time=total_time_str,
        )

        return {
            "experiment_name": cfg.experiment_name,
            "model_name": cfg.model.name,
            "best_val_accuracy": best_val_acc,
            "best_epoch": best_epoch,
            "total_time_seconds": total_time,
            "total_time_str": total_time_str,
            "num_params": self.num_params,
            "history": self.history,
        }

    def _train_one_epoch(
        self, epoch: int, total_epochs: int
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of ``(average_loss, accuracy_percentage)``.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        cfg = self.config.train

        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{total_epochs} [Train]",
            leave=False,
        )

        for step, batch in enumerate(pbar):
            images = batch[0].to(self.device, non_blocking=True)
            labels = batch[1].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / cfg.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % cfg.accumulation_steps == 0:
                if cfg.gradient_clip_value > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.gradient_clip_value
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # OneCycleLR steps per batch
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.OneCycleLR,
                ) and epoch > self.config.train.warmup_epochs:
                    self.scheduler.step()

            running_loss += loss.item() * cfg.accumulation_steps * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        """
        Run validation pass.

        Returns:
            Tuple of ``(average_loss, accuracy_percentage)``.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            images = batch[0].to(self.device, non_blocking=True)
            labels = batch[1].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def predict(
        self,
        dataloader: Optional[DataLoader] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference and return predictions.

        Args:
            dataloader: DataLoader to run inference on. Defaults to test_loader.

        Returns:
            Tuple of ``(logits, probabilities, predictions)``.
        """
        dl = dataloader or self.test_loader or self.val_loader
        logits, probs, preds, _, _ = run_inference(
            self.model, dl, str(self.device)
        )
        return logits, probs, preds

    def save_predictions(
        self,
        dataloader: Optional[DataLoader] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Run inference and save .npy cache for ensemble.

        Args:
            dataloader: DataLoader to run inference on. Defaults to test_loader.
            output_dir: Directory for .npy files. Defaults to checkpoint dir.
        """
        dl = dataloader or self.test_loader or self.val_loader
        out = output_dir or self.config.checkpoint.save_dir

        logits, probs, preds, labels, _ = run_inference(
            self.model, dl, str(self.device)
        )
        save_predictions(logits, probs, preds, labels, out)
        self.logger.info(f"Predictions saved to {out}")

    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full evaluation with reports and visualizations.

        Args:
            dataloader: DataLoader for evaluation. Defaults to test_loader.
            output_dir: Directory for outputs. Defaults to checkpoint dir.

        Returns:
            Metrics dictionary.
        """
        dl = dataloader or self.test_loader or self.val_loader
        out = output_dir or self.config.checkpoint.save_dir

        return evaluate_model(
            model=self.model,
            dataloader=dl,
            class_names=self.class_names,
            output_dir=out,
            device=str(self.device),
        )

    def load_best_model(self) -> None:
        """Load the best checkpoint weights into the model."""
        ckpt = self.ckpt_manager.load_best(device=str(self.device))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.logger.info(
            f"Loaded best model from epoch {ckpt.get('epoch', '?')}"
        )

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into ``HH:MM:SS`` or ``MM:SS``."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        return f"{m}m {s:02d}s"
