"""
Logging System
==============

Beautiful console logging with Rich tables for epoch summaries.
Supports file-based logging, CSV export, and TensorBoard integration.

Usage::

    from src.utils.logger import setup_logger, log_epoch

    logger = setup_logger("resnet50", log_dir="./outputs/resnet50")
    logger.info("Starting training...")
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

console = Console()


# ============================================================
# Logger Setup
# ============================================================

def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a logger with Rich console output and optional file handler.

    Args:
        name: Logger name (typically the experiment name).
        log_dir: Directory for log files. If ``None``, only console output.
        level: Logging level.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)

    # File handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "training.log"),
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


# ============================================================
# Epoch Summary Table
# ============================================================

def log_epoch(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    lr: float,
    elapsed: str,
    eta: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Print a beautiful epoch summary table to the console.

    Args:
        epoch: Current epoch number (1-indexed).
        total_epochs: Total number of epochs.
        train_loss: Training loss for the epoch.
        val_loss: Validation loss for the epoch.
        val_accuracy: Validation accuracy (0-100).
        lr: Current learning rate.
        elapsed: Elapsed time string.
        eta: Estimated time remaining string.
        extra: Optional additional metrics to display.
    """
    table = Table(
        title=f"Epoch {epoch}/{total_epochs}",
        show_header=True,
        header_style="bold cyan",
        border_style="bright_blue",
        pad_edge=True,
        expand=False,
    )

    table.add_column("Metric", style="bold white", min_width=14)
    table.add_column("Value", style="green", min_width=12, justify="right")

    table.add_row("Train Loss", f"{train_loss:.4f}")
    table.add_row("Val Loss", f"{val_loss:.4f}")
    table.add_row("Val Accuracy", f"{val_accuracy:.2f}%")
    table.add_row("Learning Rate", f"{lr:.2e}")
    table.add_row("Elapsed", elapsed)
    table.add_row("ETA", eta)

    if extra:
        for key, value in extra.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))

    console.print(table)


# ============================================================
# Training Startup Configuration Banner
# ============================================================

def get_hardware_info(device_str: str = "auto") -> Dict[str, str]:
    """Query and return detailed hardware/device metadata."""
    import platform
    import torch

    info: Dict[str, str] = {}
    if torch.cuda.is_available() and device_str.lower() != "cpu":
        dev_idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(dev_idx)
        vram_gb = prop.total_memory / (1024 ** 3)
        info["Compute Device"] = "CUDA (GPU Acceleration)"
        info["GPU Device Name"] = f"{prop.name} (Device ID: {dev_idx})"
        info["Total Dedicated VRAM"] = f"{vram_gb:.2f} GB"
        info["CUDA & cuDNN Version"] = (
            f"CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}"
        )
    else:
        info["Compute Device"] = "CPU (Host Processor)"
        info["CPU Threads"] = f"{os.cpu_count()} logical cores"
        info["Platform OS"] = f"{platform.system()} {platform.release()} ({platform.machine()})"

    info["Python Runtime"] = f"Python {platform.python_version()} ({platform.python_implementation()})"
    return info


def log_training_startup_banner(
    config: Any,
    num_params: int = 0,
    is_kfold: bool = False,
    n_splits: int = 5,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Print a comprehensive Rich table and log full hardware/hyperparameter specs at startup.
    """
    import torch

    device_str = getattr(config, "device", "auto")
    hw = get_hardware_info(device_str)

    cfg_train = getattr(config, "train", None)
    cfg_data = getattr(config, "data", None)
    cfg_model = getattr(config, "model", None)
    cfg_ckpt = getattr(config, "checkpoint", None)
    cfg_aug = getattr(config, "augmentation", None)

    table = Table(
        title="[bold white on blue]  TRAINING STARTUP CONFIGURATION & HARDWARE SPECIFICATIONS  [/bold white on blue]",
        show_header=True,
        header_style="bold cyan",
        border_style="bright_blue",
        expand=False,
        pad_edge=True,
    )

    table.add_column("Category", style="bold yellow", min_width=18)
    table.add_column("Parameter / Attribute", style="bold white", min_width=26)
    table.add_column("Configured Value / Specification", style="bold green", min_width=38)

    # 1. Hardware & Environment
    for idx, (k, v) in enumerate(hw.items()):
        cat_label = "HARDWARE & ENV" if idx == 0 else ""
        table.add_row(cat_label, k, v)

    table.add_section()

    # 2. Model Specifications
    m_name = getattr(cfg_model, "name", "unknown") if cfg_model else "unknown"
    pretrained = getattr(cfg_model, "pretrained", True) if cfg_model else True
    num_classes = getattr(cfg_model, "num_classes", 6) if cfg_model else 6
    exp_name = getattr(config, "experiment_name", "experiment")

    table.add_row("MODEL SPECS", "Experiment Identifier", str(exp_name))
    table.add_row("", "Architecture Backbone", f"{m_name.upper()} (Pretrained={pretrained}, Classes={num_classes})")
    if num_params > 0:
        table.add_row("", "Trainable Parameters", f"{num_params:,} ({num_params / 1e6:.2f}M)")

    protocol_str = f"5-Fold Stratified Cross Validation ({n_splits} Folds)" if is_kfold else "Default Single Split (Train/Val/Test)"
    table.add_row("", "Training Protocol", protocol_str)

    table.add_section()

    # 3. Training & Hyperparameters
    if cfg_train:
        epochs = getattr(cfg_train, "epochs", 30)
        optimizer = getattr(cfg_train, "optimizer", "adamw").upper()
        lr = getattr(cfg_train, "lr", 1e-4)
        wd = getattr(cfg_train, "weight_decay", 1e-4)
        sched = getattr(cfg_train, "scheduler", "cosine").capitalize()
        warmup = getattr(cfg_train, "warmup_epochs", 3)
        patience = getattr(cfg_train, "early_stopping_patience", 6)
        smoothing = getattr(cfg_train, "label_smoothing", 0.1)
        amp = getattr(cfg_train, "mixed_precision", True)
        grad_clip = getattr(cfg_train, "gradient_clip_value", 1.0)

        table.add_row("HYPERPARAMETERS", "Total Epoch Budget", f"{epochs} Epochs")
        table.add_row("", "Optimizer & Learning Rate", f"{optimizer} (LR: {lr:.2e}, Weight Decay: {wd:.2e})")
        table.add_row("", "Scheduler & Warmup", f"{sched} Annealing (Warmup: {warmup} epochs)")
        table.add_row("", "Early Stopping Strategy", f"Patience: {patience} epochs")
        table.add_row("", "Regularization", f"Label Smoothing: {smoothing} | Grad Clip: {grad_clip}")
        table.add_row("", "Mixed Precision (AMP)", "ENABLED (torch.amp.autocast)" if amp else "DISABLED (FP32)")

    # 4. Data & Augmentation
    if cfg_data:
        bs = getattr(cfg_data, "batch_size", 32)
        img_sz = getattr(cfg_data, "image_size", 224)
        nw = getattr(cfg_data, "num_workers", 4)
        table.add_row("DATA & INPUT", "Batch Size & Resolution", f"Batch={bs}, Resolution={img_sz}x{img_sz}, Workers={nw}")

    if cfg_aug:
        train_aug = getattr(cfg_aug, "train", {})
        hflip = getattr(train_aug, "horizontal_flip", 0.5)
        vflip = getattr(train_aug, "vertical_flip", 0.5)
        rot = getattr(train_aug, "rotation_limit", 15)
        table.add_row("", "Data Augmentation", f"H-Flip: {hflip} | V-Flip: {vflip} | Rot: +/-{rot} deg")

    # 5. Checkpoint & Loss-Gate
    if cfg_ckpt:
        mon = getattr(cfg_ckpt, "monitor", "val_accuracy")
        mode = getattr(cfg_ckpt, "mode", "max")
        tol = getattr(cfg_ckpt, "loss_gate_tolerance", 0.05)
        s_dir = getattr(cfg_ckpt, "save_dir", "./outputs")
        table.add_row("CHECKPOINTING", "Target Directory", str(s_dir))
        table.add_row("", "Monitor Metric", f"{mon} ({mode})")
        table.add_row("", "Loss-Gate Safeguard", f"ACTIVE (Max Allowed Loss Drift: {tol*100:.1f}%)")

    console.print()
    console.print(table)
    console.print()

    # Log text summary to logger file
    if logger:
        logger.info("=" * 70)
        logger.info(f"  TRAINING STARTUP: {exp_name.upper()} ({protocol_str})")
        logger.info("=" * 70)
        for k, v in hw.items():
            logger.info(f"  [HARDWARE] {k}: {v}")
        if cfg_train:
            logger.info(f"  [CONFIG] Epochs: {getattr(cfg_train, 'epochs', 30)} | Optimizer: {getattr(cfg_train, 'optimizer', 'adamw')} | LR: {getattr(cfg_train, 'lr', 1e-4)}")
            logger.info(f"  [CONFIG] Loss-Gate Tolerance: {getattr(cfg_ckpt, 'loss_gate_tolerance', 0.05)*100:.1f}% | Early Stop Patience: {getattr(cfg_train, 'early_stopping_patience', 6)}")


def log_training_start(
    experiment_name: str,
    model_name: str,
    num_params: int,
    device: str,
    epochs: int,
    batch_size: int,
) -> None:
    """Print a styled training start banner."""
    info_text = (
        f"[bold cyan]Experiment:[/bold cyan] {experiment_name}\n"
        f"[bold cyan]Model:[/bold cyan]      {model_name}\n"
        f"[bold cyan]Parameters:[/bold cyan] {num_params:,}\n"
        f"[bold cyan]Device:[/bold cyan]     {device}\n"
        f"[bold cyan]Epochs:[/bold cyan]     {epochs}\n"
        f"[bold cyan]Batch Size:[/bold cyan] {batch_size}"
    )
    panel = Panel(
        info_text,
        title="[bold white]Training Started[/bold white]",
        border_style="bright_green",
        expand=False,
        padding=(1, 2),
    )
    console.print(panel)


def log_training_end(
    experiment_name: str,
    best_accuracy: float,
    best_epoch: int,
    total_time: str,
) -> None:
    """Print a styled training completion banner."""
    info_text = (
        f"[bold cyan]Experiment:[/bold cyan]   {experiment_name}\n"
        f"[bold cyan]Best Accuracy:[/bold cyan] {best_accuracy:.2f}%\n"
        f"[bold cyan]Best Epoch:[/bold cyan]    {best_epoch}\n"
        f"[bold cyan]Total Time:[/bold cyan]    {total_time}"
    )
    panel = Panel(
        info_text,
        title="[bold white]Training Complete[/bold white]",
        border_style="bright_green",
        expand=False,
        padding=(1, 2),
    )
    console.print(panel)


# ============================================================
# CSV Logger
# ============================================================

class CSVLogger:
    """
    Append-mode CSV logger for training history.

    Writes one row per epoch with all tracked metrics.

    Args:
        filepath: Path to the CSV file.
        fieldnames: List of column names.
    """

    def __init__(
        self,
        filepath: str,
        fieldnames: List[str],
        resume: bool = False,
    ) -> None:
        self.filepath = filepath
        self.fieldnames = fieldnames

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        if not (resume and os.path.exists(filepath)):
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        """Append a row to the CSV file."""
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
