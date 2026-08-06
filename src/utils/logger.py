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
# Training Banner
# ============================================================

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
