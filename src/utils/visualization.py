"""
Visualization Utilities
=======================

Publication-quality plots for training curves and confusion matrices.

All plots use a clean, modern style suitable for academic papers and theses.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Global style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def plot_training_curves(
    history: Dict[str, List[float]],
    output_dir: str,
) -> None:
    """
    Plot and save training/validation loss and accuracy curves.

    Args:
        history: Dictionary with keys ``'train_loss'``, ``'val_loss'``,
                 ``'train_accuracy'``, ``'val_accuracy'``.
        output_dir: Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # --- Loss Curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], "o-", label="Train Loss",
            color="#4A90D9", linewidth=2, markersize=4)
    ax.plot(epochs, history["val_loss"], "s-", label="Val Loss",
            color="#E74C3C", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close(fig)

    # --- Accuracy Curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_accuracy"], "o-", label="Train Accuracy",
            color="#4A90D9", linewidth=2, markersize=4)
    ax.plot(epochs, history["val_accuracy"], "s-", label="Val Accuracy",
            color="#2ECC71", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training & Validation Accuracy")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: str,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    """
    Plot and save a publication-quality confusion matrix heatmap.

    Args:
        cm: Confusion matrix of shape ``(C, C)``.
        class_names: List of class names.
        output_dir: Directory to save the plot.
        title: Plot title.
        normalize: If ``True``, normalize rows to show percentages.
    """
    os.makedirs(output_dir, exist_ok=True)

    if normalize:
        cm_plot = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    # Dynamic figure size based on number of classes
    n = len(class_names)
    fig_size = max(8, n * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close(fig)


def plot_comparison_bar(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_name: str,
    output_dir: str,
    title: Optional[str] = None,
) -> None:
    """
    Plot a grouped bar chart comparing a metric across multiple models.

    Args:
        metrics_dict: ``{model_name: {metric: value, ...}}``.
        metric_name: The metric key to compare (e.g., ``'accuracy'``).
        output_dir: Directory to save the plot.
        title: Optional plot title.
    """
    os.makedirs(output_dir, exist_ok=True)

    models = list(metrics_dict.keys())
    values = [metrics_dict[m].get(metric_name, 0.0) for m in models]

    colors = sns.color_palette("viridis", n_colors=len(models))

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    bars = ax.bar(models, values, color=colors, edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title or f"Model Comparison — {metric_name.replace('_', ' ').title()}")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")

    fig.savefig(os.path.join(output_dir, f"comparison_{metric_name}.png"))
    plt.close(fig)
