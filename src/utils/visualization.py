"""
Visualization Utilities
=======================

Publication-quality plots for training dynamics, classification performance,
per-disease breakdown, ROC/PR curves, multi-model comparisons, and ensemble analysis.

All plots adhere to academic publishing standards (IEEE/Nature/Elsevier styling,
300 DPI, modern typography, legible color palettes).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

# Global aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})

# Curated categorical color palette (Colorblind-friendly)
DISTINCT_COLORS = [
    "#2E86AB",  # Blue
    "#F24236",  # Red
    "#38B000",  # Green
    "#F77F00",  # Orange
    "#7209B7",  # Purple
    "#4361EE",  # Royal Blue
    "#CC5A71",  # Rose
    "#0077B6",  # Deep Sky Blue
    "#8338EC",  # Violet
    "#2A9D8F",  # Teal
]


# ==============================================================================
# 1. Training Dynamics & Convergence Visualizations
# ==============================================================================

def plot_training_curves(
    history: Dict[str, List[float]],
    output_dir: str,
    model_name: Optional[str] = None,
) -> None:
    """
    Plot and save standard training/validation loss, accuracy, and LR curves,
    plus a unified 4-panel training dashboard.

    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'train_accuracy',
                 'val_accuracy', and optional 'lr'.
        output_dir: Directory to save the plots.
        model_name: Optional display name of the model.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_epochs = len(history.get("train_loss", []))
    if n_epochs == 0:
        return
    epochs = range(1, n_epochs + 1)
    prefix = f"[{model_name}] " if model_name else ""

    # --- 1. Loss Curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], "o-", label="Train Loss",
            color="#2E86AB", linewidth=2.2, markersize=4.5)
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], "s-", label="Val Loss",
                color="#F24236", linewidth=2.2, markersize=4.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Cross Entropy)")
    ax.set_title(f"{prefix}Training & Validation Loss")
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="upper right")
    ax.grid(True, alpha=0.35, linestyle="--")
    fig.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close(fig)

    # --- 2. Accuracy Curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_accuracy"], "o-", label="Train Accuracy",
            color="#2E86AB", linewidth=2.2, markersize=4.5)
    if "val_accuracy" in history and history["val_accuracy"]:
        ax.plot(epochs, history["val_accuracy"], "s-", label="Val Accuracy",
                color="#38B000", linewidth=2.2, markersize=4.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{prefix}Training & Validation Accuracy")
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="lower right")
    ax.grid(True, alpha=0.35, linestyle="--")
    fig.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close(fig)

    # --- 3. Learning Rate Curve (if available) ---
    if "lr" in history and history["lr"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, history["lr"], "d-", color="#7209B7", linewidth=2.0, markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title(f"{prefix}Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.35, linestyle="--")
        fig.savefig(os.path.join(output_dir, "learning_rate_curve.png"))
        plt.close(fig)

    # --- 4. Unified Multi-Panel Training Dashboard ---
    plot_training_dashboard(history, output_dir, model_name=model_name)


def plot_training_dashboard(
    history: Dict[str, List[float]],
    output_dir: str,
    model_name: Optional[str] = None,
) -> None:
    """
    Generate a 4-panel comprehensive training diagnostics dashboard:
    Panel A: Loss Dynamics (Train vs Val)
    Panel B: Accuracy Progression (Train vs Val)
    Panel C: Generalization Gap (Train Acc - Val Acc)
    Panel D: Learning Rate Annealing
    """
    os.makedirs(output_dir, exist_ok=True)
    n_epochs = len(history.get("train_loss", []))
    if n_epochs == 0:
        return
    epochs = np.array(list(range(1, n_epochs + 1)))
    title_model = f" — {model_name}" if model_name else ""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Comprehensive Training Dynamics Dashboard{title_model}", fontsize=16, y=0.98)

    # Panel 1: Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history["train_loss"], "o-", color="#2E86AB", label="Train Loss", linewidth=2, markersize=4)
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], "s-", color="#F24236", label="Val Loss", linewidth=2, markersize=4)
        best_loss_idx = int(np.argmin(history["val_loss"]))
        ax1.axvline(epochs[best_loss_idx], color="#F24236", linestyle=":", alpha=0.8,
                    label=f"Min Val Loss ({history['val_loss'][best_loss_idx]:.4f} @ Ep {epochs[best_loss_idx]})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title("(A) Loss Convergence")
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Panel 2: Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, history["train_accuracy"], "o-", color="#2E86AB", label="Train Accuracy", linewidth=2, markersize=4)
    if "val_accuracy" in history and history["val_accuracy"]:
        ax2.plot(epochs, history["val_accuracy"], "s-", color="#38B000", label="Val Accuracy", linewidth=2, markersize=4)
        best_acc_idx = int(np.argmax(history["val_accuracy"]))
        ax2.axvline(epochs[best_acc_idx], color="#38B000", linestyle=":", alpha=0.8,
                    label=f"Peak Val Acc ({history['val_accuracy'][best_acc_idx]:.2f}% @ Ep {epochs[best_acc_idx]})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("(B) Classification Accuracy Progression")
    ax2.legend(loc="lower right", frameon=True, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Panel 3: Generalization Gap
    ax3 = axes[1, 0]
    if "val_accuracy" in history and history["val_accuracy"]:
        train_acc = np.array(history["train_accuracy"])
        val_acc = np.array(history["val_accuracy"])
        gap = train_acc - val_acc
        ax3.fill_between(epochs, 0, gap, color="#F77F00", alpha=0.25, label="Overfitting Zone (>0)")
        ax3.plot(epochs, gap, "d-", color="#F77F00", linewidth=2, markersize=4, label="Generalization Gap (Train - Val %)")
        ax3.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Accuracy Gap (%)")
        ax3.set_title("(C) Generalization Gap & Overfitting Analysis")
        ax3.legend(loc="upper left", frameon=True, fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle="--")
    else:
        ax3.text(0.5, 0.5, "Validation Accuracy data not available", ha="center", va="center")

    # Panel 4: Learning Rate
    ax4 = axes[1, 1]
    if "lr" in history and history["lr"]:
        ax4.plot(epochs, history["lr"], "^-", color="#7209B7", linewidth=2, markersize=4, label="Learning Rate")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Learning Rate (Log Scale)")
        ax4.set_yscale("log")
        ax4.set_title("(D) Learning Rate Annealing Schedule")
        ax4.legend(loc="upper right", frameon=True, fontsize=9)
        ax4.grid(True, alpha=0.3, linestyle="--")
    else:
        ax4.text(0.5, 0.5, "Learning Rate log not available", ha="center", va="center")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "training_dashboard.png"))
    plt.close(fig)


# ==============================================================================
# 2. Confusion Matrices (Count & Normalized Heatmaps)
# ==============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: str,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    filename: Optional[str] = None,
) -> None:
    """
    Plot and save publication-grade confusion matrix heatmap.
    """
    os.makedirs(output_dir, exist_ok=True)
    fname = filename or ("confusion_matrix_normalized.png" if normalize else "confusion_matrix.png")

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Prevent division by zero
        cm_plot = (cm.astype("float") / row_sums) * 100.0
        fmt = ".1f"
        cbar_label = "Class Recall / Accuracy (%)"
        cmap = "Blues"
    else:
        cm_plot = cm
        fmt = "d"
        cbar_label = "Sample Count"
        cmap = "Blues"

    n = len(class_names)
    fig_size = max(7.5, n * 1.1)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))

    # Clean short labels for readable ticks
    clean_labels = [c.replace("_", " ").replace("leaf", "Leaf").title() for c in class_names]

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        square=True,
        linewidths=0.75,
        linecolor="#E0E0E0",
        cbar_kws={"shrink": 0.82, "label": cbar_label},
        ax=ax,
    )

    ax.set_xlabel("Predicted Disease Class", fontsize=12, labelpad=8)
    ax.set_ylabel("Ground Truth Disease Class", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)

    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)


# ==============================================================================
# 3. Per-Disease Class Performance Visualizations
# ==============================================================================

def plot_per_class_metrics(
    per_class_dict: Dict[str, Dict[str, float]],
    class_names: List[str],
    output_dir: str,
    model_name: Optional[str] = None,
    filename: str = "per_class_metrics.png",
) -> None:
    """
    Plot grouped bar chart showing Precision, Recall, and F1-Score for each
    individual tea disease class + Macro and Weighted Averages.
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"[{model_name}] " if model_name else ""

    classes_to_plot = [c for c in class_names if c in per_class_dict]
    if not classes_to_plot:
        return

    precisions = [per_class_dict[c]["precision"] * 100.0 for c in classes_to_plot]
    recalls = [per_class_dict[c]["recall"] * 100.0 for c in classes_to_plot]
    f1s = [per_class_dict[c]["f1_score"] * 100.0 for c in classes_to_plot]
    supports = [per_class_dict[c].get("support", 0) for c in classes_to_plot]

    # Calculate Macro and Weighted Averages
    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    total_supp = sum(supports) if sum(supports) > 0 else 1
    weighted_prec = float(sum(p * s for p, s in zip(precisions, supports)) / total_supp)
    weighted_rec = float(sum(r * s for r, s in zip(recalls, supports)) / total_supp)
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_supp)

    # Append averages to display
    display_classes = [c.replace("_", " ").title() for c in classes_to_plot] + ["Macro Avg", "Weighted Avg"]
    precisions_all = precisions + [macro_prec, weighted_prec]
    recalls_all = recalls + [macro_rec, weighted_rec]
    f1s_all = f1s + [macro_f1, weighted_f1]

    x = np.arange(len(display_classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(10, len(display_classes) * 1.5), 6))

    bars1 = ax.bar(x - width, precisions_all, width, label="Precision", color="#2E86AB", edgecolor="white")
    bars2 = ax.bar(x, recalls_all, width, label="Recall", color="#38B000", edgecolor="white")
    bars3 = ax.bar(x + width, f1s_all, width, label="F1-Score", color="#F77F00", edgecolor="white")

    # Add divider before summary averages
    ax.axvline(len(classes_to_plot) - 0.5, color="#888888", linestyle="--", alpha=0.7, linewidth=1.2)

    # Annotate bar values
    def autolabel(bars):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", rotation=0
            )

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title(f"{prefix}Per-Disease Class Metric Breakdown (Precision, Recall, F1)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_classes, rotation=25, ha="right")
    ax.set_ylim(0, 112)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="lower right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


# ==============================================================================
# 4. Multi-Class ROC and Precision-Recall Curves
# ==============================================================================

def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    output_dir: str,
    model_name: Optional[str] = None,
    filename: str = "roc_curves.png",
) -> None:
    """
    Plot Multi-Class One-vs-Rest (OvR) ROC curves for each disease class + Macro/Micro AUC.
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"[{model_name}] " if model_name else ""
    n_classes = len(class_names)

    # Binarize labels for OvR ROC
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fpr: Dict[Union[int, str], np.ndarray] = {}
    tpr: Dict[Union[int, str], np.ndarray] = {}
    roc_auc: Dict[Union[int, str], float] = {}

    # Per-class ROC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 7))

    # Micro and macro curves
    ax.plot(
        fpr["micro"], tpr["micro"],
        label=f"Micro-average ROC (AUC = {roc_auc['micro']:.4f})",
        color="#111111", linestyle=":", linewidth=2.5
    )
    ax.plot(
        fpr["macro"], tpr["macro"],
        label=f"Macro-average ROC (AUC = {roc_auc['macro']:.4f})",
        color="#7209B7", linestyle="--", linewidth=2.5
    )

    colors = sns.color_palette("tab10", n_colors=n_classes)
    for i, color in zip(range(n_classes), colors):
        clean_name = class_names[i].replace("_", " ").title()
        ax.plot(
            fpr[i], tpr[i], color=color, linewidth=1.8,
            label=f"{clean_name} (AUC = {roc_auc[i]:.4f})"
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.0, label="Random Guess (AUC = 0.50)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(f"{prefix}Multi-Class One-vs-Rest Receiver Operating Characteristic (ROC)", fontsize=13)
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, fontsize=9.5)
    ax.grid(True, alpha=0.35, linestyle="--")

    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    output_dir: str,
    model_name: Optional[str] = None,
    filename: str = "precision_recall_curves.png",
) -> None:
    """
    Plot Multi-Class Precision-Recall curves with Average Precision (AP) and Iso-F1 contours.
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"[{model_name}] " if model_name else ""
    n_classes = len(class_names)

    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    precision: Dict[Union[int, str], np.ndarray] = {}
    recall: Dict[Union[int, str], np.ndarray] = {}
    avg_prec: Dict[Union[int, str], float] = {}

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        avg_prec[i] = average_precision_score(y_true_bin[:, i], y_proba[:, i])

    # Micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_proba.ravel())
    avg_prec["micro"] = average_precision_score(y_true_bin, y_proba, average="micro")

    fig, ax = plt.subplots(figsize=(8.5, 7))

    # Iso-F1 curves
    f_scores = np.linspace(0.4, 0.9, num=6)
    for f_score in f_scores:
        x_iso = np.linspace(f_score / 2 + 0.005, 1.0, 100)
        y_iso = f_score * x_iso / (2 * x_iso - f_score)
        valid_idx = (y_iso >= 0) & (y_iso <= 1.0)
        ax.plot(x_iso[valid_idx], y_iso[valid_idx], color="gray", alpha=0.25, linestyle="--", linewidth=0.8)
        if (0.92 > f_score / 2):
            y_annot = f_score * 0.92 / (2 * 0.92 - f_score)
            if 0 <= y_annot <= 1.0:
                ax.annotate(f"f1={f_score:.1f}", xy=(0.92, y_annot),
                            fontsize=7.5, color="gray", alpha=0.6)

    ax.plot(
        recall["micro"], precision["micro"],
        label=f"Micro-average (AP = {avg_prec['micro']:.4f})",
        color="#111111", linestyle=":", linewidth=2.5
    )

    colors = sns.color_palette("tab10", n_colors=n_classes)
    for i, color in zip(range(n_classes), colors):
        clean_name = class_names[i].replace("_", " ").title()
        ax.plot(
            recall[i], precision[i], color=color, linewidth=1.8,
            label=f"{clean_name} (AP = {avg_prec[i]:.4f})"
        )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"{prefix}Multi-Class Precision-Recall Curves", fontsize=13)
    ax.legend(loc="lower left", frameon=True, fancybox=True, shadow=True, fontsize=9.5)
    ax.grid(True, alpha=0.35, linestyle="--")

    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


# ==============================================================================
# 5. Cross-Model & Ensemble Benchmark Comparisons
# ==============================================================================

def plot_comparison_bar(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_name: str,
    output_dir: str,
    title: Optional[str] = None,
    filename: Optional[str] = None,
) -> None:
    """
    Plot a ranked bar chart comparing a metric across all models and ensemble methods.
    """
    os.makedirs(output_dir, exist_ok=True)
    fname = filename or f"comparison_{metric_name}.png"

    # Sort models by metric value descending
    sorted_items = sorted(
        metrics_dict.items(),
        key=lambda x: x[1].get(metric_name, 0.0),
        reverse=True
    )
    models = [k for k, _ in sorted_items]
    values = [v.get(metric_name, 0.0) for _, v in sorted_items]

    # Color code: individual models in blue/teal, ensemble methods in gold/orange/purple
    colors = []
    for m in models:
        m_lower = m.lower()
        if "stacking" in m_lower:
            colors.append("#F77F00")
        elif "voting" in m_lower:
            colors.append("#2A9D8F")
        else:
            colors.append("#2E86AB")

    fig, ax = plt.subplots(figsize=(max(8.5, len(models) * 1.3), 5.5))
    bars = ax.bar(models, values, color=colors, edgecolor="white", linewidth=0.9, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"{val:.2f}%",
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold"
        )

    metric_display = metric_name.replace("_", " ").title()
    ax.set_ylabel(f"{metric_display} (%)", fontsize=12)
    ax.set_title(title or f"Model Benchmark Comparison — {metric_display}", fontsize=14, pad=10)
    ax.set_ylim(0, max(values) * 1.14 if values else 100)
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    plt.xticks(rotation=25, ha="right")

    # Legend for model categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E86AB", label="Base Backbone"),
        Patch(facecolor="#2A9D8F", label="Voting Ensemble"),
        Patch(facecolor="#F77F00", label="Stacking Ensemble"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)


def plot_per_class_comparison_heatmap(
    all_models_per_class: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "f1_score",
    class_names: Optional[List[str]] = None,
    output_dir: str = "outputs",
    filename: str = "comparison_per_class_heatmap.png",
) -> None:
    """
    Plot a 2D Heatmap of per-disease performance (e.g. F1-score) across all models.
    Rows: Models & Ensembles
    Columns: Disease Classes
    """
    os.makedirs(output_dir, exist_ok=True)
    models = list(all_models_per_class.keys())
    if not models:
        return

    if class_names is None:
        first_model = models[0]
        class_names = list(all_models_per_class[first_model].keys())

    # Build matrix (models x classes)
    matrix = np.zeros((len(models), len(class_names)))
    for i, m in enumerate(models):
        for j, c in enumerate(class_names):
            val = all_models_per_class[m].get(c, {}).get(metric, 0.0)
            matrix[i, j] = val * 100.0 if val <= 1.0 else val

    clean_cols = [c.replace("_", " ").title() for c in class_names]
    clean_rows = [m for m in models]

    fig, ax = plt.subplots(figsize=(max(9, len(class_names) * 1.6), max(6, len(models) * 0.75)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=clean_cols,
        yticklabels=clean_rows,
        linewidths=0.8,
        linecolor="#FFFFFF",
        cbar_kws={"label": f"{metric.replace('_', ' ').title()} (%)"},
        ax=ax,
        vmin=max(80, np.min(matrix) - 2),
        vmax=100.0,
    )

    ax.set_title(f"Per-Disease Class Performance Matrix ({metric.upper()}) Across All Models", fontsize=14, pad=12)
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)

    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


def plot_per_class_comparison_bar(
    all_models_per_class: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "f1_score",
    class_names: Optional[List[str]] = None,
    output_dir: str = "outputs",
    filename: str = "comparison_per_class_bar.png",
) -> None:
    """
    Plot grouped bar chart comparing each disease class across selected top models & ensembles.
    """
    os.makedirs(output_dir, exist_ok=True)
    models = list(all_models_per_class.keys())
    if not models:
        return

    if class_names is None:
        first_model = models[0]
        class_names = list(all_models_per_class[first_model].keys())

    clean_classes = [c.replace("_", " ").title() for c in class_names]
    n_models = len(models)
    n_classes = len(class_names)

    x = np.arange(n_classes)
    total_width = 0.8
    bar_width = total_width / n_models

    fig, ax = plt.subplots(figsize=(max(11, n_classes * 2.2), 6.5))
    colors = sns.color_palette("tab10", n_colors=n_models)

    for idx, (m_name, color) in enumerate(zip(models, colors)):
        values = []
        for c in class_names:
            val = all_models_per_class[m_name].get(c, {}).get(metric, 0.0)
            values.append(val * 100.0 if val <= 1.0 else val)

        offset = (idx - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=m_name, color=color, edgecolor="white", linewidth=0.5)

    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)", fontsize=12)
    ax.set_title(f"Per-Disease Class Benchmark Comparison ({metric.replace('_', ' ').title()})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_classes, rotation=20, ha="right")
    ax.set_ylim(80, 102)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=9.5)
    ax.grid(axis="y", alpha=0.35, linestyle="--")

    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


def plot_radar_chart_comparison(
    models_metrics: Dict[str, List[float]],
    categories: List[str],
    output_dir: str,
    title: str = "Model Multi-Class Radar Comparison",
    filename: str = "comparison_radar.png",
) -> None:
    """
    Plot a Radar / Spider chart comparing models across all disease classes.
    """
    os.makedirs(output_dir, exist_ok=True)
    N = len(categories)
    if N < 3:
        return

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the radar circle

    clean_categories = [c.replace("_", " ").title() for c in categories]
    clean_categories += clean_categories[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = DISTINCT_COLORS[:len(models_metrics)]

    min_val = 100.0
    for idx, (m_name, vals) in enumerate(models_metrics.items()):
        values = list(vals)
        values = [v * 100.0 if v <= 1.0 else v for v in values]
        min_val = min(min_val, min(values))
        values += values[:1]  # Close polygon

        ax.plot(angles, values, "o-", linewidth=2.0, label=m_name, color=colors[idx % len(colors)], markersize=4)
        ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(clean_categories[:-1], fontsize=10.5, fontweight="semibold")
    ax.set_ylim(max(75, min_val - 5), 101)
    ax.set_title(title, fontsize=14, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


def plot_model_tradeoffs(
    models_data: List[Dict[str, Any]],
    output_dir: str,
    filename: str = "model_tradeoffs_pareto.png",
) -> None:
    """
    Plot Pareto efficiency scatter plot: Model Complexity (Parameters in M) vs Test Accuracy (%).
    Bubble size represents Inference Time.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not models_data:
        return

    names = [d["Model"] for d in models_data]
    params_m = [d.get("Parameters_M", 0.0) for d in models_data]
    accs = [d.get("Test_Accuracy", 0.0) for d in models_data]
    latencies = [max(10, d.get("Inference_Time_s", 20) * 12) for d in models_data]

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = []
    for n in names:
        if "Stacking" in n:
            colors.append("#F77F00")
        elif "Voting" in n:
            colors.append("#2A9D8F")
        else:
            colors.append("#2E86AB")

    scatter = ax.scatter(
        params_m, accs, s=latencies, c=colors,
        alpha=0.85, edgecolors="black", linewidth=1.2
    )

    for i, name in enumerate(names):
        ax.annotate(
            name,
            (params_m[i], accs[i]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold"
        )

    ax.set_xlabel("Model Parameters (Million)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Model Trade-off: Accuracy vs Model Complexity & Latency", fontsize=14)
    ax.grid(True, alpha=0.35, linestyle="--")

    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


# ==============================================================================
# 6. Ensemble Meta-Learner & Weight Visualizations
# ==============================================================================

def plot_ensemble_weights(
    weights: Dict[str, float] | List[float],
    model_names: List[str],
    output_dir: str,
    filename: str = "ensemble_weights.png",
) -> None:
    """
    Plot the optimal weights assigned to base models in Weighted Voting.
    """
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(weights, dict):
        names = list(weights.keys())
        vals = list(weights.values())
    else:
        names = model_names
        vals = weights

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = sns.color_palette("Blues_r", n_colors=len(names))
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.8, width=0.55)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f} ({val*100:.1f}%)",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold"
        )

    ax.set_ylabel("Optimal Ensemble Weight", fontsize=12)
    ax.set_title("Weighted Voting: Optimal Backbone Weights", fontsize=13)
    ax.set_ylim(0, max(vals) * 1.25 if vals else 1.0)
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    plt.xticks(rotation=15, ha="right")

    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


def plot_stacking_feature_importance(
    importances: np.ndarray | List[float],
    feature_names: List[str],
    output_dir: str,
    meta_name: str = "Random Forest",
    filename: str = "stacking_feature_importance.png",
    top_k: int = 15,
) -> None:
    """
    Plot feature importance or coefficient magnitude for Stacking Meta-Learners.
    """
    os.makedirs(output_dir, exist_ok=True)
    importances = np.array(importances)
    indices = np.argsort(importances)[::-1][:top_k]

    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(9, max(5, top_k * 0.35)))
    y_pos = np.arange(len(sorted_names))

    ax.barh(y_pos, sorted_vals, color="#2E86AB", edgecolor="black", linewidth=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Feature Importance / Weight", fontsize=11)
    ax.set_title(f"Stacking Meta-Learner Feature Importance ({meta_name})", fontsize=13)
    ax.grid(axis="x", alpha=0.35, linestyle="--")

    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


# ==============================================================================
# 7. K-Fold Cross Validation Stability & Variance Plots
# ==============================================================================

def plot_kfold_summary(
    kfold_metrics: Dict[str, List[float]],
    output_dir: str,
    model_name: str = "Model",
    filename: str = "kfold_variance_bar.png",
) -> None:
    """
    Plot cross-validation stability: Mean ± Standard Deviation with individual fold points.
    """
    os.makedirs(output_dir, exist_ok=True)
    metric_names = [m for m in ["Accuracy", "Precision", "Recall", "F1_Score"] if m in kfold_metrics]
    if not metric_names:
        return

    means = [np.mean(kfold_metrics[m]) for m in metric_names]
    stds = [np.std(kfold_metrics[m]) for m in metric_names]

    x = np.arange(len(metric_names))
    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar chart of means with error bars
    bars = ax.bar(
        x, means, yerr=stds, capsize=6,
        color="#2E86AB", edgecolor="white", linewidth=1.0, width=0.5,
        error_kw={"elinewidth": 1.8, "ecolor": "#D90429", "capthick": 1.8}
    )

    # Scatter individual fold points
    for idx, m in enumerate(metric_names):
        fold_vals = kfold_metrics[m]
        x_jitter = np.random.normal(idx, 0.04, size=len(fold_vals))
        ax.scatter(x_jitter, fold_vals, color="#F77F00", s=40, zorder=4, edgecolor="black", linewidth=0.8)

    for bar, mean_val, std_val in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean_val + std_val + 0.5,
            f"{mean_val:.2f}% ± {std_val:.2f}%",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold"
        )

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title(f"5-Fold Cross-Validation Stability Analysis ({model_name})", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in metric_names], fontsize=11)
    ax.set_ylim(max(70, min(means) - 10), 105)
    ax.grid(axis="y", alpha=0.35, linestyle="--")

    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
