"""
Metrics Utilities
=================

Compute classification metrics using scikit-learn.

Functions return both scalar summaries and per-class breakdowns
suitable for JSON serialization and thesis reporting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    average: str = "macro",
) -> Dict[str, Any]:
    """
    Compute full classification metrics.

    Args:
        y_true: Ground truth labels of shape ``(N,)``.
        y_pred: Predicted labels of shape ``(N,)``.
        class_names: Optional list of class display names.
        average: Averaging strategy for multi-class (``'macro'``, ``'weighted'``).

    Returns:
        Dictionary with scalar metrics and per-class details::

            {
                "accuracy": float,
                "precision": float,
                "recall": float,
                "f1_score": float,
                "confusion_matrix": List[List[int]],
                "classification_report": str,
                "per_class": {class_name: {"precision": ..., "recall": ..., "f1": ...}}
            }
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    # Per-class breakdown
    per_class: Dict[str, Dict[str, float]] = {}
    if class_names is not None:
        prec_per = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec_per = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)

        for i, name in enumerate(class_names):
            per_class[name] = {
                "precision": float(prec_per[i]),
                "recall": float(rec_per[i]),
                "f1_score": float(f1_per[i]),
                "support": int(np.sum(y_true == i)),
            }

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str,
        "per_class": per_class,
    }
