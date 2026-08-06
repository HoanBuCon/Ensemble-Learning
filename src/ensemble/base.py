"""
Ensemble Base Class
===================

Abstract interface for all ensemble methods.
Every ensemble exposes ``fit()``, ``predict()``, and ``evaluate()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.metrics import compute_metrics


class EnsembleBase(ABC):
    """
    Abstract base class for ensemble methods.

    All ensemble methods operate on lists of probability arrays
    (one per base model) and never touch PyTorch models directly.

    Subclasses must implement :meth:`fit` and :meth:`predict`.
    :meth:`evaluate` is provided for free.
    """

    @staticmethod
    def validate_class_mappings(class_mappings: List[Dict[str, int]]) -> bool:
        """
        Verify that all base models share an identical class_to_idx mapping.

        Raises:
            ValueError: If class mappings differ across models.
        """
        if not class_mappings:
            return True
        ref = class_mappings[0]
        for i, cmap in enumerate(class_mappings[1:], start=1):
            if cmap != ref:
                raise ValueError(
                    f"Class mapping mismatch detected across base models! "
                    f"Model 0 has class_to_idx={ref}, "
                    f"but Model {i} has class_to_idx={cmap}."
                )
        return True

    @abstractmethod
    def fit(
        self,
        probabilities: List[np.ndarray],
        labels: np.ndarray,
    ) -> "EnsembleBase":
        """
        Fit the ensemble (e.g., learn weights or train a meta-learner).

        Args:
            probabilities: List of ``M`` arrays, each of shape ``(N, C)``,
                           where ``M`` = number of models, ``N`` = samples,
                           ``C`` = classes.
            labels: Ground truth labels of shape ``(N,)``.

        Returns:
            ``self`` for method chaining.
        """
        ...

    @abstractmethod
    def predict(
        self,
        probabilities: List[np.ndarray],
    ) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            probabilities: List of ``M`` probability arrays, each ``(N, C)``.

        Returns:
            Predicted class indices of shape ``(N,)``.
        """
        ...

    def predict_proba(
        self,
        probabilities: List[np.ndarray],
    ) -> np.ndarray:
        """
        Generate ensemble probability estimates.

        Default implementation returns averaged probabilities.
        Override in subclasses for different behavior.

        Args:
            probabilities: List of ``M`` probability arrays, each ``(N, C)``.

        Returns:
            Ensemble probabilities of shape ``(N, C)``.
        """
        return np.mean(probabilities, axis=0)

    def evaluate(
        self,
        probabilities: List[np.ndarray],
        labels: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Predict and evaluate, returning full metrics.

        Args:
            probabilities: List of ``M`` probability arrays, each ``(N, C)``.
            labels: Ground truth labels of shape ``(N,)``.
            class_names: Optional class name list for the report.

        Returns:
            Metrics dictionary (accuracy, precision, recall, F1, CM, etc.).
        """
        predictions = self.predict(probabilities)
        return compute_metrics(labels, predictions, class_names=class_names)
