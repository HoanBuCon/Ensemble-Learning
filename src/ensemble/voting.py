"""
Voting Ensembles
================

Hard Voting, Soft Voting, and Weighted Voting implementations.

All methods work on cached probability arrays (``np.ndarray``)
and have no PyTorch dependency.

Usage::

    from src.ensemble.voting import HardVoting, SoftVoting, WeightedVoting
    import numpy as np

    probs = [
        np.load("outputs/resnet50/probabilities.npy"),
        np.load("outputs/densenet121/probabilities.npy"),
        np.load("outputs/swin_tiny/probabilities.npy"),
    ]
    labels = np.load("outputs/resnet50/labels.npy")

    # Hard Voting
    hv = HardVoting()
    hv_metrics = hv.evaluate(probs, labels)

    # Soft Voting
    sv = SoftVoting()
    sv_metrics = sv.evaluate(probs, labels)

    # Weighted Voting (auto-optimize weights)
    wv = WeightedVoting()
    wv.fit(probs, labels)
    wv_metrics = wv.evaluate(probs, labels)
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import mode as scipy_mode

from src.ensemble.base import EnsembleBase


class HardVoting(EnsembleBase):
    """
    Hard Voting (Majority Vote) Ensemble.

    Each model votes with its argmax prediction. The final prediction
    is the class with the most votes (ties broken by lowest index).
    """

    def fit(
        self,
        probabilities: List[np.ndarray],
        labels: np.ndarray,
    ) -> "HardVoting":
        """No fitting needed for hard voting — returns self."""
        return self

    def predict(
        self,
        probabilities: List[np.ndarray],
    ) -> np.ndarray:
        """
        Majority vote across model predictions.

        Args:
            probabilities: List of ``M`` arrays, each ``(N, C)``.

        Returns:
            Predicted class indices of shape ``(N,)``.
        """
        # Get argmax predictions from each model: shape (M, N)
        predictions = np.array([p.argmax(axis=1) for p in probabilities])

        # Majority vote along model axis
        result = scipy_mode(predictions, axis=0, keepdims=False)
        mode_arr = np.asarray(result.mode).ravel()
        return mode_arr.astype(int)


class SoftVoting(EnsembleBase):
    """
    Soft Voting (Average Probabilities) Ensemble.

    Averages the probability vectors across all models and takes argmax.
    """

    def fit(
        self,
        probabilities: List[np.ndarray],
        labels: np.ndarray,
    ) -> "SoftVoting":
        """No fitting needed for soft voting — returns self."""
        return self

    def predict(
        self,
        probabilities: List[np.ndarray],
    ) -> np.ndarray:
        """
        Average probabilities and take argmax.

        Args:
            probabilities: List of ``M`` arrays, each ``(N, C)``.

        Returns:
            Predicted class indices of shape ``(N,)``.
        """
        avg_probs = np.mean(probabilities, axis=0)
        return np.argmax(avg_probs, axis=1)

    def predict_proba(
        self,
        probabilities: List[np.ndarray],
    ) -> np.ndarray:
        """Return averaged probability matrix."""
        return np.mean(probabilities, axis=0)


class WeightedVoting(EnsembleBase):
    """
    Weighted Voting Ensemble.

    Computes a weighted average of probability vectors. Weights can be:

    1. Set manually via the constructor.
    2. Optimized via :meth:`fit` using grid search over the validation set.

    Args:
        weights: Optional manual weights. If ``None``, :meth:`fit` will
                 find optimal weights via grid search.
        grid_resolution: Number of weight values to try in grid search (per model).
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        grid_resolution: int = 11,
    ) -> None:
        self.weights = weights
        self.grid_resolution = grid_resolution

    def fit(
        self,
        probabilities: List[np.ndarray],
        labels: np.ndarray,
    ) -> "WeightedVoting":
        """
        Optimize ensemble weights via grid search on accuracy.

        Only runs if no manual weights were provided.

        Args:
            probabilities: List of ``M`` arrays, each ``(N, C)``.
            labels: Ground truth labels of shape ``(N,)``.

        Returns:
            ``self`` with optimized ``self.weights``.
        """
        if self.weights is not None:
            return self

        num_models = len(probabilities)

        if num_models == 1:
            self.weights = [1.0]
            return self

        # Grid search
        weight_candidates = np.linspace(0, 1, self.grid_resolution)
        best_acc = -1.0
        best_weights: Optional[List[float]] = None

        for combo in product(weight_candidates, repeat=num_models):
            w = np.array(combo)
            w_sum = w.sum()
            if w_sum == 0:
                continue
            w = w / w_sum  # Normalize

            weighted_probs = sum(
                w_i * p for w_i, p in zip(w, probabilities)
            )
            preds = np.argmax(weighted_probs, axis=1)
            acc = np.mean(preds == labels)

            if acc > best_acc:
                best_acc = acc
                best_weights = w.tolist()

        self.weights = best_weights
        return self

    def predict(
        self,
        probabilities: List[np.ndarray],
    ) -> np.ndarray:
        """
        Weighted average of probabilities → argmax.

        Args:
            probabilities: List of ``M`` arrays, each ``(N, C)``.

        Returns:
            Predicted class indices of shape ``(N,)``.

        Raises:
            RuntimeError: If weights are not set (call ``fit()`` first).
        """
        if self.weights is None:
            raise RuntimeError(
                "Weights not set. Call fit() or provide weights manually."
            )

        weights = np.array(self.weights)
        weights = weights / weights.sum()

        weighted_probs = sum(
            w * p for w, p in zip(weights, probabilities)
        )
        return np.argmax(weighted_probs, axis=1)

    def predict_proba(
        self,
        probabilities: List[np.ndarray],
    ) -> np.ndarray:
        """Return weighted probability matrix."""
        if self.weights is None:
            raise RuntimeError(
                "Weights not set. Call fit() or provide weights manually."
            )

        weights = np.array(self.weights)
        weights = weights / weights.sum()

        return sum(w * p for w, p in zip(weights, probabilities))
