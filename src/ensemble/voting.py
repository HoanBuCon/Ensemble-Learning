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

import json
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
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

    def predict_proba(
        self,
        probabilities: List[np.ndarray],
    ) -> np.ndarray:
        """Return class vote fractions so argmax is the hard-vote prediction."""
        predictions = np.asarray([p.argmax(axis=1) for p in probabilities])
        num_classes = probabilities[0].shape[1]
        return np.stack(
            [
                (predictions == class_index).mean(axis=0)
                for class_index in range(num_classes)
            ],
            axis=1,
        )


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

    Weights are fitted with constrained SLSQP by minimizing multiclass
    negative log-likelihood on a non-test fit split.

    Args:
        weights: Previously fitted weights, used only when loading an artifact.
        epsilon: Probability clipping floor used by the log-loss objective.
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        epsilon: float = 1e-12,
        max_iter: int = 1000,
        ftol: float = 1e-12,
    ) -> None:
        self.weights = weights
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.ftol = float(ftol)
        self.optimizer_result: Optional[Dict[str, Any]] = None

    def fit(
        self,
        probabilities: List[np.ndarray],
        labels: np.ndarray,
    ) -> "WeightedVoting":
        """
        Optimize non-negative simplex weights with SLSQP and multiclass log-loss.

        Args:
            probabilities: List of ``M`` arrays, each ``(N, C)``.
            labels: Ground truth labels of shape ``(N,)``.

        Returns:
            ``self`` with optimized ``self.weights``.
        """
        num_models = len(probabilities)
        if num_models == 0:
            raise ValueError("WeightedVoting.fit requires at least one base model")
        reference_shape = probabilities[0].shape
        if len(reference_shape) != 2 or len(labels) != reference_shape[0]:
            raise ValueError("Probability and label shapes are incompatible")
        for index, matrix in enumerate(probabilities):
            if matrix.shape != reference_shape:
                raise ValueError(
                    f"Base probability shape mismatch at index {index}: "
                    f"{matrix.shape} != {reference_shape}"
                )
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"Non-finite probabilities at base index {index}")

        if num_models == 1:
            self.weights = [1.0]
            objective_value = self._objective(
                np.array(self.weights, dtype=np.float64), probabilities, labels
            )
            self.optimizer_result = {
                "success": True,
                "message": "Single base model; optimization not required",
                "objective_value": objective_value,
                "num_iterations": 0,
            }
            return self

        initial = np.full(num_models, 1.0 / num_models, dtype=np.float64)
        result = minimize(
            fun=self._objective,
            x0=initial,
            args=(probabilities, labels),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * num_models,
            constraints=[{
                "type": "eq",
                "fun": lambda weights: float(np.sum(weights) - 1.0),
            }],
            options={"maxiter": self.max_iter, "ftol": self.ftol},
        )
        if not result.success:
            raise RuntimeError(
                f"SLSQP weighted-voting optimization failed: {result.message}"
            )
        weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
        weights /= weights.sum()
        self.weights = weights.tolist()
        self.optimizer_result = {
            "success": bool(result.success),
            "message": str(result.message),
            "objective_value": float(result.fun),
            "num_iterations": int(result.nit),
        }
        return self

    def _objective(
        self,
        weights: np.ndarray,
        probabilities: List[np.ndarray],
        labels: np.ndarray,
    ) -> float:
        combined = np.sum(
            np.stack(probabilities, axis=0) * weights[:, None, None],
            axis=0,
        )
        true_class_probabilities = combined[
            np.arange(len(labels)), np.asarray(labels, dtype=np.int64)
        ]
        clipped = np.clip(true_class_probabilities, self.epsilon, 1.0)
        return float(-np.mean(np.log(clipped)))

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

    def save(
        self,
        path: str,
        *,
        base_model_order: List[str],
        fit_split: str,
        input_probability_hashes: Dict[str, str],
    ) -> Dict[str, Any]:
        """Persist fitted weights and optimizer provenance as structured JSON."""
        if self.weights is None or self.optimizer_result is None:
            raise RuntimeError("WeightedVoting must be fitted before serialization")
        if len(base_model_order) != len(self.weights):
            raise ValueError("base_model_order length does not match fitted weights")
        payload: Dict[str, Any] = {
            "method": "SLSQP",
            "objective": "multiclass_log_loss",
            "weights": [float(value) for value in self.weights],
            "base_model_order": list(base_model_order),
            "success": self.optimizer_result["success"],
            "optimizer_message": self.optimizer_result["message"],
            "objective_value": self.optimizer_result["objective_value"],
            "num_iterations": self.optimizer_result["num_iterations"],
            "fit_split": fit_split,
            "input_probability_hashes": dict(input_probability_hashes),
            "epsilon": self.epsilon,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return payload

    @classmethod
    def load(cls, path: str) -> "WeightedVoting":
        """Load a fitted weighted-voting artifact without refitting."""
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("method") != "SLSQP" or payload.get("objective") != "multiclass_log_loss":
            raise ValueError(f"Unsupported weighted-voting artifact: {path}")
        instance = cls(weights=payload["weights"], epsilon=payload.get("epsilon", 1e-12))
        instance.optimizer_result = {
            "success": bool(payload["success"]),
            "message": payload["optimizer_message"],
            "objective_value": float(payload["objective_value"]),
            "num_iterations": int(payload["num_iterations"]),
        }
        return instance
