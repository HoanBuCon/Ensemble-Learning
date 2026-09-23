"""
Stacking Ensemble
=================

Meta-learner ensemble that trains a second-stage classifier on
probability vectors from base models.

Supports:
    - Logistic Regression (default)
    - Random Forest
    - XGBoost (optional — degrades gracefully if not installed)

Operates on OOF (Out-of-Fold) predictions for training to avoid
data leakage, and averaged test predictions for inference.

Usage::

    from src.ensemble.stacking import StackingEnsemble
    import numpy as np

    # OOF probabilities from each base model: shape (N_train, C) each
    oof_probs = [
        np.load("outputs/resnet50/oof_probabilities.npy"),
        np.load("outputs/densenet121/oof_probabilities.npy"),
    ]
    oof_labels = np.load("outputs/resnet50/oof_labels.npy")

    # Test probabilities
    test_probs = [
        np.load("outputs/resnet50/probabilities.npy"),
        np.load("outputs/densenet121/probabilities.npy"),
    ]

    stacker = StackingEnsemble(meta_learner="logistic_regression")
    stacker.fit(oof_probs, oof_labels)
    predictions = stacker.predict(test_probs)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.ensemble.base import EnsembleBase


def _get_xgboost():
    """Lazy import XGBoost and fail explicitly when it is unavailable."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required for the canonical xgboost stacker"
        ) from exc


def load_stacking_config(path: str = "configs/ensemble.yaml") -> Dict[str, Dict[str, Any]]:
    """Load the single canonical meta-learner configuration."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    stacking = payload.get("stacking") if isinstance(payload, dict) else None
    required = {"logistic_regression", "random_forest", "xgboost"}
    if not isinstance(stacking, dict) or set(stacking) != required:
        raise ValueError(
            f"Canonical stacking config must define exactly {sorted(required)}"
        )
    return stacking


class StackingEnsemble(EnsembleBase):
    """
    Stacking Ensemble with a configurable meta-learner.

    The meta-learner is trained on concatenated probability vectors
    from all base models: input shape = ``(N, M * C)`` where
    ``M`` = number of models and ``C`` = number of classes.

    Args:
        meta_learner: Meta-learner type. One of
                      ``'logistic_regression'``, ``'random_forest'``, ``'xgboost'``.
        meta_params: Optional keyword arguments for the meta-learner constructor.
    """

    def __init__(
        self,
        meta_learner: str = "logistic_regression",
        meta_params: Optional[Dict[str, Any]] = None,
        config_path: str = "configs/ensemble.yaml",
    ) -> None:
        self.meta_learner_name = meta_learner
        canonical = load_stacking_config(config_path)
        if meta_learner not in canonical:
            raise ValueError(
                f"Unknown meta-learner '{meta_learner}'. Available: {sorted(canonical)}"
            )
        self.meta_params = dict(canonical[meta_learner])
        if meta_params:
            self.meta_params.update(meta_params)
        self.config_path = config_path
        self._model = None
        self.feature_dimension: Optional[int] = None

        self._build_model()

    def _build_model(self) -> None:
        """Instantiate the meta-learner."""
        name = self.meta_learner_name.lower()

        if name == "xgboost":
            XGBClassifier = _get_xgboost()
            self._model = XGBClassifier(**self.meta_params)
        elif name == "logistic_regression":
            self._model = LogisticRegression(**self.meta_params)
        elif name == "random_forest":
            self._model = RandomForestClassifier(**self.meta_params)
        else:
            raise AssertionError(f"Unhandled canonical meta-learner: {name}")

    @staticmethod
    def _stack_features(probabilities: List[np.ndarray]) -> np.ndarray:
        """
        Concatenate probability vectors from all models into a feature matrix.

        Args:
            probabilities: List of ``M`` arrays, each ``(N, C)``.

        Returns:
            Stacked feature matrix of shape ``(N, M * C)``.
        """
        if not probabilities:
            raise ValueError("Stacking requires at least one probability matrix")
        row_count = probabilities[0].shape[0]
        if any(
            matrix.ndim != 2 or matrix.shape[0] != row_count
            for matrix in probabilities
        ):
            raise ValueError("Stacking probability matrices are not row-aligned")
        return np.concatenate(probabilities, axis=1)

    def fit(
        self,
        probabilities: List[np.ndarray],
        labels: np.ndarray,
    ) -> "StackingEnsemble":
        """
        Train the meta-learner on stacked probability features.

        **Important**: Use OOF predictions (not regular training predictions)
        to avoid data leakage.

        Args:
            probabilities: List of ``M`` OOF probability arrays, each ``(N, C)``.
            labels: Ground truth labels of shape ``(N,)``.

        Returns:
            ``self`` with a trained meta-learner.
        """
        X = self._stack_features(probabilities)
        self.feature_dimension = int(X.shape[1])
        self._model.fit(X, labels)
        return self

    def predict(
        self,
        probabilities: List[np.ndarray],
    ) -> np.ndarray:
        """
        Generate ensemble predictions via the trained meta-learner.

        Args:
            probabilities: List of ``M`` test probability arrays, each ``(N, C)``.

        Returns:
            Predicted class indices of shape ``(N,)``.
        """
        X = self._stack_features(probabilities)
        return self._model.predict(X)

    def predict_proba(
        self,
        probabilities: List[np.ndarray],
    ) -> np.ndarray:
        """
        Generate probability estimates from the meta-learner.

        Args:
            probabilities: List of ``M`` test probability arrays, each ``(N, C)``.

        Returns:
            Meta-learner probability estimates of shape ``(N, C)``.
        """
        X = self._stack_features(probabilities)
        return self._model.predict_proba(X)

    def save(self, path: str) -> None:
        """
        Save the trained meta-learner to disk.

        Args:
            path: File path for the serialized model.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not hasattr(self._model, "classes_"):
            raise RuntimeError("Stacking model must be fitted before serialization")
        if self.meta_learner_name == "xgboost":
            if not path.lower().endswith(".json"):
                raise ValueError("Canonical XGBoost artifact must use a .json path")
            self._model.save_model(path)
        else:
            joblib.dump(self._model, path)

    def load(self, path: str) -> "StackingEnsemble":
        """
        Load a previously saved meta-learner.

        Args:
            path: File path to the serialized model.

        Returns:
            ``self`` with loaded meta-learner.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stacking artifact not found: {path}")
        if self.meta_learner_name == "xgboost":
            self._model.load_model(path)
        else:
            self._model = joblib.load(path)
        feature_count = getattr(self._model, "n_features_in_", None)
        self.feature_dimension = int(feature_count) if feature_count is not None else None
        return self
