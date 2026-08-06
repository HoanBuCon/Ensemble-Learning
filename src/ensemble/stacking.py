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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.ensemble.base import EnsembleBase


def _get_xgboost():
    """Lazy import XGBoost — returns None if not installed."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        return None


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

    _LEARNER_MAP = {
        "logistic_regression": lambda **kw: LogisticRegression(
            max_iter=1000, solver="lbfgs",
            random_state=42, **kw,
        ),
        "random_forest": lambda **kw: RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, **kw,
        ),
    }

    def __init__(
        self,
        meta_learner: str = "logistic_regression",
        meta_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.meta_learner_name = meta_learner
        self.meta_params = meta_params or {}
        self._model = None

        self._build_model()

    def _build_model(self) -> None:
        """Instantiate the meta-learner."""
        name = self.meta_learner_name.lower()

        if name == "xgboost":
            XGBClassifier = _get_xgboost()
            if XGBClassifier is None:
                raise ImportError(
                    "XGBoost is not installed. Install with: pip install xgboost"
                )
            self._model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="mlogloss",
                **self.meta_params,
            )
        elif name in self._LEARNER_MAP:
            self._model = self._LEARNER_MAP[name](**self.meta_params)
        else:
            available = list(self._LEARNER_MAP.keys()) + ["xgboost"]
            raise ValueError(
                f"Unknown meta-learner '{name}'. Available: {available}"
            )

    @staticmethod
    def _stack_features(probabilities: List[np.ndarray]) -> np.ndarray:
        """
        Concatenate probability vectors from all models into a feature matrix.

        Args:
            probabilities: List of ``M`` arrays, each ``(N, C)``.

        Returns:
            Stacked feature matrix of shape ``(N, M * C)``.
        """
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
        joblib.dump(self._model, path)

    def load(self, path: str) -> "StackingEnsemble":
        """
        Load a previously saved meta-learner.

        Args:
            path: File path to the serialized model.

        Returns:
            ``self`` with loaded meta-learner.
        """
        self._model = joblib.load(path)
        return self
