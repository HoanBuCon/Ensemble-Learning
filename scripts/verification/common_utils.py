"""Fail-closed loaders for canonical FINAL_V2 verification artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from src.ensemble.artifacts import (
    PredictionArtifact,
    load_prediction_artifact,
    validate_prediction_alignment,
)


ENSEMBLE_METHODS = [
    "hard_voting",
    "soft_voting",
    "weighted_voting",
    "stacking_logistic_regression",
    "stacking_random_forest",
    "stacking_xgboost",
]


def base_model_order(config_path: str = "configs/ensemble.yaml") -> List[str]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    order = config.get("base_model_order") if isinstance(config, dict) else None
    if not isinstance(order, list) or not order:
        raise ValueError(f"Missing base_model_order in {config_path}")
    return [str(value) for value in order]


def protocol_root(results_root: str, protocol: str) -> Path:
    if protocol not in {"single_split", "oof"}:
        raise ValueError("protocol must be exactly 'single_split' or 'oof'")
    root = Path(results_root) / protocol
    if not root.is_dir():
        raise FileNotFoundError(f"Protocol result root not found: {root}")
    return root


def load_protocol_predictions(
    protocol: str,
    results_root: str = "RESULTS/FINAL_V2",
    ensemble_config_path: str = "configs/ensemble.yaml",
) -> Tuple[Dict[str, PredictionArtifact], Dict[str, PredictionArtifact]]:
    """Load base and saved ensemble test predictions without reconstructing models."""
    root = protocol_root(results_root, protocol)
    order = base_model_order(ensemble_config_path)
    base: Dict[str, PredictionArtifact] = {}
    for model_name in order:
        model_root = root / model_name
        if protocol == "oof":
            model_root = model_root / "kfold"
        base[model_name] = load_prediction_artifact(
            str(model_root / "test_predictions.npz")
        )
    validate_prediction_alignment(
        list(base.values()), expected_protocol=protocol, expected_split="test"
    )

    ensembles: Dict[str, PredictionArtifact] = {}
    prediction_root = root / "ensembles" / "predictions"
    for method in ENSEMBLE_METHODS:
        ensembles[method] = load_prediction_artifact(
            str(prediction_root / f"{method}.npz")
        )
    validate_prediction_alignment(
        list(ensembles.values()), expected_protocol=protocol, expected_split="test"
    )
    reference = next(iter(base.values()))
    for method, artifact in ensembles.items():
        validate_prediction_alignment(
            [reference, artifact], expected_protocol=protocol, expected_split="test"
        )
        if artifact.method != method:
            raise ValueError(
                f"Prediction method identity mismatch: {artifact.method} != {method}"
            )
    return base, ensembles


def verification_output_dir(results_root: str = "RESULTS/FINAL_V2") -> Path:
    path = Path(results_root) / "verification"
    path.mkdir(parents=True, exist_ok=True)
    return path
