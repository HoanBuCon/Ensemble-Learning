"""Fail-closed loaders for canonical scientific verification artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json

import yaml

from src.ensemble.artifacts import (
    PredictionArtifact,
    load_prediction_artifact,
    validate_prediction_alignment,
)
from src.utils.provenance import load_dataset_manifest_sha256, sha256_file
from src.utils.run_identity import final_run_root, validate_run_id


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


def protocol_root(results_root: str, run_id: str, protocol: str) -> Path:
    if protocol not in {"single_split", "oof"}:
        raise ValueError("protocol must be exactly 'single_split' or 'oof'")
    root = final_run_root(results_root, validate_run_id(run_id)) / protocol
    if not root.is_dir():
        raise FileNotFoundError(f"Protocol result root not found: {root}")
    return root


def load_protocol_predictions(
    protocol: str,
    results_root: str = "RESULTS",
    ensemble_config_path: str = "configs/ensemble.yaml",
    run_id: str = "",
) -> Tuple[Dict[str, PredictionArtifact], Dict[str, PredictionArtifact]]:
    """Load base and saved ensemble test predictions without reconstructing models."""
    run_id = validate_run_id(run_id)
    root = protocol_root(results_root, run_id, protocol)
    order = base_model_order(ensemble_config_path)
    base: Dict[str, PredictionArtifact] = {}
    base_paths: Dict[str, Path] = {}
    base_config_hashes: Dict[str, str] = {}
    base_checkpoint_hashes: Dict[str, Dict[str, str]] = {}
    for model_name in order:
        model_root = root / model_name
        if protocol == "oof":
            model_root = model_root / "kfold"
        prediction_path = model_root / "test_predictions.npz"
        base_paths[model_name] = prediction_path
        base[model_name] = load_prediction_artifact(str(prediction_path))
        base_config_hashes[model_name] = sha256_file(f"configs/{model_name}.yaml")
        if protocol == "single_split":
            base_checkpoint_hashes[model_name] = {
                "checkpoint": sha256_file(model_root / "best_model.pth")
            }
        else:
            base_checkpoint_hashes[model_name] = {
                f"fold_{index}": sha256_file(model_root / f"fold_{index}" / "best_model.pth")
                for index in range(1, 6)
            }
    validate_prediction_alignment(
        list(base.values()), expected_protocol=protocol, expected_split="test",
        expected_methods=order,
        expected_run_id=run_id,
        expected_dataset_manifest_sha256=load_dataset_manifest_sha256(),
        expected_config_sha256=base_config_hashes,
        expected_artifact_hashes=base_checkpoint_hashes,
    )

    ensembles: Dict[str, PredictionArtifact] = {}
    prediction_root = root / "ensembles" / "predictions"
    for method in ENSEMBLE_METHODS:
        ensembles[method] = load_prediction_artifact(
            str(prediction_root / f"{method}.npz")
        )
    ensemble_config_sha256 = sha256_file(ensemble_config_path)
    input_hashes = {
        f"test_{name}": sha256_file(path) for name, path in base_paths.items()
    }
    artifact_root = root / "ensembles" / "ensemble_artifacts"
    ensemble_hashes: Dict[str, Dict[str, str]] = {
        "hard_voting": dict(input_hashes),
        "soft_voting": dict(input_hashes),
        "weighted_voting": {
            **input_hashes,
            "weighted_voting": sha256_file(artifact_root / "weighted_voting.json"),
        },
        "stacking_logistic_regression": {
            **input_hashes,
            "stacking_logistic_regression": sha256_file(artifact_root / "stacking_lr.joblib"),
        },
        "stacking_random_forest": {
            **input_hashes,
            "stacking_random_forest": sha256_file(artifact_root / "stacking_rf.joblib"),
        },
        "stacking_xgboost": {
            **input_hashes,
            "stacking_xgboost": sha256_file(artifact_root / "stacking_xgb.json"),
        },
    }
    validate_prediction_alignment(
        list(ensembles.values()), expected_protocol=protocol, expected_split="test",
        expected_methods=ENSEMBLE_METHODS,
        expected_run_id=run_id,
        expected_dataset_manifest_sha256=load_dataset_manifest_sha256(),
        expected_config_sha256=ensemble_config_sha256,
        expected_artifact_hashes=ensemble_hashes,
    )
    reference = next(iter(base.values()))
    manifest_path = artifact_root / "ensemble_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Ensemble manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_manifest = {
        "protocol": protocol,
        "run_id": run_id,
        "dataset_manifest_sha256": reference.dataset_manifest_sha256,
        "ensemble_config_sha256": ensemble_config_sha256,
        "source_commit": reference.source_commit,
        "class_order": reference.class_order.tolist(),
        "base_model_order": order,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"Ensemble manifest identity mismatch at {key}")
    for method, artifact in ensembles.items():
        validate_prediction_alignment(
            [reference, artifact], expected_protocol=protocol, expected_split="test"
        )
        if artifact.method != method:
            raise ValueError(
                f"Prediction method identity mismatch: {artifact.method} != {method}"
            )
    return base, ensembles


def verification_output_dir(
    results_root: str = "RESULTS", run_id: str = ""
) -> Path:
    path = final_run_root(results_root, validate_run_id(run_id)) / "verification"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_verification_output_dir(
    results_root: str, run_id: str, requested: str | None = None
) -> Path:
    expected = final_run_root(results_root, validate_run_id(run_id)) / "verification"
    if requested is not None and Path(requested).resolve() != expected.resolve():
        raise ValueError(
            f"Verification output must stay inside the selected run: {expected}"
        )
    expected.mkdir(parents=True, exist_ok=True)
    return expected
