"""Strict, protocol-aware prediction and ensemble artifact helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.utils.provenance import sha256_file, sha256_text, write_json


SUPPORTED_PROTOCOLS = {"single_split", "oof"}


def sample_id_from_path(path: str, project_root: str = ".") -> str:
    """Create the same stable ID used by the frozen dataset manifest."""
    resolved = Path(path).resolve()
    project = Path(project_root).resolve()
    try:
        relative = resolved.relative_to(project).as_posix()
    except ValueError as exc:
        raise ValueError(f"Sample path is outside project root: {resolved}") from exc
    return sha256_text(relative)


def sample_ids_from_paths(paths: Sequence[str], project_root: str = ".") -> np.ndarray:
    return np.asarray(
        [sample_id_from_path(path, project_root=project_root) for path in paths],
        dtype=str,
    )


@dataclass(frozen=True)
class PredictionArtifact:
    sample_ids: np.ndarray
    y_true: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    class_order: np.ndarray
    protocol: str
    method: str
    split: str

    def validate(self) -> None:
        if self.protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"Unsupported protocol identity: {self.protocol}")
        n_samples = len(self.sample_ids)
        if len(set(self.sample_ids.tolist())) != n_samples:
            raise ValueError("Prediction artifact contains duplicate sample_ids")
        if self.probabilities.ndim != 2:
            raise ValueError("probabilities must have shape (N, C)")
        if self.probabilities.shape != (n_samples, len(self.class_order)):
            raise ValueError("Probability shape does not match sample/class identity")
        if self.y_true.shape != (n_samples,) or self.predictions.shape != (n_samples,):
            raise ValueError("Label/prediction rows do not match sample_ids")
        if not np.all(np.isfinite(self.probabilities)):
            raise ValueError("Prediction artifact contains non-finite probabilities")
        if not np.allclose(self.probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Prediction probabilities do not sum to one")
        if not np.array_equal(
            self.predictions.astype(np.int64),
            np.argmax(self.probabilities, axis=1).astype(np.int64),
        ):
            raise ValueError("Predictions do not equal probability argmax")


def save_prediction_artifact(
    path: str,
    *,
    sample_ids: Sequence[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    class_order: Sequence[str],
    protocol: str,
    method: str,
    split: str,
) -> str:
    artifact = PredictionArtifact(
        sample_ids=np.asarray(sample_ids, dtype=str),
        y_true=np.asarray(y_true, dtype=np.int64),
        probabilities=np.asarray(probabilities, dtype=np.float64),
        predictions=np.asarray(predictions, dtype=np.int64),
        class_order=np.asarray(class_order, dtype=str),
        protocol=str(protocol),
        method=str(method),
        split=str(split),
    )
    artifact.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        sample_ids=artifact.sample_ids,
        y_true=artifact.y_true,
        probabilities=artifact.probabilities,
        predictions=artifact.predictions,
        class_order=artifact.class_order,
        protocol=np.asarray(artifact.protocol),
        method=np.asarray(artifact.method),
        split=np.asarray(artifact.split),
    )
    return sha256_file(target)


def load_prediction_artifact(path: str) -> PredictionArtifact:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Prediction artifact not found: {target}")
    with np.load(target, allow_pickle=False) as data:
        required = {
            "sample_ids", "y_true", "probabilities", "predictions",
            "class_order", "protocol", "method", "split",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Prediction artifact missing fields {sorted(missing)}: {target}")
        artifact = PredictionArtifact(
            sample_ids=data["sample_ids"].astype(str),
            y_true=data["y_true"].astype(np.int64),
            probabilities=data["probabilities"].astype(np.float64),
            predictions=data["predictions"].astype(np.int64),
            class_order=data["class_order"].astype(str),
            protocol=str(data["protocol"].item()),
            method=str(data["method"].item()),
            split=str(data["split"].item()),
        )
    artifact.validate()
    return artifact


def validate_prediction_alignment(
    artifacts: Sequence[PredictionArtifact],
    *,
    expected_protocol: str,
    expected_split: str,
) -> None:
    """Fail closed unless all base artifacts identify exactly the same rows/classes."""
    if not artifacts:
        raise ValueError("No prediction artifacts provided")
    reference = artifacts[0]
    if reference.protocol != expected_protocol or reference.split != expected_split:
        raise ValueError(
            f"Unexpected reference identity: {reference.protocol}/{reference.split}; "
            f"expected {expected_protocol}/{expected_split}"
        )
    for artifact in artifacts[1:]:
        if artifact.protocol != expected_protocol or artifact.split != expected_split:
            raise ValueError("Cross-protocol or cross-split artifact mixing rejected")
        if not np.array_equal(artifact.sample_ids, reference.sample_ids):
            raise ValueError("sample_id ordering mismatch across base models")
        if not np.array_equal(artifact.y_true, reference.y_true):
            raise ValueError("y_true mismatch across base models")
        if not np.array_equal(artifact.class_order, reference.class_order):
            raise ValueError("class_order mismatch across base models")
        if artifact.probabilities.shape != reference.probabilities.shape:
            raise ValueError("Probability shape mismatch across base models")


def save_ensemble_manifest(
    path: str,
    *,
    protocol: str,
    class_order: Sequence[str],
    base_model_order: Sequence[str],
    feature_dimension: int,
    fit_split: str,
    input_paths: Dict[str, str],
    model_artifact_paths: Dict[str, str],
) -> Dict[str, object]:
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported protocol identity: {protocol}")
    payload: Dict[str, object] = {
        "protocol": protocol,
        "class_order": list(class_order),
        "base_model_order": list(base_model_order),
        "feature_dimension": int(feature_dimension),
        "fit_split": fit_split,
        "input_hashes": {
            name: sha256_file(value) for name, value in input_paths.items()
        },
        "model_artifact_hashes": {
            name: sha256_file(value) for name, value in model_artifact_paths.items()
        },
        "model_artifact_paths": dict(model_artifact_paths),
    }
    write_json(path, payload)
    return payload
