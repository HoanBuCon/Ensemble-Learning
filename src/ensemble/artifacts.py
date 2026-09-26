"""Strict, protocol-aware prediction and ensemble artifact helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from src.utils.provenance import sha256_file, sha256_text, write_json
from src.utils.run_identity import validate_run_id


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
    run_id: str
    dataset_manifest_sha256: str
    config_sha256: str
    source_commit: str
    artifact_hashes: Dict[str, str]
    aggregation_semantics: str

    def validate(self, probability_tolerance: float = 1e-6) -> None:
        if self.protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"Unsupported protocol identity: {self.protocol}")
        n_samples = len(self.sample_ids)
        if len(set(self.sample_ids.tolist())) != n_samples:
            raise ValueError("Prediction artifact contains duplicate sample_ids")
        if len(set(self.class_order.tolist())) != len(self.class_order):
            raise ValueError("Prediction artifact contains duplicate class names")
        if not self.method or not self.split or not self.run_id:
            raise ValueError("Prediction artifact method/split/run_id must be explicit")
        validate_run_id(self.run_id)
        for field_name, value in (
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            ("config_sha256", self.config_sha256),
        ):
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
                raise ValueError(f"Invalid {field_name}: {value!r}")
        if len(self.source_commit) != 40 or any(
            ch not in "0123456789abcdef" for ch in self.source_commit.lower()
        ):
            raise ValueError("source_commit must be a full 40-character hexadecimal Git SHA")
        if not self.artifact_hashes:
            raise ValueError("checkpoint/artifact hashes are required")
        for name, value in self.artifact_hashes.items():
            if not name or len(value) != 64 or any(
                ch not in "0123456789abcdef" for ch in value.lower()
            ):
                raise ValueError(f"Invalid artifact hash for {name!r}")
        if not self.aggregation_semantics:
            raise ValueError("aggregation_semantics must be explicit")
        if self.probabilities.ndim != 2:
            raise ValueError("probabilities must have shape (N, C)")
        if self.probabilities.shape != (n_samples, len(self.class_order)):
            raise ValueError("Probability shape does not match sample/class identity")
        if self.y_true.shape != (n_samples,) or self.predictions.shape != (n_samples,):
            raise ValueError("Label/prediction rows do not match sample_ids")
        if not np.all(np.isfinite(self.probabilities)):
            raise ValueError("Prediction artifact contains non-finite probabilities")
        if np.any(self.probabilities < -probability_tolerance):
            raise ValueError("Prediction probabilities are below zero tolerance")
        if np.any(self.probabilities > 1.0 + probability_tolerance):
            raise ValueError("Prediction probabilities exceed one tolerance")
        if not np.allclose(self.probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Prediction probabilities do not sum to one")
        n_classes = len(self.class_order)
        if np.any(self.y_true < 0) or np.any(self.y_true >= n_classes):
            raise ValueError("y_true contains an out-of-range class index")
        if np.any(self.predictions < 0) or np.any(self.predictions >= n_classes):
            raise ValueError("predictions contains an out-of-range class index")
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
    run_id: str,
    dataset_manifest_sha256: str,
    config_sha256: str,
    source_commit: str,
    artifact_hashes: Mapping[str, str],
    aggregation_semantics: str,
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
        run_id=str(run_id),
        dataset_manifest_sha256=str(dataset_manifest_sha256),
        config_sha256=str(config_sha256),
        source_commit=str(source_commit),
        artifact_hashes=dict(artifact_hashes),
        aggregation_semantics=str(aggregation_semantics),
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
        run_id=np.asarray(artifact.run_id),
        dataset_manifest_sha256=np.asarray(artifact.dataset_manifest_sha256),
        config_sha256=np.asarray(artifact.config_sha256),
        source_commit=np.asarray(artifact.source_commit),
        artifact_hashes_json=np.asarray(json.dumps(artifact.artifact_hashes, sort_keys=True)),
        aggregation_semantics=np.asarray(artifact.aggregation_semantics),
    )
    return sha256_file(target)


def load_prediction_artifact(path: str) -> PredictionArtifact:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Prediction artifact not found: {target}")
    with np.load(target, allow_pickle=False) as data:
        required = {
            "sample_ids", "y_true", "probabilities", "predictions",
            "class_order", "protocol", "method", "split", "run_id",
            "dataset_manifest_sha256", "config_sha256", "source_commit",
            "artifact_hashes_json", "aggregation_semantics",
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
            run_id=str(data["run_id"].item()),
            dataset_manifest_sha256=str(data["dataset_manifest_sha256"].item()),
            config_sha256=str(data["config_sha256"].item()),
            source_commit=str(data["source_commit"].item()),
            artifact_hashes=json.loads(str(data["artifact_hashes_json"].item())),
            aggregation_semantics=str(data["aggregation_semantics"].item()),
        )
    artifact.validate()
    return artifact


def validate_prediction_alignment(
    artifacts: Sequence[PredictionArtifact],
    *,
    expected_protocol: str,
    expected_split: str,
    expected_methods: Sequence[str] | None = None,
    expected_run_id: str | None = None,
    expected_dataset_manifest_sha256: str | None = None,
    expected_config_sha256: str | Mapping[str, str] | None = None,
    expected_artifact_hashes: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    """Fail closed unless all base artifacts identify exactly the same rows/classes."""
    if not artifacts:
        raise ValueError("No prediction artifacts provided")
    reference = artifacts[0]
    for artifact in artifacts:
        artifact.validate()
    if reference.protocol != expected_protocol or reference.split != expected_split:
        raise ValueError(
            f"Unexpected reference identity: {reference.protocol}/{reference.split}; "
            f"expected {expected_protocol}/{expected_split}"
        )
    if expected_methods is not None:
        if len(expected_methods) != len(artifacts):
            raise ValueError("expected_methods length does not match artifacts")
        for artifact, method in zip(artifacts, expected_methods):
            if artifact.method != method:
                raise ValueError(
                    f"Prediction method identity mismatch: {artifact.method} != {method}"
                )
    if expected_run_id is not None and reference.run_id != expected_run_id:
        raise ValueError("Prediction run identity mismatch")
    if (
        expected_dataset_manifest_sha256 is not None
        and reference.dataset_manifest_sha256 != expected_dataset_manifest_sha256
    ):
        raise ValueError("Prediction dataset manifest identity mismatch")
    for artifact in artifacts:
        if expected_config_sha256 is not None:
            expected_config = (
                expected_config_sha256.get(artifact.method)
                if isinstance(expected_config_sha256, Mapping)
                else expected_config_sha256
            )
            if expected_config is None or artifact.config_sha256 != expected_config:
                raise ValueError(
                    f"Prediction config identity mismatch for {artifact.method}"
                )
        if expected_artifact_hashes is not None:
            expected_hashes = expected_artifact_hashes.get(artifact.method)
            if expected_hashes is None or artifact.artifact_hashes != dict(expected_hashes):
                raise ValueError(
                    f"Prediction checkpoint/artifact identity mismatch for {artifact.method}"
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
        if artifact.run_id != reference.run_id:
            raise ValueError("run_id mismatch across prediction artifacts")
        if artifact.dataset_manifest_sha256 != reference.dataset_manifest_sha256:
            raise ValueError("dataset manifest mismatch across prediction artifacts")
        if artifact.source_commit != reference.source_commit:
            raise ValueError("source commit mismatch across prediction artifacts")


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
    run_id: str,
    dataset_manifest_sha256: str,
    ensemble_config_sha256: str,
    source_commit: str,
    aggregation_semantics: str,
) -> Dict[str, object]:
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported protocol identity: {protocol}")
    validate_run_id(run_id)
    for field_name, value, length in (
        ("dataset_manifest_sha256", dataset_manifest_sha256, 64),
        ("ensemble_config_sha256", ensemble_config_sha256, 64),
        ("source_commit", source_commit, 40),
    ):
        if len(value) != length or any(
            character not in "0123456789abcdef" for character in value.lower()
        ):
            raise ValueError(f"Invalid {field_name}: {value!r}")
    if not aggregation_semantics:
        raise ValueError("aggregation_semantics must be explicit")
    payload: Dict[str, object] = {
        "protocol": protocol,
        "class_order": list(class_order),
        "base_model_order": list(base_model_order),
        "feature_dimension": int(feature_dimension),
        "fit_split": fit_split,
        "run_id": run_id,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "ensemble_config_sha256": ensemble_config_sha256,
        "source_commit": source_commit,
        "aggregation_semantics": aggregation_semantics,
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
