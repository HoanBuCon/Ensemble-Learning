"""Fit, serialize, replay, and evaluate canonical scientific ensembles."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import yaml

from src.ensemble.artifacts import (
    PredictionArtifact,
    load_prediction_artifact,
    save_ensemble_manifest,
    save_prediction_artifact,
    validate_prediction_alignment,
)
from src.ensemble.stacking import StackingEnsemble
from src.ensemble.voting import HardVoting, SoftVoting, WeightedVoting
from src.utils.metrics import compute_metrics
from src.utils.provenance import (
    load_dataset_manifest_sha256,
    require_git_commit,
    sha256_file,
    verify_dataset_snapshot,
    write_experiment_manifest,
    write_json,
)
from src.utils.config import load_config as load_experiment_config
from src.utils.run_identity import (
    final_run_root,
    resolve_backbone_config_paths,
    validate_run_id,
    validate_protocol,
)


def _load_config(path: str = "configs/ensemble.yaml") -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not config.get("base_model_order"):
        raise ValueError(f"Invalid canonical ensemble config: {path}")
    return config


def _protocol_paths(
    protocol: str,
    results_root: str,
    base_model_order: Sequence[str],
    run_id: str,
) -> Tuple[Dict[str, str], Dict[str, str], str]:
    run_root = final_run_root(results_root, run_id)
    if protocol == "single_split":
        root = run_root / "single_split"
        fit_name = "val_predictions.npz"
        test_name = "test_predictions.npz"
        fit_split = "val"
        model_dirs = {name: root / name for name in base_model_order}
    elif protocol == "oof":
        root = run_root / "oof"
        fit_name = "oof_predictions.npz"
        test_name = "test_predictions.npz"
        fit_split = "oof_train"
        model_dirs = {name: root / name / "kfold" for name in base_model_order}
    else:
        raise ValueError("protocol must be exactly 'single_split' or 'oof'")

    fit_paths = {name: str(path / fit_name) for name, path in model_dirs.items()}
    test_paths = {name: str(path / test_name) for name, path in model_dirs.items()}
    return fit_paths, test_paths, fit_split


def _load_aligned(
    paths: Dict[str, str],
    model_order: Sequence[str],
    protocol: str,
    split: str,
    run_id: str,
) -> List[PredictionArtifact]:
    artifacts = [load_prediction_artifact(paths[name]) for name in model_order]
    validate_prediction_alignment(
        artifacts,
        expected_protocol=protocol,
        expected_split=split,
        expected_methods=model_order,
        expected_run_id=run_id,
        expected_dataset_manifest_sha256=load_dataset_manifest_sha256(),
    )
    return artifacts


def _approved_split_identity(
    manifest_path: str,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the immutable ordered IDs, labels, and class order for one split."""
    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"Frozen dataset manifest has no rows for split {split!r}")
    class_by_index: Dict[int, str] = {}
    for row in rows:
        index = int(row["class_index"])
        name = row["class_name"]
        previous = class_by_index.setdefault(index, name)
        if previous != name:
            raise ValueError(f"Frozen manifest class identity conflict at index {index}")
    expected_indices = list(range(len(class_by_index)))
    if sorted(class_by_index) != expected_indices:
        raise ValueError("Frozen manifest class indices must be contiguous from zero")
    return (
        np.asarray([row["sample_id"] for row in selected], dtype=str),
        np.asarray([int(row["class_index"]) for row in selected], dtype=np.int64),
        np.asarray([class_by_index[index] for index in expected_indices], dtype=str),
    )


def validate_ensemble_fit_test_identity(
    fit_artifacts: Sequence[PredictionArtifact],
    test_artifacts: Sequence[PredictionArtifact],
    *,
    model_order: Sequence[str],
    protocol: str,
    run_id: str,
    current_source_commit: str,
    expected_config_sha256: Dict[str, str],
    manifest_path: str = "artifacts/manifests/dataset_manifest.csv",
    n_splits: int = 5,
) -> None:
    """Prove approved split membership and one model lineage before any fit."""
    if len(fit_artifacts) != len(model_order) or len(test_artifacts) != len(model_order):
        raise ValueError("Fit/test backbone count does not match canonical model order")
    manifest_sha256 = sha256_file(manifest_path)
    fit_manifest_split = "validation" if protocol == "single_split" else "train"
    fit_ids, fit_labels, class_order = _approved_split_identity(
        manifest_path, fit_manifest_split
    )
    test_ids, test_labels, test_class_order = _approved_split_identity(
        manifest_path, "test"
    )
    if not np.array_equal(class_order, test_class_order):
        raise ValueError("Frozen manifest class order differs across fit/test splits")
    if set(fit_ids.tolist()).intersection(test_ids.tolist()):
        raise ValueError("Frozen fit/test populations are not disjoint")

    expected_fit_semantics = (
        "single_checkpoint_inference"
        if protocol == "single_split"
        else "one_outer_holdout_checkpoint_per_training_row"
    )
    expected_test_semantics = (
        "single_checkpoint_inference"
        if protocol == "single_split"
        else "mean_fold_probabilities"
    )
    expected_fold_keys = {f"fold_{index}" for index in range(1, n_splits + 1)}

    for method, fit_artifact, test_artifact in zip(
        model_order, fit_artifacts, test_artifacts
    ):
        for role, artifact, expected_ids, expected_labels, expected_semantics in (
            ("fit", fit_artifact, fit_ids, fit_labels, expected_fit_semantics),
            ("test", test_artifact, test_ids, test_labels, expected_test_semantics),
        ):
            artifact.validate()
            if artifact.method != method:
                raise ValueError(f"{role} method identity mismatch for {method}")
            if artifact.run_id != run_id:
                raise ValueError(f"{role} run identity mismatch for {method}")
            if artifact.dataset_manifest_sha256 != manifest_sha256:
                raise ValueError(f"{role} dataset identity mismatch for {method}")
            if artifact.source_commit != current_source_commit:
                raise ValueError(f"{role} source identity mismatch for {method}")
            if artifact.config_sha256 != expected_config_sha256.get(method):
                raise ValueError(f"{role} config identity mismatch for {method}")
            if not np.array_equal(artifact.class_order, class_order):
                raise ValueError(f"{role} class order mismatch for {method}")
            if artifact.aggregation_semantics != expected_semantics:
                raise ValueError(f"{role} aggregation semantics mismatch for {method}")
            if not np.array_equal(artifact.sample_ids, expected_ids):
                raise ValueError(f"{role} sample IDs do not match approved {fit_manifest_split if role == 'fit' else 'test'} rows for {method}")
            if not np.array_equal(artifact.y_true, expected_labels):
                raise ValueError(f"{role} labels do not match approved manifest for {method}")

        if fit_artifact.config_sha256 != test_artifact.config_sha256:
            raise ValueError(f"Fit/test config lineage mismatch for {method}")
        if fit_artifact.source_commit != test_artifact.source_commit:
            raise ValueError(f"Fit/test source lineage mismatch for {method}")
        if fit_artifact.artifact_hashes != test_artifact.artifact_hashes:
            raise ValueError(f"Fit/test checkpoint lineage mismatch for {method}")
        if protocol == "single_split":
            if set(fit_artifact.artifact_hashes) != {"checkpoint"}:
                raise ValueError(f"Single-split checkpoint lineage is incomplete for {method}")
        elif set(fit_artifact.artifact_hashes) != expected_fold_keys:
            raise ValueError(f"OOF five-fold checkpoint lineage is incomplete for {method}")


def _backbone_config_hashes(model_order: Sequence[str]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for config_path in resolve_backbone_config_paths():
        model_name = load_experiment_config(config_path).model.name
        if model_name in model_order:
            hashes[model_name] = sha256_file(config_path)
    missing = [name for name in model_order if name not in hashes]
    if missing:
        raise ValueError(f"Missing canonical backbone configs for {missing}")
    return hashes


def _save_method_prediction(
    path: Path,
    method: str,
    protocol: str,
    reference: PredictionArtifact,
    probabilities: np.ndarray,
    *,
    config_sha256: str,
    artifact_hashes: Dict[str, str],
    aggregation_semantics: str,
) -> None:
    save_prediction_artifact(
        str(path),
        sample_ids=reference.sample_ids,
        y_true=reference.y_true,
        probabilities=probabilities,
        predictions=np.argmax(probabilities, axis=1),
        class_order=reference.class_order,
        protocol=protocol,
        method=method,
        split="test",
        run_id=reference.run_id,
        dataset_manifest_sha256=reference.dataset_manifest_sha256,
        config_sha256=config_sha256,
        source_commit=reference.source_commit,
        artifact_hashes=artifact_hashes,
        aggregation_semantics=aggregation_semantics,
    )


def _metrics_row(method: str, method_type: str, metrics: Dict[str, object]) -> Dict[str, object]:
    return {
        "Method": method,
        "Type": method_type,
        "Accuracy": float(metrics["accuracy"]),
        "Precision": float(metrics["precision"]),
        "Recall": float(metrics["recall"]),
        "F1_Score": float(metrics["f1_score"]),
    }


def _record_method(
    method: str,
    method_type: str,
    probabilities: np.ndarray,
    protocol: str,
    reference: PredictionArtifact,
    prediction_dir: Path,
    class_order: List[str],
    rows: List[Dict[str, object]],
    full_metrics: Dict[str, object],
    *,
    config_sha256: str,
    artifact_hashes: Dict[str, str],
    aggregation_semantics: str,
) -> None:
    predictions = np.argmax(probabilities, axis=1)
    metrics = compute_metrics(reference.y_true, predictions, class_names=class_order)
    full_metrics[method] = metrics
    rows.append(_metrics_row(method, method_type, metrics))
    _save_method_prediction(
        prediction_dir / f"{method}.npz",
        method,
        protocol,
        reference,
        probabilities,
        config_sha256=config_sha256,
        artifact_hashes=artifact_hashes,
        aggregation_semantics=aggregation_semantics,
    )


def run_ensemble_evaluation(
    protocol: str,
    results_root: str = "RESULTS",
    ensemble_config_path: str = "configs/ensemble.yaml",
    *,
    run_id: str,
) -> pd.DataFrame:
    """Run a strict protocol-isolated ensemble evaluation without hidden refits."""
    run_id = validate_run_id(run_id)
    protocol = validate_protocol(protocol)
    verify_dataset_snapshot()
    config = _load_config(ensemble_config_path)
    model_order = [str(value) for value in config["base_model_order"]]
    fit_paths, test_paths, fit_split = _protocol_paths(
        protocol, results_root, model_order, run_id
    )
    fit_artifacts = _load_aligned(
        fit_paths, model_order, protocol=protocol, split=fit_split, run_id=run_id
    )
    test_artifacts = _load_aligned(
        test_paths, model_order, protocol=protocol, split="test", run_id=run_id
    )
    current_commit = require_git_commit()
    validate_ensemble_fit_test_identity(
        fit_artifacts,
        test_artifacts,
        model_order=model_order,
        protocol=protocol,
        run_id=run_id,
        current_source_commit=current_commit,
        expected_config_sha256=_backbone_config_hashes(model_order),
    )

    fit_probabilities = [artifact.probabilities for artifact in fit_artifacts]
    test_probabilities = [artifact.probabilities for artifact in test_artifacts]
    fit_labels = fit_artifacts[0].y_true
    test_reference = test_artifacts[0]
    class_order = test_reference.class_order.tolist()

    output_dir = final_run_root(results_root, run_id) / protocol / "ensembles"
    artifact_dir = output_dir / "ensemble_artifacts"
    prediction_dir = output_dir / "predictions"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    input_hashes = {name: sha256_file(fit_paths[name]) for name in model_order}
    test_input_hashes = {
        f"test_{name}": sha256_file(test_paths[name]) for name in model_order
    }
    ensemble_config_sha256 = sha256_file(ensemble_config_path)
    aggregation_semantics = (
        "mean_fold_probs_then_meta" if protocol == "oof"
        else "single_checkpoint_probs_then_meta"
    )
    model_artifact_paths: Dict[str, str] = {}
    rows: List[Dict[str, object]] = []
    full_metrics: Dict[str, object] = {}

    for name, artifact in zip(model_order, test_artifacts):
        metrics = compute_metrics(
            artifact.y_true, artifact.predictions, class_names=class_order
        )
        full_metrics[f"base_{name}"] = metrics
        rows.append(_metrics_row(f"base_{name}", "base", metrics))

    hard_probabilities = HardVoting().predict_proba(test_probabilities)
    _record_method(
        "hard_voting", "voting", hard_probabilities, protocol,
        test_reference, prediction_dir, class_order, rows, full_metrics,
        config_sha256=ensemble_config_sha256,
        artifact_hashes=test_input_hashes,
        aggregation_semantics=aggregation_semantics,
    )

    soft_probabilities = SoftVoting().predict_proba(test_probabilities)
    _record_method(
        "soft_voting", "voting", soft_probabilities, protocol,
        test_reference, prediction_dir, class_order, rows, full_metrics,
        config_sha256=ensemble_config_sha256,
        artifact_hashes=test_input_hashes,
        aggregation_semantics=aggregation_semantics,
    )

    weighted_config = config["weighted_voting"]
    weighted = WeightedVoting(
        epsilon=float(weighted_config["epsilon"]),
        max_iter=int(weighted_config["max_iterations"]),
        ftol=float(weighted_config["ftol"]),
    ).fit(fit_probabilities, fit_labels)
    weighted_path = artifact_dir / "weighted_voting.json"
    weighted.save(
        str(weighted_path),
        base_model_order=model_order,
        fit_split=fit_split,
        input_probability_hashes=input_hashes,
    )
    model_artifact_paths["weighted_voting"] = str(weighted_path)
    weighted_probabilities = WeightedVoting.load(str(weighted_path)).predict_proba(
        test_probabilities
    )
    _record_method(
        "weighted_voting", "voting", weighted_probabilities, protocol,
        test_reference, prediction_dir, class_order, rows, full_metrics,
        config_sha256=ensemble_config_sha256,
        artifact_hashes={**test_input_hashes, "weighted_voting": sha256_file(weighted_path)},
        aggregation_semantics=aggregation_semantics,
    )

    stacking_artifacts = {
        "logistic_regression": artifact_dir / "stacking_lr.joblib",
        "random_forest": artifact_dir / "stacking_rf.joblib",
        "xgboost": artifact_dir / "stacking_xgb.json",
    }
    for learner_name, artifact_path in stacking_artifacts.items():
        StackingEnsemble(
            learner_name, config_path=ensemble_config_path
        ).fit(fit_probabilities, fit_labels).save(str(artifact_path))
        model_artifact_paths[f"stacking_{learner_name}"] = str(artifact_path)
        replayed = StackingEnsemble(
            learner_name, config_path=ensemble_config_path
        ).load(str(artifact_path))
        probabilities = replayed.predict_proba(test_probabilities)
        _record_method(
            f"stacking_{learner_name}", "stacking", probabilities, protocol,
            test_reference, prediction_dir, class_order, rows, full_metrics,
            config_sha256=ensemble_config_sha256,
            artifact_hashes={**test_input_hashes, f"stacking_{learner_name}": sha256_file(artifact_path)},
            aggregation_semantics=aggregation_semantics,
        )

    save_ensemble_manifest(
        str(artifact_dir / "ensemble_manifest.json"),
        protocol=protocol,
        class_order=class_order,
        base_model_order=model_order,
        feature_dimension=sum(
            artifact.probabilities.shape[1] for artifact in fit_artifacts
        ),
        fit_split=fit_split,
        input_paths=fit_paths,
        model_artifact_paths=model_artifact_paths,
        run_id=run_id,
        dataset_manifest_sha256=test_reference.dataset_manifest_sha256,
        ensemble_config_sha256=ensemble_config_sha256,
        source_commit=test_reference.source_commit,
        aggregation_semantics=aggregation_semantics,
    )
    write_experiment_manifest(
        output_dir / "experiment_manifest.json",
        protocol=protocol,
        model="canonical_ensemble_suite",
        config_path=ensemble_config_path,
        seed=42,
        checkpoint_path=artifact_dir / "ensemble_manifest.json",
        prediction_path=prediction_dir / "weighted_voting.npz",
        arguments={
            "base_model_order": model_order,
            "fit_split": fit_split,
        },
        run_id=run_id,
    )
    write_json(str(output_dir / "ensemble_full_metrics.json"), full_metrics)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ensemble_comparison.csv", index=False)
    with (output_dir / "ensemble_comparison.md").open("w", encoding="utf-8") as handle:
        handle.write(f"# Canonical ensemble evaluation ({protocol})\n\n")
        handle.write(frame.to_markdown(index=False))
        handle.write("\n")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run canonical, protocol-isolated scientific ensembles"
    )
    parser.add_argument("--protocol", required=True, choices=["single_split", "oof"])
    parser.add_argument("--results-root", default="RESULTS")
    parser.add_argument("--ensemble-config", default="configs/ensemble.yaml")
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    run_ensemble_evaluation(
        protocol=arguments.protocol,
        results_root=arguments.results_root,
        ensemble_config_path=arguments.ensemble_config,
        run_id=arguments.run_id,
    )
