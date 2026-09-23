"""Fit, serialize, replay, and evaluate canonical FINAL_V2 ensembles."""

from __future__ import annotations

import argparse
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
    sha256_file,
    verify_dataset_snapshot,
    write_experiment_manifest,
    write_json,
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
) -> Tuple[Dict[str, str], Dict[str, str], str]:
    if protocol == "single_split":
        root = Path(results_root) / "single_split"
        fit_name = "val_predictions.npz"
        test_name = "test_predictions.npz"
        fit_split = "val"
        model_dirs = {name: root / name for name in base_model_order}
    elif protocol == "oof":
        root = Path(results_root) / "oof"
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
) -> List[PredictionArtifact]:
    artifacts = [load_prediction_artifact(paths[name]) for name in model_order]
    validate_prediction_alignment(
        artifacts,
        expected_protocol=protocol,
        expected_split=split,
    )
    return artifacts


def _save_method_prediction(
    path: Path,
    method: str,
    protocol: str,
    reference: PredictionArtifact,
    probabilities: np.ndarray,
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
    )


def run_ensemble_evaluation(
    protocol: str | None = None,
    results_root: str = "RESULTS/FINAL_V2",
    ensemble_config_path: str = "configs/ensemble.yaml",
    *,
    mode: str | None = None,
    outputs_dir: str | None = None,
) -> pd.DataFrame:
    """Run a strict protocol-isolated ensemble evaluation without hidden refits."""
    verify_dataset_snapshot()
    if protocol is None:
        aliases = {"val": "single_split", "single_split": "single_split", "oof": "oof"}
        if mode not in aliases:
            raise ValueError("An explicit single_split or oof protocol is required")
        protocol = aliases[mode]
    if outputs_dir is not None:
        normalized = os.path.normpath(outputs_dir)
        expected = os.path.normpath(results_root)
        if normalized != expected:
            raise ValueError(
                "Historical/cross-protocol output roots are rejected; "
                f"expected {expected}, received {normalized}"
            )
    config = _load_config(ensemble_config_path)
    model_order = [str(value) for value in config["base_model_order"]]
    fit_paths, test_paths, fit_split = _protocol_paths(
        protocol, results_root, model_order
    )
    fit_artifacts = _load_aligned(
        fit_paths, model_order, protocol=protocol, split=fit_split
    )
    test_artifacts = _load_aligned(
        test_paths, model_order, protocol=protocol, split="test"
    )
    if not np.array_equal(fit_artifacts[0].class_order, test_artifacts[0].class_order):
        raise ValueError("Fit/test class_order mismatch")

    fit_probabilities = [artifact.probabilities for artifact in fit_artifacts]
    test_probabilities = [artifact.probabilities for artifact in test_artifacts]
    fit_labels = fit_artifacts[0].y_true
    test_reference = test_artifacts[0]
    class_order = test_reference.class_order.tolist()

    output_dir = Path(results_root) / protocol / "ensembles"
    artifact_dir = output_dir / "ensemble_artifacts"
    prediction_dir = output_dir / "predictions"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    input_hashes = {name: sha256_file(fit_paths[name]) for name in model_order}
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
    )

    soft_probabilities = SoftVoting().predict_proba(test_probabilities)
    _record_method(
        "soft_voting", "voting", soft_probabilities, protocol,
        test_reference, prediction_dir, class_order, rows, full_metrics,
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
        description="Run canonical, protocol-isolated FINAL_V2 ensembles"
    )
    parser.add_argument("--protocol", required=True, choices=["single_split", "oof"])
    parser.add_argument("--results-root", default="RESULTS/FINAL_V2")
    parser.add_argument("--ensemble-config", default="configs/ensemble.yaml")
    arguments = parser.parse_args()
    run_ensemble_evaluation(
        protocol=arguments.protocol,
        results_root=arguments.results_root,
        ensemble_config_path=arguments.ensemble_config,
    )
