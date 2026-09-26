"""
Evaluator
=========

Standalone evaluation pipeline that takes a trained model and dataloader,
computes metrics, generates reports, and saves all artifacts.

Usage::

    from src.engine.evaluator import evaluate_model

    metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        class_names=class_names,
        output_dir="./outputs/resnet50",
        device="cuda",
    )
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.ensemble.artifacts import sample_ids_from_paths, save_prediction_artifact
from src.utils.metrics import compute_metrics
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_precision_recall_curves,
    plot_roc_curves,
)


def run_inference(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], float]:
    """
    Run inference on a dataloader and collect all outputs.

    Args:
        model: Trained model in eval mode.
        dataloader: DataLoader to run inference on.
        device: Device string.

    Returns:
        Tuple of ``(logits, probabilities, predictions, labels, paths, time)``.
    """
    model.eval()
    model.to(device)

    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_paths: List[str] = []

    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1]

            logits = model(images)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
            if len(batch) < 3:
                raise ValueError("Scientific inference requires source paths for sample identity")
            all_paths.extend(str(path) for path in batch[2])

    inference_time = time.time() - start_time

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Softmax probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    predictions = np.argmax(probabilities, axis=1)

    return logits, probabilities, predictions, labels, all_paths, inference_time


def save_predictions(
    logits: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    paths: List[str],
    output_dir: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    split: str = "test",
    protocol: str = "single_split",
    *,
    method: str,
    run_id: str,
    dataset_manifest_sha256: str,
    config_sha256: str,
    source_commit: str,
    artifact_hashes: Dict[str, str],
    aggregation_semantics: str,
    write_legacy_arrays: bool = False,
) -> None:
    """
    Save prediction arrays as ``.npy`` files for ensemble caching.

    Args:
        logits: Raw model outputs of shape ``(N, C)``.
        probabilities: Softmax probabilities of shape ``(N, C)``.
        predictions: Predicted class indices of shape ``(N,)``.
        labels: Ground truth labels of shape ``(N,)``.
        output_dir: Directory to save the arrays.
        class_to_idx: Optional class-to-index mapping dictionary.
        split: Dataset split identifier ('test', 'val', or 'train').
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{split}_" if split else ""

    if write_legacy_arrays:
        np.save(os.path.join(output_dir, f"{prefix}logits.npy"), logits)
        np.save(os.path.join(output_dir, f"{prefix}probabilities.npy"), probabilities)
        np.save(os.path.join(output_dir, f"{prefix}predictions.npy"), predictions)
        np.save(os.path.join(output_dir, f"{prefix}labels.npy"), labels)

    if class_to_idx is not None:
        class_mapping_path = os.path.join(output_dir, f"{split}_class_to_idx.json")
        with open(class_mapping_path, "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, indent=2)
        class_order = [
            name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
        ]
    else:
        class_order = [str(index) for index in range(probabilities.shape[1])]

    save_prediction_artifact(
        os.path.join(output_dir, f"{split}_predictions.npz"),
        sample_ids=sample_ids_from_paths(paths),
        y_true=labels,
        probabilities=probabilities,
        predictions=predictions,
        class_order=class_order,
        protocol=protocol,
        method=method,
        split=split,
        run_id=run_id,
        dataset_manifest_sha256=dataset_manifest_sha256,
        config_sha256=config_sha256,
        source_commit=source_commit,
        artifact_hashes=artifact_hashes,
        aggregation_semantics=aggregation_semantics,
    )


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    output_dir: str,
    device: str = "cpu",
    split: str = "test",
    protocol: str = "single_split",
    *,
    method: str,
    run_id: str,
    dataset_manifest_sha256: str,
    config_sha256: str,
    source_commit: str,
    artifact_hashes: Dict[str, str],
    aggregation_semantics: str = "single_checkpoint_inference",
) -> Dict[str, Any]:
    """
    Full evaluation pipeline: inference → metrics → reports → saved artifacts.

    Generates:
        - ``metrics.json``
        - ``confusion_matrix.png``
        - ``classification_report.txt``
        - ``class_to_idx.json``
        - ``<split>_probabilities.npy``, ``<split>_predictions.npy``, ``<split>_labels.npy``, ``<split>_logits.npy``

    Args:
        model: Trained model.
        dataloader: Evaluation DataLoader.
        class_names: List of class names.
        output_dir: Directory for output artifacts.
        device: Device string.
        split: Dataset split name ('test' or 'val').

    Returns:
        Metrics dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract class_to_idx if available in dataset
    dataset = getattr(dataloader, "dataset", None)
    class_to_idx = getattr(dataset, "class_to_idx", None)

    # Run inference
    logits, probabilities, predictions, labels, paths, inference_time = run_inference(
        model, dataloader, device
    )

    # Save prediction cache
    save_predictions(
        logits, probabilities, predictions, labels, paths, output_dir,
        class_to_idx=class_to_idx, split=split, protocol=protocol,
        method=method,
        run_id=run_id,
        dataset_manifest_sha256=dataset_manifest_sha256,
        config_sha256=config_sha256,
        source_commit=source_commit,
        artifact_hashes=artifact_hashes,
        aggregation_semantics=aggregation_semantics,
    )

    # Compute metrics
    metrics = compute_metrics(labels, predictions, class_names=class_names)
    metrics["inference_time_seconds"] = inference_time
    metrics["num_samples"] = len(labels)

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, f"{split}_metrics.json")
    serializable = {
        k: v for k, v in metrics.items()
        if k != "classification_report"
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)

    # Save classification report text
    report_path = os.path.join(output_dir, f"{split}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(metrics["classification_report"])

    # Generate comprehensive visualizations
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm, class_names, output_dir,
        title=f"Confusion Matrix ({split.title()} Split)",
        normalize=False, filename=f"{split}_confusion_matrix.png",
    )
    plot_confusion_matrix(
        cm, class_names, output_dir,
        title=f"Normalized Confusion Matrix ({split.title()} Split)",
        normalize=True, filename=f"{split}_confusion_matrix_normalized.png",
    )

    per_class = metrics.get("per_class", {})
    if per_class:
        plot_per_class_metrics(
            per_class, class_names, output_dir,
            filename=f"{split}_per_class_metrics.png",
        )

    plot_roc_curves(
        labels, probabilities, class_names, output_dir,
        filename=f"{split}_roc_curves.png",
    )
    plot_precision_recall_curves(
        labels, probabilities, class_names, output_dir,
        filename=f"{split}_precision_recall_curves.png",
    )

    return metrics
