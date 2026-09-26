"""t-SNE analysis with explicit, non-fallback feature semantics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from torch.utils.data import DataLoader

from scripts.verification.common_utils import (
    base_model_order,
    load_protocol_predictions,
    resolve_verification_output_dir,
)
from src.datasets.dataset import ImageFolderDataset
from src.datasets.transforms import build_transforms
from src.models.factory import create_model
from src.utils.config import load_config
from src.utils.provenance import write_json
from src.utils.run_identity import final_run_root


def feature_label(name: str, feature_source: str, matrix: np.ndarray) -> str:
    """Build plot text only from the actual feature matrix metadata."""
    return f"{name}: {feature_source} ({matrix.shape[1]}-D)"


def _probability_features(
    protocol: str,
    results_root: str,
    run_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    base, _ = load_protocol_predictions(protocol, results_root=results_root, run_id=run_id)
    order = list(base)
    reference = base[order[0]]
    single = base["swin_tiny"].probabilities
    ensemble = np.concatenate([base[name].probabilities for name in order], axis=1)
    return single, ensemble, reference.y_true, reference.class_order.tolist()


def _penultimate_features(
    results_root: str,
    device: torch.device,
    run_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    protocol = "single_split"
    base, _ = load_protocol_predictions(protocol, results_root=results_root, run_id=run_id)
    reference = next(iter(base.values()))
    order = base_model_order()
    config = load_config("configs/swin_tiny.yaml")
    transform = build_transforms(
        config.augmentation.test, image_size=config.data.image_size, stage="test"
    )
    dataset = ImageFolderDataset(root=os.path.join(config.data.root, "test"), transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    features: Dict[str, np.ndarray] = {}
    for model_name in order:
        checkpoint = final_run_root(results_root, run_id) / protocol / model_name / "best_model.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Deep-latent t-SNE requested but checkpoint is missing: {checkpoint}"
            )
        model = create_model(model_name, pretrained=False, num_classes=len(dataset.classes))
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        if model_name == "resnet50":
            model.fc = nn.Identity()
        elif model_name == "densenet121":
            model.classifier = nn.Identity()
        else:
            model.reset_classifier(0)
        model.to(device).eval()
        batches = []
        with torch.no_grad():
            for images, _, _ in loader:
                batches.append(model(images.to(device)).cpu().numpy())
        features[model_name] = np.concatenate(batches, axis=0)
    single = features["swin_tiny"]
    ensemble = np.concatenate([features[name] for name in order], axis=1)
    return single, ensemble, reference.y_true, reference.class_order.tolist()


def run_tsne_analysis(
    feature_source: str,
    protocol: str = "oof",
    results_root: str = "RESULTS",
    save_dir: Optional[str] = None,
    run_id: str = "",
) -> pd.DataFrame:
    if feature_source == "probability_vector":
        single, ensemble, y_true, class_order = _probability_features(protocol, results_root, run_id)
    elif feature_source == "penultimate_embedding":
        if protocol != "single_split":
            raise ValueError(
                "penultimate_embedding currently requires protocol=single_split; "
                "no OOF fold checkpoint may be selected implicitly"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        single, ensemble, y_true, class_order = _penultimate_features(results_root, device, run_id)
    else:
        raise ValueError(
            "feature_source must be 'probability_vector' or 'penultimate_embedding'"
        )

    projections = []
    rows = []
    for name, matrix in (("swin_tiny", single), ("ensemble_concatenation", ensemble)):
        projection = TSNE(
            n_components=2,
            perplexity=35,
            max_iter=1000,
            random_state=42,
        ).fit_transform(matrix)
        projections.append((name, projection, matrix.shape[1]))
        rows.append({
            "representation": name,
            "feature_source": feature_source,
            "feature_dimension": int(matrix.shape[1]),
            "silhouette_score": float(silhouette_score(matrix, y_true)),
            "davies_bouldin_index": float(davies_bouldin_score(matrix, y_true)),
            "calinski_harabasz_index": float(calinski_harabasz_score(matrix, y_true)),
        })

    output = resolve_verification_output_dir(results_root, run_id, save_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    palette = ["#ef4444", "#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#06b6d4"]
    for axis, (name, projection, dimension) in zip(axes, projections):
        for class_index, class_name in enumerate(class_order):
            mask = y_true == class_index
            axis.scatter(
                projection[mask, 0], projection[mask, 1],
                color=palette[class_index], label=class_name, alpha=0.75, s=26,
            )
        matrix = single if name == "swin_tiny" else ensemble
        axis.set_title(feature_label(name, feature_source, matrix))
        axis.set_xlabel("t-SNE dimension 1")
        axis.set_ylabel("t-SNE dimension 2")
        axis.legend(fontsize=8)
    figure.tight_layout()
    plot_path = output / f"tsne_{protocol}_{feature_source}.png"
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"tsne_{protocol}_{feature_source}_metrics.csv", index=False)
    write_json(
        output / f"tsne_{protocol}_{feature_source}_metadata.json",
        {
            "protocol": protocol,
            "feature_source": feature_source,
            "representations": rows,
            "plot": str(plot_path),
        },
    )
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explicit-source t-SNE analysis")
    parser.add_argument(
        "--feature-source",
        required=True,
        choices=["probability_vector", "penultimate_embedding"],
    )
    parser.add_argument("--protocol", choices=["single_split", "oof"], default="oof")
    parser.add_argument("--results-root", default="RESULTS")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_tsne_analysis(
        args.feature_source, args.protocol, args.results_root, args.save_dir,
        run_id=args.run_id,
    )
