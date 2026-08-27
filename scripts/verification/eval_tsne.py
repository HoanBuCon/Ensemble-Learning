"""
Latent Feature Space Quality & t-SNE 2D Manifold Visualizer
==========================================================

Generates 2D t-SNE projections comparing:
1. Best Single Backbone (Swin-Tiny, 768-D Deep Penultimate Latent Space)
2. Ensemble Multi-Model Concatenated Fusion Space (5120-D Deep Latent Embedding)

Computes Quantitative Clustering Separation Metrics:
- Silhouette Score ([-1, +1] - Higher is Better)
- Davies-Bouldin Index (Lower is Better)
- Calinski-Harabasz Index (Higher is Better)

Usage:
    python scripts/verification/eval_tsne.py
    python scripts/verification/eval_tsne.py --save-dir outputs/verification
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_dataset_config
from src.datasets.dataset import ImageFolderDataset
from src.models.factory import create_model
from scripts.verification.common_utils import resolve_outputs_dirs


def extract_penultimate_features(
    def_dir: str,
    oof_dir: str,
    test_dir: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Extract deep penultimate embeddings for all backbones on the test set."""
    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    test_ds = ImageFolderDataset(root=test_dir, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    y_true = np.array([s[1] for s in test_ds.samples])
    class_names = test_ds.classes

    models = ["densenet121", "efficientnet_b0", "resnet50", "swin_tiny"]
    all_features = {}

    for m in models:
        ckpt_path = None
        candidates = [
            os.path.join(def_dir, m, "best_model.pth"),
            os.path.join(oof_dir, m, "kfold", "fold_1", "best_model.pth"),
            os.path.join(oof_dir, m, "best_model.pth"),
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break

        if not ckpt_path:
            print(f"Warning: Checkpoint for {m} not found. Searching outputs/...")
            continue

        print(f"-> Extracting penultimate feature embeddings for {m} from '{ckpt_path}'...")
        model = create_model(m, pretrained=False, num_classes=len(class_names))
        state = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)

        # Strip final classification head to expose deep latent representation
        if m == "resnet50":
            model.fc = nn.Identity()
        elif m == "densenet121":
            model.classifier = nn.Identity()
        elif m in ["efficientnet_b0", "swin_tiny"]:
            model.reset_classifier(0)

        model.to(device)
        model.eval()

        feats_list = []
        with torch.no_grad():
            for imgs, _, _ in test_loader:
                imgs = imgs.to(device)
                feat = model(imgs)
                feats_list.append(feat.cpu().numpy())

        all_features[m] = np.vstack(feats_list)
        print(f"   [Feature Extracted] {m} shape: {all_features[m].shape}")

    return all_features, y_true, class_names


def run_tsne_analysis(
    default_dir: Optional[str] = None,
    oof_dir: Optional[str] = None,
    save_dir: Optional[str] = "outputs/verification",
) -> pd.DataFrame:
    """Execute t-SNE projection on deep penultimate feature embeddings and compute cluster separation."""
    def_dir, oof_dir = resolve_outputs_dirs(default_dir, oof_dir)
    ds_raw = load_dataset_config()
    test_dir = ds_raw.get("test_dir", os.path.join(PROJECT_ROOT, "data", "test"))

    print("\n" + "=" * 90)
    print("      t-SNE 2D DEEP LATENT FEATURE EMBEDDING & CLUSTER SEPARATION BENCHMARK")
    print("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    all_features, y_true, class_names = extract_penultimate_features(
        def_dir=def_dir,
        oof_dir=oof_dir,
        test_dir=test_dir,
        device=device,
    )

    models = ["densenet121", "efficientnet_b0", "resnet50", "swin_tiny"]
    if "swin_tiny" in all_features and len(all_features) == 4:
        swin_feats = all_features["swin_tiny"]  # (1546, 768)
        ensemble_feats = np.hstack([all_features[m] for m in models])  # (1546, 5120)
        swin_dim_label = "768-D Penultimate Latent Space"
        ens_dim_label = "5120-D Concatenated Multi-Backbone Feature Space"
    else:
        # Fallback to probability vectors if checkpoints are unavailable
        print("Fallback: Using cached probability matrices for t-SNE projection.")
        oof_probs = [
            np.load(os.path.join(oof_dir, m, "kfold", "test_probabilities.npy") if os.path.exists(os.path.join(oof_dir, m, "kfold", "test_probabilities.npy")) else os.path.join(oof_dir, m, "test_probabilities.npy")) 
            for m in models
        ]
        swin_feats = oof_probs[3]
        ensemble_feats = np.hstack(oof_probs)
        swin_dim_label = "6-D Softmax Vector"
        ens_dim_label = "24-D Meta-Probability Vector"

    # Compute t-SNE 2D Projections
    print("-> Computing 2D t-SNE manifold projections (Perplexity=35, n_iter=1000)...")
    tsne_swin_model = TSNE(n_components=2, perplexity=35, n_iter=1000, random_state=42)
    tsne_swin = tsne_swin_model.fit_transform(swin_feats)

    tsne_ens_model = TSNE(n_components=2, perplexity=35, n_iter=1000, random_state=42)
    tsne_ensemble = tsne_ens_model.fit_transform(ensemble_feats)

    # Compute Quantitative Clustering Metrics
    print("-> Computing quantitative cluster separation metrics...")
    sil_swin = float(silhouette_score(swin_feats, y_true))
    db_swin = float(davies_bouldin_score(swin_feats, y_true))
    ch_swin = float(calinski_harabasz_score(swin_feats, y_true))

    sil_ens = float(silhouette_score(ensemble_feats, y_true))
    db_ens = float(davies_bouldin_score(ensemble_feats, y_true))
    ch_ens = float(calinski_harabasz_score(ensemble_feats, y_true))

    metrics_rows = [
        {
            "Representation Space": "Single Backbone (Swin-Tiny)",
            "Dimension": swin_dim_label,
            "Silhouette Score": f"{sil_swin:.4f}",
            "Davies-Bouldin Index": f"{db_swin:.4f}",
            "Calinski-Harabasz": f"{ch_swin:.1f}",
        },
        {
            "Representation Space": "Ensemble Fusion (4 Multi-Scale Backbones)",
            "Dimension": ens_dim_label,
            "Silhouette Score": f"{sil_ens:.4f}",
            "Davies-Bouldin Index": f"{db_ens:.4f}",
            "Calinski-Harabasz": f"{ch_ens:.1f}",
        }
    ]
    df_metrics = pd.DataFrame(metrics_rows)
    print("\n" + df_metrics.to_string(index=False))
    print("-" * 90)

    # Plot 2D Comparison Figure
    palette = ["#ef4444", "#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#06b6d4"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))

    # Subplot 1: Single Model
    for c_idx, c_name in enumerate(class_names):
        mask = (y_true == c_idx)
        axes[0].scatter(
            tsne_swin[mask, 0],
            tsne_swin[mask, 1],
            c=palette[c_idx],
            label=f"{c_name}",
            alpha=0.75,
            s=28,
            edgecolors="none",
        )
    axes[0].set_title(
        f"(a) Single Backbone Deep Latent Space (Swin-Tiny, {swin_dim_label})\nSilhouette: {sil_swin:.4f} | Davies-Bouldin: {db_swin:.4f}",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[0].set_xlabel("t-SNE Dimension 1", fontsize=10)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(loc="best", fontsize=8.5, framealpha=0.85)

    # Subplot 2: Ensemble Fusion Space
    for c_idx, c_name in enumerate(class_names):
        mask = (y_true == c_idx)
        axes[1].scatter(
            tsne_ensemble[mask, 0],
            tsne_ensemble[mask, 1],
            c=palette[c_idx],
            label=f"{c_name}",
            alpha=0.8,
            s=32,
            edgecolors="none",
        )
    axes[1].set_title(
        f"(b) Ensemble Multi-Backbone Concatenated Space (5120-D Deep Latent)\nSilhouette: {sil_ens:.4f} | Davies-Bouldin: {db_ens:.4f}",
        fontsize=11.5,
        fontweight="bold",
        color="#10b981",
    )
    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=10)
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(loc="best", fontsize=8.5, framealpha=0.85)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_plot = os.path.join(save_dir, "tsne_latent_space.png")
    out_svg = os.path.join(save_dir, "tsne_latent_space.svg")
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.savefig(out_plot, format="png", dpi=300, bbox_inches="tight")
    plt.close()

    csv_p = os.path.join(save_dir, "tsne_clustering_metrics.csv")
    md_p = os.path.join(save_dir, "tsne_clustering_metrics.md")
    df_metrics.to_csv(csv_p, index=False)
    with open(md_p, "w", encoding="utf-8") as f:
        f.write("# Latent Feature Space Clustering Benchmark (t-SNE Quality)\n\n")
        f.write(df_metrics.to_markdown(index=False))
        f.write("\n\n![t-SNE 2D Feature Space](tsne_latent_space.png)\n")

    print(f"[t-SNE Plot Saved] -> '{out_svg}' and '{out_plot}'")
    print(f"[Metrics Saved]    -> '{csv_p}' and '{md_p}'")
    print("=" * 90)
    return df_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE Feature Space Analysis")
    parser.add_argument("--save-dir", default="outputs/verification", help="Directory to save output reports")
    args = parser.parse_args()
    run_tsne_analysis(save_dir=args.save_dir)
