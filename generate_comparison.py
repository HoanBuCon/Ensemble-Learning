"""
Base Model Comparison Generator
===============================

Standalone script & module to extract model metrics (Validation & Test),
format percentage comparison tables, export CSV/Markdown artifacts, and
plot metric comparison bar charts.

CLI Usage::

    python generate_comparison.py
    python generate_comparison.py --outputs-dir outputs
    python generate_comparison.py --configs configs/resnet50.yaml configs/swin_tiny.yaml

Python API::

    from generate_comparison import generate_base_comparison_report

    df = generate_base_comparison_report(outputs_dir="outputs")
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.factory import create_model
from src.utils.config import load_config
from src.utils.metrics import compute_metrics
from src.utils.visualization import plot_comparison_bar


def count_parameters(model_name: str, num_classes: int = 6) -> int:
    """Count trainable parameters for a registered model."""
    import torch

    model = create_model(model_name, pretrained=False, num_classes=num_classes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n_params


def get_model_size_mb(save_dir: str) -> float:
    """Get the size of the best model checkpoint in MB."""
    path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def extract_model_history_info(save_dir: str) -> Dict[str, Any]:
    """Extract best_epoch and best_val_accuracy from best_model.pth or training_history.json."""
    import torch

    best_pth = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_pth):
        try:
            ckpt = torch.load(best_pth, map_location="cpu", weights_only=False)
            epoch = ckpt.get("epoch", 0)
            best_val = ckpt.get("best_value", 0.0)
            if epoch > 0 and best_val > 0.0:
                return {
                    "best_epoch": int(epoch),
                    "best_val_accuracy": float(best_val),
                }
        except Exception:
            pass

    history_path = os.path.join(save_dir, "training_history.json")
    best_epoch = 0
    best_val_acc = 0.0

    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            val_accs = data.get("val_accuracy", [])
            if val_accs:
                best_idx = int(np.argmax(val_accs))
                best_epoch = best_idx + 1  # 1-indexed epoch
                best_val_acc = float(val_accs[best_idx])
        except Exception:
            pass

    return {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
    }


def extract_model_val_metrics(save_dir: str) -> Dict[str, float]:
    """Compute full validation metrics (Acc, Precision, Recall, F1) from val_probabilities.npy."""
    val_prob_path = os.path.join(save_dir, "val_probabilities.npy")
    val_label_path = os.path.join(save_dir, "val_labels.npy")

    if os.path.exists(val_prob_path) and os.path.exists(val_label_path):
        try:
            val_probs = np.load(val_prob_path)
            val_labels = np.load(val_label_path)
            preds = np.argmax(val_probs, axis=1)
            m = compute_metrics(val_labels, preds)
            return {
                "Val_Accuracy": m["accuracy"] * 100,
                "Val_Precision": m["precision"] * 100,
                "Val_Recall": m["recall"] * 100,
                "Val_F1_Score": m["f1_score"] * 100,
            }
        except Exception:
            pass

    hist_info = extract_model_history_info(save_dir)
    best_val = hist_info["best_val_accuracy"]
    return {
        "Val_Accuracy": best_val,
        "Val_Precision": best_val,
        "Val_Recall": best_val,
        "Val_F1_Score": best_val,
    }


def generate_base_comparison_report(
    outputs_dir: str = "outputs",
    config_paths: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate complete base model comparison metrics, plots, and CSV/Markdown reports.

    Args:
        outputs_dir: Main output directory (default: 'outputs').
        config_paths: Optional list of specific YAML config paths to inspect.

    Returns:
        DataFrame containing formatted comparison metrics.
    """
    import re

    # Discover candidate model directories
    if config_paths:
        model_dirs = []
        for p in config_paths:
            cfg = load_config(p)
            save_d = cfg.checkpoint.save_dir
            if os.path.isdir(save_d):
                model_dirs.append(save_d)
    else:
        all_dirs = sorted(glob.glob(os.path.join(outputs_dir, "*")))
        valid_dirs = [
            d for d in all_dirs
            if os.path.isdir(d) and (
                os.path.exists(os.path.join(d, "metrics.json")) or
                os.path.exists(os.path.join(d, "probabilities.npy")) or
                os.path.exists(os.path.join(d, "test_probabilities.npy"))
            )
        ]

        # Group by base model name to select latest version if versioned
        model_groups: Dict[str, List[tuple[int, str]]] = {}
        for d in valid_dirs:
            folder_name = os.path.basename(d)
            if folder_name in ["val", "oof"]:
                continue
            match = re.match(r"^(.*?)(?:_(\d+))?$", folder_name)
            if match:
                base_name = match.group(1)
                version = int(match.group(2)) if match.group(2) else 0
                if base_name not in model_groups:
                    model_groups[base_name] = []
                model_groups[base_name].append((version, d))

        model_dirs = []
        for base_name, versions in sorted(model_groups.items()):
            versions.sort(key=lambda x: x[0], reverse=True)
            model_dirs.append(versions[0][1])

    if not model_dirs:
        print(f"No completed model outputs found in '{outputs_dir}'. Train models first.")
        return pd.DataFrame()

    results_list: List[Dict[str, Any]] = []

    for m_dir in model_dirs:
        m_name = os.path.basename(m_dir)
        # Extract base model name if versioned
        base_name = re.sub(r"_\d+$", "", m_name)

        metrics_path = os.path.join(m_dir, "metrics.json")
        eval_metrics: Dict[str, Any] = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    eval_metrics = json.load(f)
            except Exception:
                pass

        n_params = count_parameters(base_name)
        val_metrics = extract_model_val_metrics(m_dir)
        hist_info = extract_model_history_info(m_dir)

        row = {
            "Model": base_name,
            "Experiment": f"{m_name}_baseline",
            "Best_Epoch": hist_info["best_epoch"],
            "Val_Accuracy": val_metrics["Val_Accuracy"],
            "Val_Precision": val_metrics["Val_Precision"],
            "Val_Recall": val_metrics["Val_Recall"],
            "Val_F1_Score": val_metrics["Val_F1_Score"],
            "Test_Accuracy": eval_metrics.get("accuracy", 0.0) * 100,
            "Test_Precision": eval_metrics.get("precision", 0.0) * 100,
            "Test_Recall": eval_metrics.get("recall", 0.0) * 100,
            "Test_F1_Score": eval_metrics.get("f1_score", 0.0) * 100,
            "Parameters": n_params,
            "Parameters_M": n_params / 1e6,
            "Training_Time_s": 0.0,
            "Inference_Time_s": eval_metrics.get("inference_time_seconds", 0.0),
            "Model_Size_MB": get_model_size_mb(m_dir),
        }
        results_list.append(row)

    df = pd.DataFrame(results_list)
    if len(df) == 0:
        return pd.DataFrame()

    df = df.sort_values("Test_Accuracy", ascending=False).reset_index(drop=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # Plot comparison bar charts for primary metrics
    metrics_dict = {}
    for _, row in df.iterrows():
        metrics_dict[row["Model"]] = {
            "accuracy": row["Test_Accuracy"],
            "precision": row["Test_Precision"],
            "recall": row["Test_Recall"],
            "f1_score": row["Test_F1_Score"],
        }

    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        plot_comparison_bar(metrics_dict, metric, outputs_dir)

    # Format percentage strings for Markdown & CSV output
    formatted_df = df.copy()
    for col in [
        "Val_Accuracy", "Val_Precision", "Val_Recall", "Val_F1_Score",
        "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1_Score"
    ]:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}%")

    if "Parameters_M" in formatted_df.columns:
        formatted_df["Parameters_M"] = formatted_df["Parameters_M"].apply(lambda x: f"{x:.2f}M")
    if "Model_Size_MB" in formatted_df.columns:
        formatted_df["Model_Size_MB"] = formatted_df["Model_Size_MB"].apply(lambda x: f"{x:.2f}MB")

    csv_path = os.path.join(outputs_dir, "comparison_table.csv")
    formatted_df.to_csv(csv_path, index=False)

    md_path = os.path.join(outputs_dir, "comparison_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Base Model Comparison Results (Validation vs Test Metrics)\n\n")
        f.write(formatted_df.to_markdown(index=False))

    print(f"\n\n{'='*60}")
    print("  EXPERIMENT COMPARISON SUMMARY (BASE MODELS)")
    print(f"{'='*60}\n")

    display_cols = [
        "Model", "Best_Epoch",
        "Val_Accuracy", "Val_Precision", "Val_Recall", "Val_F1_Score",
        "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1_Score",
        "Parameters_M", "Model_Size_MB",
    ]
    display_cols = [c for c in display_cols if c in formatted_df.columns]
    display_df = formatted_df[display_cols]

    print(display_df.to_string(index=False))
    print(f"\n  Results saved to: {csv_path}")
    print(f"  Markdown table: {md_path}\n")

    return formatted_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Base Model Comparison Report and Bar Charts"
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing model outputs (default: 'outputs')",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional list of YAML config files to inspect",
    )
    args = parser.parse_args()

    generate_base_comparison_report(
        outputs_dir=args.outputs_dir,
        config_paths=args.configs,
    )
