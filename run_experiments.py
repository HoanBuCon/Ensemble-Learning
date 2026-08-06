"""
Multi-Experiment Runner
=======================

Train all configured models sequentially, cache predictions, and
generate a comparison table for thesis analysis.

CLI::

    python run_experiments.py
    python run_experiments.py --configs configs/resnet50.yaml configs/swin_tiny.yaml

Output::

    outputs/comparison_table.csv
    outputs/comparison_accuracy.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List

import pandas as pd

from train import train as train_model
from src.utils.visualization import plot_comparison_bar


def count_parameters(model_name: str, num_classes: int = 6) -> int:
    """Count trainable parameters for a model."""
    from src.models.factory import create_model
    import torch

    model = create_model(model_name, pretrained=False, num_classes=num_classes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    torch.cuda.empty_cache()
    return n_params


def get_model_size_mb(save_dir: str) -> float:
    """Get the size of the best model checkpoint in MB."""
    path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def run_experiments(config_paths: List[str]) -> pd.DataFrame:
    """
    Train all models and generate a comparison table.

    Args:
        config_paths: List of YAML config file paths.

    Returns:
        DataFrame with the comparison table.
    """
    results_list: List[Dict[str, Any]] = []

    print(f"\n{'='*60}")
    print(f"  Running {len(config_paths)} experiments")
    print(f"{'='*60}\n")

    for i, config_path in enumerate(config_paths):
        print(f"\n{'-'*60}")
        print(f"  Experiment {i+1}/{len(config_paths)}: {config_path}")
        print(f"{'-'*60}\n")

        start_time = time.time()

        try:
            result = train_model(config_path)
            total_time = time.time() - start_time

            eval_metrics = result.get("eval_metrics", {})

            row = {
                "Model": result.get("model_name", "unknown"),
                "Experiment": result.get("experiment_name", "unknown"),
                "Accuracy": eval_metrics.get("accuracy", 0.0),
                "Precision": eval_metrics.get("precision", 0.0),
                "Recall": eval_metrics.get("recall", 0.0),
                "F1_Score": eval_metrics.get("f1_score", 0.0),
                "Best_Val_Accuracy": result.get("best_val_accuracy", 0.0),
                "Best_Epoch": result.get("best_epoch", 0),
                "Parameters": result.get("num_params", 0),
                "Parameters_M": result.get("num_params", 0) / 1e6,
                "Training_Time_s": total_time,
                "Inference_Time_s": eval_metrics.get("inference_time_seconds", 0.0),
                "Model_Size_MB": get_model_size_mb(
                    result.get("history", {}).get("save_dir", "")
                    if isinstance(result.get("history"), dict)
                    else ""
                ),
            }

            # Try to get model size from config
            from src.utils.config import load_config
            cfg = load_config(config_path)
            row["Model_Size_MB"] = get_model_size_mb(cfg.checkpoint.save_dir)

            results_list.append(row)

            print(f"\n  [OK] {row['Model']} - Accuracy: {row['Accuracy']:.4f}, "
                  f"F1: {row['F1_Score']:.4f}, Time: {total_time:.0f}s")

        except Exception as e:
            print(f"\n  [FAIL] {config_path}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Build comparison DataFrame
    df = pd.DataFrame(results_list)

    if len(df) > 0:
        # Sort by accuracy
        df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

        # Save comparison table
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, "comparison_table.csv")
        df.to_csv(csv_path, index=False)

        # Save as formatted markdown (useful for thesis)
        md_path = os.path.join(output_dir, "comparison_table.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Model Comparison Results\n\n")
            f.write(df.to_markdown(index=False))

        # Generate comparison plots
        metrics_dict = {}
        for _, row in df.iterrows():
            metrics_dict[row["Model"]] = {
                "accuracy": row["Accuracy"] * 100,
                "precision": row["Precision"] * 100,
                "recall": row["Recall"] * 100,
                "f1_score": row["F1_Score"] * 100,
            }

        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            plot_comparison_bar(metrics_dict, metric, output_dir)

        # Print summary table
        print(f"\n\n{'='*60}")
        print("  EXPERIMENT COMPARISON SUMMARY")
        print(f"{'='*60}\n")

        display_cols = [
            "Model", "Accuracy", "Precision", "Recall", "F1_Score",
            "Parameters_M", "Training_Time_s",
        ]
        display_df = df[display_cols].copy()
        display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x:.4f}")
        display_df["Precision"] = display_df["Precision"].apply(lambda x: f"{x:.4f}")
        display_df["Recall"] = display_df["Recall"].apply(lambda x: f"{x:.4f}")
        display_df["F1_Score"] = display_df["F1_Score"].apply(lambda x: f"{x:.4f}")
        display_df["Parameters_M"] = display_df["Parameters_M"].apply(lambda x: f"{x:.2f}M")
        display_df["Training_Time_s"] = display_df["Training_Time_s"].apply(lambda x: f"{x:.0f}s")

        print(display_df.to_string(index=False))
        print(f"\n  Results saved to: {csv_path}")
        print(f"  Markdown table: {md_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multiple training experiments"
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="List of config files. Defaults to all configs/*.yaml",
    )
    args = parser.parse_args()

    if args.configs:
        config_paths = args.configs
    else:
        config_paths = sorted(glob.glob("configs/*.yaml"))

    if not config_paths:
        print("No config files found. Provide --configs or add files to configs/")
        sys.exit(1)

    run_experiments(config_paths)
