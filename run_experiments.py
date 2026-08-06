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


def inspect_model_status(config_path: str) -> Dict[str, Any]:
    """
    Inspect the training status of a model from its output directory.

    Returns:
        Dict with status ('COMPLETED', 'RESUMABLE', 'NOT_STARTED'), last_epoch,
        total_epochs, and save_dir.
    """
    from src.utils.config import load_config
    import torch

    cfg = load_config(config_path)
    save_dir = cfg.checkpoint.save_dir
    total_epochs = cfg.train.epochs
    metrics_path = os.path.join(save_dir, "metrics.json")
    last_ckpt_path = os.path.join(save_dir, "last_model.pth")

    if os.path.exists(metrics_path):
        return {
            "config_path": config_path,
            "model_name": cfg.model.name,
            "experiment_name": cfg.experiment_name,
            "status": "COMPLETED",
            "last_epoch": total_epochs,
            "total_epochs": total_epochs,
            "save_dir": save_dir,
        }

    if os.path.exists(last_ckpt_path):
        try:
            ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
            last_epoch = ckpt.get("epoch", 0)
            return {
                "config_path": config_path,
                "model_name": cfg.model.name,
                "experiment_name": cfg.experiment_name,
                "status": "RESUMABLE",
                "last_epoch": last_epoch,
                "total_epochs": total_epochs,
                "save_dir": save_dir,
            }
        except Exception:
            pass

    return {
        "config_path": config_path,
        "model_name": cfg.model.name,
        "experiment_name": cfg.experiment_name,
        "status": "NOT_STARTED",
        "last_epoch": 0,
        "total_epochs": total_epochs,
        "save_dir": save_dir,
    }


def display_status_table(statuses: List[Dict[str, Any]]) -> None:
    """Display a Rich table of model training statuses."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        title="[bold white]Experiment Status Overview[/bold white]",
        header_style="bold cyan",
        border_style="bright_blue",
    )
    table.add_column("Model Name", style="bold white")
    table.add_column("Config Path", style="dim white")
    table.add_column("Status", justify="center")
    table.add_column("Progress", justify="right")
    table.add_column("Action", style="dim white")

    for s in statuses:
        status_str = s["status"]
        if status_str == "COMPLETED":
            status_style = "[bold green]COMPLETED[/bold green]"
            action_style = "Skip (Already evaluated)"
            progress_str = f"{s['last_epoch']}/{s['total_epochs']}"
        elif status_str == "RESUMABLE":
            status_style = "[bold yellow]RESUMABLE[/bold yellow]"
            action_style = f"Resume at Epoch {s['last_epoch'] + 1}"
            progress_str = f"{s['last_epoch']}/{s['total_epochs']}"
        else:
            status_style = "[dim white]NOT STARTED[/dim white]"
            action_style = "Train from scratch (Epoch 1)"
            progress_str = f"0/{s['total_epochs']}"

        table.add_row(
            s["model_name"],
            s["config_path"],
            status_style,
            progress_str,
            action_style,
        )

    console.print(table)
    console.print()


def run_experiments(config_paths: List[str], mode: str = "auto") -> pd.DataFrame:
    """
    Train all models and generate a comparison table.

    Args:
        config_paths: List of YAML config file paths.
        mode: Training mode ('scratch', 'resume', 'auto').

    Returns:
        DataFrame with the comparison table.
    """
    results_list: List[Dict[str, Any]] = []

    statuses = [inspect_model_status(p) for p in config_paths]
    display_status_table(statuses)

    # Determine mode if interactive
    has_resumable = any(s["status"] == "RESUMABLE" for s in statuses)

    if mode == "auto" and has_resumable:
        print("Interrupted experiments detected!")
        print("  [1] Resume interrupted experiments (Recommended)")
        print("  [2] Train all from scratch")
        try:
            choice = input("Select mode [1/2] (default: 1): ").strip()
            if choice == "2":
                mode = "scratch"
            else:
                mode = "resume"
        except (KeyboardInterrupt, EOFError):
            mode = "resume"
    elif mode == "auto":
        mode = "scratch"

    is_resume_mode = (mode == "resume")

    print(f"\n{'='*60}")
    print(f"  Running {len(config_paths)} experiments (Mode: {mode.upper()})")
    print(f"{'='*60}\n")

    for i, config_path in enumerate(config_paths):
        print(f"\n{'-'*60}")
        print(f"  Experiment {i+1}/{len(config_paths)}: {config_path}")
        print(f"{'-'*60}\n")

        start_time = time.time()

        try:
            result = train_model(config_path, resume=is_resume_mode)
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
        df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, "comparison_table.csv")
        df.to_csv(csv_path, index=False)

        md_path = os.path.join(output_dir, "comparison_table.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Model Comparison Results\n\n")
            f.write(df.to_markdown(index=False))

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
    parser.add_argument(
        "--mode",
        choices=["scratch", "resume", "auto"],
        default="auto",
        help="Training mode: 'scratch' (train from epoch 1), 'resume' (continue interrupted runs), 'auto' (prompt or auto-detect)",
    )
    args = parser.parse_args()

    if args.configs:
        config_paths = args.configs
    else:
        config_paths = sorted(glob.glob("configs/*.yaml"))

    if not config_paths:
        print("No config files found. Provide --configs or add files to configs/")
        sys.exit(1)

    run_experiments(config_paths, mode=args.mode)
