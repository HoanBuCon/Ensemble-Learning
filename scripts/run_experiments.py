"""
Multi-Experiment Runner Script
==============================

Train all configured models sequentially, cache predictions, and
generate a comparison table for thesis analysis.

CLI::

    python scripts/run_experiments.py
    python main.py benchmark --mode auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from scripts.train import train as train_model
from src.utils.report import (
    count_parameters,
    get_model_size_mb,
    extract_model_history_info,
    extract_model_val_metrics,
)
from src.utils.run_identity import (
    final_run_root,
    model_run_root,
    resolve_backbone_config_paths,
    validate_run_id,
)


def _required_metric(metrics: Dict[str, Any], key: str) -> float:
    if key not in metrics or metrics[key] is None:
        raise KeyError(f"Required metric is missing: {key}")
    return float(metrics[key])


def inspect_model_status(
    config_path: str, *, run_id: str, results_root: str = "RESULTS"
) -> Dict[str, Any]:
    """
    Inspect the training status of a model from its output directory.

    Returns:
        Dict with status ('COMPLETED', 'RESUMABLE', 'NOT_STARTED'), last_epoch,
        accepted_checkpoint_epoch, accepted_checkpoint_val_accuracy,
        total_epochs, and save_dir.
    """
    from src.utils.config import load_config
    import torch

    cfg = load_config(config_path)
    save_dir = str(model_run_root(results_root, run_id, "single_split", cfg.model.name))
    total_epochs = cfg.train.epochs
    metrics_path = os.path.join(save_dir, "test_metrics.json")
    last_ckpt_path = os.path.join(save_dir, "last_model.pth")

    hist_info = extract_model_history_info(save_dir)

    if os.path.exists(metrics_path):
        return {
            "config_path": config_path,
            "model_name": cfg.model.name,
            "experiment_name": cfg.experiment_name,
            "status": "COMPLETED",
            "last_epoch": total_epochs,
            "accepted_checkpoint_epoch": hist_info["accepted_checkpoint_epoch"],
            "accepted_checkpoint_val_accuracy": hist_info["accepted_checkpoint_val_accuracy"],
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
                "accepted_checkpoint_epoch": hist_info["accepted_checkpoint_epoch"],
                "accepted_checkpoint_val_accuracy": hist_info["accepted_checkpoint_val_accuracy"],
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
        "accepted_checkpoint_epoch": None,
        "accepted_checkpoint_val_accuracy": None,
        "total_epochs": total_epochs,
        "save_dir": save_dir,
    }


def display_status_table(statuses: List[Dict[str, Any]]) -> None:
    """Print clean Rich table showing status of all experiments."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        title="Experiment Status Overview",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Model Name", style="cyan")
    table.add_column("Config Path", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Progress", justify="right")
    table.add_column("Action", style="green")

    for s in statuses:
        status_str = s["status"]
        if status_str == "COMPLETED":
            status_style = "[bold green]COMPLETED[/bold green]"
            action_style = "Skip (Already evaluated)"
            progress_str = f"{s['total_epochs']}/{s['total_epochs']}"
        elif status_str == "RESUMABLE":
            status_style = "[bold yellow]RESUMABLE[/bold yellow]"
            action_style = f"Resume from epoch {s['last_epoch'] + 1}"
            progress_str = f"{s['last_epoch']}/{s['total_epochs']}"
        else:
            status_style = "[dim]NOT STARTED[/dim]"
            action_style = "Train from scratch"
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


def run_experiments(
    config_paths: List[str], mode: str = "auto", *, run_id: str,
    results_root: str = "RESULTS",
) -> pd.DataFrame:
    """
    Train all models and generate a comparison table.

    Args:
        config_paths: List of YAML config file paths.
        mode: Training mode ('scratch', 'resume', 'auto').

    Returns:
        DataFrame with the comparison table.
    """
    run_id = validate_run_id(run_id)
    config_paths = resolve_backbone_config_paths(config_paths)
    if not config_paths:
        print("No valid model config files provided.")
        return pd.DataFrame()

    results_list: List[Dict[str, Any]] = []

    statuses = [inspect_model_status(p, run_id=run_id, results_root=results_root) for p in config_paths]
    display_status_table(statuses)

    # Determine mode if interactive
    has_resumable = any(s["status"] == "RESUMABLE" for s in statuses)
    all_completed = all(s["status"] == "COMPLETED" for s in statuses)

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
        if all_completed:
            mode = "resume"
        else:
            mode = "scratch"

    is_resume_mode = (mode == "resume")

    print(f"\n{'='*60}")
    print(f"  Running {len(config_paths)} experiments (Mode: {mode.upper()})")
    print(f"{'='*60}\n")

    for i, config_path in enumerate(config_paths):
        status_info = statuses[i]

        print(f"\n{'-'*60}")
        print(f"  Experiment {i+1}/{len(config_paths)}: {config_path}")
        print(f"{'-'*60}\n")

        # Skip already completed models in resume mode
        if is_resume_mode and status_info["status"] == "COMPLETED":
            metrics_path = os.path.join(status_info["save_dir"], "test_metrics.json")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        eval_metrics = json.load(f)

                    n_params = count_parameters(status_info["model_name"])
                    val_metrics = extract_model_val_metrics(status_info["save_dir"])

                    row = {
                        "Model": status_info["model_name"],
                        "Experiment": status_info["experiment_name"],
                        "Best_Epoch": status_info.get("accepted_checkpoint_epoch"),
                        "Val_Accuracy": val_metrics["Val_Accuracy"],
                        "Val_Precision": val_metrics["Val_Precision"],
                        "Val_Recall": val_metrics["Val_Recall"],
                        "Val_F1_Score": val_metrics["Val_F1_Score"],
                        "Test_Accuracy": _required_metric(eval_metrics, "accuracy") * 100,
                        "Test_Precision": _required_metric(eval_metrics, "precision") * 100,
                        "Test_Recall": _required_metric(eval_metrics, "recall") * 100,
                        "Test_F1_Score": _required_metric(eval_metrics, "f1_score") * 100,
                        "Parameters": n_params,
                        "Parameters_M": n_params / 1e6,
                        "Training_Time_s": None,
                        "Inference_Time_s": _required_metric(eval_metrics, "inference_time_seconds"),
                        "Model_Size_MB": get_model_size_mb(status_info["save_dir"]),
                    }
                    results_list.append(row)
                    val_display = (
                        "N/A" if row["Val_Accuracy"] is None
                        else f"{float(row['Val_Accuracy']):.2f}%"
                    )
                    print(
                        f"  [SKIP] {status_info['model_name']} - Already COMPLETED "
                        f"(Test Acc: {row['Test_Accuracy']:.2f}%, Val Acc: {val_display})"
                    )
                    continue
                except Exception as e:
                    raise RuntimeError(
                        f"Completed run has invalid canonical metrics: {metrics_path}"
                    ) from e

        start_time = time.time()

        try:
            result = train_model(
                config_path, resume=is_resume_mode,
                run_id=run_id, results_root=results_root,
            )
            total_time = time.time() - start_time

            eval_metrics = result.get("eval_metrics", {})
            save_d = result["save_dir"]
            val_metrics = extract_model_val_metrics(save_d)

            row = {
                "Model": result["model_name"],
                "Experiment": result["experiment_name"],
                "Best_Epoch": result["accepted_checkpoint_epoch"],
                "Val_Accuracy": val_metrics["Val_Accuracy"],
                "Val_Precision": val_metrics["Val_Precision"],
                "Val_Recall": val_metrics["Val_Recall"],
                "Val_F1_Score": val_metrics["Val_F1_Score"],
                "Test_Accuracy": _required_metric(eval_metrics, "accuracy") * 100,
                "Test_Precision": _required_metric(eval_metrics, "precision") * 100,
                "Test_Recall": _required_metric(eval_metrics, "recall") * 100,
                "Test_F1_Score": _required_metric(eval_metrics, "f1_score") * 100,
                "Parameters": result["num_params"],
                "Parameters_M": result["num_params"] / 1e6,
                "Training_Time_s": total_time,
                "Inference_Time_s": _required_metric(eval_metrics, "inference_time_seconds"),
                "Model_Size_MB": get_model_size_mb(save_d),
            }

            results_list.append(row)

            print(f"\n  [OK] {row['Model']} - Test Acc: {row['Test_Accuracy']:.2f}%, "
                  f"Test F1: {row['Test_F1_Score']:.2f}%, Time: {total_time:.0f}s")

        except Exception as e:
            print(f"\n  [FAIL] {config_path}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            raise

    df = pd.DataFrame(results_list)
    report_dir = final_run_root(results_root, run_id) / "single_split"
    report_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_dir / "base_model_comparison.csv", index=False)
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", default="RESULTS")
    parser.add_argument(
        "--mode",
        choices=["scratch", "resume", "auto"],
        default="auto",
        help="Training mode: 'scratch' (train from epoch 1), 'resume' (continue interrupted runs), 'auto' (prompt or auto-detect)",
    )
    args = parser.parse_args()

    if args.configs:
        config_paths = resolve_backbone_config_paths(args.configs)
    else:
        config_paths = resolve_backbone_config_paths()

    if not config_paths:
        print("No config files found. Provide --configs or add files to configs/")
        sys.exit(1)

    run_experiments(
        config_paths, mode=args.mode, run_id=args.run_id,
        results_root=args.results_root,
    )
