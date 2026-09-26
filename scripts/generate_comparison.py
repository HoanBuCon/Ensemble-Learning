"""
Base Model Comparison Generator Script
======================================

Standalone script & module to extract model metrics (Validation & Test),
format percentage comparison tables, export CSV/Markdown artifacts, and
plot metric comparison bar charts.

CLI::

    python scripts/generate_comparison.py
    python main.py report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.utils.config import load_config
from src.utils.report import (
    count_parameters,
    get_model_size_mb,
    extract_model_history_info,
    extract_model_val_metrics,
    extract_model_training_time,
)
from src.utils.visualization import (
    plot_comparison_bar,
    plot_per_class_comparison_heatmap,
    plot_per_class_comparison_bar,
    plot_radar_chart_comparison,
    plot_model_tradeoffs,
)


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
    if Path(outputs_dir).name.upper() == "RESULTS":
        raise ValueError(
            "Legacy comparison generator rejects canonical RESULTS; use canonical prediction records"
        )
    # Discover candidate model directories
    if config_paths:
        model_dirs = []
        for p in config_paths:
            if os.path.basename(p) == "dataset.yaml":
                continue
            cfg = load_config(p)
            save_d = cfg.checkpoint.save_dir
            if "RESULTS" in {part.upper() for part in Path(save_d).parts}:
                raise ValueError(
                    "Legacy comparison generator rejects canonical RESULTS config roots; "
                    "use canonical prediction records"
                )
            if os.path.isdir(save_d):
                model_dirs.append(save_d)
    else:
        # Check target outputs_dir, and fallback to RESULTS / OOF_Results / Default_Results if empty
        search_dirs = [
            outputs_dir,
            "RESULTS/DEFAULT_TRAINING/outputs",
            "RESULTS/OOF_TRAINING/outputs",
            "RESULTS/DEFAULT_TRAINING",
            "RESULTS/OOF_TRAINING",
            "Default_Result_V2/outputs",
            "OOF_Results/outputs",
            "Default_Results/outputs",
        ]
        valid_dirs = []
        resolved_out_dir = outputs_dir

        for s_dir in search_dirs:
            if not os.path.exists(s_dir):
                continue
            all_dirs = sorted(glob.glob(os.path.join(s_dir, "*")))
            candidate_dirs = [
                d for d in all_dirs
                if os.path.isdir(d) and (
                    os.path.exists(os.path.join(d, "metrics.json")) or
                    os.path.exists(os.path.join(d, "probabilities.npy")) or
                    os.path.exists(os.path.join(d, "test_probabilities.npy")) or
                    os.path.exists(os.path.join(d, "kfold", "oof_metrics.json")) or
                    os.path.exists(os.path.join(d, "kfold", "oof_probabilities.npy")) or
                    os.path.exists(os.path.join(d, "kfold", "test_probabilities.npy")) or
                    os.path.exists(os.path.join(d, "kfold", "kfold_summary.csv"))
                )
            ]
            if candidate_dirs:
                valid_dirs = candidate_dirs
                resolved_out_dir = s_dir
                break

        # Group by base model name to select latest version if versioned
        model_groups: Dict[str, List[tuple[int, str]]] = {}
        for d in valid_dirs:
            folder_name = os.path.basename(d)
            if folder_name in ["val", "oof", "default", "ensemble", "tensorboard"]:
                continue
            match = re.match(r"^(.*?)(?:_(\d+))?$", folder_name)
            if match:
                base_name = match.group(1)
                if base_name in ["val", "oof", "default", "ensemble", "tensorboard"]:
                    continue
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
        base_name = re.sub(r"_\d+$", "", m_name)

        metrics_path = os.path.join(m_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            metrics_path = os.path.join(m_dir, "kfold", "oof_metrics.json")
        if not os.path.exists(metrics_path):
            metrics_path = os.path.join(m_dir, "kfold", "fold_1", "metrics.json")

        eval_metrics: Dict[str, Any] = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    eval_metrics = json.load(f)
            except Exception:
                pass

        # If test metrics missing, compute from test_probabilities.npy
        if "accuracy" not in eval_metrics or eval_metrics.get("accuracy", 0.0) == 0.0:
            test_prob_path = os.path.join(m_dir, "test_probabilities.npy")
            if not os.path.exists(test_prob_path):
                test_prob_path = os.path.join(m_dir, "kfold", "test_probabilities.npy")
            test_lbl_path = os.path.join(m_dir, "test_labels.npy")
            if not os.path.exists(test_lbl_path):
                test_lbl_path = os.path.join(m_dir, "kfold", "test_labels.npy")
            
            if os.path.exists(test_prob_path):
                try:
                    t_probs = np.load(test_prob_path)
                    t_preds = np.argmax(t_probs, axis=1)
                    if os.path.exists(test_lbl_path):
                        t_labels = np.load(test_lbl_path)
                    else:
                        from src.utils.config import load_dataset_config
                        from src.datasets.dataset import ImageFolderDataset
                        ds_raw = load_dataset_config()
                        t_dir = ds_raw.get("test_dir", "./data/test")
                        t_ds = ImageFolderDataset(root=t_dir)
                        t_labels = np.array([s[1] for s in t_ds.samples])
                    
                    comp_m = compute_metrics(t_labels, t_preds)
                    eval_metrics.update(comp_m)
                except Exception:
                    pass

        n_params = count_parameters(base_name)
        val_metrics = extract_model_val_metrics(m_dir)
        hist_info = extract_model_history_info(m_dir)
        train_time_str = extract_model_training_time(m_dir)

        row = {
            "Model": base_name,
            "Experiment": f"{m_name}_baseline",
            "Best_Epoch": hist_info["accepted_checkpoint_epoch"],
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
            "Training_Time": train_time_str,
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

    # Per-class heatmaps, bars, and radar
    try:
        from src.utils.config import load_dataset_config
        ds_cfg = load_dataset_config()
        class_names = ds_cfg.get("classes", [])
        all_models_per_class = {}
        radar_data = {}

        for m_dir in model_dirs:
            base_name = re.sub(r"_\d+$", "", os.path.basename(m_dir))
            m_path = os.path.join(m_dir, "metrics.json")
            if not os.path.exists(m_path):
                m_path = os.path.join(m_dir, "kfold", "oof_metrics.json")
            if not os.path.exists(m_path):
                m_path = os.path.join(m_dir, "kfold", "fold_1", "metrics.json")

            if os.path.exists(m_path):
                with open(m_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                per_cls = d.get("per_class", {})
                if per_cls:
                    all_models_per_class[base_name] = per_cls
                    radar_data[base_name] = [per_cls.get(c, {}).get("f1_score", 0.0) * 100.0 for c in class_names]

        if all_models_per_class:
            plot_per_class_comparison_heatmap(
                all_models_per_class, metric="f1_score", class_names=class_names,
                output_dir=outputs_dir, filename="comparison_per_class_heatmap.png"
            )
            plot_per_class_comparison_bar(
                all_models_per_class, metric="f1_score", class_names=class_names,
                output_dir=outputs_dir, filename="comparison_per_class_bar.png"
            )
        if radar_data:
            plot_radar_chart_comparison(
                radar_data, categories=class_names, output_dir=outputs_dir,
                title="Disease Class F1-Score Radar Comparison (Base Models)",
                filename="comparison_radar.png"
            )
    except Exception as e:
        print(f"Warning: Per-class comparison plot generation failed: {e}")

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
        "Parameters_M", "Training_Time", "Model_Size_MB",
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
