"""
Comprehensive Plot Generator Script
===================================

Generates publication-quality visualizations across all trained backbone models,
dataset splits, individual disease classes, and ensemble methods.

Generates:
1. Base Model Training Dynamics (Loss, Accuracy, Learning Rate, 4-Panel Dashboard)
2. Base Model Test/Val Visualizations (Raw & Normalized Confusion Matrices, Per-Class Bar Charts, ROC & PR Curves)
3. Ensemble Visualizations (Per-ensemble Confusion Matrices, Per-Class Bar Charts, ROC & PR Curves)
4. Cross-Model Benchmark Comparisons (Ranked Metric Bar Charts, Per-Disease Heatmaps & Grouped Bars, Radar Chart, Pareto Trade-off Plot)
5. Ensemble Meta-Analysis (Optimal Voting Weights, Stacking Meta-Learner Feature Importances)

CLI::

    python scripts/generate_all_plots.py --outputs-dir outputs
    python main.py report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

from src.utils.config import load_config, load_dataset_config
from src.utils.metrics import compute_metrics
from src.utils.report import count_parameters, get_model_size_mb
from src.utils.visualization import (
    plot_comparison_bar,
    plot_confusion_matrix,
    plot_ensemble_weights,
    plot_model_tradeoffs,
    plot_per_class_comparison_bar,
    plot_per_class_comparison_heatmap,
    plot_per_class_metrics,
    plot_precision_recall_curves,
    plot_radar_chart_comparison,
    plot_roc_curves,
    plot_stacking_feature_importance,
    plot_training_curves,
    plot_training_dashboard,
)


def get_dataset_class_names() -> List[str]:
    """Load canonical class list from configs/dataset.yaml."""
    ds_cfg = load_dataset_config()
    classes = ds_cfg.get("classes", [])
    if not classes:
        classes = [
            "Brown_Blight",
            "Gray_Blight",
            "Green_mirid_bug",
            "Healthy_leaf",
            "Helopeltis",
            "Tea_algal_leaf_spot",
        ]
    return classes


def generate_base_model_plots(model_dir: str, class_names: List[str]) -> None:
    """Generate all charts for a single base model output folder."""
    model_name = os.path.basename(model_dir)
    print(f"--> Generating full visualization suite for Base Model: {model_name}...")

    # 1. Training dynamics & dashboard
    history_json = os.path.join(model_dir, "training_history.json")
    if os.path.exists(history_json):
        try:
            with open(history_json, "r", encoding="utf-8") as f:
                hist = json.load(f)
            plot_training_curves(hist, model_dir, model_name=model_name)
        except Exception as e:
            print(f"    Warning: Failed to plot training curves for {model_name}: {e}")

    # 2. Test set predictions, confusion matrices, ROC & PR curves
    test_prob_file = os.path.join(model_dir, "test_probabilities.npy")
    if not os.path.exists(test_prob_file):
        test_prob_file = os.path.join(model_dir, "probabilities.npy")

    test_lbl_file = os.path.join(model_dir, "test_labels.npy")
    if not os.path.exists(test_lbl_file):
        test_lbl_file = os.path.join(model_dir, "labels.npy")

    if os.path.exists(test_prob_file) and os.path.exists(test_lbl_file):
        try:
            probs = np.load(test_prob_file)
            labels = np.load(test_lbl_file)
            preds = np.argmax(probs, axis=1)

            metrics = compute_metrics(labels, preds, class_names=class_names)
            cm = np.array(metrics["confusion_matrix"])

            # Confusion matrices (raw count & normalized recall)
            plot_confusion_matrix(
                cm, class_names, model_dir,
                title=f"Confusion Matrix — {model_name} (Test Set)",
                normalize=False, filename="confusion_matrix.png",
            )
            plot_confusion_matrix(
                cm, class_names, model_dir,
                title=f"Normalized Confusion Matrix — {model_name} (Test Set)",
                normalize=True, filename="confusion_matrix_normalized.png",
            )

            # Per-class Precision, Recall, F1 breakdown
            plot_per_class_metrics(
                metrics.get("per_class", {}), class_names, model_dir,
                model_name=model_name, filename="per_class_metrics.png",
            )

            # ROC & PR curves
            plot_roc_curves(
                labels, probs, class_names, model_dir,
                model_name=model_name, filename="roc_curves.png",
            )
            plot_precision_recall_curves(
                labels, probs, class_names, model_dir,
                model_name=model_name, filename="precision_recall_curves.png",
            )
        except Exception as e:
            print(f"    Warning: Failed to generate test plots for {model_name}: {e}")

    # 3. Validation set visualizations (if available)
    val_prob_file = os.path.join(model_dir, "val_probabilities.npy")
    val_lbl_file = os.path.join(model_dir, "val_labels.npy")
    if os.path.exists(val_prob_file) and os.path.exists(val_lbl_file):
        try:
            val_probs = np.load(val_prob_file)
            val_labels = np.load(val_lbl_file)
            val_preds = np.argmax(val_probs, axis=1)
            val_metrics = compute_metrics(val_labels, val_preds, class_names=class_names)
            val_cm = np.array(val_metrics["confusion_matrix"])

            val_plots_dir = os.path.join(model_dir, "val_evaluation")
            os.makedirs(val_plots_dir, exist_ok=True)

            plot_confusion_matrix(
                val_cm, class_names, val_plots_dir,
                title=f"Confusion Matrix — {model_name} (Validation Set)",
                normalize=False, filename="val_confusion_matrix.png",
            )
            plot_confusion_matrix(
                val_cm, class_names, val_plots_dir,
                title=f"Normalized Confusion Matrix — {model_name} (Validation Set)",
                normalize=True, filename="val_confusion_matrix_normalized.png",
            )
            plot_per_class_metrics(
                val_metrics.get("per_class", {}), class_names, val_plots_dir,
                model_name=model_name, filename="val_per_class_metrics.png",
            )
            plot_roc_curves(
                val_labels, val_probs, class_names, val_plots_dir,
                model_name=model_name, filename="val_roc_curves.png",
            )
            plot_precision_recall_curves(
                val_labels, val_probs, class_names, val_plots_dir,
                model_name=model_name, filename="val_precision_recall_curves.png",
            )
        except Exception as e:
            print(f"    Warning: Failed to generate val evaluation plots for {model_name}: {e}")


def generate_ensemble_plots(
    outputs_dir: str = "outputs",
    mode: str = "val",
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Generate complete visualization suite for all ensemble methods
    (Hard/Soft/Weighted Voting & Stacking Meta-Learners).
    """
    if class_names is None:
        class_names = get_dataset_class_names()

    mode_dir = os.path.join(outputs_dir, mode)
    if not os.path.exists(mode_dir):
        print(f"Ensemble mode directory '{mode_dir}' does not exist. Run ensemble evaluation first.")
        return

    json_path = os.path.join(mode_dir, "ensemble_full_metrics.json")
    if not os.path.exists(json_path):
        print(f"Ensemble metrics JSON '{json_path}' not found.")
        return

    print(f"\n--> Generating full visualization suite for Ensemble Methods (Mode: {mode.upper()})...")

    with open(json_path, "r", encoding="utf-8") as f:
        full_metrics = json.load(f)

    # 1. Base test probabilities & labels
    model_dirs = sorted([
        d for d in glob.glob(os.path.join(outputs_dir, "*"))
        if os.path.isdir(d) and os.path.basename(d) not in ["val", "oof", "reports"]
    ])
    base_models = [os.path.basename(d) for d in model_dirs]

    test_probs_dict = {}
    test_labels = None

    for m in base_models:
        m_dir = os.path.join(outputs_dir, m)
        p_file = os.path.join(m_dir, "test_probabilities.npy")
        if not os.path.exists(p_file):
            p_file = os.path.join(m_dir, "probabilities.npy")
        l_file = os.path.join(m_dir, "test_labels.npy")
        if not os.path.exists(l_file):
            l_file = os.path.join(m_dir, "labels.npy")

        if os.path.exists(p_file) and os.path.exists(l_file):
            test_probs_dict[m] = np.load(p_file)
            if test_labels is None:
                test_labels = np.load(l_file)

    # Compute probability matrices for voting and stacking to render ROC/PR curves
    ensemble_probs_map: Dict[str, np.ndarray] = {}

    if test_probs_dict and test_labels is not None:
        ordered_probs = [test_probs_dict[m] for m in base_models if m in test_probs_dict]

        # Soft Voting Probs
        ensemble_probs_map["Soft Voting Ensemble"] = np.mean(ordered_probs, axis=0)

        # Weighted Voting Probs (re-fit on val)
        val_probs_list = []
        val_labels = None
        for m in base_models:
            v_p = os.path.join(outputs_dir, m, "val_probabilities.npy")
            v_l = os.path.join(outputs_dir, m, "val_labels.npy")
            if os.path.exists(v_p) and os.path.exists(v_l):
                val_probs_list.append(np.load(v_p))
                if val_labels is None:
                    val_labels = np.load(v_l)

        if val_probs_list and val_labels is not None:
            from src.ensemble.voting import WeightedVoting
            from src.ensemble.stacking import StackingEnsemble

            wv = WeightedVoting()
            wv.fit(val_probs_list, val_labels)
            ensemble_probs_map["Weighted Voting Ensemble"] = wv.predict_proba(ordered_probs)

            # Plot optimal weights bar chart
            plot_ensemble_weights(
                weights=dict(zip(base_models, wv.weights)),
                model_names=base_models,
                output_dir=mode_dir,
                filename="ensemble_weights.png",
            )

            # Stacking Meta-learners Probs & Feature Importances
            for meta in ["logistic_regression", "random_forest", "xgboost"]:
                m_label = f"Stacking ({meta.replace('_', ' ').title()})"
                try:
                    stk = StackingEnsemble(meta_learner=meta)
                    stk.fit(val_probs_list, val_labels)
                    ensemble_probs_map[m_label] = stk.predict_proba(ordered_probs)

                    # Feature importance for tree-based models
                    if hasattr(stk._model, "feature_importances_"):
                        feature_names = [
                            f"{m} : {c.replace('_', ' ')}"
                            for m in base_models
                            for c in class_names
                        ]
                        plot_stacking_feature_importance(
                            stk._model.feature_importances_,
                            feature_names=feature_names,
                            output_dir=mode_dir,
                            meta_name=meta.replace("_", " ").title(),
                            filename=f"stacking_{meta}_feature_importance.png",
                        )
                except Exception as e:
                    print(f"    Warning: Stacking {meta} prob estimation failed: {e}")

    # Generate per-ensemble method charts (Confusion Matrices, Per-Class Bar, ROC/PR)
    plots_subfolder = os.path.join(mode_dir, "method_breakdowns")
    os.makedirs(plots_subfolder, exist_ok=True)

    all_models_per_class: Dict[str, Dict[str, Dict[str, float]]] = {}
    overview_metrics_dict: Dict[str, Dict[str, float]] = {}

    for method_name, metrics in full_metrics.items():
        clean_method_folder = re.sub(r"[^\w\-]", "_", method_name.lower())
        m_dir = os.path.join(plots_subfolder, clean_method_folder)
        os.makedirs(m_dir, exist_ok=True)

        overview_metrics_dict[method_name] = {
            "accuracy": metrics.get("accuracy", 0.0) * 100.0,
            "precision": metrics.get("precision", 0.0) * 100.0,
            "recall": metrics.get("recall", 0.0) * 100.0,
            "f1_score": metrics.get("f1_score", 0.0) * 100.0,
        }

        per_cls = metrics.get("per_class", {})
        all_models_per_class[method_name] = per_cls

        cm = np.array(metrics.get("confusion_matrix", []))
        if cm.size > 0:
            plot_confusion_matrix(
                cm, class_names, m_dir,
                title=f"Confusion Matrix — {method_name}",
                normalize=False, filename="confusion_matrix.png",
            )
            plot_confusion_matrix(
                cm, class_names, m_dir,
                title=f"Normalized Confusion Matrix — {method_name}",
                normalize=True, filename="confusion_matrix_normalized.png",
            )

        if per_cls:
            plot_per_class_metrics(
                per_cls, class_names, m_dir,
                model_name=method_name, filename="per_class_metrics.png",
            )

        if method_name in ensemble_probs_map and test_labels is not None:
            plot_roc_curves(
                test_labels, ensemble_probs_map[method_name], class_names, m_dir,
                model_name=method_name, filename="roc_curves.png",
            )
            plot_precision_recall_curves(
                test_labels, ensemble_probs_map[method_name], class_names, m_dir,
                model_name=method_name, filename="precision_recall_curves.png",
            )

    # Cross-Model Benchmark Comparison Visualizations
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        plot_comparison_bar(
            overview_metrics_dict, metric, mode_dir,
            title=f"Benchmark Comparison (Base Models vs Ensembles) — {metric.replace('_', ' ').title()}",
            filename=f"ensemble_comparison_{metric}.png",
        )

    # Per-Class Disease Heatmap & Grouped Bar Chart
    plot_per_class_comparison_heatmap(
        all_models_per_class, metric="f1_score", class_names=class_names,
        output_dir=mode_dir, filename="ensemble_per_class_f1_heatmap.png",
    )
    plot_per_class_comparison_heatmap(
        all_models_per_class, metric="recall", class_names=class_names,
        output_dir=mode_dir, filename="ensemble_per_class_recall_heatmap.png",
    )
    plot_per_class_comparison_bar(
        all_models_per_class, metric="f1_score", class_names=class_names,
        output_dir=mode_dir, filename="ensemble_per_class_f1_bar.png",
    )

    # Radar Comparison Chart (Top Base Models vs Top Ensembles)
    top_radar_methods = [
        "Single Model (densenet121)",
        "Single Model (swin_tiny)",
        "Soft Voting Ensemble",
        "Weighted Voting Ensemble",
        "Stacking (Random Forest)",
    ]
    radar_data = {}
    for m in top_radar_methods:
        if m in all_models_per_class:
            radar_data[m] = [
                all_models_per_class[m].get(c, {}).get("f1_score", 0.0) * 100.0
                for c in class_names
            ]

    if radar_data:
        plot_radar_chart_comparison(
            radar_data, categories=class_names, output_dir=mode_dir,
            title="Disease Class F1-Score Radar Comparison (Top Models vs Ensembles)",
            filename="ensemble_comparison_radar.png",
        )

    # Pareto Efficiency / Model Trade-Off Plot
    tradeoff_rows = []
    for m in base_models:
        m_dir = os.path.join(outputs_dir, m)
        metrics_p = os.path.join(m_dir, "metrics.json")
        if os.path.exists(metrics_p):
            with open(metrics_p, "r", encoding="utf-8") as f:
                d = json.load(f)
            n_params = count_parameters(m)
            tradeoff_rows.append({
                "Model": m,
                "Parameters_M": n_params / 1e6,
                "Test_Accuracy": d.get("accuracy", 0.0) * 100.0,
                "Inference_Time_s": d.get("inference_time_seconds", 45.0),
            })

    # Add Ensembles to Trade-off plot
    total_base_params = sum(r["Parameters_M"] for r in tradeoff_rows)
    for ens_name in ["Soft Voting Ensemble", "Weighted Voting Ensemble", "Stacking (Random Forest)"]:
        if ens_name in overview_metrics_dict:
            tradeoff_rows.append({
                "Model": ens_name.replace(" Ensemble", ""),
                "Parameters_M": total_base_params,
                "Test_Accuracy": overview_metrics_dict[ens_name]["accuracy"],
                "Inference_Time_s": sum(r["Inference_Time_s"] for r in tradeoff_rows if r["Model"] in base_models),
            })

    if tradeoff_rows:
        plot_model_tradeoffs(
            tradeoff_rows, output_dir=mode_dir,
            filename="model_tradeoffs_pareto.png",
        )

    print(f"--> All Ensemble visualizations successfully generated in '{mode_dir}' and '{plots_subfolder}'\n")


def generate_all_plots(outputs_dir: str = "outputs") -> None:
    """Master routine to generate all plots across base models and ensembles."""
    class_names = get_dataset_class_names()

    # Discover base model directories
    all_dirs = sorted(glob.glob(os.path.join(outputs_dir, "*")))
    model_dirs = [
        d for d in all_dirs
        if os.path.isdir(d) and os.path.basename(d) not in ["val", "oof", "reports"]
        and (os.path.exists(os.path.join(d, "metrics.json")) or os.path.exists(os.path.join(d, "history.csv")))
    ]

    print(f"\n{'='*75}")
    print(f"  RUNNING MASTER VISUALIZATION GENERATOR (Found {len(model_dirs)} Base Models)")
    print(f"{'='*75}\n")

    for m_dir in model_dirs:
        generate_base_model_plots(m_dir, class_names)

    # Generate ensemble plots for both 'val' and 'oof' modes if they exist
    for mode in ["val", "oof"]:
        if os.path.exists(os.path.join(outputs_dir, mode)):
            generate_ensemble_plots(outputs_dir, mode=mode, class_names=class_names)

    # Generate base comparison plots in root outputs/
    from scripts.generate_comparison import generate_base_comparison_report
    generate_base_comparison_report(outputs_dir=outputs_dir)

    print(f"\n{'='*75}")
    print(f"  ALL PLOTS AND DIAGNOSTIC REPORTS COMPLETED SUCCESSFULLY!")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comprehensive visualization suite")
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Main outputs directory (default: 'outputs')",
    )
    args = parser.parse_args()

    generate_all_plots(outputs_dir=args.outputs_dir)
