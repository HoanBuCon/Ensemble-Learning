# -*- coding: utf-8 -*-
"""
Master Chart Regeneration Script
=================================
Regenerates ALL publication-quality figures across the entire pipeline
in both scalable vector format (.svg) and high-resolution raster (.png).
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_dataset_config
from src.utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_ensemble_confusion_matrix_grid,
    plot_per_class_metrics,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_comparison_bar,
    plot_per_class_comparison_heatmap,
    plot_per_class_comparison_bar,
    plot_radar_chart_comparison,
    plot_model_tradeoffs,
    plot_kfold_summary,
)
from scripts.verification.eval_tsne import run_tsne_analysis


def regenerate_model_charts(model_dir: str, class_names: List[str], model_name: str) -> None:
    """Regenerate all figures for a single model directory."""
    if not os.path.exists(model_dir):
        return

    # 1. Training Curves
    hist_json = os.path.join(model_dir, "training_history.json")
    hist_csv = os.path.join(model_dir, "history.csv")
    history = {}
    if os.path.exists(hist_json):
        try:
            with open(hist_json, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass
    elif os.path.exists(hist_csv):
        try:
            df = pd.read_csv(hist_csv)
            history = {col: df[col].tolist() for col in df.columns}
        except Exception:
            pass

    if history:
        print(f"  -> Plotting training curves for {model_name}...")
        plot_training_curves(history, model_dir, model_name=model_name)

    # 2. Confusion Matrices & Predictions
    test_probs_path = os.path.join(model_dir, "test_probabilities.npy")
    test_labels_path = os.path.join(model_dir, "test_labels.npy")
    if not os.path.exists(test_probs_path):
        test_probs_path = os.path.join(model_dir, "probabilities.npy")
    if not os.path.exists(test_labels_path):
        test_labels_path = os.path.join(model_dir, "labels.npy")

    if os.path.exists(test_probs_path) and os.path.exists(test_labels_path):
        probs = np.load(test_probs_path)
        labels = np.load(test_labels_path)
        preds = np.argmax(probs, axis=1)

        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(labels, preds)

        print(f"  -> Plotting confusion matrices for {model_name}...")
        plot_confusion_matrix(cm, class_names, model_dir, title=f"Confusion Matrix — {model_name}", normalize=False)
        plot_confusion_matrix(cm, class_names, model_dir, title=f"Normalized Confusion Matrix — {model_name}", normalize=True)

        print(f"  -> Plotting ROC & PR curves for {model_name}...")
        plot_roc_curves(labels, probs, class_names, model_dir, model_name=model_name)
        plot_precision_recall_curves(labels, probs, class_names, model_dir, model_name=model_name)

        # Per-class metrics
        rep = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
        per_class_dict = {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"], "f1_score": rep[c]["f1-score"], "support": rep[c]["support"]} for c in class_names if c in rep}
        plot_per_class_metrics(per_class_dict, class_names, model_dir, model_name=model_name)


def regenerate_pipeline_directory(root_dir: str, class_names: List[str]) -> None:
    """Regenerate all figures in a pipeline run folder (e.g. RESULTS/DEFAULT_TRAINING/outputs)."""
    if not os.path.exists(root_dir):
        return

    print(f"\n==================================================================")
    print(f" REGENERATING CHARTS IN: {root_dir}")
    print(f"==================================================================")

    model_names = ["resnet50", "densenet121", "efficientnet_b0", "swin_tiny"]
    all_models_per_class = {}
    metrics_dict = {}
    radar_metrics = {}

    for m in model_names:
        m_dir = os.path.join(root_dir, m)
        if os.path.exists(m_dir):
            regenerate_model_charts(m_dir, class_names, m)

            # Load metrics for comparison
            m_json = os.path.join(m_dir, "metrics.json")
            if os.path.exists(m_json):
                with open(m_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    metrics_dict[m] = {
                        "accuracy": data.get("accuracy", 0.0) * 100.0 if data.get("accuracy", 0.0) <= 1.0 else data.get("accuracy", 0.0),
                        "f1_score": data.get("macro_f1", data.get("f1_score", 0.0)) * 100.0 if data.get("macro_f1", data.get("f1_score", 0.0)) <= 1.0 else data.get("macro_f1", data.get("f1_score", 0.0)),
                        "precision": data.get("macro_precision", data.get("precision", 0.0)) * 100.0 if data.get("macro_precision", data.get("precision", 0.0)) <= 1.0 else data.get("macro_precision", data.get("precision", 0.0)),
                        "recall": data.get("macro_recall", data.get("recall", 0.0)) * 100.0 if data.get("macro_recall", data.get("recall", 0.0)) <= 1.0 else data.get("macro_recall", data.get("recall", 0.0)),
                    }
                    if "per_class" in data:
                        all_models_per_class[m] = data["per_class"]
                        radar_metrics[m] = [data["per_class"][c]["f1_score"] for c in class_names if c in data["per_class"]]

    # Ensemble models
    ens_cms = {}
    ens_names = ["hard_voting", "soft_voting", "weighted_voting", "stacking_logistic_regression", "stacking_random_forest", "stacking_xgboost"]
    ens_display_names = {
        "hard_voting": "Hard Voting Ensemble",
        "soft_voting": "Soft Voting Ensemble",
        "weighted_voting": "Weighted Voting Ensemble",
        "stacking_logistic_regression": "Stacking (Logistic Regression)",
        "stacking_random_forest": "Stacking (Random Forest)",
        "stacking_xgboost": "Stacking (XGBoost)",
    }

    for e_key in ens_names:
        e_dir = os.path.join(root_dir, e_key)
        if os.path.exists(e_dir):
            disp_name = ens_display_names.get(e_key, e_key)
            probs_p = os.path.join(e_dir, "test_probabilities.npy")
            labels_p = os.path.join(e_dir, "test_labels.npy")
            preds_p = os.path.join(e_dir, "test_predictions.npy")

            if os.path.exists(labels_p) and (os.path.exists(probs_p) or os.path.exists(preds_p)):
                labels = np.load(labels_p)
                if os.path.exists(probs_p):
                    probs = np.load(probs_p)
                    preds = np.argmax(probs, axis=1) if probs.ndim == 2 and probs.shape[1] > 1 else np.load(preds_p)
                else:
                    preds = np.load(preds_p)
                    probs = None

                from sklearn.metrics import confusion_matrix, classification_report
                cm = confusion_matrix(labels, preds)
                ens_cms[disp_name] = cm

                plot_confusion_matrix(cm, class_names, e_dir, title=f"Confusion Matrix — {disp_name}", normalize=False)
                plot_confusion_matrix(cm, class_names, e_dir, title=f"Normalized Confusion Matrix — {disp_name}", normalize=True)

                if probs is not None and probs.shape[1] == len(class_names):
                    plot_roc_curves(labels, probs, class_names, e_dir, model_name=disp_name)
                    plot_precision_recall_curves(labels, probs, class_names, e_dir, model_name=disp_name)

                rep = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
                per_class_dict = {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"], "f1_score": rep[c]["f1-score"], "support": rep[c]["support"]} for c in class_names if c in rep}
                plot_per_class_metrics(per_class_dict, class_names, e_dir, model_name=disp_name)

                # Add to comparisons
                acc = (preds == labels).mean() * 100.0
                f1 = rep["macro avg"]["f1-score"] * 100.0
                prec = rep["macro avg"]["precision"] * 100.0
                rec = rep["macro avg"]["recall"] * 100.0
                metrics_dict[disp_name] = {"accuracy": acc, "f1_score": f1, "precision": prec, "recall": rec}
                all_models_per_class[disp_name] = per_class_dict
                radar_metrics[disp_name] = [per_class_dict[c]["f1_score"] for c in class_names if c in per_class_dict]

    # Global Comparisons
    if metrics_dict:
        print(f"  -> Plotting global comparison charts in {root_dir}...")
        for metric_name in ["accuracy", "f1_score", "precision", "recall"]:
            plot_comparison_bar(metrics_dict, metric_name, root_dir)

    if all_models_per_class:
        plot_per_class_comparison_heatmap(all_models_per_class, metric="f1_score", class_names=class_names, output_dir=root_dir)
        plot_per_class_comparison_bar(all_models_per_class, metric="f1_score", class_names=class_names, output_dir=root_dir)

    if radar_metrics:
        plot_radar_chart_comparison(radar_metrics, categories=class_names, output_dir=root_dir)

    if ens_cms:
        plot_ensemble_confusion_matrix_grid(ens_cms, class_names, root_dir)


def main():
    ds_raw = load_dataset_config()
    class_names = ds_raw.get("classes", [
        "Brown_Blight", "Gray_Blight", "Green_mirid_bug",
        "Healthy_leaf", "Helopeltis", "Tea_algal_leaf_spot"
    ])

    print("\n" + "=" * 90)
    print("      REGENERATING ALL PIPELINE FIGURES IN SVG + PNG FORMATS")
    print("=" * 90)

    # 1. Regenerate DEFAULT_TRAINING
    def_dir = os.path.join("RESULTS", "DEFAULT_TRAINING", "outputs")
    if not os.path.exists(def_dir):
        def_dir = "outputs"
    regenerate_pipeline_directory(def_dir, class_names)

    # 2. Regenerate OOF_TRAINING
    oof_dir = os.path.join("RESULTS", "OOF_TRAINING", "outputs")
    if os.path.exists(oof_dir):
        regenerate_pipeline_directory(oof_dir, class_names)

    # 3. Regenerate Verification Visuals (t-SNE)
    print("\n==================================================================")
    print(" REGENERATING VERIFICATION VISUALS (t-SNE) IN SVG")
    print("==================================================================")
    
    verif_dir = os.path.join("outputs", "verification")
    run_tsne_analysis(save_dir=verif_dir)

    # Sync to RESULTS/verification
    res_verif = os.path.join("RESULTS", "verification")
    if os.path.exists(res_verif):
        import shutil
        for f in os.listdir(verif_dir):
            src_f = os.path.join(verif_dir, f)
            if os.path.isfile(src_f):
                shutil.copy2(src_f, os.path.join(res_verif, f))
        print(f"[SYNC] Mirrored all verification SVGs and PNGs to '{res_verif}'")

    print("\n" + "=" * 90)
    print("[SUCCESS] ALL CHARTS & VISUALIZATIONS HAVE BEEN REGENERATED IN .SVG FORMAT!")
    print("=" * 90)


if __name__ == "__main__":
    main()
