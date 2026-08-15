"""
Ensemble Evaluation Runner Script
=================================

Evaluates all ensemble methods (Hard Voting, Soft Voting, Weighted Voting,
and Stacking Ensembles) using the prediction probabilities of trained 
backbone models.

CLI::

    python scripts/run_ensemble_eval.py --mode val
    python scripts/run_ensemble_eval.py --mode oof
    python main.py ensemble --mode val
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
import torch

from src.datasets.dataset import create_dataloaders
from src.engine.evaluator import run_inference
from src.ensemble import (
    HardVoting,
    SoftVoting,
    WeightedVoting,
    StackingEnsemble,
)
from src.ensemble.base import EnsembleBase
from src.ensemble.oof import OOFGenerator
from src.models.factory import create_model
from src.utils.config import load_config, load_dataset_config
from src.utils.metrics import compute_metrics


def get_latest_model_dirs(outputs_dir: str = "outputs") -> List[str]:
    """Find latest directory for each base model in outputs_dir (supports standard and kfold)."""
    all_dirs = sorted(glob.glob(os.path.join(outputs_dir, "*")))
    valid_dirs = [
        d for d in all_dirs
        if os.path.isdir(d) and os.path.basename(d) not in ["oof", "val", "ensemble"] and (
            os.path.exists(os.path.join(d, "probabilities.npy"))
            or os.path.exists(os.path.join(d, "test_probabilities.npy"))
            or os.path.exists(os.path.join(d, "kfold", "oof_probabilities.npy"))
            or os.path.exists(os.path.join(d, "kfold", "test_probabilities.npy"))
        )
    ]

    model_groups: Dict[str, List[tuple[int, str]]] = {}
    for d in valid_dirs:
        folder_name = os.path.basename(d)
        match = re.match(r"^(.*?)(?:_(\d+))?$", folder_name)
        if match:
            base_name = match.group(1)
            version = int(match.group(2)) if match.group(2) else 0
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append((version, d))

    latest_dirs = []
    for base_name, versions in sorted(model_groups.items()):
        versions.sort(key=lambda x: x[0], reverse=True)
        latest_dirs.append(versions[0][1])

    return sorted(latest_dirs)


def generate_val_predictions_if_missing(model_dirs: List[str]) -> None:
    """Generate val_probabilities.npy and val_labels.npy if missing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for m_dir in model_dirs:
        val_prob_path = os.path.join(m_dir, "val_probabilities.npy")
        val_label_path = os.path.join(m_dir, "val_labels.npy")

        if os.path.exists(val_prob_path) and os.path.exists(val_label_path):
            continue

        m_name = os.path.basename(m_dir)
        config_name = re.sub(r"_\d+$", "", m_name)
        config_path = os.path.join("configs", f"{config_name}.yaml")

        if not os.path.exists(config_path):
            print(f"Config file for {m_name} not found at {config_path}. Skipping.")
            continue

        print(f"--> Generating Validation predictions for {m_name}...")
        cfg = load_config(config_path)

        ckpt_path = os.path.join(m_dir, "best_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint for {m_name} not found at {ckpt_path}. Skipping.")
            continue

        model = create_model(
            model_name=cfg.model.name,
            pretrained=False,
            num_classes=cfg.model.num_classes,
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        _, val_loader, _, _ = create_dataloaders(cfg)
        val_probs, val_labels = run_inference(model, val_loader, device=device)

        np.save(val_prob_path, val_probs)
        np.save(val_label_path, val_labels)
        print(f"  Saved Validation predictions: {val_probs.shape}")


def run_ensemble_evaluation(
    mode: str = "val",
    outputs_dir: str = "outputs",
) -> pd.DataFrame:
    """
    Run ensemble evaluation pipeline.

    Args:
        mode: Protocol mode ('val' or 'oof').
        outputs_dir: Base output directory.

    Returns:
        DataFrame summarizing ensemble metrics.
    """
    mode = mode.lower()
    if mode not in ["val", "oof"]:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'val' or 'oof'.")

    print(f"\n>>> SELECTED MODE: {mode.upper()} <<<\n")

    model_dirs = get_latest_model_dirs(outputs_dir)
    if not model_dirs:
        print(f"No trained model outputs found in '{outputs_dir}'. Train models first.")
        return pd.DataFrame()

    base_models = [re.sub(r"_\d+$", "", os.path.basename(d)) for d in model_dirs]
    class_mappings = []
    for d in model_dirs:
        cmap_path = os.path.join(d, "class_to_idx.json")
        if not os.path.exists(cmap_path):
            cmap_path = os.path.join(d, "kfold", "class_to_idx.json")
        if os.path.exists(cmap_path):
            with open(cmap_path, "r", encoding="utf-8") as f:
                class_mappings.append(json.load(f))
    if class_mappings:
        EnsembleBase.validate_class_mappings(class_mappings)

    # Load test probabilities & labels
    base_test_probs_list = []
    base_test_labels = None

    for m_dir in model_dirs:
        # Check kfold/test_probabilities.npy first, then root
        prob_path = os.path.join(m_dir, "kfold", "test_probabilities.npy")
        if not os.path.exists(prob_path):
            prob_path = os.path.join(m_dir, "test_probabilities.npy")
        if not os.path.exists(prob_path):
            prob_path = os.path.join(m_dir, "probabilities.npy")

        label_path = os.path.join(m_dir, "test_labels.npy")
        if not os.path.exists(label_path):
            label_path = os.path.join(m_dir, "labels.npy")

        if os.path.exists(prob_path):
            probs = np.load(prob_path)
            base_test_probs_list.append(probs)

        if base_test_labels is None and os.path.exists(label_path):
            base_test_labels = np.load(label_path)

    # Fallback to load test labels from dataset if missing in output folders
    ds_cfg_raw = load_dataset_config()
    ds_dict = ds_cfg_raw.get("dataset", ds_cfg_raw)
    class_names = ds_dict.get("classes", [])

    if base_test_labels is None:
        from src.datasets.dataset import ImageFolderDataset
        test_dir = ds_dict.get("test_dir", os.path.join(ds_dict.get("data_root", "./data"), "test"))
        if not os.path.isdir(test_dir):
            test_dir = os.path.join("data", "test")

        if os.path.isdir(test_dir):
            test_ds = ImageFolderDataset(root=test_dir)
            base_test_labels = np.array([s[1] for s in test_ds.samples])

    test_labels = base_test_labels

    # Prepare training probabilities & labels for Meta-Learner
    if mode == "val":
        generate_val_predictions_if_missing(model_dirs)
        val_probs_list = []
        val_labels = None
        for m_dir in model_dirs:
            v_prob = np.load(os.path.join(m_dir, "val_probabilities.npy"))
            v_lbl = np.load(os.path.join(m_dir, "val_labels.npy"))
            val_probs_list.append(v_prob)
            if val_labels is None:
                val_labels = v_lbl

        meta_train_probs = val_probs_list
        meta_train_labels = val_labels
    else: # mode == 'oof'
        meta_train_probs = []
        meta_train_labels = None
        for m_dir in model_dirs:
            oof_p_path = os.path.join(m_dir, "kfold", "oof_probabilities.npy")
            oof_l_path = os.path.join(m_dir, "kfold", "oof_labels.npy")
            if not os.path.exists(oof_p_path):
                oof_p_path = os.path.join(m_dir, "oof", "oof_probabilities.npy")
                oof_l_path = os.path.join(m_dir, "oof", "oof_labels.npy")

            if os.path.exists(oof_p_path) and os.path.exists(oof_l_path):
                oof_probs = np.load(oof_p_path)
                oof_lbls = np.load(oof_l_path)
                meta_train_probs.append(oof_probs)
                if meta_train_labels is None:
                    meta_train_labels = oof_lbls
            else:
                m_name = re.sub(r"_\d+$", "", os.path.basename(m_dir))
                cfg_path = os.path.join("configs", f"{m_name}.yaml")
                if not os.path.exists(cfg_path):
                    print(f"Config path for {m_name} not found at {cfg_path}. Skipping.")
                    continue
                oof_gen = OOFGenerator(config=cfg_path, output_dir=os.path.join(m_dir, "kfold"))
                oof_probs, oof_lbls, _ = oof_gen.generate()
                meta_train_probs.append(oof_probs)
                if meta_train_labels is None:
                    meta_train_labels = oof_lbls

    meta_test_probs = base_test_probs_list

    # Single Model Baseline Results on Test Set
    single_model_results = []
    full_metrics_store = {}

    for m_name, probs in zip(base_models, meta_test_probs):
        m_metrics = compute_metrics(test_labels, np.argmax(probs, axis=1), class_names=class_names)
        full_metrics_store[f"Single Model ({m_name})"] = m_metrics
        single_model_results.append({
            "Method": f"Single Model ({m_name})",
            "Type": "Individual",
            "Accuracy": m_metrics["accuracy"] * 100,
            "Precision": m_metrics["precision"] * 100,
            "Recall": m_metrics["recall"] * 100,
            "F1_Score": m_metrics["f1_score"] * 100,
            "Details": "-",
        })

    best_single_acc = max(r["Accuracy"] for r in single_model_results)

    # Ensemble Methods Evaluation
    ensemble_evaluations = []
    per_class_rows = []

    # 1. Hard Voting
    hv = HardVoting()
    hv_metrics = hv.evaluate(meta_test_probs, test_labels, class_names=class_names)
    full_metrics_store["Hard Voting Ensemble"] = hv_metrics
    ensemble_evaluations.append({
        "Method": "Hard Voting Ensemble",
        "Type": "Ensemble (Voting)",
        "Accuracy": hv_metrics["accuracy"] * 100,
        "Precision": hv_metrics["precision"] * 100,
        "Recall": hv_metrics["recall"] * 100,
        "F1_Score": hv_metrics["f1_score"] * 100,
        "Details": "Majority vote across predictions",
    })

    # 2. Soft Voting
    sv = SoftVoting()
    sv_metrics = sv.evaluate(meta_test_probs, test_labels, class_names=class_names)
    full_metrics_store["Soft Voting Ensemble"] = sv_metrics
    ensemble_evaluations.append({
        "Method": "Soft Voting Ensemble",
        "Type": "Ensemble (Voting)",
        "Accuracy": sv_metrics["accuracy"] * 100,
        "Precision": sv_metrics["precision"] * 100,
        "Recall": sv_metrics["recall"] * 100,
        "F1_Score": sv_metrics["f1_score"] * 100,
        "Details": "Equal weight probability averaging",
    })

    # 3. Weighted Voting
    wv = WeightedVoting()
    wv.fit(meta_train_probs, meta_train_labels)
    wv_metrics = wv.evaluate(meta_test_probs, test_labels, class_names=class_names)
    full_metrics_store["Weighted Voting Ensemble"] = wv_metrics
    w_str = ", ".join(f"{name}: {w:.2f}" for name, w in zip(base_models, wv.weights))
    ensemble_evaluations.append({
        "Method": "Weighted Voting Ensemble",
        "Type": "Ensemble (Voting)",
        "Accuracy": wv_metrics["accuracy"] * 100,
        "Precision": wv_metrics["precision"] * 100,
        "Recall": wv_metrics["recall"] * 100,
        "F1_Score": wv_metrics["f1_score"] * 100,
        "Details": f"Optimized weights ({w_str})",
    })

    # 4. Stacking Ensembles
    meta_learners = ["logistic_regression", "random_forest", "xgboost"]
    fit_label = "Validation Set" if mode == "val" else "5-Fold OOF Train Set"

    for meta in meta_learners:
        m_label = f"Stacking ({meta.replace('_', ' ').title()})"
        stk = StackingEnsemble(meta_learner=meta)
        stk.fit(meta_train_probs, meta_train_labels)
        stk_metrics = stk.evaluate(meta_test_probs, test_labels, class_names=class_names)
        full_metrics_store[m_label] = stk_metrics
        ensemble_evaluations.append({
            "Method": m_label,
            "Type": "Ensemble (Stacking)",
            "Accuracy": stk_metrics["accuracy"] * 100,
            "Precision": stk_metrics["precision"] * 100,
            "Recall": stk_metrics["recall"] * 100,
            "F1_Score": stk_metrics["f1_score"] * 100,
            "Details": f"Meta-learner: {meta} (Fit on {fit_label})",
        })

    # Build Per-Class Breakdown DataFrame
    for method_name, m_dict in full_metrics_store.items():
        per_cls = m_dict.get("per_class", {})
        for c_name, c_data in per_cls.items():
            per_class_rows.append({
                "Method": method_name,
                "Disease_Class": c_name,
                "Precision": f"{c_data['precision'] * 100:.2f}%",
                "Recall": f"{c_data['recall'] * 100:.2f}%",
                "F1_Score": f"{c_data['f1_score'] * 100:.2f}%",
                "Support": c_data['support'],
            })
    df_per_class = pd.DataFrame(per_class_rows)

    # Combine overall results
    all_results = single_model_results + ensemble_evaluations
    df_results = pd.DataFrame(all_results)

    # Compute improvement over best single model
    improvements = []
    for _, row in df_results.iterrows():
        diff = row["Accuracy"] - best_single_acc
        if abs(diff) < 1e-6 and row["Type"] == "Individual":
            improvements.append("Base (0.00%)")
        else:
            sign = "+" if diff >= 0 else ""
            improvements.append(f"{sign}{diff:.2f}%")

    df_results.insert(3, "Improvement", improvements)

    # Output save directory
    mode_output_dir = os.path.join(outputs_dir, mode)
    os.makedirs(mode_output_dir, exist_ok=True)

    csv_path = os.path.join(mode_output_dir, "ensemble_comparison.csv")
    md_path = os.path.join(mode_output_dir, "ensemble_comparison.md")
    per_class_csv_path = os.path.join(mode_output_dir, "ensemble_per_class_report.csv")
    per_class_md_path = os.path.join(mode_output_dir, "ensemble_per_class_report.md")
    full_json_path = os.path.join(mode_output_dir, "ensemble_full_metrics.json")

    # Format percentage strings for saved tables
    formatted_df = df_results.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1_Score"]:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}%")

    formatted_df.to_csv(csv_path, index=False)
    df_per_class.to_csv(per_class_csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Ensemble Methods Structured Benchmark Comparison (Mode: {mode.upper()})\n\n")
        f.write(formatted_df.to_markdown(index=False))

    with open(per_class_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Ensemble Methods Per-Class Disease Detailed Report (Mode: {mode.upper()})\n\n")
        f.write(df_per_class.to_markdown(index=False))

    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics_store, f, indent=2)

    print(f"\n\n{'='*60}")
    print(f"  ENSEMBLE FINAL METRICS SUMMARY (MODE: {mode.upper()})")
    print(f"{'='*60}\n")
    print(formatted_df.to_string(index=False))
    print(f"\n  Results saved to:")
    print(f"   - Main Summary CSV: {csv_path}")
    print(f"   - Per-Class Report CSV: {per_class_csv_path}")
    print(f"   - Per-Class Report Markdown: {per_class_md_path}")
    print(f"   - Full JSON Dump: {full_json_path}\n")

    return df_results

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ensemble Evaluation Pipeline")
    parser.add_argument(
        "--mode",
        choices=["val", "oof"],
        default="val",
        help="Evaluation mode: 'val' (Fast validation split ~30s) or 'oof' (Full 5-Fold OOF ~10-13 hrs)",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Main output directory (default: 'outputs')",
    )
    args = parser.parse_args()

    run_ensemble_evaluation(mode=args.mode, outputs_dir=args.outputs_dir)
