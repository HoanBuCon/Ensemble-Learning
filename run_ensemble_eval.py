"""
Ensemble Evaluation Runner
==========================

Evaluates all ensemble methods (Hard Voting, Soft Voting, Weighted Voting,
and Stacking Ensembles) using the prediction probabilities of 4 trained 
backbone models (ResNet-50, DenseNet-121, EfficientNet-B0, Swin-Tiny).

Supports two leak-free protocols:
  Mode 1: Train Meta-Learner on Validation Set (1,568 samples) -> Evaluate on Test Set (1,546 samples) [FAST ~1 min]
  Mode 2: Train Meta-Learner on 5-Fold OOF Train Set (7,000 samples) -> Evaluate on Test Set (1,546 samples) [FULL ~10-13 hrs]

CLI::

    python run_ensemble_eval.py --mode val
    python run_ensemble_eval.py --mode oof
    python run_ensemble_eval.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
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
from src.ensemble.oof import OOFGenerator
from src.models.factory import create_model
from src.utils.config import load_config
from src.utils.metrics import compute_metrics


def generate_val_predictions_if_missing(model_dirs: list[str]) -> None:
    """Generate val_probabilities.npy and val_labels.npy if missing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for m_dir in model_dirs:
        val_prob_path = os.path.join(m_dir, "val_probabilities.npy")
        val_label_path = os.path.join(m_dir, "val_labels.npy")

        if os.path.exists(val_prob_path) and os.path.exists(val_label_path):
            continue

        m_name = os.path.basename(m_dir)
        config_path = os.path.join("configs", f"{m_name}.yaml")

        if not os.path.exists(config_path):
            print(f"Config file for {m_name} not found at {config_path}. Skipping.")
            continue

        print(f"--> Generating Validation predictions for {m_name}...")
        cfg = load_config(config_path)

        _, val_loader, _, _ = create_dataloaders(cfg)

        model = create_model(
            model_name=cfg.model.name,
            pretrained=False,
            num_classes=cfg.model.num_classes,
            drop_rate=cfg.model.drop_rate,
        )

        weights_path = os.path.join(m_dir, "best_model.pth")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)

        _, val_probs, _, val_labels, _ = run_inference(model, val_loader, device)

        np.save(val_prob_path, val_probs)
        np.save(val_label_path, val_labels)

        del model
        torch.cuda.empty_cache()


def run_ensemble_evaluation(mode: str = "interactive") -> pd.DataFrame:
    """Run ensemble evaluation based on selected mode (val or oof)."""
    
    if mode not in ["val", "oof"]:
        print("\n" + "=" * 80)
        print("              SELECT STACKING TRAINING MODE")
        print("=" * 80)
        print("\n [1] Validation Mode (Fast - ~30 seconds):")
        print("     - Run inference on Validation set (1,568 images).")
        print("     - Train Meta-Learner & Weighted Voting on Validation predictions.")
        print("     - Evaluate independently on Test set (1,546 images).")
        print("     -> Good for quick, leak-free results.\n")
        print(" [2] Full 5-Fold OOF Mode (Gold Standard - ~10-13 hours):")
        print("     - Run OOFGenerator: retrain each model 5 times on Train folds (7,000 images).")
        print("     - Train Meta-Learner on 7,000 OOF predictions.")
        print("     - Evaluate independently on Test set (1,546 images).")
        print("     -> Best practice for thesis / research papers.")
        print("=" * 80)

        choice = input("\nSelect mode [1/2] (Default: 1): ").strip()
        mode = "oof" if choice == "2" else "val"

    print(f"\n>>> SELECTED MODE: {'FULL 5-FOLD OOF' if mode == 'oof' else 'VALIDATION SPLIT'} <<<\n")

    model_dirs = sorted(glob.glob("outputs/*"))
    model_dirs = [d for d in model_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, "probabilities.npy"))]

    if len(model_dirs) == 0:
        print("No model predictions found in outputs/. Train models first.")
        return pd.DataFrame()

    model_names = [os.path.basename(d) for d in model_dirs]
    test_probs_list = [np.load(os.path.join(d, "probabilities.npy")) for d in model_dirs]
    test_labels = np.load(os.path.join(model_dirs[0], "labels.npy"))

    print(f"Loaded {len(model_names)} models: {', '.join(model_names)}")
    print(f"Test dataset size: {len(test_labels)} samples, {test_probs_list[0].shape[1]} classes\n")

    fit_probs_list = []
    fit_labels = None

    if mode == "val":
        print("--> Generating / Loading Validation predictions for training Meta-Learner...")
        generate_val_predictions_if_missing(model_dirs)
        fit_probs_list = [np.load(os.path.join(d, "val_probabilities.npy")) for d in model_dirs]
        fit_labels = np.load(os.path.join(model_dirs[0], "val_labels.npy"))
        print(f"Validation dataset size for fitting Meta-Learner: {len(fit_labels)} samples\n")
    else:  # mode == "oof"
        print("--> Running 5-Fold OOF Generator across all models...")
        print("    (Each model is retrained 5 times from scratch on Train folds)")
        print("    (Test predictions are averaged across 5 fold models)\n")
        fit_probs_list = []
        oof_test_probs_list = []
        for m_name in model_names:
            cfg_path = os.path.join("configs", f"{m_name}.yaml")
            oof_gen = OOFGenerator(config_path=cfg_path, n_splits=5)
            oof_probs, oof_lbls, oof_test_probs = oof_gen.generate()
            fit_probs_list.append(oof_probs)
            if oof_test_probs is not None:
                oof_test_probs_list.append(oof_test_probs)
            if fit_labels is None:
                fit_labels = oof_lbls
        print(f"OOF dataset size for fitting Meta-Learner: {len(fit_labels)} samples")

        # Override test predictions with OOF-averaged test predictions
        # This ensures the same fold models that generated OOF also predict test
        if len(oof_test_probs_list) == len(model_names):
            test_probs_list = oof_test_probs_list
            print("Test predictions: Using OOF-averaged test probabilities (5-fold averaged)\n")
        else:
            print("Warning: OOF test predictions incomplete. Falling back to fully-trained model predictions.\n")

    results = []

    # 1. Individual Models Baseline
    for name, test_prob in zip(model_names, test_probs_list):
        preds = np.argmax(test_prob, axis=1)
        m = compute_metrics(test_labels, preds)
        results.append({
            "Method": f"Single Model ({name})",
            "Type": "Individual",
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1_Score": m["f1_score"],
            "Details": "-",
        })
        print(f"  [Single Model] {name:<18} Acc: {m['accuracy']*100:.2f}%, F1: {m['f1_score']*100:.2f}%")

    print("-" * 60)

    # 2. Hard Voting
    hv = HardVoting()
    hv_metrics = hv.evaluate(test_probs_list, test_labels)
    results.append({
        "Method": "Hard Voting Ensemble",
        "Type": "Ensemble (Voting)",
        "Accuracy": hv_metrics["accuracy"],
        "Precision": hv_metrics["precision"],
        "Recall": hv_metrics["recall"],
        "F1_Score": hv_metrics["f1_score"],
        "Details": "Majority vote across predictions",
    })
    print(f"  [Ensemble] Hard Voting          Acc: {hv_metrics['accuracy']*100:.2f}%, F1: {hv_metrics['f1_score']*100:.2f}%")

    # 3. Soft Voting
    sv = SoftVoting()
    sv_metrics = sv.evaluate(test_probs_list, test_labels)
    results.append({
        "Method": "Soft Voting Ensemble",
        "Type": "Ensemble (Voting)",
        "Accuracy": sv_metrics["accuracy"],
        "Precision": sv_metrics["precision"],
        "Recall": sv_metrics["recall"],
        "F1_Score": sv_metrics["f1_score"],
        "Details": "Equal weight probability averaging",
    })
    print(f"  [Ensemble] Soft Voting          Acc: {sv_metrics['accuracy']*100:.2f}%, F1: {sv_metrics['f1_score']*100:.2f}%")

    # 4. Weighted Voting (Fitted on Val/OOF, Evaluated on Test)
    wv = WeightedVoting()
    wv.fit(fit_probs_list, fit_labels)
    wv_metrics = wv.evaluate(test_probs_list, test_labels)
    weights_str = ", ".join([f"{n}: {w:.2f}" for n, w in zip(model_names, wv.weights)])
    results.append({
        "Method": "Weighted Voting Ensemble",
        "Type": "Ensemble (Voting)",
        "Accuracy": wv_metrics["accuracy"],
        "Precision": wv_metrics["precision"],
        "Recall": wv_metrics["recall"],
        "F1_Score": wv_metrics["f1_score"],
        "Details": f"Optimized weights ({weights_str})",
    })
    print(f"  [Ensemble] Weighted Voting      Acc: {wv_metrics['accuracy']*100:.2f}%, F1: {wv_metrics['f1_score']*100:.2f}%")

    # 5. Stacking Ensembles (Fitted on Val/OOF, Evaluated on Test)
    for meta_learner in ["logistic_regression", "random_forest", "xgboost"]:
        try:
            st = StackingEnsemble(meta_learner=meta_learner)
            st.fit(fit_probs_list, fit_labels)
            st_metrics = st.evaluate(test_probs_list, test_labels)
            
            method_name = f"Stacking ({meta_learner.replace('_', ' ').title()})"
            fit_source_name = "5-Fold OOF Train" if mode == "oof" else "Validation Set"
            
            results.append({
                "Method": method_name,
                "Type": "Ensemble (Stacking)",
                "Accuracy": st_metrics["accuracy"],
                "Precision": st_metrics["precision"],
                "Recall": st_metrics["recall"],
                "F1_Score": st_metrics["f1_score"],
                "Details": f"Meta-learner: {meta_learner} (Fit on {fit_source_name})",
            })
            print(f"  [Ensemble] {method_name:<25} Acc: {st_metrics['accuracy']*100:.2f}%, F1: {st_metrics['f1_score']*100:.2f}% (Fit: {fit_source_name})")
        except Exception as e:
            print(f"  [Skip] Stacking ({meta_learner}): {e}")

    df = pd.DataFrame(results)

    # Calculate Improvement (+-%) relative to best single base model
    single_models_df = df[df["Type"] == "Individual"]
    best_single_acc = single_models_df["Accuracy"].max()

    improvements = []
    for _, row in df.iterrows():
        diff = (row["Accuracy"] - best_single_acc) * 100
        if row["Accuracy"] == best_single_acc and row["Type"] == "Individual":
            improvements.append("Base (0.00%)")
        else:
            improvements.append(f"{diff:+.2f}%")

    df["Improvement"] = improvements

    cols = ["Method", "Type", "Accuracy", "Improvement", "Precision", "Recall", "F1_Score", "Details"]
    df = df[cols]

    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "ensemble_comparison.csv")
    md_path = os.path.join(output_dir, "ensemble_comparison.md")

    df.to_csv(csv_path, index=False)

    md_df = df.copy()
    md_df["Accuracy"] = md_df["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
    md_df["Precision"] = md_df["Precision"].apply(lambda x: f"{x*100:.2f}%")
    md_df["Recall"] = md_df["Recall"].apply(lambda x: f"{x*100:.2f}%")
    md_df["F1_Score"] = md_df["F1_Score"].apply(lambda x: f"{x*100:.2f}%")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Ensemble Methods Structured Benchmark Comparison (Mode: {mode.upper()})\n\n")
        f.write(md_df.to_markdown(index=False))

    print(f"\n\n{'='*60}")
    print(f"  ENSEMBLE FINAL METRICS SUMMARY (MODE: {mode.upper()})")
    print(f"{'='*60}\n")
    print(md_df.to_string(index=False))

    print(f"\n  Results saved to: {csv_path}")
    print(f"  Markdown table: {md_path}\n")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ensemble Evaluation")
    parser.add_argument(
        "--mode",
        choices=["val", "oof", "interactive"],
        default="interactive",
        help="Training source for Stacking & Weighted Voting (val or oof)",
    )
    args = parser.parse_args()

    run_ensemble_evaluation(mode=args.mode)
