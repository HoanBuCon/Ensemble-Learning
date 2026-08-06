"""
Ensemble Evaluation Runner
==========================

Evaluates all ensemble methods (Hard Voting, Soft Voting, Weighted Voting,
and Stacking Ensembles) using the saved test probability predictions of all
4 trained backbone models (ResNet-50, DenseNet-121, EfficientNet-B0, Swin-Tiny).

CLI::

    python run_ensemble_eval.py

Outputs::

    outputs/ensemble_comparison.csv
    outputs/ensemble_comparison.md
"""

from __future__ import annotations

import glob
import json
import os
import numpy as np
import pandas as pd

from src.ensemble import (
    HardVoting,
    SoftVoting,
    WeightedVoting,
    StackingEnsemble,
)


def run_ensemble_evaluation() -> pd.DataFrame:
    """Load predictions from all models and benchmark all ensemble strategies."""
    model_dirs = sorted(glob.glob("outputs/*"))
    model_dirs = [d for d in model_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, "probabilities.npy"))]

    if len(model_dirs) == 0:
        print("No model predictions found in outputs/. Train models first.")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"  Evaluating Ensemble Methods on {len(model_dirs)} Models")
    print(f"{'='*60}\n")

    model_names = []
    probs_list = []
    labels = None

    for m_dir in model_dirs:
        m_name = os.path.basename(m_dir)
        prob_path = os.path.join(m_dir, "probabilities.npy")
        label_path = os.path.join(m_dir, "labels.npy")

        prob = np.load(prob_path)
        lbl = np.load(label_path)

        model_names.append(m_name)
        probs_list.append(prob)

        if labels is None:
            labels = lbl

    print(f"Loaded models: {', '.join(model_names)}")
    print(f"Test dataset size: {len(labels)} samples, {probs_list[0].shape[1]} classes\n")

    results = []

    # Individual Models Baseline
    for name, prob in zip(model_names, probs_list):
        preds = np.argmax(prob, axis=1)
        acc = float(np.mean(preds == labels))
        results.append({
            "Method": f"Single Model ({name})",
            "Type": "Individual",
            "Accuracy": acc,
            "Details": "-",
        })
        print(f"  [Single Model] {name:<18} Accuracy: {acc*100:.2f}%")

    print("-" * 60)

    # 1. Hard Voting
    hv = HardVoting()
    hv_metrics = hv.evaluate(probs_list, labels)
    results.append({
        "Method": "Hard Voting Ensemble",
        "Type": "Ensemble (Voting)",
        "Accuracy": hv_metrics["accuracy"],
        "Details": "Majority vote across predictions",
    })
    print(f"  [Ensemble] Hard Voting          Accuracy: {hv_metrics['accuracy']*100:.2f}%")

    # 2. Soft Voting
    sv = SoftVoting()
    sv_metrics = sv.evaluate(probs_list, labels)
    results.append({
        "Method": "Soft Voting Ensemble",
        "Type": "Ensemble (Voting)",
        "Accuracy": sv_metrics["accuracy"],
        "Details": "Equal weight probability averaging",
    })
    print(f"  [Ensemble] Soft Voting          Accuracy: {sv_metrics['accuracy']*100:.2f}%")

    # 3. Weighted Voting
    wv = WeightedVoting()
    wv.fit(probs_list, labels)
    wv_metrics = wv.evaluate(probs_list, labels)
    weights_str = ", ".join([f"{n}: {w:.2f}" for n, w in zip(model_names, wv.weights)])
    results.append({
        "Method": "Weighted Voting Ensemble",
        "Type": "Ensemble (Voting)",
        "Accuracy": wv_metrics["accuracy"],
        "Details": f"Optimized weights ({weights_str})",
    })
    print(f"  [Ensemble] Weighted Voting      Accuracy: {wv_metrics['accuracy']*100:.2f}% ({weights_str})")

    # 4. Stacking Ensembles
    for meta_learner in ["logistic_regression", "random_forest", "xgboost"]:
        try:
            st = StackingEnsemble(meta_learner=meta_learner)
            st.fit(probs_list, labels)
            st_metrics = st.evaluate(probs_list, labels)
            method_name = f"Stacking ({meta_learner.replace('_', ' ').title()})"
            results.append({
                "Method": method_name,
                "Type": "Ensemble (Stacking)",
                "Accuracy": st_metrics["accuracy"],
                "Details": f"Meta-learner: {meta_learner}",
            })
            print(f"  [Ensemble] {method_name:<20} Accuracy: {st_metrics['accuracy']*100:.2f}%")
        except Exception as e:
            print(f"  [Skip] Stacking ({meta_learner}): {e}")

    df = pd.DataFrame(results)
    df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    output_dir = "./outputs"
    csv_path = os.path.join(output_dir, "ensemble_comparison.csv")
    md_path = os.path.join(output_dir, "ensemble_comparison.md")

    df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Ensemble Methods Comparison\n\n")
        f.write(df.to_markdown(index=False))

    print(f"\n\n{'='*60}")
    print("  ENSEMBLE FINAL RESULTS SUMMARY")
    print(f"{'='*60}\n")

    display_df = df.copy()
    display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
    print(display_df.to_string(index=False))

    print(f"\n  Results saved to: {csv_path}")
    print(f"  Markdown table: {md_path}\n")

    return df


if __name__ == "__main__":
    run_ensemble_evaluation()
