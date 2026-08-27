"""
Probability Calibration and Uncertainty Evaluation
==================================================

Computes and exports:
1. Expected Calibration Error (ECE) with 15 uniform bins
2. Multi-class Brier Score (Strictly proper scoring rule)
3. Negative Log-Likelihood (NLL / Log-Loss with epsilon clipping)

Usage:
    python scripts/verification/eval_calibration.py
    python scripts/verification/eval_calibration.py --save-dir outputs/verification
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_dataset_config
from src.datasets.dataset import ImageFolderDataset
from scripts.verification.common_utils import resolve_outputs_dirs


def calc_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    N = len(y_true)
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    acc = (preds == y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_idx = np.where((conf > bins[i]) & (conf <= bins[i+1]))[0]
        if len(bin_idx) > 0:
            bin_acc = np.mean(acc[bin_idx])
            bin_conf = np.mean(conf[bin_idx])
            ece += (len(bin_idx) / N) * np.abs(bin_acc - bin_conf)
    return float(ece)


def calc_brier(probs: np.ndarray, y_true: np.ndarray, num_classes: int = 6) -> float:
    N = len(y_true)
    y_one_hot = np.zeros((N, num_classes))
    for i in range(N):
        y_one_hot[i, y_true[i]] = 1.0
    return float(np.mean(np.sum((probs - y_one_hot)**2, axis=1)))


def calc_nll(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-15) -> float:
    N = len(y_true)
    probs_c = np.clip(probs, eps, 1.0)
    return float(-np.mean([np.log(probs_c[i, y_true[i]]) for i in range(N)]))


def evaluate_all_calibration(
    default_dir: str | None = None, 
    oof_dir: str | None = None,
    save_dir: str | None = "outputs/verification",
) -> pd.DataFrame:
    """Run calibration evaluation and export CSV/MD reports."""
    def_dir, oof_dir = resolve_outputs_dirs(default_dir, oof_dir)
    print(f"-> Using Default Results Directory: '{def_dir}'")
    print(f"-> Using 5-Fold OOF Results Directory: '{oof_dir}'")

    ds_raw = load_dataset_config()
    test_ds = ImageFolderDataset(root=ds_raw.get("test_dir", "./data/test"))
    y_true = np.array([s[1] for s in test_ds.samples])
    
    models = ["densenet121", "efficientnet_b0", "resnet50", "swin_tiny"]
    results = []

    print("\n" + "=" * 90)
    print("      PROBABILITY CALIBRATION & UNCERTAINTY BENCHMARK (ECE, BRIER, NLL)")
    print("=" * 90)

    # 1. Base models Default
    for m in models:
        p_path = os.path.join(def_dir, m, "test_probabilities.npy")
        if not os.path.exists(p_path):
            p_path = os.path.join(def_dir, m, "probabilities.npy")
        if os.path.exists(p_path):
            probs = np.load(p_path)
            results.append({
                "Protocol": "Single-Split",
                "Model / Ensemble": f"Base Model ({m})",
                "ECE (15 bins)": f"{calc_ece(probs, y_true):.4f}",
                "Brier Score": f"{calc_brier(probs, y_true):.4f}",
                "NLL": f"{calc_nll(probs, y_true):.4f}",
            })

    # 2. Base models OOF
    for m in models:
        p_path = os.path.join(oof_dir, m, "kfold", "test_probabilities.npy")
        if not os.path.exists(p_path):
            p_path = os.path.join(oof_dir, m, "test_probabilities.npy")
        if os.path.exists(p_path):
            probs = np.load(p_path)
            results.append({
                "Protocol": "5-Fold OOF",
                "Model / Ensemble": f"Base Model ({m})",
                "ECE (15 bins)": f"{calc_ece(probs, y_true):.4f}",
                "Brier Score": f"{calc_brier(probs, y_true):.4f}",
                "NLL": f"{calc_nll(probs, y_true):.4f}",
            })

    # 3. Ensembles
    def_probs = [
        np.load(os.path.join(def_dir, m, "test_probabilities.npy") if os.path.exists(os.path.join(def_dir, m, "test_probabilities.npy")) else os.path.join(def_dir, m, "probabilities.npy")) 
        for m in models
    ]
    oof_probs = [
        np.load(os.path.join(oof_dir, m, "kfold", "test_probabilities.npy") if os.path.exists(os.path.join(oof_dir, m, "kfold", "test_probabilities.npy")) else os.path.join(oof_dir, m, "test_probabilities.npy")) 
        for m in models
    ]

    # Soft Vote
    soft_def = np.mean(def_probs, axis=0)
    soft_oof = np.mean(oof_probs, axis=0)
    results.append({"Protocol": "Single-Split", "Model / Ensemble": "Soft Voting Ensemble", "ECE (15 bins)": f"{calc_ece(soft_def, y_true):.4f}", "Brier Score": f"{calc_brier(soft_def, y_true):.4f}", "NLL": f"{calc_nll(soft_def, y_true):.4f}"})
    results.append({"Protocol": "5-Fold OOF", "Model / Ensemble": "Soft Voting Ensemble", "ECE (15 bins)": f"{calc_ece(soft_oof, y_true):.4f}", "Brier Score": f"{calc_brier(soft_oof, y_true):.4f}", "NLL": f"{calc_nll(soft_oof, y_true):.4f}"})

    # Hard Vote (Vote Proportion)
    def hard_vote_probs(p_list):
        preds = [np.argmax(p, axis=1) for p in p_list]
        N_samples = len(p_list[0])
        C = 6
        hp = np.zeros((N_samples, C))
        for i in range(N_samples):
            for m_idx in range(len(p_list)):
                hp[i, preds[m_idx][i]] += 1.0 / len(p_list)
        return hp

    hard_def = hard_vote_probs(def_probs)
    hard_oof = hard_vote_probs(oof_probs)
    results.append({"Protocol": "Single-Split", "Model / Ensemble": "Hard Voting Ensemble", "ECE (15 bins)": f"{calc_ece(hard_def, y_true):.4f}", "Brier Score": f"{calc_brier(hard_def, y_true):.4f}", "NLL": f"{calc_nll(hard_def, y_true, eps=1e-7):.4f}"})
    results.append({"Protocol": "5-Fold OOF", "Model / Ensemble": "Hard Voting Ensemble", "ECE (15 bins)": f"{calc_ece(hard_oof, y_true):.4f}", "Brier Score": f"{calc_brier(hard_oof, y_true):.4f}", "NLL": f"{calc_nll(hard_oof, y_true, eps=1e-7):.4f}"})

    # Stacking
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    # Default fit on Val
    val_p_list = [np.load(os.path.join(def_dir, m, "val_probabilities.npy")) for m in models]
    val_probs_def = np.hstack(val_p_list)
    val_labels_def = np.load(os.path.join(def_dir, "densenet121", "val_labels.npy"))
    test_feat_def = np.hstack(def_probs)

    lr_def = LogisticRegression(max_iter=1000, random_state=42).fit(val_probs_def, val_labels_def)
    rf_def = RandomForestClassifier(n_estimators=200, random_state=42).fit(val_probs_def, val_labels_def)
    xgb_def = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, eval_metric="mlogloss").fit(val_probs_def, val_labels_def)

    results.append({"Protocol": "Single-Split", "Model / Ensemble": "Stacking (Logistic Regression)", "ECE (15 bins)": f"{calc_ece(lr_def.predict_proba(test_feat_def), y_true):.4f}", "Brier Score": f"{calc_brier(lr_def.predict_proba(test_feat_def), y_true):.4f}", "NLL": f"{calc_nll(lr_def.predict_proba(test_feat_def), y_true):.4f}"})
    results.append({"Protocol": "Single-Split", "Model / Ensemble": "Stacking (Random Forest)", "ECE (15 bins)": f"{calc_ece(rf_def.predict_proba(test_feat_def), y_true):.4f}", "Brier Score": f"{calc_brier(rf_def.predict_proba(test_feat_def), y_true):.4f}", "NLL": f"{calc_nll(rf_def.predict_proba(test_feat_def), y_true):.4f}"})
    results.append({"Protocol": "Single-Split", "Model / Ensemble": "Stacking (XGBoost)", "ECE (15 bins)": f"{calc_ece(xgb_def.predict_proba(test_feat_def), y_true):.4f}", "Brier Score": f"{calc_brier(xgb_def.predict_proba(test_feat_def), y_true):.4f}", "NLL": f"{calc_nll(xgb_def.predict_proba(test_feat_def), y_true):.4f}"})

    # OOF fit on OOF
    oof_p_list = [np.load(os.path.join(oof_dir, m, "kfold", "oof_probabilities.npy")) for m in models]
    oof_feat = np.hstack(oof_p_list)
    oof_labels = np.load(os.path.join(oof_dir, "densenet121", "kfold", "oof_labels.npy"))
    test_feat_oof = np.hstack(oof_probs)

    lr_oof = LogisticRegression(max_iter=1000, random_state=42).fit(oof_feat, oof_labels)
    rf_oof = RandomForestClassifier(n_estimators=200, random_state=42).fit(oof_feat, oof_labels)
    xgb_oof = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, eval_metric="mlogloss").fit(oof_feat, oof_labels)

    results.append({"Protocol": "5-Fold OOF", "Model / Ensemble": "Stacking (Logistic Regression)", "ECE (15 bins)": f"{calc_ece(lr_oof.predict_proba(test_feat_oof), y_true):.4f}", "Brier Score": f"{calc_brier(lr_oof.predict_proba(test_feat_oof), y_true):.4f}", "NLL": f"{calc_nll(lr_oof.predict_proba(test_feat_oof), y_true):.4f}"})
    results.append({"Protocol": "5-Fold OOF", "Model / Ensemble": "Stacking (Random Forest)", "ECE (15 bins)": f"{calc_ece(rf_oof.predict_proba(test_feat_oof), y_true):.4f}", "Brier Score": f"{calc_brier(rf_oof.predict_proba(test_feat_oof), y_true):.4f}", "NLL": f"{calc_nll(rf_oof.predict_proba(test_feat_oof), y_true):.4f}"})
    results.append({"Protocol": "5-Fold OOF", "Model / Ensemble": "Stacking (XGBoost)", "ECE (15 bins)": f"{calc_ece(xgb_oof.predict_proba(test_feat_oof), y_true):.4f}", "Brier Score": f"{calc_brier(xgb_oof.predict_proba(test_feat_oof), y_true):.4f}", "NLL": f"{calc_nll(xgb_oof.predict_proba(test_feat_oof), y_true):.4f}"})

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_p = os.path.join(save_dir, "calibration_benchmark.csv")
        md_p = os.path.join(save_dir, "calibration_benchmark.md")
        df.to_csv(csv_p, index=False)
        with open(md_p, "w", encoding="utf-8") as f:
            f.write("# Probability Calibration & Uncertainty Benchmark\n\n")
            f.write(df.to_markdown(index=False))
        print(f"\n[Saved Calibration Reports] -> '{csv_p}' and '{md_p}'")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Calibration and Predictive Uncertainty")
    parser.add_argument("--default-dir", default=None, help="Path to Single-Split outputs directory")
    parser.add_argument("--oof-dir", default=None, help="Path to 5-Fold OOF outputs directory")
    parser.add_argument("--save-dir", default="outputs/verification", help="Directory to save output reports")
    args = parser.parse_args()
    evaluate_all_calibration(default_dir=args.default_dir, oof_dir=args.oof_dir, save_dir=args.save_dir)
