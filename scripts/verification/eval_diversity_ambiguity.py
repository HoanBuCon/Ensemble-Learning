"""
Model Diversity and Ambiguity Decomposition Benchmark
====================================================

Computes and exports:
1. Pairwise Disagreement Rate
2. Pairwise Yule's Q-Statistic
3. Pairwise Cohen's Kappa
4. Global Disagreement Rate
5. Continuous Probability Vector Ambiguity (A_prob)

Usage:
    python scripts/verification/eval_diversity_ambiguity.py
    python scripts/verification/eval_diversity_ambiguity.py --save-dir outputs/verification
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_dataset_config
from src.datasets.dataset import ImageFolderDataset
from scripts.verification.common_utils import resolve_outputs_dirs


def compute_yules_q(preds_i: np.ndarray, preds_j: np.ndarray, y_true: np.ndarray) -> float:
    correct_i = (preds_i == y_true)
    correct_j = (preds_j == y_true)
    N11 = np.sum(correct_i & correct_j)
    N00 = np.sum((~correct_i) & (~correct_j))
    N10 = np.sum(correct_i & (~correct_j))
    N01 = np.sum((~correct_i) & correct_j)
    det = (N11 * N00) - (N10 * N01)
    denom = (N11 * N00) + (N10 * N01)
    return float(det / denom) if denom != 0 else 1.0


def compute_prob_ambiguity(probs_list: list[np.ndarray]) -> float:
    stacked = np.stack(probs_list, axis=0)
    mean_p = np.mean(stacked, axis=0)
    sq_diff = np.sum((stacked - mean_p)**2, axis=2)
    return float(np.mean(sq_diff))


def evaluate_diversity(
    default_dir: str | None = None, 
    oof_dir: str | None = None,
    save_dir: str | None = "outputs/verification",
) -> pd.DataFrame:
    """Compute and export diversity & ambiguity tables."""
    def_dir, oof_dir = resolve_outputs_dirs(default_dir, oof_dir)
    print(f"-> Using Default Results Directory: '{def_dir}'")
    print(f"-> Using 5-Fold OOF Results Directory: '{oof_dir}'")

    ds_raw = load_dataset_config()
    test_ds = ImageFolderDataset(root=ds_raw.get("test_dir", "./data/test"))
    y_true = np.array([s[1] for s in test_ds.samples])
    N = len(y_true)

    model_names = ["densenet121", "efficientnet_b0", "resnet50", "swin_tiny"]
    display_names = ["DenseNet-121", "EfficientNet-B0", "ResNet-50", "Swin-Tiny"]

    def_probs = [
        np.load(os.path.join(def_dir, m, "test_probabilities.npy") if os.path.exists(os.path.join(def_dir, m, "test_probabilities.npy")) else os.path.join(def_dir, m, "probabilities.npy")) 
        for m in model_names
    ]
    oof_probs = [
        np.load(os.path.join(oof_dir, m, "kfold", "test_probabilities.npy") if os.path.exists(os.path.join(oof_dir, m, "kfold", "test_probabilities.npy")) else os.path.join(oof_dir, m, "test_probabilities.npy")) 
        for m in model_names
    ]

    def_preds = [np.argmax(p, axis=1) for p in def_probs]
    oof_preds = [np.argmax(p, axis=1) for p in oof_probs]

    pairs = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3),
        (2, 3)
    ]

    rows = []
    print("\n" + "=" * 90)
    print("          MODEL DIVERSITY & AMBIGUITY DECOMPOSITION BENCHMARK")
    print("=" * 90)

    for i, j in pairs:
        dis_def = np.mean(def_preds[i] != def_preds[j]) * 100
        dis_def_cnt = np.sum(def_preds[i] != def_preds[j])
        q_def = compute_yules_q(def_preds[i], def_preds[j], y_true)
        kap_def = cohen_kappa_score(def_preds[i], def_preds[j])

        dis_oof = np.mean(oof_preds[i] != oof_preds[j]) * 100
        dis_oof_cnt = np.sum(oof_preds[i] != oof_preds[j])
        q_oof = compute_yules_q(oof_preds[i], oof_preds[j], y_true)
        kap_oof = cohen_kappa_score(oof_preds[i], oof_preds[j])

        rows.append({
            "Model Pair": f"{display_names[i]} vs {display_names[j]}",
            "Disagreement Single": f"{dis_def:.2f}% ({dis_def_cnt})",
            "Disagreement OOF": f"{dis_oof:.2f}% ({dis_oof_cnt})",
            "Yule's Q Single": f"{q_def:.4f}",
            "Yule's Q OOF": f"{q_oof:.4f}",
            "Kappa Single": f"{kap_def:.4f}",
            "Kappa OOF": f"{kap_oof:.4f}",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    global_def_cnt = np.sum([len(set([def_preds[m][idx] for m in range(4)])) > 1 for idx in range(N)])
    global_oof_cnt = np.sum([len(set([oof_preds[m][idx] for m in range(4)])) > 1 for idx in range(N)])
    amb_def = compute_prob_ambiguity(def_probs)
    amb_oof = compute_prob_ambiguity(oof_probs)

    print("-" * 90)
    print(f"Global Disagreement Rate:  Single-Split = {global_def_cnt/N*100:.2f}% ({global_def_cnt}/{N}) | 5-Fold OOF = {global_oof_cnt/N*100:.2f}% ({global_oof_cnt}/{N})")
    print(f"Probability Ambiguity (A): Single-Split = {amb_def:.6f} | 5-Fold OOF = {amb_oof:.6f} | Ratio = {amb_def/amb_oof:.3f}x")
    print("=" * 90)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_p = os.path.join(save_dir, "diversity_benchmark.csv")
        md_p = os.path.join(save_dir, "diversity_benchmark.md")
        df.to_csv(csv_p, index=False)
        with open(md_p, "w", encoding="utf-8") as f:
            f.write("# Model Diversity & Ambiguity Decomposition Benchmark\n\n")
            f.write(df.to_markdown(index=False))
            f.write(f"\n\n- **Global Disagreement Rate**: Single-Split = {global_def_cnt/N*100:.2f}% ({global_def_cnt}/{N}) | 5-Fold OOF = {global_oof_cnt/N*100:.2f}% ({global_oof_cnt}/{N})\n")
            f.write(f"- **Probability Ambiguity (A)**: Single-Split = {amb_def:.6f} | 5-Fold OOF = {amb_oof:.6f} (Ratio = {amb_def/amb_oof:.3f}x)\n")
        print(f"\n[Saved Diversity Reports] -> '{csv_p}' and '{md_p}'")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Diversity and Ambiguity Decomposition")
    parser.add_argument("--default-dir", default=None, help="Path to Single-Split outputs directory")
    parser.add_argument("--oof-dir", default=None, help="Path to 5-Fold OOF outputs directory")
    parser.add_argument("--save-dir", default="outputs/verification", help="Directory to save output reports")
    args = parser.parse_args()
    evaluate_diversity(default_dir=args.default_dir, oof_dir=args.oof_dir, save_dir=args.save_dir)
