"""
Advanced Discrimination and Ranking Benchmark (Macro ROC-AUC & MCC)
===================================================================

Computes:
1. Multi-Class Macro ROC-AUC (One-vs-Rest Area Under Receiver Operating Characteristic)
2. Matthews Correlation Coefficient (MCC, robust across class distribution)
3. Macro Precision, Recall, Macro F1, Weighted F1, Cohen's Kappa, Accuracy

Usage:
    python scripts/verification/eval_advanced_metrics.py
    python scripts/verification/eval_advanced_metrics.py --save-dir outputs/verification
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_dataset_config
from src.datasets.dataset import ImageFolderDataset
from scripts.verification.common_utils import resolve_outputs_dirs


def evaluate_advanced_metrics(
    default_dir: Optional[str] = None,
    oof_dir: Optional[str] = None,
    save_dir: Optional[str] = "outputs/verification",
) -> pd.DataFrame:
    """Compute Macro ROC-AUC, MCC, F1, and Kappa across all 10 models in Single-Split vs. 5-Fold OOF."""
    def_dir, oof_dir = resolve_outputs_dirs(default_dir, oof_dir)
    print(f"-> Using Default Results Directory: '{def_dir}'")
    print(f"-> Using 5-Fold OOF Results Directory: '{oof_dir}'")

    ds_raw = load_dataset_config()
    test_ds = ImageFolderDataset(root=ds_raw.get("test_dir", "./data/test"))
    y_true = np.array([s[1] for s in test_ds.samples])
    num_classes = len(test_ds.classes)
    y_true_one_hot = label_binarize(y_true, classes=list(range(num_classes)))

    models = ["densenet121", "efficientnet_b0", "resnet50", "swin_tiny"]
    display_names = {
        "densenet121": "DenseNet-121",
        "efficientnet_b0": "EfficientNet-B0",
        "resnet50": "ResNet-50",
        "swin_tiny": "Swin-Tiny",
    }

    # Load Single-Split probabilities
    def_probs = [
        np.load(os.path.join(def_dir, m, "test_probabilities.npy") if os.path.exists(os.path.join(def_dir, m, "test_probabilities.npy")) else os.path.join(def_dir, m, "probabilities.npy")) 
        for m in models
    ]
    # Load 5-Fold OOF probabilities
    oof_probs = [
        np.load(os.path.join(oof_dir, m, "kfold", "test_probabilities.npy") if os.path.exists(os.path.join(oof_dir, m, "kfold", "test_probabilities.npy")) else os.path.join(oof_dir, m, "test_probabilities.npy")) 
        for m in models
    ]

    # Stacking models setup
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    val_p_list = [np.load(os.path.join(def_dir, m, "val_probabilities.npy")) for m in models]
    val_probs_def = np.hstack(val_p_list)
    val_labels_def = np.load(os.path.join(def_dir, "densenet121", "val_labels.npy"))
    test_feat_def = np.hstack(def_probs)

    lr_def = LogisticRegression(max_iter=1000, random_state=42).fit(val_probs_def, val_labels_def)
    rf_def = RandomForestClassifier(n_estimators=200, random_state=42).fit(val_probs_def, val_labels_def)
    xgb_def = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, eval_metric="mlogloss").fit(val_probs_def, val_labels_def)

    oof_p_list = [np.load(os.path.join(oof_dir, m, "kfold", "oof_probabilities.npy")) for m in models]
    oof_feat = np.hstack(oof_p_list)
    oof_labels = np.load(os.path.join(oof_dir, "densenet121", "kfold", "oof_labels.npy"))
    test_feat_oof = np.hstack(oof_probs)

    lr_oof = LogisticRegression(max_iter=1000, random_state=42).fit(oof_feat, oof_labels)
    rf_oof = RandomForestClassifier(n_estimators=200, random_state=42).fit(oof_feat, oof_labels)
    xgb_oof = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, eval_metric="mlogloss").fit(oof_feat, oof_labels)

    # Hard Voting probabilities
    def get_hard_vote_probs(p_list):
        preds = [np.argmax(p, axis=1) for p in p_list]
        N_samples = len(p_list[0])
        hp = np.zeros((N_samples, num_classes))
        for i in range(N_samples):
            for m_idx in range(len(p_list)):
                hp[i, preds[m_idx][i]] += 1.0 / len(p_list)
        return hp

    # Build model dictionary
    all_experiments = []

    # 1. Single-Split Base
    for idx, m in enumerate(models):
        p = def_probs[idx]
        pred = np.argmax(p, axis=1)
        all_experiments.append(("Single-Split", f"Base Model ({display_names[m]})", p, pred))

    # 2. Single-Split Ensembles
    soft_def = np.mean(def_probs, axis=0)
    all_experiments.append(("Single-Split", "Soft Voting Ensemble", soft_def, np.argmax(soft_def, axis=1)))
    hard_def = get_hard_vote_probs(def_probs)
    all_experiments.append(("Single-Split", "Hard Voting Ensemble", hard_def, np.argmax(hard_def, axis=1)))
    p_lr_def = lr_def.predict_proba(test_feat_def)
    all_experiments.append(("Single-Split", "Stacking (Logistic Regression)", p_lr_def, np.argmax(p_lr_def, axis=1)))
    p_rf_def = rf_def.predict_proba(test_feat_def)
    all_experiments.append(("Single-Split", "Stacking (Random Forest)", p_rf_def, np.argmax(p_rf_def, axis=1)))
    p_xgb_def = xgb_def.predict_proba(test_feat_def)
    all_experiments.append(("Single-Split", "Stacking (XGBoost)", p_xgb_def, np.argmax(p_xgb_def, axis=1)))

    # 3. 5-Fold OOF Base
    for idx, m in enumerate(models):
        p = oof_probs[idx]
        pred = np.argmax(p, axis=1)
        all_experiments.append(("5-Fold OOF", f"Base Model ({display_names[m]})", p, pred))

    # 4. 5-Fold OOF Ensembles
    soft_oof = np.mean(oof_probs, axis=0)
    all_experiments.append(("5-Fold OOF", "Soft Voting Ensemble", soft_oof, np.argmax(soft_oof, axis=1)))
    hard_oof = get_hard_vote_probs(oof_probs)
    all_experiments.append(("5-Fold OOF", "Hard Voting Ensemble", hard_oof, np.argmax(hard_oof, axis=1)))
    p_lr_oof = lr_oof.predict_proba(test_feat_oof)
    all_experiments.append(("5-Fold OOF", "Stacking (Logistic Regression)", p_lr_oof, np.argmax(p_lr_oof, axis=1)))
    p_rf_oof = rf_oof.predict_proba(test_feat_oof)
    all_experiments.append(("5-Fold OOF", "Stacking (Random Forest)", p_rf_oof, np.argmax(p_rf_oof, axis=1)))
    p_xgb_oof = xgb_oof.predict_proba(test_feat_oof)
    all_experiments.append(("5-Fold OOF", "Stacking (XGBoost)", p_xgb_oof, np.argmax(p_xgb_oof, axis=1)))

    results = []
    print("\n" + "=" * 105)
    print("      ADVANCED DISCRIMINATION & RANKING BENCHMARK (MACRO ROC-AUC, MCC, KAPPA, F1)")
    print("=" * 105)

    for protocol, name, prob_mat, pred_vec in all_experiments:
        acc = accuracy_score(y_true, pred_vec) * 100
        macro_f1 = f1_score(y_true, pred_vec, average="macro") * 100
        weighted_f1 = f1_score(y_true, pred_vec, average="weighted") * 100
        mcc = matthews_corrcoef(y_true, pred_vec)
        kappa = cohen_kappa_score(y_true, pred_vec)

        try:
            roc_auc = roc_auc_score(y_true_one_hot, prob_mat, average="macro", multi_class="ovr")
        except Exception:
            roc_auc = 0.0

        results.append({
            "Protocol": protocol,
            "Model / Ensemble Method": name,
            "Accuracy (%)": f"{acc:.2f}%",
            "Macro F1 (%)": f"{macro_f1:.2f}%",
            "Weighted F1 (%)": f"{weighted_f1:.2f}%",
            "MCC": f"{mcc:.4f}",
            "Cohen's Kappa": f"{kappa:.4f}",
            "Macro ROC-AUC": f"{roc_auc:.4f}",
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("=" * 105)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_p = os.path.join(save_dir, "advanced_metrics_benchmark.csv")
        md_p = os.path.join(save_dir, "advanced_metrics_benchmark.md")
        df.to_csv(csv_p, index=False)
        with open(md_p, "w", encoding="utf-8") as f:
            f.write("# Advanced Discrimination & Ranking Benchmark (ROC-AUC & MCC)\n\n")
            f.write(df.to_markdown(index=False))
        print(f"\n[Saved Advanced Metrics Reports] -> '{csv_p}' and '{md_p}'")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Macro ROC-AUC and MCC")
    parser.add_argument("--default-dir", default=None, help="Path to Single-Split outputs directory")
    parser.add_argument("--oof-dir", default=None, help="Path to 5-Fold OOF outputs directory")
    parser.add_argument("--save-dir", default="outputs/verification", help="Directory to save output reports")
    args = parser.parse_args()
    evaluate_advanced_metrics(default_dir=args.default_dir, oof_dir=args.oof_dir, save_dir=args.save_dir)
