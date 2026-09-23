"""Replay saved FINAL_V2 predictions for advanced metrics; never refit."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from scripts.verification.common_utils import (
    load_protocol_predictions,
    verification_output_dir,
)


def evaluate_advanced_metrics(
    results_root: str = "RESULTS/FINAL_V2",
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    for protocol in ("single_split", "oof"):
        base, ensembles = load_protocol_predictions(protocol, results_root=results_root)
        for method, artifact in {**base, **ensembles}.items():
            y_true = artifact.y_true
            predicted = artifact.predictions
            probabilities = artifact.probabilities
            class_indices = np.arange(len(artifact.class_order))
            one_hot = np.eye(len(class_indices), dtype=np.float64)[y_true]
            rows.append({
                "Protocol": protocol,
                "Method": method,
                "Accuracy": accuracy_score(y_true, predicted),
                "Macro_precision": precision_score(y_true, predicted, average="macro", zero_division=0),
                "Macro_recall": recall_score(y_true, predicted, average="macro", zero_division=0),
                "Macro_F1": f1_score(y_true, predicted, average="macro", zero_division=0),
                "Weighted_F1": f1_score(y_true, predicted, average="weighted", zero_division=0),
                "MCC": matthews_corrcoef(y_true, predicted),
                "Cohens_kappa": cohen_kappa_score(y_true, predicted),
                "Macro_ROC_AUC_OVR": roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"),
            })
    frame = pd.DataFrame(rows)
    output = Path(save_dir) if save_dir else verification_output_dir(results_root)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "advanced_metrics_benchmark.csv", index=False)
    with (output / "advanced_metrics_benchmark.md").open("w", encoding="utf-8") as handle:
        handle.write("# Advanced Metrics Replay\n\n")
        handle.write(frame.to_markdown(index=False))
        handle.write("\n")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay advanced metrics")
    parser.add_argument("--results-root", default="RESULTS/FINAL_V2")
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()
    evaluate_advanced_metrics(args.results_root, args.save_dir)
