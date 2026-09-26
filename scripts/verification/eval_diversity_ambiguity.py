"""Replay aligned base predictions for diversity and ambiguity metrics."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from scripts.verification.common_utils import (
    load_protocol_predictions,
    resolve_verification_output_dir,
)


def compute_yules_q(pred_a: np.ndarray, pred_b: np.ndarray, y_true: np.ndarray) -> float:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    n11 = int(np.sum(correct_a & correct_b))
    n00 = int(np.sum(~correct_a & ~correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    denominator = n11 * n00 + n10 * n01
    return float((n11 * n00 - n10 * n01) / denominator) if denominator else 1.0


def compute_prob_ambiguity(probabilities: list[np.ndarray]) -> float:
    stacked = np.stack(probabilities, axis=0)
    mean_probability = stacked.mean(axis=0)
    return float(np.mean(np.sum((stacked - mean_probability) ** 2, axis=2)))


def evaluate_diversity(
    results_root: str = "RESULTS",
    save_dir: Optional[str] = None,
    run_id: str = "",
) -> pd.DataFrame:
    rows = []
    summaries = []
    for protocol in ("single_split", "oof"):
        base, _ = load_protocol_predictions(protocol, results_root=results_root, run_id=run_id)
        names = list(base)
        reference = base[names[0]]
        for left, right in itertools.combinations(names, 2):
            pred_left = base[left].predictions
            pred_right = base[right].predictions
            rows.append({
                "Protocol": protocol,
                "Model_A": left,
                "Model_B": right,
                "Disagreement_rate": float(np.mean(pred_left != pred_right)),
                "Yules_Q": compute_yules_q(pred_left, pred_right, reference.y_true),
                "Cohens_kappa": cohen_kappa_score(pred_left, pred_right),
            })
        prediction_matrix = np.stack([base[name].predictions for name in names], axis=0)
        summaries.append({
            "Protocol": protocol,
            "Global_disagreement_rate": float(np.mean(np.ptp(prediction_matrix, axis=0) != 0)),
            "Probability_ambiguity": compute_prob_ambiguity(
                [base[name].probabilities for name in names]
            ),
        })
    frame = pd.DataFrame(rows)
    output = resolve_verification_output_dir(results_root, run_id, save_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "diversity_benchmark.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "diversity_summary.csv", index=False)
    with (output / "diversity_benchmark.md").open("w", encoding="utf-8") as handle:
        handle.write("# Base-model Diversity Replay\n\n")
        handle.write(frame.to_markdown(index=False))
        handle.write("\n\n")
        handle.write(pd.DataFrame(summaries).to_markdown(index=False))
        handle.write("\n")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay diversity metrics")
    parser.add_argument("--results-root", default="RESULTS")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    evaluate_diversity(args.results_root, args.save_dir, run_id=args.run_id)
