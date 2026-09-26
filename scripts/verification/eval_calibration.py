"""Replay saved canonical probabilities for calibration metrics; never refit."""

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

from scripts.verification.common_utils import (
    load_protocol_predictions,
    resolve_verification_output_dir,
)


def calc_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    confidence = np.max(probs, axis=1)
    correct = np.argmax(probs, axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        lower_inclusive = index == 0
        mask = (
            (confidence >= edges[index]) if lower_inclusive
            else (confidence > edges[index])
        ) & (confidence <= edges[index + 1])
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def calc_brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def calc_nll(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-15) -> float:
    selected = probs[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(np.clip(selected, eps, 1.0))))


def evaluate_all_calibration(
    results_root: str = "RESULTS",
    save_dir: Optional[str] = None,
    run_id: str = "",
) -> pd.DataFrame:
    rows = []
    for protocol in ("single_split", "oof"):
        base, ensembles = load_protocol_predictions(protocol, results_root=results_root, run_id=run_id)
        for method, artifact in {**base, **ensembles}.items():
            rows.append({
                "Protocol": protocol,
                "Method": method,
                "ECE_15_bins": calc_ece(artifact.probabilities, artifact.y_true),
                "Brier_score": calc_brier(artifact.probabilities, artifact.y_true),
                "NLL": calc_nll(artifact.probabilities, artifact.y_true),
            })
    frame = pd.DataFrame(rows)
    output = resolve_verification_output_dir(results_root, run_id, save_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "calibration_benchmark.csv", index=False)
    with (output / "calibration_benchmark.md").open("w", encoding="utf-8") as handle:
        handle.write("# Probability Calibration Replay\n\n")
        handle.write(frame.to_markdown(index=False))
        handle.write("\n")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay calibration metrics")
    parser.add_argument("--results-root", default="RESULTS")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    evaluate_all_calibration(args.results_root, args.save_dir, run_id=args.run_id)
