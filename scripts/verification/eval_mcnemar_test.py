"""Paired McNemar replay over saved canonical ensemble predictions."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from scripts.verification.common_utils import (
    ENSEMBLE_METHODS,
    load_protocol_predictions,
    verification_output_dir,
)
from src.utils.provenance import write_json


def compute_mcnemar_metrics(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    y_true: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Compute exact McNemar, Edwards correction, paired RD, and labeled effects."""
    if not (len(pred_a) == len(pred_b) == len(y_true)):
        raise ValueError("Paired McNemar inputs must have identical row counts")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    n11 = int(np.sum(correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n00 = int(np.sum(~correct_a & ~correct_b))
    n_samples = len(y_true)
    discordant = n10 + n01

    p_a = float(correct_a.mean())
    p_b = float(correct_b.mean())
    risk_difference = p_a - p_b
    variance = (
        n10 + n01 - ((n10 - n01) ** 2) / n_samples
    ) / (n_samples ** 2)
    standard_error = math.sqrt(max(variance, 0.0))
    z_critical = float(norm.ppf(1.0 - alpha / 2.0))
    ci_low = risk_difference - z_critical * standard_error
    ci_high = risk_difference + z_critical * standard_error

    marginal_cohens_h = (
        2.0 * math.asin(math.sqrt(p_a))
        - 2.0 * math.asin(math.sqrt(p_b))
    )

    if discordant:
        exact_p_value = float(
            binomtest(n10, discordant, p=0.5, alternative="two-sided").pvalue
        )
        edwards_chi_square = float(
            max(0.0, abs(n10 - n01) - 1.0) ** 2 / discordant
        )
        discordant_rate_a = n10 / discordant
        null_se = math.sqrt(0.25 / discordant)
        alternative_se = math.sqrt(
            max(discordant_rate_a * (1.0 - discordant_rate_a) / discordant, 1e-12)
        )
        z_beta = (
            abs(discordant_rate_a - 0.5) - z_critical * null_se
        ) / alternative_se
        supplementary_posthoc_power = float(norm.cdf(z_beta))
    else:
        exact_p_value = 1.0
        edwards_chi_square = 0.0
        supplementary_posthoc_power = 0.0

    return {
        "alpha": alpha,
        "n11": n11,
        "n10_a_correct_b_wrong": n10,
        "n01_a_wrong_b_correct": n01,
        "n00": n00,
        "discordant": discordant,
        "accuracy_a": p_a,
        "accuracy_b": p_b,
        "paired_risk_difference": risk_difference,
        "paired_risk_difference_variance": variance,
        "paired_risk_difference_ci_low": ci_low,
        "paired_risk_difference_ci_high": ci_high,
        "marginal_accuracy_cohens_h": marginal_cohens_h,
        "edwards_corrected_chi_square": edwards_chi_square,
        "exact_binomial_p_value": exact_p_value,
        "decision": "REJECT" if exact_p_value <= alpha else "FAIL_TO_REJECT",
        "supplementary_posthoc_power": supplementary_posthoc_power,
        "power_used_for_decision": False,
    }


def run_mcnemar_analysis(
    results_root: str = "RESULTS/FINAL_V2",
    save_dir: Optional[str] = None,
    family_alpha: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    _, single = load_protocol_predictions("single_split", results_root=results_root)
    _, oof = load_protocol_predictions("oof", results_root=results_root)
    family_size = len(ENSEMBLE_METHODS)
    adjusted_alpha = family_alpha / family_size
    rows = []
    details: Dict[str, object] = {}
    for method in ENSEMBLE_METHODS:
        single_artifact = single[method]
        oof_artifact = oof[method]
        if not np.array_equal(single_artifact.sample_ids, oof_artifact.sample_ids):
            raise ValueError(f"Cross-protocol sample identity mismatch for {method}")
        if not np.array_equal(single_artifact.y_true, oof_artifact.y_true):
            raise ValueError(f"Cross-protocol y_true mismatch for {method}")
        metrics = compute_mcnemar_metrics(
            single_artifact.predictions,
            oof_artifact.predictions,
            single_artifact.y_true,
            alpha=adjusted_alpha,
        )
        details[method] = metrics
        rows.append({"method": method, **metrics})

    frame = pd.DataFrame(rows)
    summary: Dict[str, object] = {
        "family_alpha": family_alpha,
        "bonferroni_family_size": family_size,
        "bonferroni_adjusted_alpha": adjusted_alpha,
        "cohens_h_label": "marginal-accuracy Cohen's h",
        "posthoc_power_role": "supplementary_only_not_used_for_decision",
        "comparisons": details,
    }
    output = Path(save_dir) if save_dir else verification_output_dir(results_root)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "mcnemar_cross_protocol_matrix.csv", index=False)
    write_json(output / "mcnemar_test_results.json", summary)
    with (output / "mcnemar_cross_protocol_matrix.md").open("w", encoding="utf-8") as handle:
        handle.write("# Cross-protocol McNemar Replay\n\n")
        handle.write(
            f"Bonferroni family: {family_size}; family alpha: {family_alpha}; "
            f"adjusted alpha: {adjusted_alpha}. Decisions are `REJECT` or "
            "`FAIL_TO_REJECT`; failure to reject is not an equivalence claim.\n\n"
        )
        handle.write(frame.to_markdown(index=False))
        handle.write("\n")
    return frame, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay paired McNemar statistics")
    parser.add_argument("--results-root", default="RESULTS/FINAL_V2")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    run_mcnemar_analysis(args.results_root, args.save_dir, args.alpha)
