"""Fail-closed orchestration for FINAL_V2 scientific verification replay."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.verification.eval_advanced_metrics import evaluate_advanced_metrics
from scripts.verification.eval_calibration import evaluate_all_calibration
from scripts.verification.eval_diversity_ambiguity import evaluate_diversity
from scripts.verification.eval_latency_throughput import run_hardware_benchmark
from scripts.verification.eval_mcnemar_test import run_mcnemar_analysis
from scripts.verification.eval_tsne import run_tsne_analysis
from src.utils.provenance import verify_dataset_snapshot, write_json


def run_full_scientific_verification(
    results_root: str = "RESULTS/FINAL_V2",
    save_dir: Optional[str] = None,
    skip_tsne: bool = False,
) -> None:
    """Replay final predictions and fitted artifacts; any missing input is fatal."""
    verify_dataset_snapshot()
    output = Path(save_dir) if save_dir else Path(results_root) / "verification"
    output.mkdir(parents=True, exist_ok=True)

    calibration = evaluate_all_calibration(results_root, str(output))
    diversity = evaluate_diversity(results_root, str(output))
    mcnemar, mcnemar_summary = run_mcnemar_analysis(results_root, str(output))
    advanced = evaluate_advanced_metrics(results_root, str(output))
    latency_single = run_hardware_benchmark(
        "single_split", results_root, save_dir=str(output)
    )
    latency_oof = run_hardware_benchmark("oof", results_root, save_dir=str(output))
    tsne_outputs = []
    if not skip_tsne:
        for protocol in ("single_split", "oof"):
            run_tsne_analysis(
                "probability_vector", protocol, results_root, str(output)
            )
            tsne_outputs.append(f"tsne_{protocol}_probability_vector_metadata.json")

    report_path = output / "FULL_SCIENTIFIC_VERIFICATION_REPORT.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# FINAL_V2 Scientific Verification Replay\n\n")
        handle.write(
            "All tables below were computed from saved, identity-checked prediction "
            "artifacts. Verification did not fit or reconstruct an ensemble.\n\n"
        )
        for title, frame in (
            ("Advanced metrics", advanced),
            ("Calibration", calibration),
            ("Base-model diversity", diversity),
            ("Cross-protocol McNemar", mcnemar),
            ("Single-split latency", latency_single),
            ("OOF latency", latency_oof),
        ):
            handle.write(f"## {title}\n\n")
            handle.write(frame.to_markdown(index=False))
            handle.write("\n\n")
        handle.write(
            "McNemar decisions use the stored unrounded exact p-values. "
            "`FAIL_TO_REJECT` is not interpreted as equivalence. "
            "Cohen's h is labeled as a marginal-accuracy quantity.\n"
        )
    write_json(
        output / "verification_manifest.json",
        {
            "results_root": results_root,
            "report": str(report_path),
            "mcnemar_family_alpha": mcnemar_summary["family_alpha"],
            "mcnemar_family_size": mcnemar_summary["bonferroni_family_size"],
            "tsne_outputs": tsne_outputs,
            "status": "COMPLETE",
            "ensemble_refit_performed": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay FINAL_V2 verification")
    parser.add_argument("--results-root", default="RESULTS/FINAL_V2")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--skip-tsne", action="store_true")
    args = parser.parse_args()
    run_full_scientific_verification(
        args.results_root, args.save_dir, args.skip_tsne
    )
