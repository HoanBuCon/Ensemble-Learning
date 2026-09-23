"""
Scientific Verification, Calibration & Explainability Package
=============================================================
"""

from .common_utils import load_protocol_predictions, protocol_root
from .eval_calibration import evaluate_all_calibration
from .eval_diversity_ambiguity import evaluate_diversity
from .eval_mcnemar_test import run_mcnemar_analysis
from .eval_advanced_metrics import evaluate_advanced_metrics
from .eval_latency_throughput import run_hardware_benchmark
from .eval_tsne import run_tsne_analysis
from .verify_all_metrics import run_full_scientific_verification

__all__ = [
    "load_protocol_predictions",
    "protocol_root",
    "evaluate_all_calibration",
    "evaluate_diversity",
    "run_mcnemar_analysis",
    "evaluate_advanced_metrics",
    "run_hardware_benchmark",
    "run_tsne_analysis",
    "run_full_scientific_verification",
]
