"""
5-Fold Cross-Validation Benchmark Runner
========================================

Executes Stratified 5-Fold Cross Validation for one or all backbone models,
saves per-fold checkpoints, metrics, and visualization curves, and outputs
academic benchmark tables (Mean ± Std).

CLI::

    python scripts/run_kfold.py --configs configs/resnet50.yaml --folds 5
    python scripts/run_kfold.py --folds 5
    python main.py kfold configs/resnet50.yaml
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ensemble.oof import OOFGenerator
from src.utils.config import load_config


def run_kfold_experiment(
    config_path: str,
    n_splits: int = 5,
    split_seed: int = 42,
    output_dir: Optional[str] = None,
    force_retrain: bool = False,
) -> None:
    """Run 5-Fold Cross Validation for a single configuration."""
    cfg = load_config(config_path)
    model_name = cfg.model.name

    print(f"\n{'='*75}")
    print(f"  RUNNING {n_splits}-FOLD CROSS VALIDATION: {model_name.upper()} ({config_path})")
    print(f"{'='*75}\n")

    out_d = output_dir or os.path.join("RESULTS", "OOF_TRAINING", "outputs", model_name, "kfold")

    generator = OOFGenerator(
        config=cfg,
        n_splits=n_splits,
        split_seed=split_seed,
        output_dir=out_d,
        force_retrain=force_retrain,
    )
    generator.generate()


def run_all_kfold_experiments(
    config_paths: Optional[List[str]] = None,
    n_splits: int = 5,
    split_seed: int = 42,
    force_retrain: bool = False,
    eval_ensemble: bool = True,
    outputs_dir: str = "RESULTS/OOF_TRAINING/outputs",
) -> None:
    """Run 5-Fold Cross Validation sequentially across all configured backbone models and evaluate Meta-Learners."""
    if not config_paths:
        config_paths = sorted(glob.glob("configs/*.yaml"))
        config_paths = [c for c in config_paths if not c.endswith("dataset.yaml")]

    print(f"\n{'='*75}")
    print(f"  EXECUTING {n_splits}-FOLD CROSS VALIDATION ACROSS {len(config_paths)} BACKBONES")
    print(f"{'='*75}\n")

    for i, cfg_path in enumerate(config_paths, 1):
        print(f"\n>>> Model {i}/{len(config_paths)}: {cfg_path}")
        cfg = load_config(cfg_path)
        m_name = cfg.model.name
        m_out_d = os.path.join(outputs_dir, m_name, "kfold")
        run_kfold_experiment(
            cfg_path,
            n_splits=n_splits,
            split_seed=split_seed,
            output_dir=m_out_d,
            force_retrain=force_retrain,
        )

    if eval_ensemble:
        print(f"\n{'='*75}")
        print("  STEP 2: TRAINING META-LEARNERS & ENSEMBLE BENCHMARK (OOF MODE)")
        print(f"{'='*75}\n")
        try:
            from scripts.run_ensemble_eval import run_ensemble_evaluation
            run_ensemble_evaluation(mode="oof", outputs_dir=outputs_dir)
        except Exception as e:
            print(f"Warning: Automatic ensemble evaluation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-Fold Cross Validation Runner")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="List of YAML config files (default: all configs/*.yaml)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified K-Fold splitting (default: 42)",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retrain all folds even if cached",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Skip automatic Meta-Learner training after K-Fold completion",
    )
    parser.add_argument(
        "--outputs-dir",
        default="RESULTS/OOF_TRAINING/outputs",
        help="Directory to save K-Fold outputs (default: 'RESULTS/OOF_TRAINING/outputs')",
    )
    args = parser.parse_args()

    if args.configs:
        valid_configs = [c for c in args.configs if os.path.basename(c) != "dataset.yaml"]
        for c in valid_configs:
            cfg = load_config(c)
            m_name = cfg.model.name
            m_out_d = os.path.join(args.outputs_dir, m_name, "kfold")
            run_kfold_experiment(
                c,
                n_splits=args.folds,
                split_seed=args.seed,
                output_dir=m_out_d,
                force_retrain=args.force_retrain,
            )
        if not args.no_ensemble:
            from scripts.run_ensemble_eval import run_ensemble_evaluation
            run_ensemble_evaluation(mode="oof", outputs_dir=args.outputs_dir)
    else:
        run_all_kfold_experiments(
            config_paths=None,
            n_splits=args.folds,
            split_seed=args.seed,
            force_retrain=args.force_retrain,
            eval_ensemble=not args.no_ensemble,
            outputs_dir=args.outputs_dir,
        )
