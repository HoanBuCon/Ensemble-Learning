"""
Unified Master CLI Controller
=============================

Master CLI entrypoint for controlling all pipeline tasks: training single models,
evaluating checkpoints, running multi-model benchmarks, performing ensemble evaluations,
and generating comparison reports & plots.

CLI Commands::

    python main.py all-in-one
    python main.py train configs/resnet50.yaml
    python main.py evaluate configs/resnet50.yaml --split test
    python main.py benchmark --mode auto
    python main.py ensemble --mode val
    python main.py report
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional

# Ensure project root directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.train import train as run_train
from scripts.evaluate import evaluate as run_evaluate
from scripts.run_experiments import run_experiments
from scripts.run_ensemble_eval import run_ensemble_evaluation
from scripts.generate_comparison import generate_base_comparison_report


def prompt_ensemble_mode(current_mode: Optional[str] = None) -> str:
    """
    Prompt user interactively to select the Ensemble Evaluation Protocol
    if no explicit CLI flag was provided.

    Args:
        current_mode: Explicit mode provided by CLI ('val' or 'oof'), or None.

    Returns:
        Selected mode string ('val' or 'oof').
    """
    if current_mode in ["val", "oof"]:
        return current_mode

    print("\n" + "=" * 80)
    print("                    SELECT ENSEMBLE EVALUATION PROTOCOL")
    print("=" * 80)
    print("  [1] Fast Validation Mode (val) ~30 sec")
    print("      - Fits Stacking meta-learners on cached Validation set predictions (1,568 samples).")
    print("      - Extremely fast, ideal for rapid prototyping & testing code pipeline.\n")
    print("  [2] Full 5-Fold OOF Mode (oof) ~10-13 hrs")
    print("      - Trains 5 folds for each of 4 base models (20 models total).")
    print("      - Fits Stacking meta-learners on full 7,000 Out-of-Fold (OOF) train predictions.")
    print("      - Gold-standard paper quality with zero data leakage for academic thesis.")
    print("-" * 80)

    try:
        choice = input("Select Ensemble Protocol [1/2] (default: 1): ").strip()
        return "oof" if choice == "2" else "val"
    except (KeyboardInterrupt, EOFError):
        print("\nDefaulting to Fast Validation Mode (val).")
        return "val"


def resolve_config_paths(configs: Optional[List[str]]) -> List[str]:
    """Resolve list of YAML config files or default to all configs/*.yaml."""
    paths = configs if configs else sorted(glob.glob("configs/*.yaml"))
    if not paths:
        print("Error: No YAML config files found in configs/ directory.")
        sys.exit(1)
    return paths


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for the Master CLI."""
    description = """\
================================================================================
          Tea Leaf Ensemble Learning Pipeline - Master CLI Controller
================================================================================

Framework overview & command guide for training base Computer Vision backbones,
evaluating checkpoints, running multi-model benchmarks, performing Ensemble Learning
(Hard/Soft/Weighted Voting & Stacking Meta-learners), and generating comparison reports.

EXAMPLES & COMMON WORKFLOWS:

  1. Run Full End-to-End Pipeline (Train Base -> Report -> Ensemble):
     $ python main.py all-in-one                       # Interactive prompt for val vs oof mode
     $ python main.py all-in-one --ensemble-mode val   # Fast validation mode (~30s)
     $ python main.py all-in-one --ensemble-mode oof   # Full 5-Fold OOF mode (~10-13 hrs)

  2. Train & Evaluate ALL Base Models Only (No Ensemble):
     $ python main.py train-all                 # Default: auto-detect & skip completed models
     $ python main.py train-all --mode scratch  # Re-train all base models from scratch (Epoch 1)
     $ python main.py train-all --mode resume   # Resume training from last saved checkpoint

  3. Train a Single Base Model:
     $ python main.py train configs/resnet50.yaml
     $ python main.py train configs/resnet50.yaml --resume

  4. Evaluate a Trained Model Checkpoint:
     $ python main.py evaluate configs/resnet50.yaml                # Evaluate best_model.pth on Test (Default)
     $ python main.py evaluate configs/resnet50.yaml --split val    # Evaluate best_model.pth on Val

  5. Perform Ensemble Evaluation Only (Voting & Stacking):
     $ python main.py ensemble                          # Interactive prompt for val vs oof mode
     $ python main.py ensemble --mode val               # Fast Validation Mode (~30s)
     $ python main.py ensemble --mode oof               # Full 5-Fold OOF Mode (~10-13 hrs)

  6. Re-generate Comparison Tables & Plots Only:
     $ python main.py report

AUTOMATIC EVALUATION NOTE:
  Training commands ('train', 'train-all', 'all-in-one') automatically evaluate the best model
  checkpoint upon completion and cache probability predictions (.npy) required for ensembling.

ENSEMBLE PROTOCOLS EXPLAINED:
  - Fast Validation Mode ('val') ~30 sec:
    Fits Stacking meta-learners on cached Validation set predictions (1,568 samples).
    Ideal for rapid prototyping, pipeline verification, and fast experiment loops.

  - Full 5-Fold OOF Mode ('oof') ~10-13 hrs:
    Trains 5 folds per base backbone (20 models total) and builds Out-of-Fold predictions.
    Fits Stacking meta-learners on 7,000 OOF samples with zero data leakage.
    Gold-standard quality required for academic thesis and paper publication.
"""

    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    # 1. Full End-to-End Pipeline Subcommand
    pipe_parser = subparsers.add_parser(
        "all-in-one",
        aliases=["pipeline", "full-pipeline"],
        help="Run full end-to-end pipeline: Train all base models -> Base report -> Ensemble evaluation",
    )
    pipe_parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="List of YAML config files (default: all configs/*.yaml)",
    )
    pipe_parser.add_argument(
        "--train-mode",
        choices=["scratch", "resume", "auto"],
        default="auto",
        help="Training mode: 'scratch', 'resume', or 'auto'",
    )
    pipe_parser.add_argument(
        "--ensemble-mode",
        choices=["val", "oof"],
        default=None,
        help="Ensemble protocol: 'val' or 'oof'. Prompts interactively if omitted.",
    )
    pipe_parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Output directory (default: 'outputs')",
    )

    # 2. Train All Base Models Subcommand (No Ensemble)
    bench_parser = subparsers.add_parser(
        "train-all",
        aliases=["benchmark", "train-base"],
        help="Train & evaluate all base models & generate baseline comparison report (No ensemble)",
    )
    bench_parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="List of YAML config files (default: all configs/*.yaml)",
    )
    bench_parser.add_argument(
        "--mode",
        choices=["scratch", "resume", "auto"],
        default="auto",
        help="Training mode: 'scratch', 'resume', or 'auto'",
    )

    # 3. Train Single Subcommand
    train_parser = subparsers.add_parser(
        "train",
        aliases=["train-single"],
        help="Train a single model architecture (e.g. python main.py train configs/resnet50.yaml)",
    )
    train_parser.add_argument("config", help="Path to YAML config file (e.g. configs/resnet50.yaml)")
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last_model.pth if available",
    )

    # 4. Evaluate Subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model checkpoint")
    eval_parser.add_argument("config", help="Path to YAML config file (e.g. configs/resnet50.yaml)")
    eval_parser.add_argument("--checkpoint", default=None, help="Path to specific checkpoint file")
    eval_parser.add_argument("--split", default="test", choices=["test", "val"], help="Dataset split to evaluate")

    # 5. Ensemble Subcommand
    ens_parser = subparsers.add_parser("ensemble", help="Run ensemble evaluation on base model predictions (Voting & Stacking)")
    ens_parser.add_argument(
        "--mode",
        choices=["val", "oof"],
        default=None,
        help="Ensemble mode: 'val' or 'oof'. Prompts interactively if omitted.",
    )
    ens_parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Output directory (default: 'outputs')",
    )

    # 6. Report Subcommand
    report_parser = subparsers.add_parser("report", help="Re-generate comparison Markdown/CSV reports & bar charts")
    report_parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Output directory (default: 'outputs')",
    )
    report_parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional list of YAML config files to inspect",
    )

    return parser


def prompt_base_models_training(outputs_dir: str = "outputs") -> str:
    """
    Check if valid base model outputs exist in outputs_dir, and interactively
    prompt the user to choose between using existing outputs or re-training from scratch.

    Returns:
        Training mode string: 'auto' (use existing outputs/skip) or 'scratch' (re-train from scratch).
    """
    from scripts.run_experiments import inspect_model_status
    config_paths = sorted(glob.glob("configs/*.yaml"))
    if not config_paths:
        return "auto"

    statuses = [inspect_model_status(p) for p in config_paths]
    completed_models = [s for s in statuses if s["status"] == "COMPLETED"]

    if completed_models:
        print("\n" + "=" * 80)
        print("           EXISTING TRAINED BASE MODEL OUTPUTS DETECTED")
        print("=" * 80)
        print("  Found completed trained base model outputs in 'outputs/':")
        for m in completed_models:
            acc = m.get("best_val_accuracy", 0.0) * 100 if m.get("best_val_accuracy", 0.0) <= 1.0 else m.get("best_val_accuracy", 0.0)
            print(f"  - {m['model_name']} ({m['config_path']}): Val Acc: {acc:.2f}%")
        print("\n  Select Base Model Training Action:")
        print("  [1] Use existing outputs & proceed directly to Ensemble Evaluation (Fast) [Default]")
        print("  [2] Re-train all 4 base models from scratch (Epoch 1)")
        print("-" * 80)
        try:
            choice = input("Select option [1/2] (default: 1): ").strip()
            if choice == "2":
                return "scratch"
            else:
                return "auto"
        except (KeyboardInterrupt, EOFError):
            return "auto"

    return "auto"


def main() -> None:
    """Main CLI execution handler."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command in ["train", "train-single"]:
        run_train(args.config, resume=args.resume)

    elif args.command == "evaluate":
        run_evaluate(args.config, checkpoint_path=args.checkpoint, split=args.split)

    elif args.command in ["train-all", "benchmark", "train-base"]:
        config_paths = resolve_config_paths(args.configs)
        run_experiments(config_paths, mode=args.mode)

    elif args.command == "ensemble":
        mode = prompt_ensemble_mode(args.mode)
        run_ensemble_evaluation(mode=mode, outputs_dir=args.outputs_dir)

    elif args.command == "report":
        generate_base_comparison_report(outputs_dir=args.outputs_dir, config_paths=args.configs)

    elif args.command in ["all-in-one", "pipeline", "full-pipeline"]:
        config_paths = resolve_config_paths(args.configs)
        train_mode = prompt_base_models_training(outputs_dir=args.outputs_dir)
        ensemble_mode = prompt_ensemble_mode(args.ensemble_mode)

        print("\n" + "=" * 80)
        print(f"  STEP 1/2: BASE MODELS BENCHMARK & REPORT GENERATION (MODE: {train_mode.upper()})")
        print("=" * 80 + "\n")
        run_experiments(config_paths, mode=train_mode)

        print("\n" + "=" * 80)
        print(f"  STEP 2/2: ENSEMBLE EVALUATION PIPELINE (MODE: {ensemble_mode.upper()})")
        print("=" * 80 + "\n")
        run_ensemble_evaluation(mode=ensemble_mode, outputs_dir=args.outputs_dir)


if __name__ == "__main__":
    main()
