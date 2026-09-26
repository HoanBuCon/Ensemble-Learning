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
from scripts.generate_all_plots import generate_all_plots
from scripts.run_kfold import run_kfold_experiment, run_all_kfold_experiments
from scripts.verification.verify_all_metrics import run_full_scientific_verification as run_verification_suite
from scripts.verification.eval_calibration import evaluate_all_calibration
from scripts.verification.eval_diversity_ambiguity import evaluate_diversity
from scripts.verification.eval_mcnemar_test import run_mcnemar_analysis
from scripts.verification.eval_advanced_metrics import evaluate_advanced_metrics as evaluate_advanced_benchmark
from scripts.verification.eval_latency_throughput import run_hardware_benchmark as evaluate_hardware_latency
from scripts.verification.eval_tsne import run_tsne_analysis as evaluate_tsne_clustering
from src.utils.run_identity import resolve_backbone_config_paths


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
    if current_mode is not None:
        raise ValueError(f"Unknown ensemble mode: {current_mode}")

    print("\n" + "=" * 80)
    print("                    SELECT ENSEMBLE EVALUATION PROTOCOL")
    print("=" * 80)
    print("  [1] Fast Validation Mode (val) ~30 sec")
    print("      - Fits Stacking meta-learners on cached Validation set predictions (1,568 samples).")
    print("      - Extremely fast, ideal for rapid prototyping & testing code pipeline.\n")
    print("  [2] Full 5-Fold OOF Mode (oof) ~10-13 hrs")
    print("      - Trains 5 folds for each of 4 base models (20 models total).")
    print("      - Fits Stacking meta-learners on full 7,000 Out-of-Fold (OOF) train predictions.")
    print("      - Fixed external validation selects checkpoints; outer folds are inference-only.")
    print("-" * 80)

    try:
        choice = input("Select Ensemble Protocol [1/2] (default: 1): ").strip()
        return "oof" if choice == "2" else "val"
    except (KeyboardInterrupt, EOFError):
        print("\nDefaulting to Fast Validation Mode (val).")
        return "val"


def resolve_config_paths(configs: Optional[List[str]]) -> List[str]:
    """Resolve only the four registered executable backbone configs."""
    return resolve_backbone_config_paths(configs)


def canonical_protocol_from_mode(mode: str) -> str:
    """Map the CLI alias onto the only persisted protocol identities."""
    if mode == "val":
        return "single_split"
    if mode == "oof":
        return "oof"
    raise ValueError(f"Unknown ensemble mode: {mode}")


def dispatch_verification_task(
    task: str,
    *,
    results_root: str,
    run_id: str,
    save_dir: Optional[str],
    protocol: Optional[str] = None,
    skip_tsne: bool = False,
):
    """Current verification API contract; safe to smoke-test with mocked callees."""
    if task == "calibration":
        return evaluate_all_calibration(results_root, save_dir, run_id=run_id)
    if task == "diversity":
        return evaluate_diversity(results_root, save_dir, run_id=run_id)
    if task == "mcnemar":
        return run_mcnemar_analysis(results_root, save_dir, run_id=run_id)
    if task == "advanced":
        return evaluate_advanced_benchmark(results_root, save_dir, run_id=run_id)
    if task == "latency":
        if protocol not in {"single_split", "oof"}:
            raise ValueError("--protocol is required for latency")
        return evaluate_hardware_latency(
            protocol, results_root, save_dir=save_dir, run_id=run_id
        )
    if task == "tsne":
        if protocol not in {"single_split", "oof"}:
            raise ValueError("--protocol is required for t-SNE")
        return evaluate_tsne_clustering(
            "probability_vector", protocol, results_root, save_dir, run_id=run_id
        )
    if task == "all":
        return run_verification_suite(
            results_root, save_dir, skip_tsne, run_id=run_id
        )
    raise ValueError(f"Unknown verification task: {task}")


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
    Fits stacking meta-learners on identity-checked validation predictions.
    Ideal for rapid prototyping, pipeline verification, and fast experiment loops.

  - Full 5-Fold OOF Mode ('oof') ~10-13 hrs:
    Trains 5 folds per base backbone (20 models total) and builds Out-of-Fold predictions.
    Fits stacking meta-learners on cross-fitted training predictions.
    Fixed external validation selects checkpoints; outer folds are inference-only.
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
        help="Ensemble protocol: 'val' (Single-Split) or 'oof' (5-Fold CV). Prompts interactively if omitted.",
    )
    pipe_parser.add_argument("--run-id", required=True)
    pipe_parser.add_argument("--results-root", default="RESULTS")

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

    bench_parser.add_argument("--run-id", required=True)
    bench_parser.add_argument("--results-root", default="RESULTS")

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
    train_parser.add_argument("--run-id", required=True)
    train_parser.add_argument("--results-root", default="RESULTS")

    # 4. Evaluate Subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model checkpoint")
    eval_parser.add_argument("config", help="Path to YAML config file (e.g. configs/resnet50.yaml)")
    eval_parser.add_argument("--checkpoint", default=None, help="Path to specific checkpoint file")
    eval_parser.add_argument("--split", default="test", choices=["test", "val"], help="Dataset split to evaluate")
    eval_parser.add_argument("--run-id", required=True)
    eval_parser.add_argument("--results-root", default="RESULTS")

    # 5. Ensemble Subcommand
    ens_parser = subparsers.add_parser("ensemble", help="Run ensemble evaluation on base model predictions (Voting & Stacking)")
    ens_parser.add_argument(
        "--mode",
        choices=["val", "oof"],
        default=None,
        help="Ensemble mode: 'val' (Single-Split) or 'oof' (5-Fold CV). Prompts interactively if omitted.",
    )
    ens_parser.add_argument("--run-id", required=True)
    ens_parser.add_argument("--results-root", default="RESULTS")

    # 6. Report Subcommand
    report_parser = subparsers.add_parser("report", help="Re-generate comparison Markdown/CSV reports & bar charts")
    report_parser.add_argument("--results-root", default="RESULTS")
    report_parser.add_argument("--run-id", required=True)

    # 7. Plot Subcommand (Master Plot Generator)
    plot_parser = subparsers.add_parser("plot", aliases=["plots", "visualize"], help="Generate complete visualization suite (Training, ROC, PR, CM, Heatmaps, Radar)")
    plot_parser.add_argument("--results-root", default="RESULTS")
    plot_parser.add_argument("--run-id", required=True)

    # 8. K-Fold Subcommand (5-Fold Cross Validation for Single Model)
    kfold_parser = subparsers.add_parser("kfold", help="Run 5-Fold Cross Validation for a single backbone")
    kfold_parser.add_argument("config", help="Path to YAML config file (e.g. configs/resnet50.yaml)")
    kfold_parser.add_argument("--run-id", required=True)
    kfold_parser.add_argument("--results-root", default="RESULTS")
    kfold_parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    kfold_parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting (default: 42)")
    kfold_parser.add_argument("--force-retrain", action="store_true", help="Force retrain all folds")

    # 9. K-Fold All Subcommand (5-Fold Cross Validation for All Models)
    kfold_all_parser = subparsers.add_parser("kfold-all", help="Run 5-Fold Cross Validation for ALL backbone models")
    kfold_all_parser.add_argument("--run-id", required=True)
    kfold_all_parser.add_argument("--results-root", default="RESULTS")
    kfold_all_parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    kfold_all_parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting (default: 42)")
    kfold_all_parser.add_argument("--force-retrain", action="store_true", help="Force retrain all folds")

    # 10. Serve Subcommand (FastAPI Web Server)
    serve_parser = subparsers.add_parser(
        "serve",
        aliases=["api", "server"],
        help="Start FastAPI REST API & Web Dashboard server on localhost:8000",
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # 11. Scientific Verification Subcommand
    verify_parser = subparsers.add_parser(
        "verify",
        aliases=["benchmark-verify", "audit"],
        help="Run Scientific Verification Suite (ECE, Brier, NLL, Ambiguity, McNemar test)",
    )
    verify_parser.add_argument(
        "--task",
        choices=["all", "calibration", "diversity", "mcnemar", "advanced", "latency", "tsne"],
        default="all",
        help="Verification task: 'all', 'calibration', 'diversity', or 'mcnemar' (default: 'all')",
    )
    verify_parser.add_argument("--results-root", default="RESULTS")
    verify_parser.add_argument("--run-id", required=True)
    verify_parser.add_argument("--protocol", choices=["single_split", "oof"], default=None)
    verify_parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional explicit verification output directory inside the selected run",
    )
    verify_parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip t-SNE plot generation in the full suite",
    )

    return parser


def prompt_base_models_training(
    run_id: str, results_root: str = "RESULTS"
) -> str:
    """
    Check if valid base model outputs exist in outputs_dir, and interactively
    prompt the user to choose between using existing outputs or re-training from scratch.

    Returns:
        Training mode string: 'auto' (use existing outputs/skip) or 'scratch' (re-train from scratch).
    """
    from scripts.run_experiments import inspect_model_status
    config_paths = resolve_config_paths(None)
    if not config_paths:
        return "auto"

    statuses = [
        inspect_model_status(p, run_id=run_id, results_root=results_root)
        for p in config_paths
    ]
    completed_models = [s for s in statuses if s["status"] == "COMPLETED"]

    if completed_models:
        print("\n" + "=" * 80)
        print("           EXISTING TRAINED BASE MODEL OUTPUTS DETECTED")
        print("=" * 80)
        print("  Found completed trained base model outputs:")
        for m in completed_models:
            acc = m.get("accepted_checkpoint_val_accuracy")
            display = "N/A" if acc is None else f"{float(acc):.2f}%"
            print(f"  - {m['model_name']} ({m['config_path']}): Val Acc: {display}")
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


def select_menu_option(options: List[str], header_title: str) -> int:
    """
    Interactive menu selector allowing navigation via:
    - Arrow Keys (Up/Down) + Enter
    - Direct Number Keys (1, 2, 3...)
    - Ctrl+C to cancel
    """
    import os
    import sys

    if os.name == "nt":
        try:
            import msvcrt
            selected_idx = 0
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                print("\n" + "=" * 80)
                print(f"  {header_title}")
                print("=" * 80)
                print("  Navigation: Use [↑/↓] Arrow Keys + ENTER, or press Number Keys [1-{}]:\n".format(len(options)))

                for i, opt in enumerate(options):
                    if i == selected_idx:
                        print(f"  👉 \033[1;36m[{i+1}] {opt}\033[0m")
                    else:
                        print(f"     [{i+1}] {opt}")

                print("\n" + "-" * 80)

                ch = msvcrt.getch()
                if ch in (b"\x00", b"\xe0"):
                    arrow = msvcrt.getch()
                    if arrow == b"H":  # Up key
                        selected_idx = (selected_idx - 1) % len(options)
                    elif arrow == b"P":  # Down key
                        selected_idx = (selected_idx + 1) % len(options)
                elif ch in (b"\r", b"\n"):  # Enter key
                    return selected_idx
                elif ch.isdigit():
                    num = int(ch.decode("ascii"))
                    if 1 <= num <= len(options):
                        return num - 1
                elif ch == b"\x03":  # Ctrl+C
                    raise KeyboardInterrupt
        except Exception:
            pass

    # Fallback standard input prompt for non-interactive environments
    print("\n" + "=" * 80)
    print(f"  {header_title}")
    print("=" * 80)
    for i, opt in enumerate(options):
        print(f"  [{i+1}] {opt}")
    print("-" * 80)

    while True:
        try:
            choice = input(f"Select option [1-{len(options)}]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return int(choice) - 1
        except (KeyboardInterrupt, EOFError):
            raise KeyboardInterrupt


def run_interactive_cli_menu() -> None:
    """Run interactive menu loop for Master CLI."""
    run_id = input("Scientific run_id (required): ").strip()
    from src.utils.run_identity import validate_run_id
    run_id = validate_run_id(run_id)
    results_root = "RESULTS"
    menu_options = [
        "🚀 Run Full End-to-End Pipeline (Train Base -> Report -> Ensemble)",
        "🏋️ Train a Single Backbone Model (ResNet-50 / DenseNet-121 / EfficientNet-B0 / Swin-Tiny)",
        "⚡ Run Ensemble Benchmark Evaluation (Voting & Stacking Meta-Learners)",
        "🔁 Run 5-Fold Cross Validation (Single Backbone or All Backbones)",
        "📈 Generate Complete Diagnostic Visualization Suite (Plots & Heatmaps)",
        "🌐 Start FastAPI Web Server & Visualizer Dashboard (http://127.0.0.1:8000)",
        "🔬 Evaluate Trained Model Checkpoint on Test/Val Split",
        "📊 Re-generate Comparison Tables & Benchmark Reports",
        "🛡️ Run Scientific Verification Suite (ECE, Brier, Ambiguity, McNemar)",
        "❌ Exit CLI",
    ]

    while True:
        try:
            choice_idx = select_menu_option(
                menu_options,
                "TEA LEAF ENSEMBLE PIPELINE - INTERACTIVE MASTER CLI CONTROLLER",
            )

            if choice_idx == 0:
                train_mode = prompt_base_models_training(run_id, results_root)
                ensemble_mode = prompt_ensemble_mode()
                config_paths = resolve_config_paths(None)
                print("\n" + "=" * 80)
                print(f"  STEP 1/3: BASE MODELS BENCHMARK & REPORT GENERATION (MODE: {train_mode.upper()})")
                print("=" * 80 + "\n")
                run_experiments(
                    config_paths, mode=train_mode, run_id=run_id,
                    results_root=results_root,
                )
                print("\n" + "=" * 80)
                print(f"  STEP 2/3: ENSEMBLE EVALUATION PIPELINE (MODE: {ensemble_mode.upper()})")
                print("=" * 80 + "\n")
                run_ensemble_evaluation(
                    protocol=canonical_protocol_from_mode(ensemble_mode),
                    results_root=results_root, run_id=run_id,
                )
                print("\n" + "=" * 80)
                print("  STEP 3/3: SCIENTIFIC VERIFICATION & CALIBRATION SUITE")
                print("=" * 80 + "\n")
                run_verification_suite(results_root, None, False, run_id=run_id)
                input("\nPress ENTER to return to main menu...")

            elif choice_idx == 1:
                configs = resolve_config_paths(None)
                cfg_options = [f"{os.path.basename(c).replace('.yaml','').upper()} ({c})" for c in configs]
                cfg_idx = select_menu_option(cfg_options, "SELECT BACKBONE MODEL TO TRAIN")
                target_cfg = configs[cfg_idx]
                resume_choice = select_menu_option(
                    ["Auto-resume / Scratch mode [Default]", "Force resume from last_model.pth"],
                    f"TRAINING OPTIONS: {target_cfg}",
                )
                run_train(
                    target_cfg, resume=(resume_choice == 1),
                    run_id=run_id, results_root=results_root,
                )
                input("\nPress ENTER to return to main menu...")

            elif choice_idx == 2:
                mode = prompt_ensemble_mode()
                run_ensemble_evaluation(
                    protocol=canonical_protocol_from_mode(mode),
                    results_root=results_root, run_id=run_id,
                )
                input("\nPress ENTER to return to main menu...")

            elif choice_idx == 3:
                configs = resolve_config_paths(None)
                kfold_options = ["Run 5-Fold CV on ALL Backbones"] + [f"Run 5-Fold CV on {os.path.basename(c).replace('.yaml','').upper()}" for c in configs]
                k_idx = select_menu_option(kfold_options, "5-FOLD CROSS VALIDATION MENU")
                if k_idx == 0:
                    run_all_kfold_experiments(
                        config_paths=configs, n_splits=5, run_id=run_id,
                        results_root=results_root,
                    )
                else:
                    target_cfg = configs[k_idx - 1]
                    run_kfold_experiment(
                        target_cfg, n_splits=5, run_id=run_id,
                        results_root=results_root,
                    )
                input("\nPress ENTER to return to main menu...")

            elif choice_idx == 4:
                generate_all_plots(outputs_dir=results_root, run_id=run_id)
                input("\nPress ENTER to return to main menu...")

            elif choice_idx == 5:
                import uvicorn
                print("\nStarting FastAPI Web Server on http://127.0.0.1:8000 ...\n")
                uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)

            elif choice_idx == 6:
                configs = resolve_config_paths(None)
                cfg_options = [f"{os.path.basename(c).replace('.yaml','').upper()} ({c})" for c in configs]
                cfg_idx = select_menu_option(cfg_options, "SELECT MODEL TO EVALUATE")
                target_cfg = configs[cfg_idx]
                split_idx = select_menu_option(["Test Split (Default)", "Validation Split"], "SELECT DATASET SPLIT")
                split_name = "test" if split_idx == 0 else "val"
                run_evaluate(
                    target_cfg, checkpoint_path=None, split=split_name,
                    run_id=run_id, results_root=results_root,
                )
                input("\nPress ENTER to return to main menu...")

            elif choice_idx == 7:
                print("\n" + "=" * 80)
                print("       RE-GENERATE COMPARISON TABLES & BENCHMARK REPORTS")
                print("=" * 80)
                generate_all_plots(outputs_dir=results_root, run_id=run_id)

                input("\nPress ENTER to return to main menu...")

            elif choice_idx == 8:
                print("\n" + "=" * 80)
                print("      RUNNING SCIENTIFIC VERIFICATION & BENCHMARKING SUITE")
                print("=" * 80)
                run_verification_suite(results_root, None, False, run_id=run_id)
                input("\nPress ENTER to return to main menu...")

            elif choice_idx == 9:
                print("\nExiting Interactive Master CLI. Goodbye!\n")
                break

        except KeyboardInterrupt:
            print("\nExiting Interactive Master CLI. Goodbye!\n")
            break


def main() -> None:
    """Main CLI execution handler."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        run_interactive_cli_menu()
        sys.exit(0)

    if args.command in ["serve", "api", "server"]:
        import uvicorn
        print(f"\nStarting FastAPI Web Server on http://{args.host}:{args.port} ...\n")
        uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)

    elif args.command in ["train", "train-single"]:
        run_train(args.config, resume=args.resume, run_id=args.run_id, results_root=args.results_root)

    elif args.command == "evaluate":
        run_evaluate(
            args.config, checkpoint_path=args.checkpoint, split=args.split,
            run_id=args.run_id, results_root=args.results_root,
        )

    elif args.command in ["train-all", "benchmark", "train-base"]:
        config_paths = resolve_config_paths(args.configs)
        run_experiments(
            config_paths, mode=args.mode, run_id=args.run_id,
            results_root=args.results_root,
        )

    elif args.command == "ensemble":
        mode = prompt_ensemble_mode(args.mode)
        run_ensemble_evaluation(
            protocol=canonical_protocol_from_mode(mode),
            results_root=args.results_root, run_id=args.run_id,
        )

    elif args.command == "report":
        generate_all_plots(outputs_dir=args.results_root, run_id=args.run_id)

    elif args.command in ["plot", "plots", "visualize"]:
        generate_all_plots(outputs_dir=args.results_root, run_id=args.run_id)

    elif args.command == "kfold":
        run_kfold_experiment(
            args.config,
            n_splits=args.folds,
            split_seed=args.seed,
            force_retrain=args.force_retrain,
            run_id=args.run_id,
            results_root=args.results_root,
        )

    elif args.command == "kfold-all":
        run_all_kfold_experiments(
            config_paths=None,
            n_splits=args.folds,
            split_seed=args.seed,
            force_retrain=args.force_retrain,
            run_id=args.run_id,
            results_root=args.results_root,
        )

    elif args.command in ["verify", "benchmark-verify", "audit"]:
        dispatch_verification_task(
            args.task,
            results_root=args.results_root,
            run_id=args.run_id,
            save_dir=args.save_dir,
            protocol=args.protocol,
            skip_tsne=args.skip_visuals,
        )

    elif args.command in ["all-in-one", "pipeline", "full-pipeline"]:
        config_paths = resolve_config_paths(args.configs)
        train_mode = args.train_mode
        ensemble_mode = prompt_ensemble_mode(args.ensemble_mode)

        print("\n" + "=" * 80)
        print(f"  STEP 1/3: BASE MODELS BENCHMARK & REPORT GENERATION (MODE: {train_mode.upper()})")
        print("=" * 80 + "\n")
        run_experiments(
            config_paths, mode=train_mode, run_id=args.run_id,
            results_root=args.results_root,
        )

        print("\n" + "=" * 80)
        print(f"  STEP 2/3: ENSEMBLE EVALUATION PIPELINE (MODE: {ensemble_mode.upper()})")
        print("=" * 80 + "\n")
        run_ensemble_evaluation(
            protocol=canonical_protocol_from_mode(ensemble_mode),
            results_root=args.results_root, run_id=args.run_id,
        )

        print("\n" + "=" * 80)
        print("  STEP 3/3: SCIENTIFIC VERIFICATION & CALIBRATION SUITE")
        print("=" * 80 + "\n")
        run_verification_suite(
            args.results_root, None, False, run_id=args.run_id
        )


if __name__ == "__main__":
    main()
