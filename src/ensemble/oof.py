"""
Out-of-Fold (OOF) Prediction Generator & 5-Fold Cross-Validation Engine
======================================================================

Generates OOF predictions for Stacking training — the critical step
that prevents data leakage in stacked ensembles, and provides full
K-Fold cross-validation statistical benchmarking.

Features:
    - Stratified K-Fold cross validation with deterministic alignment.
    - Full training hyperparameter inheritance from ExperimentConfig (LR, warmup, scheduler, early stopping, AMP, grad clip).
    - Per-fold artifact persistence (best_model.pth, metrics.json, history.csv, confusion matrices, ROC/PR curves).
    - Fold-level resumption support (skips completed folds on rerun).
    - Comprehensive K-Fold summary tables (Mean ± Std) and variance visualization plots.
    - Out-of-fold probability caching for Stacking Meta-Learner training.

Usage::

    from src.ensemble.oof import OOFGenerator

    oof_gen = OOFGenerator(config="configs/resnet50.yaml", n_splits=5)
    oof_probs, oof_labels, test_probs = oof_gen.generate()
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.datasets.dataset import ImageFolderDataset
from src.datasets.transforms import build_transforms
from src.ensemble.artifacts import sample_ids_from_paths, save_prediction_artifact
from src.engine.checkpoint import CheckpointManager
from src.models.factory import create_model
from src.utils.config import ExperimentConfig, load_config, load_dataset_config
from src.utils.logger import CSVLogger, log_training_startup_banner, setup_logger
from src.utils.metrics import compute_metrics
from src.utils.provenance import verify_dataset_snapshot, write_experiment_manifest
from src.utils.reproducibility import get_generator, seed_worker, set_seed
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_kfold_summary,
    plot_per_class_metrics,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_training_curves,
)


class OOFGenerator:
    """
    Generate Out-of-Fold predictions and execute 5-Fold Cross Validation.

    Uses ``StratifiedKFold`` to split training data, trains a fresh
    backbone on each fold with full hyperparameter support, and collects
    held-out predictions. Test predictions are averaged across all folds.

    Args:
        config: Experiment configuration (path or object).
        n_splits: Number of cross-validation folds (default: 5).
        split_seed: Random seed for stratified splitting.
        output_dir: Directory to save OOF and fold artifacts.
        force_retrain: If True, ignores existing cached fold models.
    """

    def __init__(
        self,
        config: ExperimentConfig | str,
        n_splits: int = 5,
        split_seed: int = 42,
        output_dir: Optional[str] = None,
        force_retrain: bool = False,
    ) -> None:
        self.config_path = config if isinstance(config, str) else None
        if isinstance(config, str):
            config = load_config(config)

        self.config = config
        self.n_splits = n_splits
        self.split_seed = split_seed
        self.force_retrain = force_retrain
        self.device = torch.device(config.device)
        if output_dir:
            self.output_dir = output_dir
        else:
            base_sd = config.checkpoint.save_dir
            if "DEFAULT_TRAINING" in base_sd:
                self.output_dir = os.path.join(base_sd.replace("DEFAULT_TRAINING", "OOF_TRAINING"), "kfold")
            elif "Default_Result" in base_sd:
                self.output_dir = os.path.join(base_sd.replace("Default_Result", "OOF_Results"), "kfold")
            elif base_sd.startswith("./outputs") or base_sd.startswith("outputs"):
                model_name = getattr(config.model, "name", "model")
                self.output_dir = os.path.join("RESULTS", "OOF_TRAINING", "outputs", model_name, "kfold")
            else:
                self.output_dir = os.path.join(base_sd, "kfold")

        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = setup_logger(
            f"{config.experiment_name}_oof",
            log_dir=self.output_dir,
        )

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build optimizer using config training settings."""
        cfg = self.config.train
        params = model.parameters()

        optimizers = {
            "adam": lambda: torch.optim.Adam(
                params, lr=cfg.lr, weight_decay=cfg.weight_decay
            ),
            "adamw": lambda: torch.optim.AdamW(
                params, lr=cfg.lr, weight_decay=cfg.weight_decay
            ),
            "sgd": lambda: torch.optim.SGD(
                params, lr=cfg.lr, momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            ),
        }
        name = cfg.optimizer.lower()
        if name not in optimizers:
            return optimizers["adamw"]()
        return optimizers[name]()

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, total_epochs: int
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Build LR scheduler using config settings."""
        cfg = self.config.train
        schedulers = {
            "cosine": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_epochs - cfg.warmup_epochs), eta_min=1e-7
            ),
            "step": lambda: torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.step_size, gamma=cfg.step_gamma
            ),
            "plateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=self.config.checkpoint.mode,
                factor=cfg.plateau_factor, patience=cfg.plateau_patience
            ),
            "none": lambda: None,
        }
        name = cfg.scheduler.lower()
        if name not in schedulers:
            return schedulers["cosine"]()
        return schedulers[name]()

    def generate(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Run the full 5-Fold OOF generation & Cross-Validation pipeline.

        Returns:
            Tuple of:
                - ``oof_probabilities``: shape ``(N_train, C)``
                - ``oof_labels``: shape ``(N_train,)``
                - ``test_probabilities``: shape ``(N_test, C)`` or ``None``
        """
        set_seed(self.config.seed)
        verify_dataset_snapshot()
        if self.config_path is None:
            raise ValueError(
                "FINAL_V2 OOF provenance requires OOFGenerator(config=<yaml path>)"
            )
        cfg = self.config

        oof_prob_file = os.path.join(self.output_dir, "oof_probabilities.npy")
        oof_lbl_file = os.path.join(self.output_dir, "oof_labels.npy")
        test_prob_file = os.path.join(self.output_dir, "test_probabilities.npy")

        fold_manifests = [
            os.path.join(self.output_dir, f"fold_{index}", "experiment_manifest.json")
            for index in range(1, self.n_splits + 1)
        ]
        if (
            not self.force_retrain
            and os.path.exists(oof_prob_file)
            and os.path.exists(oof_lbl_file)
            and os.path.exists(os.path.join(self.output_dir, "oof_predictions.npz"))
            and all(os.path.exists(path) for path in fold_manifests)
        ):
            self.logger.info(f"Existing OOF predictions found in '{self.output_dir}'. Loading cached arrays.")
            oof_probs = np.load(oof_prob_file)
            oof_lbls = np.load(oof_lbl_file)
            test_probs = np.load(test_prob_file) if os.path.exists(test_prob_file) else None
            return oof_probs, oof_lbls, test_probs

        image_size = cfg.data.image_size
        train_transform = build_transforms(
            cfg.augmentation.train, image_size=image_size, stage="train"
        )
        val_transform = build_transforms(
            cfg.augmentation.val, image_size=image_size, stage="val"
        )

        full_train_dataset = ImageFolderDataset(
            root=os.path.join(cfg.data.root, "train"),
            transform=train_transform,
        )
        full_train_dataset_val = ImageFolderDataset(
            root=os.path.join(cfg.data.root, "train"),
            transform=val_transform,
        )

        validation_candidates = [
            os.path.join(cfg.data.root, "val"),
            os.path.join(cfg.data.root, "valid"),
        ]
        validation_dirs = [path for path in validation_candidates if os.path.isdir(path)]
        if len(validation_dirs) != 1:
            raise RuntimeError(
                "OOF training requires exactly one fixed external validation split "
                f"('data/val' or 'data/valid'); found: {validation_dirs or 'none'}"
            )
        external_val_dataset = ImageFolderDataset(
            root=validation_dirs[0],
            transform=val_transform,
        )

        num_classes = full_train_dataset.num_classes
        num_samples = len(full_train_dataset)
        class_names = full_train_dataset.classes
        all_labels = np.array([s[1] for s in full_train_dataset.samples])
        all_sample_ids = sample_ids_from_paths(
            [sample[0] for sample in full_train_dataset.samples]
        )
        if external_val_dataset.classes != class_names:
            raise RuntimeError(
                "External validation class order does not match the training split: "
                f"{external_val_dataset.classes} != {class_names}"
            )

        external_val_loader = DataLoader(
            external_val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )

        # Test set loader (if available)
        test_dir = os.path.join(cfg.data.root, "test")
        test_loader = None
        test_labels = None
        test_sample_ids = None
        if os.path.isdir(test_dir):
            test_transform = build_transforms(
                cfg.augmentation.test, image_size=image_size, stage="test"
            )
            test_dataset = ImageFolderDataset(
                root=test_dir, transform=test_transform
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.data.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
            )
            test_labels = np.array([s[1] for s in test_dataset.samples])
            test_sample_ids = sample_ids_from_paths(
                [sample[0] for sample in test_dataset.samples]
            )

        oof_probabilities = np.zeros((num_samples, num_classes), dtype=np.float32)
        oof_assignment_counts = np.zeros(num_samples, dtype=np.int8)
        test_probabilities_list: List[np.ndarray] = []
        fold_metrics_list: List[Dict[str, float]] = []

        # Model param count for startup banner
        temp_model = create_model(cfg.model.name, pretrained=False, num_classes=num_classes)
        num_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
        del temp_model

        log_training_startup_banner(
            config=self.config,
            num_params=num_params,
            is_kfold=True,
            n_splits=self.n_splits,
            logger=self.logger,
        )

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.split_seed
        )

        self.logger.info(
            f"Starting 5-Fold Cross Validation: {self.n_splits} folds, "
            f"{num_samples} samples, {num_classes} classes (Split Seed: {self.split_seed})"
        )

        for fold_idx, (train_indices, outer_holdout_indices) in enumerate(
            skf.split(np.zeros(num_samples), all_labels)
        ):
            fold_num = fold_idx + 1
            fold_dir = os.path.join(self.output_dir, f"fold_{fold_num}")
            os.makedirs(fold_dir, exist_ok=True)

            self.logger.info(f"\n{'='*55}\n  STARTING FOLD {fold_num}/{self.n_splits}\n{'='*55}")

            # Check if this fold was already completed
            fold_prob_path = os.path.join(fold_dir, "outer_holdout_probabilities.npy")
            fold_metrics_path = os.path.join(fold_dir, "metrics.json")
            fold_ckpt_path = os.path.join(fold_dir, "best_model.pth")

            if (
                not self.force_retrain
                and os.path.exists(fold_prob_path)
                and os.path.exists(fold_metrics_path)
                and os.path.exists(fold_ckpt_path)
                and os.path.exists(os.path.join(fold_dir, "experiment_manifest.json"))
            ):
                self.logger.info(f"Fold {fold_num} already completed. Loading cached results.")
                fold_probs = np.load(fold_prob_path)
                if len(fold_probs) != len(outer_holdout_indices):
                    raise RuntimeError(
                        f"Fold {fold_num} cached outer-holdout row count does not match "
                        "the current deterministic split"
                    )
                self._place_oof_rows(
                    oof_probabilities,
                    oof_assignment_counts,
                    outer_holdout_indices,
                    fold_probs,
                )

                with open(fold_metrics_path, "r", encoding="utf-8") as f:
                    f_metrics = json.load(f)
                fold_metrics_list.append({
                    "Fold": fold_num,
                    "Accuracy": f_metrics.get("accuracy", 0.0) * 100.0,
                    "Precision": f_metrics.get("precision", 0.0) * 100.0,
                    "Recall": f_metrics.get("recall", 0.0) * 100.0,
                    "F1_Score": f_metrics.get("f1_score", 0.0) * 100.0,
                })

                if test_loader is not None:
                    # Run inference with cached checkpoint on test set
                    model = create_model(
                        model_name=cfg.model.name,
                        pretrained=False,
                        num_classes=cfg.model.num_classes,
                    ).to(self.device)
                    ckpt = torch.load(fold_ckpt_path, map_location=self.device, weights_only=False)
                    model.load_state_dict(ckpt["model_state_dict"])
                    t_probs = self._predict(model, test_loader)
                    test_probabilities_list.append(t_probs)
                    del model
                    torch.cuda.empty_cache()
                continue

            set_seed(cfg.seed + fold_idx)

            self._assert_fold_boundary(train_indices, outer_holdout_indices)
            train_subset = Subset(full_train_dataset, train_indices.tolist())
            outer_holdout_subset = Subset(
                full_train_dataset_val, outer_holdout_indices.tolist()
            )

            g = get_generator(cfg.seed + fold_idx)

            fold_train_loader = DataLoader(
                train_subset,
                batch_size=cfg.data.batch_size,
                shuffle=True,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
                worker_init_fn=seed_worker,
                generator=g,
                drop_last=True,
            )
            outer_holdout_loader = DataLoader(
                outer_holdout_subset,
                batch_size=cfg.data.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
            )

            model = create_model(
                model_name=cfg.model.name,
                pretrained=cfg.model.pretrained,
                num_classes=cfg.model.num_classes,
                drop_rate=cfg.model.drop_rate,
            ).to(self.device)

            model = self._train_fold(
                model=model,
                fold_train_loader=fold_train_loader,
                external_val_loader=external_val_loader,
                fold_idx=fold_idx,
                fold_dir=fold_dir,
            )

            fold_probs, f_metrics = self._evaluate_outer_holdout(
                frozen_model=model,
                outer_holdout_loader=outer_holdout_loader,
                outer_holdout_labels=all_labels[outer_holdout_indices],
                outer_holdout_sample_ids=all_sample_ids[outer_holdout_indices],
                class_names=class_names,
                fold_idx=fold_idx,
                fold_dir=fold_dir,
            )
            test_probs_fold = None
            if test_loader is not None:
                test_probs_fold = self._predict(model, test_loader)
                np.save(os.path.join(fold_dir, "test_probabilities.npy"), test_probs_fold)

            write_experiment_manifest(
                os.path.join(fold_dir, "experiment_manifest.json"),
                protocol="oof",
                model=cfg.model.name,
                fold=fold_num,
                config_path=self.config_path,
                seed=cfg.seed + fold_idx,
                checkpoint_path=os.path.join(fold_dir, "best_model.pth"),
                prediction_path=os.path.join(
                    fold_dir, "outer_holdout_predictions.npz"
                ),
                arguments={
                    "n_splits": self.n_splits,
                    "shuffle": True,
                    "split_random_state": self.split_seed,
                    "external_validation_directory": os.path.basename(validation_dirs[0]),
                },
            )

            self._place_oof_rows(
                oof_probabilities,
                oof_assignment_counts,
                outer_holdout_indices,
                fold_probs,
            )
            fold_metrics_list.append({
                "Fold": fold_num,
                "Accuracy": f_metrics["accuracy"] * 100.0,
                "Precision": f_metrics["precision"] * 100.0,
                "Recall": f_metrics["recall"] * 100.0,
                "F1_Score": f_metrics["f1_score"] * 100.0,
            })

            if test_probs_fold is not None:
                test_probabilities_list.append(test_probs_fold)

            del model
            torch.cuda.empty_cache()

        if not np.all(oof_assignment_counts == 1):
            raise AssertionError(
                "OOF generation did not assign every training sample exactly once"
            )

        # Average test predictions across folds
        test_probabilities = None
        if test_probabilities_list:
            test_probabilities = np.mean(test_probabilities_list, axis=0)

        # Save OOF arrays
        np.save(os.path.join(self.output_dir, "oof_probabilities.npy"), oof_probabilities)
        np.save(os.path.join(self.output_dir, "oof_labels.npy"), all_labels)
        if test_probabilities is not None:
            np.save(os.path.join(self.output_dir, "test_probabilities.npy"), test_probabilities)

        save_prediction_artifact(
            os.path.join(self.output_dir, "oof_predictions.npz"),
            sample_ids=all_sample_ids,
            y_true=all_labels,
            probabilities=oof_probabilities,
            predictions=np.argmax(oof_probabilities, axis=1),
            class_order=class_names,
            protocol="oof",
            method=cfg.model.name,
            split="oof_train",
        )
        if test_probabilities is not None:
            if test_labels is None or test_sample_ids is None:
                raise AssertionError("Test identity is unavailable for OOF aggregation")
            save_prediction_artifact(
                os.path.join(self.output_dir, "test_predictions.npz"),
                sample_ids=test_sample_ids,
                y_true=test_labels,
                probabilities=test_probabilities,
                predictions=np.argmax(test_probabilities, axis=1),
                class_order=class_names,
                protocol="oof",
                method=cfg.model.name,
                split="test",
            )

        # Save class mappings and experiment config
        class_to_idx = getattr(full_train_dataset, "class_to_idx", {c: i for i, c in enumerate(class_names)})
        with open(os.path.join(self.output_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, indent=2)

        # Compute overall OOF Metrics
        oof_preds = np.argmax(oof_probabilities, axis=1)
        oof_metrics = compute_metrics(all_labels, oof_preds, class_names=class_names)
        with open(os.path.join(self.output_dir, "oof_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in oof_metrics.items() if k != "classification_report"}, f, indent=2)
        with open(os.path.join(self.output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(oof_metrics.get("classification_report", ""))

        # Build K-Fold Cross Validation Summary Table
        df_kfold = pd.DataFrame(fold_metrics_list)
        mean_row = {
            "Fold": "Mean ± Std",
            "Accuracy": f"{df_kfold['Accuracy'].mean():.2f}% ± {df_kfold['Accuracy'].std():.2f}%",
            "Precision": f"{df_kfold['Precision'].mean():.2f}% ± {df_kfold['Precision'].std():.2f}%",
            "Recall": f"{df_kfold['Recall'].mean():.2f}% ± {df_kfold['Recall'].std():.2f}%",
            "F1_Score": f"{df_kfold['F1_Score'].mean():.2f}% ± {df_kfold['F1_Score'].std():.2f}%",
        }

        # Format rows
        formatted_df = df_kfold.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1_Score"]:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}%")
        formatted_df = pd.concat([formatted_df, pd.DataFrame([mean_row])], ignore_index=True)

        summary_csv = os.path.join(self.output_dir, "kfold_summary.csv")
        summary_md = os.path.join(self.output_dir, "kfold_summary.md")
        formatted_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write(f"# 5-Fold Cross-Validation Performance Summary ({cfg.model.name})\n\n")
            f.write(formatted_df.to_markdown(index=False))
            f.write(f"\n\n**Global OOF Accuracy**: {oof_metrics['accuracy']*100:.2f}%\n")
            f.write(f"**Global OOF Macro F1**: {oof_metrics['f1_score']*100:.2f}%\n")

        # Plot K-Fold Stability variance bar chart
        kfold_dict = {
            "Accuracy": df_kfold["Accuracy"].tolist(),
            "Precision": df_kfold["Precision"].tolist(),
            "Recall": df_kfold["Recall"].tolist(),
            "F1_Score": df_kfold["F1_Score"].tolist(),
        }
        plot_kfold_summary(
            kfold_dict, output_dir=self.output_dir,
            model_name=cfg.model.name, filename="kfold_variance_bar.png"
        )

        self.logger.info(f"\n{'='*65}\n  5-FOLD CROSS VALIDATION RESULTS SUMMARY\n{'='*65}")
        self.logger.info(f"\n{formatted_df.to_string(index=False)}")
        self.logger.info(f"\nResults saved to: {self.output_dir}")

        return oof_probabilities, all_labels, test_probabilities

    @staticmethod
    def _assert_fold_boundary(
        fold_train_indices: np.ndarray,
        outer_holdout_indices: np.ndarray,
    ) -> None:
        """Assert that the cross-fitting optimization and inference sets are disjoint."""
        overlap = np.intersect1d(fold_train_indices, outer_holdout_indices)
        if overlap.size:
            raise AssertionError(
                f"Outer-holdout leakage: {overlap.size} indices occur in fold training"
            )

    @staticmethod
    def _place_oof_rows(
        target: np.ndarray,
        assignment_counts: np.ndarray,
        outer_holdout_indices: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        """Place held-out predictions once at their deterministic source indices."""
        if len(outer_holdout_indices) != len(predictions):
            raise AssertionError("OOF prediction row count does not match holdout indices")
        if np.any(assignment_counts[outer_holdout_indices] != 0):
            raise AssertionError("An OOF row was assigned more than once")
        target[outer_holdout_indices] = predictions
        assignment_counts[outer_holdout_indices] += 1

    def _train_fold(
        self,
        model: nn.Module,
        fold_train_loader: DataLoader,
        external_val_loader: DataLoader,
        fold_idx: int,
        fold_dir: str,
    ) -> nn.Module:
        """Train one fold using only the fixed external validation split for selection.

        The outer holdout loader is intentionally absent from this API. It cannot
        influence validation metrics, scheduler decisions, checkpoint acceptance,
        best epoch, or early stopping.
        """
        cfg = self.config.train
        fold_num = fold_idx + 1

        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer, cfg.epochs)

        use_amp = cfg.mixed_precision and self.device.type == "cuda"
        scaler = GradScaler("cuda", enabled=use_amp)

        early_stop_counter = 0
        checkpoint_manager = CheckpointManager(
            save_dir=fold_dir,
            monitor=self.config.checkpoint.monitor,
            mode=self.config.checkpoint.mode,
            loss_gate_tolerance=getattr(
                self.config.checkpoint, "loss_gate_tolerance", 0.05
            ),
        )

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "lr": [],
        }

        for epoch in range(1, cfg.epochs + 1):
            if epoch <= cfg.warmup_epochs:
                warmup_lr = cfg.lr * (epoch / max(1, cfg.warmup_epochs))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            # --- Training ---
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            optimizer.zero_grad()
            for step, batch in enumerate(tqdm(
                fold_train_loader,
                desc=f"Fold {fold_num} Ep {epoch}/{cfg.epochs} [Train]",
                leave=False,
            )):
                images = batch[0].to(self.device, non_blocking=True)
                labels = batch[1].to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss = loss / cfg.accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % cfg.accumulation_steps == 0:
                    if cfg.gradient_clip_value > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.gradient_clip_value
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                running_loss += loss.item() * cfg.accumulation_steps * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / total
            train_acc = 100.0 * correct / total

            # --- Fixed external validation (never the outer held-out fold) ---
            val_loss, val_acc = self._validate_epoch(
                model, external_val_loader, criterion, use_amp
            )
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["lr"].append(current_lr)

            if scheduler is not None and epoch > cfg.warmup_epochs:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            metrics_snapshot = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
            current_metric = (
                val_acc
                if self.config.checkpoint.monitor == "val_accuracy"
                else val_loss
            )
            accepted = checkpoint_manager.save_if_best(
                model=model,
                current_metric=current_metric,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics_snapshot,
                history=history,
            )
            if accepted:
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            self.logger.info(
                f"Fold {fold_num} | Ep {epoch:02d}/{cfg.epochs:02d} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                f"Accepted: {checkpoint_manager.best_value:.2f} "
                f"(Ep {checkpoint_manager.best_epoch})"
            )

            if early_stop_counter >= cfg.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered for Fold {fold_num} at epoch {epoch} "
                    f"(patience={cfg.early_stopping_patience})"
                )
                break

        if checkpoint_manager.best_epoch is None:
            raise RuntimeError(
                f"Fold {fold_num} completed without an accepted checkpoint"
            )
        best_checkpoint = checkpoint_manager.load_best(device=str(self.device))
        model.load_state_dict(best_checkpoint["model_state_dict"])

        # Save history CSV, JSON & plots
        history_df = pd.DataFrame({
            "epoch": list(range(1, len(history["train_loss"]) + 1)),
            "train_loss": history["train_loss"],
            "train_accuracy": history["train_accuracy"],
            "val_loss": history["val_loss"],
            "val_accuracy": history["val_accuracy"],
            "lr": history["lr"],
        })
        history_df.to_csv(os.path.join(fold_dir, "history.csv"), index=False)

        with open(os.path.join(fold_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        plot_training_curves(history, fold_dir, model_name=f"{self.config.model.name} Fold {fold_num}")

        return model

    def _evaluate_outer_holdout(
        self,
        frozen_model: nn.Module,
        outer_holdout_loader: DataLoader,
        outer_holdout_labels: np.ndarray,
        outer_holdout_sample_ids: np.ndarray,
        class_names: List[str],
        fold_idx: int,
        fold_dir: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run inference-only evaluation on a fold after its checkpoint is frozen."""
        fold_num = fold_idx + 1
        outer_holdout_probs = self._predict(frozen_model, outer_holdout_loader)
        np.save(
            os.path.join(fold_dir, "outer_holdout_probabilities.npy"),
            outer_holdout_probs,
        )
        save_prediction_artifact(
            os.path.join(fold_dir, "outer_holdout_predictions.npz"),
            sample_ids=outer_holdout_sample_ids,
            y_true=outer_holdout_labels,
            probabilities=outer_holdout_probs,
            predictions=np.argmax(outer_holdout_probs, axis=1),
            class_order=class_names,
            protocol="oof",
            method=self.config.model.name,
            split="outer_holdout",
        )

        outer_holdout_preds = np.argmax(outer_holdout_probs, axis=1)
        metrics = compute_metrics(
            outer_holdout_labels,
            outer_holdout_preds,
            class_names=class_names,
        )

        with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=2)
        with open(os.path.join(fold_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(metrics["classification_report"])

        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(
            cm, class_names, fold_dir,
            title=f"Confusion Matrix — Fold {fold_num}",
            normalize=False, filename="confusion_matrix.png"
        )
        plot_confusion_matrix(
            cm, class_names, fold_dir,
            title=f"Normalized Confusion Matrix — Fold {fold_num}",
            normalize=True, filename="confusion_matrix_normalized.png"
        )
        plot_per_class_metrics(
            metrics.get("per_class", {}), class_names, fold_dir,
            model_name=f"Fold {fold_num}", filename="per_class_metrics.png"
        )
        plot_roc_curves(
            outer_holdout_labels, outer_holdout_probs, class_names, fold_dir,
            model_name=f"Fold {fold_num}", filename="roc_curves.png"
        )
        plot_precision_recall_curves(
            outer_holdout_labels, outer_holdout_probs, class_names, fold_dir,
            model_name=f"Fold {fold_num}", filename="precision_recall_curves.png"
        )

        return outer_holdout_probs, metrics

    @torch.no_grad()
    def _validate_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        use_amp: bool,
    ) -> Tuple[float, float]:
        """Validate one epoch and return (avg_loss, accuracy_percentage)."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            images = batch[0].to(self.device, non_blocking=True)
            labels = batch[1].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return running_loss / total, 100.0 * correct / total

    @torch.no_grad()
    def _predict(
        self, model: nn.Module, dataloader: DataLoader
    ) -> np.ndarray:
        """Run inference and return softmax probability matrix."""
        model.eval()
        all_logits: List[np.ndarray] = []

        for batch in dataloader:
            images = batch[0].to(self.device, non_blocking=True)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())

        logits = np.concatenate(all_logits, axis=0)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return probabilities
