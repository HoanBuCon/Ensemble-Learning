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
from src.models.factory import create_model
from src.utils.config import ExperimentConfig, load_config, load_dataset_config
from src.utils.logger import CSVLogger, log_training_startup_banner, setup_logger
from src.utils.metrics import compute_metrics
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
        if isinstance(config, str):
            config = load_config(config)

        self.config = config
        self.n_splits = n_splits
        self.split_seed = split_seed
        self.force_retrain = force_retrain
        self.device = torch.device(config.device)
        self.output_dir = output_dir or os.path.join(
            config.checkpoint.save_dir, "oof"
        )

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
        cfg = self.config

        oof_prob_file = os.path.join(self.output_dir, "oof_probabilities.npy")
        oof_lbl_file = os.path.join(self.output_dir, "oof_labels.npy")
        test_prob_file = os.path.join(self.output_dir, "test_probabilities.npy")

        if not self.force_retrain and os.path.exists(oof_prob_file) and os.path.exists(oof_lbl_file):
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

        num_classes = full_train_dataset.num_classes
        num_samples = len(full_train_dataset)
        class_names = full_train_dataset.classes
        all_labels = np.array([s[1] for s in full_train_dataset.samples])

        # Test set loader (if available)
        test_dir = os.path.join(cfg.data.root, "test")
        test_loader = None
        test_labels = None
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

        oof_probabilities = np.zeros((num_samples, num_classes), dtype=np.float32)
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

        for fold_idx, (train_indices, val_indices) in enumerate(
            skf.split(np.zeros(num_samples), all_labels)
        ):
            fold_num = fold_idx + 1
            fold_dir = os.path.join(self.output_dir, f"fold_{fold_num}")
            os.makedirs(fold_dir, exist_ok=True)

            self.logger.info(f"\n{'='*55}\n  STARTING FOLD {fold_num}/{self.n_splits}\n{'='*55}")

            # Check if this fold was already completed
            fold_prob_path = os.path.join(fold_dir, "val_probabilities.npy")
            fold_metrics_path = os.path.join(fold_dir, "metrics.json")
            fold_ckpt_path = os.path.join(fold_dir, "best_model.pth")

            if (
                not self.force_retrain
                and os.path.exists(fold_prob_path)
                and os.path.exists(fold_metrics_path)
                and os.path.exists(fold_ckpt_path)
            ):
                self.logger.info(f"Fold {fold_num} already completed. Loading cached results.")
                fold_probs = np.load(fold_prob_path)
                oof_probabilities[val_indices] = fold_probs

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

            train_subset = Subset(full_train_dataset, train_indices.tolist())
            val_subset = Subset(full_train_dataset_val, val_indices.tolist())

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
            fold_val_loader = DataLoader(
                val_subset,
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

            fold_probs, f_metrics, test_probs_fold = self._train_and_evaluate_fold(
                model=model,
                train_loader=fold_train_loader,
                val_loader=fold_val_loader,
                test_loader=test_loader,
                val_labels=all_labels[val_indices],
                class_names=class_names,
                fold_idx=fold_idx,
                fold_dir=fold_dir,
            )

            oof_probabilities[val_indices] = fold_probs
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

        # Average test predictions across folds
        test_probabilities = None
        if test_probabilities_list:
            test_probabilities = np.mean(test_probabilities_list, axis=0)

        # Save OOF arrays
        np.save(os.path.join(self.output_dir, "oof_probabilities.npy"), oof_probabilities)
        np.save(os.path.join(self.output_dir, "oof_labels.npy"), all_labels)
        if test_probabilities is not None:
            np.save(os.path.join(self.output_dir, "test_probabilities.npy"), test_probabilities)

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
        formatted_df.to_csv(summary_csv, index=False)
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

    def _train_and_evaluate_fold(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader],
        val_labels: np.ndarray,
        class_names: List[str],
        fold_idx: int,
        fold_dir: str,
    ) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
        """Execute full training, validation, checkpointing, and evaluation for a single fold."""
        cfg = self.config.train
        fold_num = fold_idx + 1

        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer, cfg.epochs)

        use_amp = cfg.mixed_precision and self.device.type == "cuda"
        scaler = GradScaler("cuda", enabled=use_amp)

        best_val_acc = 0.0
        best_state = None
        best_epoch = 0
        early_stop_counter = 0

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "lr": [],
        }

        min_val_loss = float("inf")
        loss_gate_tol = getattr(self.config.checkpoint, "loss_gate_tolerance", 0.05)

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
                train_loader,
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

            # --- Validation ---
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion, use_amp)
            current_lr = optimizer.param_groups[0]["lr"]

            # Track minimum validation loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss

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

            # Checkpoint best with Dual-Metric Loss-Gate Safeguard
            if val_acc > best_val_acc:
                max_allowed_loss = (1.0 + loss_gate_tol) * min_val_loss
                if loss_gate_tol > 0 and val_loss > max_allowed_loss:
                    self.logger.warning(
                        f"Fold {fold_num} | Ep {epoch}: Val Acc improved ({val_acc:.2f}%) but rejected by Loss-Gate Safeguard "
                        f"(Val Loss {val_loss:.4f} > {max_allowed_loss:.4f} [min: {min_val_loss:.4f}])."
                    )
                    early_stop_counter += 1
                else:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    early_stop_counter = 0
            else:
                early_stop_counter += 1

            self.logger.info(
                f"Fold {fold_num} | Ep {epoch:02d}/{cfg.epochs:02d} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}% (Ep {best_epoch})"
            )

            if early_stop_counter >= cfg.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered for Fold {fold_num} at epoch {epoch} "
                    f"(patience={cfg.early_stopping_patience})"
                )
                break

        # Load best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Save fold checkpoint
        torch.save(
            {
                "epoch": best_epoch,
                "best_value": best_val_acc,
                "model_state_dict": model.state_dict(),
                "history": history,
            },
            os.path.join(fold_dir, "best_model.pth"),
        )

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

        # Compute validation predictions and full metrics
        val_probs = self._predict(model, val_loader)
        np.save(os.path.join(fold_dir, "val_probabilities.npy"), val_probs)

        val_preds = np.argmax(val_probs, axis=1)
        metrics = compute_metrics(val_labels, val_preds, class_names=class_names)

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
            val_labels, val_probs, class_names, fold_dir,
            model_name=f"Fold {fold_num}", filename="roc_curves.png"
        )
        plot_precision_recall_curves(
            val_labels, val_probs, class_names, fold_dir,
            model_name=f"Fold {fold_num}", filename="precision_recall_curves.png"
        )

        test_probs_fold = None
        if test_loader is not None:
            test_probs_fold = self._predict(model, test_loader)
            np.save(os.path.join(fold_dir, "test_probabilities.npy"), test_probs_fold)

        return val_probs, metrics, test_probs_fold

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
