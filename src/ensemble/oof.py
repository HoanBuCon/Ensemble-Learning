"""
Out-of-Fold (OOF) Prediction Generator
=======================================

Generates OOF predictions for Stacking training — the critical step
that prevents data leakage in stacked ensembles.

For each fold:
    1. Train the model on K-1 folds.
    2. Predict on the held-out fold → OOF predictions.
    3. Predict on the test set → averaged later for final stacking input.

Usage::

    from src.ensemble.oof import OOFGenerator

    oof_gen = OOFGenerator(config_path="configs/resnet50.yaml", n_splits=5)
    oof_probs, oof_labels, test_probs = oof_gen.generate()
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.datasets.dataset import ImageFolderDataset, create_dataloaders
from src.datasets.transforms import build_transforms
from src.models.factory import create_model
from src.utils.config import ExperimentConfig, load_config
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed, get_generator, seed_worker


class OOFGenerator:
    """
    Generate Out-of-Fold predictions for Stacking ensembles.

    Uses ``StratifiedKFold`` to split the training data, trains a fresh
    model on each fold, and collects held-out predictions. Test predictions
    are averaged across all folds.

    Args:
        config: Experiment configuration (path or object).
        n_splits: Number of cross-validation folds.
        output_dir: Directory to save OOF arrays. Defaults to config checkpoint dir.
    """

    def __init__(
        self,
        config: ExperimentConfig | str,
        n_splits: int = 5,
        output_dir: Optional[str] = None,
    ) -> None:
        if isinstance(config, str):
            config = load_config(config)

        self.config = config
        self.n_splits = n_splits
        self.device = torch.device(config.device)
        self.output_dir = output_dir or os.path.join(
            config.checkpoint.save_dir, "oof"
        )

        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = setup_logger(
            f"{config.experiment_name}_oof",
            log_dir=self.output_dir,
        )

    def generate(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Run the full OOF generation pipeline.

        Returns:
            Tuple of:
                - ``oof_probabilities``: shape ``(N_train, C)``
                - ``oof_labels``: shape ``(N_train,)``
                - ``test_probabilities``: shape ``(N_test, C)`` or ``None``
        """
        set_seed(self.config.seed)
        cfg = self.config

        # Build full training dataset (no shuffle in dataset itself)
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

        # We also need a version with val transforms for OOF prediction
        full_train_dataset_val = ImageFolderDataset(
            root=os.path.join(cfg.data.root, "train"),
            transform=val_transform,
        )

        num_classes = full_train_dataset.num_classes
        num_samples = len(full_train_dataset)

        # Extract labels for stratification
        all_labels = np.array([s[1] for s in full_train_dataset.samples])

        # Test loader (optional)
        test_dir = os.path.join(cfg.data.root, "test")
        test_loader = None
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

        # Initialize OOF arrays
        oof_probabilities = np.zeros((num_samples, num_classes), dtype=np.float32)
        test_probabilities_list: List[np.ndarray] = []

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=cfg.seed
        )

        self.logger.info(
            f"Starting OOF generation: {self.n_splits} folds, "
            f"{num_samples} samples, {num_classes} classes"
        )

        for fold_idx, (train_indices, val_indices) in enumerate(
            skf.split(np.zeros(num_samples), all_labels)
        ):
            self.logger.info(
                f"--- Fold {fold_idx + 1}/{self.n_splits} ---"
            )

            # Create fold dataloaders
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

            # Train fresh model for this fold
            model = create_model(
                model_name=cfg.model.name,
                pretrained=cfg.model.pretrained,
                num_classes=cfg.model.num_classes,
                drop_rate=cfg.model.drop_rate,
            ).to(self.device)

            fold_probs = self._train_and_predict_fold(
                model=model,
                train_loader=fold_train_loader,
                val_loader=fold_val_loader,
                fold_idx=fold_idx,
            )

            # Store OOF predictions
            oof_probabilities[val_indices] = fold_probs

            # Test predictions for this fold
            if test_loader is not None:
                test_probs = self._predict(model, test_loader)
                test_probabilities_list.append(test_probs)

            # Cleanup
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

        self.logger.info(
            f"OOF generation complete. Files saved to: {self.output_dir}"
        )

        return oof_probabilities, all_labels, test_probabilities

    def _train_and_predict_fold(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold_idx: int,
    ) -> np.ndarray:
        """
        Train model on fold training data and predict on fold validation data.

        Returns:
            OOF probabilities for the validation subset: shape ``(N_val, C)``.
        """
        cfg = self.config.train

        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=1e-7
        )

        use_amp = cfg.mixed_precision and self.device.type == "cuda"
        scaler = GradScaler("cuda", enabled=use_amp)

        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, cfg.epochs + 1):
            # Train
            model.train()
            for batch in tqdm(
                train_loader,
                desc=f"Fold {fold_idx + 1} Epoch {epoch}/{cfg.epochs}",
                leave=False,
            ):
                images = batch[0].to(self.device, non_blocking=True)
                labels = batch[1].to(self.device, non_blocking=True)

                optimizer.zero_grad()
                with autocast(device_type=self.device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()

                if cfg.gradient_clip_value > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.gradient_clip_value
                    )

                scaler.step(optimizer)
                scaler.update()

            scheduler.step()

            # Validate
            val_acc = self._validate_accuracy(model, val_loader)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            self.logger.info(
                f"  Fold {fold_idx + 1} | Epoch {epoch}/{cfg.epochs} | "
                f"Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}%"
            )

        # Load best weights and predict on validation set
        if best_state is not None:
            model.load_state_dict(best_state)

        return self._predict(model, val_loader)

    @torch.no_grad()
    def _validate_accuracy(
        self, model: nn.Module, dataloader: DataLoader
    ) -> float:
        """Compute validation accuracy."""
        model.eval()
        correct = 0
        total = 0
        for batch in dataloader:
            images = batch[0].to(self.device, non_blocking=True)
            labels = batch[1].to(self.device, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return 100.0 * correct / total

    @torch.no_grad()
    def _predict(
        self, model: nn.Module, dataloader: DataLoader
    ) -> np.ndarray:
        """Run inference and return softmax probabilities."""
        model.eval()
        all_logits: List[np.ndarray] = []

        for batch in dataloader:
            images = batch[0].to(self.device, non_blocking=True)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())

        logits = np.concatenate(all_logits, axis=0)

        # Stable softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return probabilities
