"""
Dataset & DataLoader Factory
=============================

Generic ImageFolder dataset with Albumentations support and
a factory function that creates train/val/test dataloaders from config.

Usage::

    from src.datasets.dataset import create_dataloaders

    train_loader, val_loader, test_loader, class_names = create_dataloaders(config)
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from src.datasets.transforms import build_transforms
from src.utils.config import ExperimentConfig, load_dataset_config
from src.utils.reproducibility import get_generator, seed_worker


class ImageFolderDataset(Dataset):
    """
    Generic image classification dataset backed by an ``ImageFolder`` layout.

    Wraps ``torchvision.datasets.ImageFolder`` and applies Albumentations
    transforms. Returns ``(image_tensor, label, image_path)``.

    Args:
        root: Path to the split directory (e.g., ``data/train``).
        transform: An :class:`albumentations.Compose` pipeline.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[A.Compose] = None,
    ) -> None:
        super().__init__()
        self._dataset = ImageFolder(root)
        self.transform = transform
        self.classes: List[str] = self._dataset.classes
        self.class_to_idx = self._dataset.class_to_idx
        self.samples = self._dataset.samples

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[Any, int, str]:
        """
        Returns:
            Tuple of ``(image_tensor, label, image_path)``.
        """
        path, label = self._dataset.samples[index]

        # Read image with OpenCV (Albumentations expects numpy HWC BGR)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label, path

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset."""
        return len(self.classes)


def create_dataloaders(
    config: ExperimentConfig,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], List[str]]:
    """
    Create train, validation, and optional test dataloaders from config.

    Expects the dataset root to contain ``train/``, ``val/`` subdirectories,
    and optionally a ``test/`` subdirectory.

    Args:
        config: Full experiment configuration.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader, class_names)``.
        ``test_loader`` is ``None`` if no ``test/`` directory exists.
    """
    image_size = config.data.image_size

    train_transform = build_transforms(
        config.augmentation.train, image_size=image_size, stage="train"
    )
    val_transform = build_transforms(
        config.augmentation.val, image_size=image_size, stage="val"
    )

    # Datasets
    train_dataset = ImageFolderDataset(
        root=os.path.join(config.data.root, "train"),
        transform=train_transform,
    )
    dataset_config = load_dataset_config()
    val_dir = str(dataset_config.get("val_dir", ""))
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Configured validation directory not found: {val_dir}")

    val_dataset = ImageFolderDataset(
        root=val_dir,
        transform=val_transform,
    )

    class_names = train_dataset.classes

    # Generator for reproducible shuffling
    g = get_generator(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    # Test set (optional)
    test_loader = None
    test_dir = os.path.join(config.data.root, "test")
    if os.path.isdir(test_dir):
        test_transform = build_transforms(
            config.augmentation.test, image_size=image_size, stage="test"
        )
        test_dataset = ImageFolderDataset(
            root=test_dir,
            transform=test_transform,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
        )

    return train_loader, val_loader, test_loader, class_names
