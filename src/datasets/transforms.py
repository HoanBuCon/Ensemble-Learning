"""
Transform Pipeline Builder
===========================

Builds Albumentations transform pipelines from YAML configuration.

All pipelines end with ``Normalize`` + ``ToTensorV2`` to produce
correctly shaped float32 tensors for PyTorch.

Usage::

    from src.datasets.transforms import build_transforms

    train_transforms = build_transforms(config.augmentation.train, image_size=224, stage="train")
    val_transforms   = build_transforms(config.augmentation.val,   image_size=224, stage="val")
"""

from __future__ import annotations

from typing import List, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.config import AugmentStageConfig


def build_transforms(
    aug_config: AugmentStageConfig,
    image_size: int = 224,
    stage: str = "train",
    seed: Optional[int] = None,
) -> A.Compose:
    """
    Build an Albumentations ``Compose`` pipeline from config.

    Args:
        aug_config: Augmentation parameters for this stage.
        image_size: Target image size (height = width).
        stage: One of ``'train'``, ``'val'``, ``'test'``.
        seed: Owner seed for Albumentations' independent random generator.
              DataLoader workers replace this with their deterministic worker
              seed when multiprocessing is active.

    Returns:
        An :class:`albumentations.Compose` instance.
    """
    transforms: List[A.BasicTransform] = []

    # Resize is always applied
    transforms.append(A.Resize(height=image_size, width=image_size))

    if stage == "train":
        # Spatial augmentations
        if aug_config.horizontal_flip > 0:
            transforms.append(A.HorizontalFlip(p=aug_config.horizontal_flip))

        if aug_config.vertical_flip > 0:
            transforms.append(A.VerticalFlip(p=aug_config.vertical_flip))

        if aug_config.rotation_limit > 0:
            transforms.append(
                A.Rotate(limit=aug_config.rotation_limit, p=0.5)
            )

        # Color augmentations
        if aug_config.brightness_limit > 0 or aug_config.contrast_limit > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=aug_config.brightness_limit,
                    contrast_limit=aug_config.contrast_limit,
                    p=0.5,
                )
            )

        if aug_config.hue_saturation_limit > 0:
            transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=int(aug_config.hue_saturation_limit * 20),
                    sat_shift_limit=int(aug_config.hue_saturation_limit * 30),
                    val_shift_limit=int(aug_config.hue_saturation_limit * 20),
                    p=0.5,
                )
            )

        # Blur
        if aug_config.gaussian_blur_prob > 0:
            transforms.append(
                A.GaussianBlur(
                    blur_limit=aug_config.gaussian_blur_limit,
                    p=aug_config.gaussian_blur_prob,
                )
            )

        # Dropout / Cutout
        if aug_config.coarse_dropout_prob > 0 and aug_config.coarse_dropout_max_holes > 0:
            hole_size = max(1, image_size // 8)
            transforms.append(
                A.CoarseDropout(
                    max_holes=aug_config.coarse_dropout_max_holes,
                    max_height=hole_size,
                    max_width=hole_size,
                    fill_value=0,
                    p=aug_config.coarse_dropout_prob,
                )
            )

    # Normalize (always applied)
    transforms.append(
        A.Normalize(
            mean=aug_config.normalize.mean,
            std=aug_config.normalize.std,
        )
    )

    # Convert to tensor (always last)
    transforms.append(ToTensorV2())

    return A.Compose(transforms, seed=seed)
