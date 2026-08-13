"""
Configuration System
====================

YAML-based configuration with typed dataclasses.
Every experiment parameter is config-driven — no hardcoded values.

Usage::

    from src.utils.config import load_config

    config = load_config("configs/resnet50.yaml")
    print(config.model.name)        # "resnet50"
    print(config.train.lr)          # 1e-4
    print(config.data.batch_size)   # 32
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

import torch


# ============================================================
# Nested Config Dataclasses
# ============================================================

@dataclass
class NormalizeConfig:
    """Image normalization parameters (ImageNet defaults)."""

    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class AugmentStageConfig:
    """Augmentation settings for a single stage (train / val / test)."""

    horizontal_flip: float = 0.0
    vertical_flip: float = 0.0
    rotation_limit: int = 0
    brightness_limit: float = 0.0
    contrast_limit: float = 0.0
    hue_saturation_limit: float = 0.0
    gaussian_blur_limit: Tuple[int, int] = (3, 7)
    gaussian_blur_prob: float = 0.0
    coarse_dropout_max_holes: int = 0
    coarse_dropout_prob: float = 0.0
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)


@dataclass
class AugmentConfig:
    """Full augmentation configuration for all stages."""

    train: AugmentStageConfig = field(default_factory=AugmentStageConfig)
    val: AugmentStageConfig = field(default_factory=AugmentStageConfig)
    test: AugmentStageConfig = field(default_factory=AugmentStageConfig)


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    name: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 6
    drop_rate: float = 0.0


@dataclass
class DataConfig:
    """Dataset and dataloader configuration."""

    root: str = "./data"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    epochs: int = 50
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    step_size: int = 10  # For StepLR
    step_gamma: float = 0.1  # For StepLR
    plateau_factor: float = 0.1  # For ReduceLROnPlateau
    plateau_patience: int = 5  # For ReduceLROnPlateau
    early_stopping_patience: int = 10
    mixed_precision: bool = True
    gradient_clip_value: float = 1.0
    label_smoothing: float = 0.1
    accumulation_steps: int = 1  # Gradient accumulation


@dataclass
class CheckpointConfig:
    """Checkpoint saving configuration."""

    save_dir: str = "./outputs/default"
    save_best: bool = True
    save_last: bool = True
    monitor: str = "val_accuracy"
    mode: str = "max"  # "max" for accuracy, "min" for loss


@dataclass
class LoggingConfig:
    """Logging configuration."""

    tensorboard: bool = True
    csv: bool = True
    console: bool = True


# ============================================================
# Top-Level Experiment Config
# ============================================================

@dataclass
class ExperimentConfig:
    """
    Top-level experiment configuration.

    Aggregates all sub-configs and resolves the device.
    This is the single source of truth for every experiment.
    """

    experiment_name: str = "default_experiment"
    seed: int = 42
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        """Resolve 'auto' device and ensure output directory exists."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Ensure checkpoint directory is created
        os.makedirs(self.checkpoint.save_dir, exist_ok=True)


# ============================================================
# Config Loading
# ============================================================

def _build_augment_stage(raw: Dict[str, Any]) -> AugmentStageConfig:
    """Build an AugmentStageConfig from a raw dictionary."""
    if not raw:
        return AugmentStageConfig()

    normalize_raw = raw.pop("normalize", None)
    normalize = NormalizeConfig(**normalize_raw) if normalize_raw else NormalizeConfig()

    # Filter out unknown keys to prevent TypeError
    valid_keys = {f.name for f in AugmentStageConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}

    return AugmentStageConfig(normalize=normalize, **filtered)


def _build_dataclass(cls: type, raw: Dict[str, Any]) -> Any:
    """Build a dataclass from a raw dictionary, ignoring unknown keys."""
    if not raw:
        return cls()

    valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return cls(**filtered)


def load_config(path: str) -> ExperimentConfig:
    """
    Load an experiment configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A fully resolved ``ExperimentConfig`` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    if raw is None:
        raw = {}

    # Build nested configs
    model = _build_dataclass(ModelConfig, raw.get("model", {}))
    data = _build_dataclass(DataConfig, raw.get("data", {}))
    train_cfg = _build_dataclass(TrainConfig, raw.get("train", {}))
    checkpoint = _build_dataclass(CheckpointConfig, raw.get("checkpoint", {}))
    logging_cfg = _build_dataclass(LoggingConfig, raw.get("logging", {}))

    # Dynamically inherit num_classes from configs/dataset.yaml
    try:
        ds_cfg = load_dataset_config()
        if ds_cfg and "num_classes" in ds_cfg:
            model.num_classes = ds_cfg["num_classes"]
    except Exception:
        pass

    # Augmentation requires special handling
    aug_raw = raw.get("augmentation", {})
    aug = AugmentConfig(
        train=_build_augment_stage(dict(aug_raw.get("train", {}) or {})),
        val=_build_augment_stage(dict(aug_raw.get("val", {}) or {})),
        test=_build_augment_stage(dict(aug_raw.get("test", {}) or {})),
    )

    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", "default_experiment"),
        seed=raw.get("seed", 42),
        device=raw.get("device", "auto"),
        model=model,
        data=data,
        augmentation=aug,
        train=train_cfg,
        checkpoint=checkpoint,
        logging=logging_cfg,
    )


def load_dataset_config(dataset_cfg_path: str = "configs/dataset.yaml") -> Dict[str, Any]:
    """
    Load dataset specification from YAML config file or auto-discover from data directories / outputs.

    Args:
        dataset_cfg_path: Path to dataset YAML configuration file.

    Returns:
        Dictionary containing dataset metadata, class names, display names, and paths.

    Raises:
        ValueError: If dataset classes cannot be resolved from YAML or filesystem.
    """
    cfg_path = Path(dataset_cfg_path)
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            ds_info = raw.get("dataset", {})
            classes = ds_info.get("classes", [])
            if classes:
                display_names = ds_info.get("display_names", {})
                for c in classes:
                    if c not in display_names:
                        display_names[c] = str(c).replace("_", " ")
                return {
                    "name": ds_info.get("name", "custom_dataset"),
                    "display_title": ds_info.get("display_title", "Custom Image Classification Benchmark Dataset"),
                    "task_type": ds_info.get("task_type", "multi_class_classification"),
                    "num_classes": len(classes),
                    "data_root": ds_info.get("data_root", "./data"),
                    "classes": classes,
                    "display_names": display_names,
                }
        except Exception as e:
            print(f"Warning: Failed to parse {dataset_cfg_path}: {e}")

    # Auto-discovery Strategy 1: Check class_to_idx.json in outputs/
    import glob
    import json
    outputs_dir = Path("outputs")
    for json_file in outputs_dir.glob("**/class_to_idx.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                c2i = json.load(f)
                discovered_classes = sorted(c2i.keys(), key=lambda k: c2i[k])
                if discovered_classes:
                    return {
                        "name": "auto_discovered_dataset",
                        "display_title": "Auto-Discovered Dataset (from Checkpoints)",
                        "task_type": "multi_class_classification",
                        "num_classes": len(discovered_classes),
                        "data_root": "./data",
                        "classes": discovered_classes,
                        "display_names": {c: c.replace("_", " ") for c in discovered_classes},
                    }
        except Exception:
            pass

    # Auto-discovery Strategy 2: Check folder names in data/train, data/val, data/test, or data/
    for sub in ["train", "val", "test", ""]:
        data_dir = Path("./data") / sub if sub else Path("./data")
        if data_dir.exists() and data_dir.is_dir():
            dirs = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if dirs:
                discovered_classes = sorted(dirs)
                return {
                    "name": "auto_discovered_dataset",
                    "display_title": "Auto-Discovered Dataset (from Data Directory)",
                    "task_type": "multi_class_classification",
                    "num_classes": len(discovered_classes),
                    "data_root": "./data",
                    "classes": discovered_classes,
                    "display_names": {c: c.replace("_", " ") for c in discovered_classes},
                }

    raise ValueError(
        f"Could not resolve dataset classes. Neither valid 'classes' list found in {dataset_cfg_path} "
        "nor subdirectories present under ./data/"
    )
