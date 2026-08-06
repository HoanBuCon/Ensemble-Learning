# Comprehensive Codebase Documentation: Tea Leaf Ensemble Research Framework

---

## 1. System Overview & Architectural Goals

The **Tea Leaf Ensemble Research Framework** is an enterprise-grade, modular, and reproducible Computer Vision research pipeline built using PyTorch. Designed specifically for academic thesis research and comparative studies in **Ensemble Learning**, the framework enforces strict separation between model training, prediction caching, and meta-learning/voting ensembles.

### Core Engineering Principles
- **Config-Driven Architecture**: Every experiment parameter (model selection, learning rate, augmentation intensity, loss smoothing, hardware device) is managed via YAML configuration files. No hardcoded magic values exist in the python logic.
- **Open/Closed Principle (OCP)**: Adding new model backbones (e.g., ConvNeXt, ViT) or custom ensemble techniques does not require altering any existing training or evaluation scripts.
- **Zero-PyTorch Ensemble Layer**: The ensemble algorithms (`Voting`, `Stacking`) operate purely on cached NumPy probability arrays (`.npy`), eliminating PyTorch overhead and enabling high-performance post-hoc experiment iterations.
- **Leakage-Free Stacking**: Out-of-Fold (OOF) prediction generation uses `StratifiedKFold` to ensure second-stage meta-learners (Logistic Regression, Random Forest, XGBoost) are trained exclusively on held-out predictions.
- **Cross-Platform Compatibility**: Fully compatible across Windows, Linux, macOS, Google Colab, and Kaggle Notebooks.

---

## 2. Directory Layout & File Overview

```
pipeline/
├── configs/                  # YAML experiment configuration files
│   ├── resnet50.yaml         # ResNet-50 baseline configuration
│   ├── densenet121.yaml      # DenseNet-121 configuration
│   ├── efficientnet_b0.yaml  # EfficientNet-B0 configuration
│   └── swin_tiny.yaml        # Swin Transformer Tiny configuration
│
├── src/                      # Main Python package
│   ├── __init__.py
│   │
│   ├── datasets/             # Data loading & transformation package
│   │   ├── __init__.py
│   │   ├── dataset.py        # ImageFolder dataset wrapper & DataLoader factory
│   │   └── transforms.py     # Config-driven Albumentations augmentation pipeline
│   │
│   ├── models/               # Model backbones & factory
│   │   ├── __init__.py
│   │   └── factory.py        # Registry-based ModelFactory (@register_model)
│   │
│   ├── engine/               # Training, evaluation & checkpointing engine
│   │   ├── __init__.py
│   │   ├── trainer.py        # Generic unified Trainer (AMP, Early Stopping, Warmup)
│   │   ├── evaluator.py      # Standalone model inference & report evaluator
│   │   └── checkpoint.py     # CheckpointManager (best/last weight persistence)
│   │
│   ├── ensemble/             # Ensemble Learning algorithms
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract EnsembleBase class
│   │   ├── voting.py         # Hard Voting, Soft Voting & Weighted Voting
│   │   ├── stacking.py       # Stacking Ensemble (Meta-learner wrapper)
│   │   └── oof.py            # Out-of-Fold (OOF) prediction generator
│   │
│   └── utils/                # Utility & helper functions
│       ├── __init__.py
│       ├── config.py         # Dataclass-based YAML parser
│       ├── logger.py         # Rich console logging & CSV logger
│       ├── metrics.py        # Classification metrics calculation (Accuracy, F1, CM)
│       ├── visualization.py  # Loss/Accuracy curves & Confusion Matrix plotting
│       └── reproducibility.py# Seed setting & deterministic CUDA configuration
│
├── train.py                  # Entry point for single-model training
├── evaluate.py               # Entry point for standalone evaluation
├── run_experiments.py        # Multi-experiment runner & automated summary table generator
├── requirements.txt          # Exact pinned environment dependencies
├── README.md                 # Bilingual documentation (English & Tiếng Việt)
└── .gitignore                # Git exclusion rules
```

---

## 3. Module Breakdown & Technical Specification

### 3.1 `src/utils/` — Utility Package

#### `config.py`
- Parses YAML files into typed dataclasses (`ExperimentConfig`, `ModelConfig`, `DataConfig`, `TrainConfig`, `AugmentConfig`, `CheckpointConfig`, `LoggingConfig`).
- Implements `load_config(path)` with automatic fallback handling and `auto` device resolution (`cuda` or `cpu`).

#### `reproducibility.py`
- Provides `set_seed(seed)` to lock seeds across Python `random`, `numpy`, and `torch` (CPU & CUDA).
- Configures deterministic CuDNN behavior (`cudnn.deterministic = True`, `cudnn.benchmark = False`).
- Provides `seed_worker` and `get_generator()` to ensure multi-process `DataLoader` shuffling is completely reproducible.

#### `metrics.py`
- Implements `compute_metrics(y_true, y_pred, class_names)` using `scikit-learn`.
- Calculates Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1-Score (Macro/Weighted), Confusion Matrix, and detailed Per-class Classification Reports.

#### `visualization.py`
- Generates publication-ready figures using `matplotlib` and `seaborn`.
- Functions include `plot_training_curves()` (Loss & Accuracy), `plot_confusion_matrix()`, and `plot_comparison_bar()`.

#### `logger.py`
- Sets up Rich-formatted console logging with clean table displays for training progress.
- Implements `CSVLogger` for real-time per-epoch epoch metric recording to `history.csv`.

---

### 3.2 `src/models/` — Model Factory & Backbones

#### `factory.py`
- Uses a global registry dictionary `_MODEL_REGISTRY` and decorator `@register_model(name)`.
- `create_model(model_name, pretrained, num_classes, **kwargs)` creates and returns any registered backbone initialized with the custom classification head.
- **Built-in Registered Models**:
  1. `resnet50`: Torchvision ResNet-50 with replaced `fc` linear layer.
  2. `densenet121`: Torchvision DenseNet-121 with replaced `classifier` linear layer.
  3. `efficientnet_b0`: `timm` EfficientNet-B0.
  4. `swin_tiny`: `timm` Swin Transformer Tiny (`swin_tiny_patch4_window7_224`).

---

### 3.3 `src/datasets/` — Data Pipeline

#### `transforms.py`
- `build_transforms(aug_config, image_size, stage)`: Constructs an `albumentations.Compose` pipeline.
- Includes Spatial transforms (Horizontal/Vertical Flip, Rotation), Color jitter (Brightness, Contrast, Hue/Saturation), Blur (GaussianBlur), and Dropout (`CoarseDropout`/Cutout).
- Always appends `A.Normalize` and `ToTensorV2()`.

#### `dataset.py`
- `ImageFolderDataset`: Subclasses `torch.utils.data.Dataset`, wrapping `torchvision.datasets.ImageFolder` while reading images via `cv2` for fast NumPy-native Albumentations compatibility.
- `create_dataloaders(config)`: Builds `train_loader`, `val_loader`, and optional `test_loader` configured with seeded worker initialization.

---

### 3.4 `src/engine/` — Core Training & Evaluation Engine

#### `trainer.py`
- `Trainer`: Generic, model-agnostic training controller.
- Features:
  - Automatic Mixed Precision (`torch.cuda.amp.autocast`, `GradScaler`).
  - Learning rate warmup & LR schedulers (`CosineAnnealingLR`, `StepLR`, `ReduceLROnPlateau`, `OneCycleLR`).
  - Gradient clipping & Gradient accumulation.
  - Early stopping based on monitored metric (`val_accuracy` or `val_loss`).
  - Integrated CSV, TensorBoard, and Rich console logging.

#### `checkpoint.py`
- `CheckpointManager`: Handles saving `best_model.pth` and `last_model.pth`.
- Serializes model `state_dict`, optimizer `state_dict`, scheduler `state_dict`, current epoch, and training history JSON.

#### `evaluator.py`
- `run_inference(model, dataloader, device)`: Runs model inference without gradients and extracts logits, probabilities, predictions, labels, and timing.
- `evaluate_model()`: Computes metrics, saves `.npy` probability caches (`logits.npy`, `probabilities.npy`, `predictions.npy`, `labels.npy`), saves `metrics.json`, `classification_report.txt`, and draws confusion matrices.

---

### 3.5 `src/ensemble/` — Ensemble Learning Package

#### `base.py`
- `EnsembleBase`: Abstract base class defining `fit()`, `predict()`, `predict_proba()`, and `evaluate()`.

#### `voting.py`
- `HardVoting`: Computes majority vote across model predictions using `scipy.stats.mode`.
- `SoftVoting`: Computes average probability vectors across models and returns `argmax`.
- `WeightedVoting`: Computes weighted average probability vectors. Includes an automated `fit()` method that performs grid-search optimization to identify optimal model weights.

#### `stacking.py`
- `StackingEnsemble`: Meta-learner classifier trained on concatenated probability vectors `(N, M * C)`.
- Supports Meta-Learners: `logistic_regression`, `random_forest`, and `xgboost`. Includes model serialization methods `save()` and `load()`.

#### `oof.py`
- `OOFGenerator`: Uses `StratifiedKFold` (5-fold default) to train models on `K-1` folds and collect out-of-fold predictions on held-out splits.
- Eliminates data leakage during Stacking meta-learner training.

---

## 4. Execution Flow & Lifecycle

```
[YAML Config] --> load_config
                       │
                       ▼
            set_seed & Setup Logger
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
create_dataloaders        create_model (ModelFactory)
         │                           │
         └─────────────┬─────────────┘
                       ▼
                    Trainer
                       │
                       ▼
        Training Loop (AMP + Warmup + EarlyStop)
                       │
                       ▼
          CheckpointManager: best_model.pth
                       │
                       ▼
          Evaluator: run_inference
                       │
                       ▼
        Save .npy Caches & Confusion Matrix
                       │
                       ▼
        Ensemble Module: Voting & Stacking
```

---

## 5. Artifacts Output Structure

After executing experiments, outputs are automatically generated into structured folders:

```
outputs/<experiment_name>/
├── best_model.pth              # Checkpoint with highest validation metric
├── last_model.pth              # Final epoch checkpoint
├── history.csv                 # Per-epoch CSV metrics log
├── training_history.json       # Metric history in JSON format
├── metrics.json                # Complete evaluation metrics
├── classification_report.txt   # Detailed text report per class
├── confusion_matrix.png        # Rendered confusion matrix image
├── loss_curve.png              # Loss progression plot
├── accuracy_curve.png          # Accuracy progression plot
├── probabilities.npy           # Raw softmax probability matrix (N, C)
├── predictions.npy             # Argmax class prediction array (N,)
├── labels.npy                  # Ground truth labels array (N,)
├── logits.npy                  # Raw unnormalized model outputs (N, C)
└── tensorboard/                # TensorBoard event logs
```

---

## 6. Verification & Validation Summary

All components were systematically tested and verified:

1. **Syntax & Compilation**: All 24 Python source files pass `py_compile` without warnings or syntax errors.
2. **Model Instantiation**: Verified that all registered models (`resnet50`, `densenet121`, `efficientnet_b0`, `swin_tiny`) instantiate correctly with specified output class counts.
3. **Ensemble Methods Execution**:
   - `HardVoting`: Passed verification.
   - `SoftVoting`: Passed verification.
   - `WeightedVoting`: Passed verification with grid-search weight optimization.
   - `StackingEnsemble`: Verified using Logistic Regression, Random Forest, and XGBoost.
4. **Environment Isolation**: Virtual environment created with exact dependency constraints frozen into `requirements.txt`.
