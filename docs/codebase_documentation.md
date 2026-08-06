# Comprehensive Codebase Documentation / Tài liệu Chi tiết mã nguồn

**Project:** Tea Leaf Ensemble Research Framework  
**Target:** System Architecture, Data Pipeline, Master CLI, Ensemble Engine & Exhaustive File Specifications  

[English](#english) | [Tiếng Việt](#tiếng-việt)

---

<a name="english"></a>
## English Documentation

### 1. System Overview & Architectural Goals

The **Tea Leaf Ensemble Research Framework** is an enterprise-grade, modular, and reproducible Computer Vision research pipeline built using PyTorch. Designed specifically for academic thesis research and comparative studies in **Ensemble Learning**, the framework enforces strict separation between base model training, prediction caching, and meta-learning/voting ensembles.

#### Core Engineering Principles
- **Unified Master CLI (`main.py`)**: Centralized command-line entrypoint for orchestrating training, evaluation, benchmarking, ensemble, and reporting with self-descriptive subcommand names.
- **Modular Scripts Folder (`scripts/`)**: All single-purpose execution scripts (`train.py`, `evaluate.py`, `run_experiments.py`, `run_ensemble_eval.py`, `generate_comparison.py`) are organized neatly inside `scripts/`.
- **Config-Driven Architecture**: Every experiment parameter (model selection, learning rate, augmentation intensity, label smoothing, hardware device) is managed via YAML configuration files in `configs/`. No hardcoded magic numbers exist in the python code.
- **Open/Closed Principle (OCP)**: Adding new model backbones (e.g., ConvNeXt, ViT) or custom ensemble techniques requires zero modifications to existing training or evaluation code.
- **Zero-PyTorch Ensemble Layer**: Ensemble algorithms (`Voting`, `Stacking`) operate purely on cached NumPy probability arrays (`.npy`), eliminating PyTorch CUDA overhead and allowing rapid post-hoc experiment iterations.
- **Class Index Alignment Verification**: `EnsembleBase.validate_class_mappings()` ensures all base models share identical `class_to_idx` mappings prior to ensembling, preventing silent column misalignment bugs.
- **Leakage-Free Stacking**: Out-of-Fold (OOF) prediction generation uses a unified `split_seed` with `StratifiedKFold` to ensure second-stage meta-learners (Logistic Regression, Random Forest, XGBoost) are trained exclusively on held-out predictions.
- **Dual Stacking Execution Modes**: Supports both `mode="val"` (Fast Prototyping ~30 sec) and `mode="oof"` (Gold Standard 5-Fold Evaluation ~10-13 hrs) with an interactive CLI prompt menu.

---

### 2. Directory Layout & File Overview

```
pipeline/
├── main.py                   # Master CLI controller
├── configs/                  # YAML experiment configuration files
│   ├── resnet50.yaml         # ResNet-50 baseline configuration
│   ├── densenet121.yaml      # DenseNet-121 configuration
│   ├── efficientnet_b0.yaml  # EfficientNet-B0 configuration
│   └── swin_tiny.yaml        # Swin Transformer Tiny configuration
│
├── scripts/                  # Modular entrypoint scripts
│   ├── train.py              # Single-model training script
│   ├── evaluate.py           # Standalone evaluation script
│   ├── run_experiments.py   # Multi-model benchmark runner
│   ├── run_ensemble_eval.py  # Ensemble evaluation pipeline
│   └── generate_comparison.py# Comparison report & plot generator
│
├── src/                      # Main Python package
│   ├── __init__.py           # Package root initialization
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
│   │   ├── base.py           # Abstract EnsembleBase class & class_to_idx validator
│   │   ├── voting.py         # Hard Voting, Soft Voting & Weighted Voting
│   │   ├── stacking.py       # Stacking Ensemble (Meta-learner wrapper)
│   │   └── oof.py            # Unified-seed Out-of-Fold (OOF) prediction generator
│   │
│   └── utils/                # Utility & helper functions
│       ├── __init__.py
│       ├── config.py         # Dataclass-based YAML parser
│       ├── logger.py         # Rich console logging & CSV logger
│       ├── metrics.py        # Classification metrics calculation (Accuracy, F1, CM)
│       ├── report.py         # Consolidated parameter counter, model sizing, & history parser
│       ├── visualization.py  # Loss/Accuracy curves & Confusion Matrix plotting
│       └── reproducibility.py# Seed setting & deterministic CUDA configuration
│
├── requirements.txt          # Frozen dependencies
├── README.md                 # Bilingual documentation (English & Tiếng Việt)
└── docs/
    └── codebase_documentation.md # Exhaustive technical reference document
```

---

### 3. Exhaustive File-by-File Specifications

#### 3.1 Root & CLI Entrypoints

- **`main.py`**
  - **Purpose**: Unified Master CLI Controller and central entrypoint for the framework.
  - **Key Classes / Functions**: `build_parser()`, `prompt_ensemble_mode()`, `resolve_config_paths()`, `main()`.
  - **Commands Supported**: `all-in-one` (`pipeline`, `full-pipeline`), `train-all` (`benchmark`, `train-base`), `train` (`train-single`), `evaluate`, `ensemble`, `report`.
  - **Description**: Parses command-line subcommands, provides an interactive selection prompt for `val` vs `oof` mode, and delegates execution to module scripts.

- **`requirements.txt`**
  - **Purpose**: Dependency locking specification.
  - **Description**: Contains exact version requirements for PyTorch, torchvision, timm, albumentations, scikit-learn, xgboost, pandas, numpy, matplotlib, seaborn, pyyaml, rich, and tabulate.

- **`README.md`**
  - **Purpose**: Primary project landing page and quick start guide in both English and Tiếng Việt.
  - **Description**: Documents features, installation commands, dataset structure, CLI workflows, and model extension steps.

#### 3.2 Configuration Files (`configs/`)

- **`configs/resnet50.yaml`**
  - **Purpose**: Experiment configuration for ResNet-50 backbone.
  - **Key Sections**: `model` (`name: resnet50`, `num_classes: 6`), `dataset`, `training` (`epochs: 50`, `lr: 0.0001`, `batch_size: 32`), `checkpoint`.

- **`configs/densenet121.yaml`**
  - **Purpose**: Experiment configuration for DenseNet-121 backbone.
  - **Key Sections**: Model parameters, learning rate schedule, data augmentation parameters, and output paths for DenseNet-121.

- **`configs/efficientnet_b0.yaml`**
  - **Purpose**: Experiment configuration for EfficientNet-B0 backbone.
  - **Key Sections**: Model resolution, image size (224x224), optimizer configuration (AdamW), and checkpoint directories for EfficientNet-B0.

- **`configs/swin_tiny.yaml`**
  - **Purpose**: Experiment configuration for Swin Transformer Tiny backbone.
  - **Key Sections**: Swin-specific image size (224x224), patch size, initial learning rate with warmup schedule, and output paths.

#### 3.3 Execution Scripts (`scripts/`)

- **`scripts/train.py`**
  - **Purpose**: Standalone entry script to train a single backbone model.
  - **Key Functions**: `train(config_path, resume)`.
  - **Description**: Loads YAML config, instantiates `Trainer`, runs training loop, automatically evaluates `best_model.pth` upon completion, and saves probability cache arrays (`.npy`).

- **`scripts/evaluate.py`**
  - **Purpose**: Standalone entry script to evaluate a trained model checkpoint.
  - **Key Functions**: `evaluate(config_path, checkpoint_path, split)`.
  - **Description**: Loads checkpoint (`best_model.pth` or specified path like `last_model.pth`), executes inference on `test` or `val` dataloaders, computes classification metrics, and outputs serialized predictions.

- **`scripts/run_experiments.py`**
  - **Purpose**: Benchmark runner script for training and evaluating all base models sequentially.
  - **Key Functions**: `run_experiments(config_paths, mode)`, `get_experiment_status()`, `display_status_table()`.
  - **Description**: Scans model output directories, checks completion status, skips/resumes/re-trains models based on `--mode`, and triggers comparison report generation at the end.

- **`scripts/run_ensemble_eval.py`**
  - **Purpose**: Ensemble evaluation pipeline script.
  - **Key Functions**: `run_ensemble_evaluation(mode, outputs_dir)`, `get_latest_model_dirs()`, `generate_val_predictions_if_missing()`.
  - **Description**: Evaluates Hard Voting, Soft Voting, Weighted Voting, and Stacking meta-learners (Logistic Regression, Random Forest, XGBoost) in `val` or `oof` modes.

- **`scripts/generate_comparison.py`**
  - **Purpose**: Standalone report & metric bar plot generator.
  - **Key Functions**: `generate_base_comparison_report(outputs_dir, config_paths)`.
  - **Description**: Extracts Validation and Test metrics (`Val_Accuracy`, `Val_Precision`, `Val_Recall`, `Val_F1_Score`, `Test_Accuracy`, `Test_Precision`, `Test_Recall`, `Test_F1_Score`), formats `%` strings, saves `comparison_table.csv` and `comparison_table.md`, and plots metric comparison bar charts.

#### 3.4 Data & Transforms Package (`src/datasets/`)

- **`src/datasets/__init__.py`**
  - **Purpose**: Package initialization exporting dataset loaders and transform factories.

- **`src/datasets/dataset.py`**
  - **Purpose**: Custom PyTorch dataset wrapper and DataLoader factory.
  - **Key Classes / Functions**: `TeaLeafDataset`, `create_dataloaders(config)`.
  - **Description**: Wraps torchvision `ImageFolder` dataset, attaches Albumentations transforms, and creates seeded training, validation, and test DataLoaders.

- **`src/datasets/transforms.py`**
  - **Purpose**: Config-driven Albumentations image transformation pipeline.
  - **Key Functions**: `get_train_transforms(config)`, `get_val_transforms(config)`.
  - **Description**: Applies random rotation, horizontal/vertical flip, color jitter, shift-scale-rotate, resize, and ImageNet normalization to training samples, and deterministic resize/normalization to val/test samples.

#### 3.5 Models Package (`src/models/`)

- **`src/models/__init__.py`**
  - **Purpose**: Package initialization exporting model factory functions.

- **`src/models/factory.py`**
  - **Purpose**: Registry-based Model Factory for modular architecture additions.
  - **Key Classes / Functions**: `ModelFactory`, `register_model(name)`, `create_model(model_name, pretrained, num_classes)`.
  - **Description**: Maintains global `_MODEL_REGISTRY` dict. Decorates model constructor functions (`_resnet50`, `_densenet121`, `_efficientnet_b0`, `_swin_tiny`) to allow adding backbones without modifying engine code.

#### 3.6 Engine Package (`src/engine/`)

- **`src/engine/__init__.py`**
  - **Purpose**: Package initialization exporting training engine components.

- **`src/engine/trainer.py`**
  - **Purpose**: Generic unified PyTorch Trainer controller.
  - **Key Classes / Functions**: `Trainer`.
  - **Description**: Handles complete training lifecycle: Mixed Precision (AMP `GradScaler`), Gradient Clipping, Gradient Accumulation, Learning Rate scheduling with Warmup, Early Stopping, and per-epoch CSV history logging.

- **`src/engine/evaluator.py`**
  - **Purpose**: Standalone inference runner and evaluation serializer.
  - **Key Functions**: `evaluate_model()`, `run_inference()`, `save_predictions()`.
  - **Description**: Runs model inference over DataLoader, computes accuracy/precision/recall/f1, plots confusion matrix, and serializes `test_probabilities.npy`, `test_labels.npy`, `val_probabilities.npy`, `val_labels.npy`, and `class_to_idx.json`.

- **`src/engine/checkpoint.py`**
  - **Purpose**: Checkpoint persistence manager.
  - **Key Classes / Functions**: `CheckpointManager`.
  - **Description**: Manages saving and loading of `best_model.pth` (highest val accuracy) and `last_model.pth` (latest epoch), serializing model state_dict, optimizer state, epoch, and metric history.

#### 3.7 Ensemble Package (`src/ensemble/`)

- **`src/ensemble/__init__.py`**
  - **Purpose**: Package initialization exporting ensemble classes.

- **`src/ensemble/base.py`**
  - **Purpose**: Abstract base class and class mapping validator.
  - **Key Classes / Functions**: `EnsembleBase`, `validate_class_mappings(class_mappings)`.
  - **Description**: Enforces `fit()`, `predict()`, and `evaluate()` abstract signatures, and validates identical `class_to_idx` mappings across all base models.

- **`src/ensemble/voting.py`**
  - **Purpose**: Voting ensemble implementations (Hard, Soft, Weighted).
  - **Key Classes / Functions**: `HardVoting`, `SoftVoting`, `WeightedVoting`.
  - **Description**: Implements majority vote (`HardVoting`), probability vector averaging (`SoftVoting`), and grid-search weight optimization ($\sum w_i = 1$) (`WeightedVoting`).

- **`src/ensemble/stacking.py`**
  - **Purpose**: Stacking Meta-Learner wrapper.
  - **Key Classes / Functions**: `StackingEnsemble`.
  - **Description**: Concatenates probability prediction matrices into shape $(N, M \cdot C)$ and trains meta-learners (`logistic_regression`, `random_forest`, `xgboost`) using fixed random seeds.

- **`src/ensemble/oof.py`**
  - **Purpose**: Out-of-Fold (OOF) prediction generator.
  - **Key Classes / Functions**: `OOFGenerator`.
  - **Description**: Performs 5-Fold Stratified K-Fold cross-validation (`split_seed=42`) to generate leak-free out-of-fold probability matrices for second-stage meta-learner training.

#### 3.8 Utilities Package (`src/utils/`)

- **`src/utils/__init__.py`**
  - **Purpose**: Package initialization exporting utility helpers.

- **`src/utils/config.py`**
  - **Purpose**: Dataclass-based YAML configuration parser.
  - **Key Classes / Functions**: `load_config(config_path)`, dataclasses (`ExperimentConfig`, `ModelConfig`, `DatasetConfig`, `TrainingConfig`, `CheckpointConfig`).
  - **Description**: Parses YAML into strongly typed Python dataclasses and auto-resolves hardware device (`cuda` or `cpu`).

- **`src/utils/logger.py`**
  - **Purpose**: Logging and progress bar utilities.
  - **Key Classes / Functions**: `get_logger()`, `CSVLogger`.
  - **Description**: Configures Rich console logger for beautiful formatted terminal logging and manages epoch-by-epoch CSV history metrics logging.

- **`src/utils/metrics.py`**
  - **Purpose**: Classification performance evaluation metrics.
  - **Key Functions**: `compute_metrics(y_true, y_pred, class_names)`.
  - **Description**: Computes overall Accuracy, Macro/Weighted Precision, Recall, F1-Score, Confusion Matrix, and detailed per-class metrics dictionary.

- **`src/utils/report.py`**
  - **Purpose**: Metrics extraction, parameter counting, and model sizing utilities.
  - **Key Functions**: `count_parameters()`, `get_model_size_mb()`, `extract_model_history_info()`, `extract_model_val_metrics()`.
  - **Description**: Inspects model checkpoints and history logs to calculate parameter counts (in Millions), model disk size (MB), best epoch, and validation metrics (`Val_Accuracy`, `Val_Precision`, `Val_Recall`, `Val_F1_Score`).

- **`src/utils/visualization.py`**
  - **Purpose**: Metric plotting and data visualization.
  - **Key Functions**: `plot_history()`, `plot_confusion_matrix()`, `plot_comparison_bar()`.
  - **Description**: Generates per-epoch loss/accuracy curves, Seaborn confusion matrix heatmaps, and metric comparison bar charts across base models.

- **`src/utils/reproducibility.py`**
  - **Purpose**: Seed locking and CUDA determinism enforcer.
  - **Key Functions**: `set_seed(seed)`.
  - **Description**: Sets identical seed across Python `random`, `numpy`, `torch`, `torch.cuda`, and configures CuDNN deterministic operations (`torch.backends.cudnn.deterministic = True`).

---

### 4. Output Artifacts Directory Structure

Outputs are saved in dedicated structured directories:

```
outputs/<experiment_name>/
├── best_model.pth              # Checkpoint with highest validation accuracy
├── last_model.pth              # Final epoch checkpoint
├── history.csv                 # Per-epoch metrics CSV
├── class_to_idx.json           # Class-to-index mapping dictionary
├── test_probabilities.npy      # Test set softmax probabilities (N, C)
├── test_labels.npy             # Test set ground truth labels (N,)
├── val_probabilities.npy       # Validation set probabilities (N, C)
├── val_labels.npy              # Validation set labels (N,)
├── metrics.json                # Complete evaluation metrics JSON
├── classification_report.txt   # Detailed text classification report
└── confusion_matrix.png        # Confusion matrix plot
```

---

<a name="tiếng-việt"></a>
## Tiếng Việt Documentation (Tài liệu Tiếng Việt)

### 1. Tổng quan Hệ thống & Mục tiêu Kiến trúc

**Tea Leaf Ensemble Research Framework** là khung nghiên cứu Thị giác máy tính (Computer Vision) cấp doanh nghiệp, dạng mô-đun và tái lập cao được xây dựng trên PyTorch. Được thiết kế chuyên biệt cho luận văn và các bài báo nghiên cứu về **Học kết hợp (Ensemble Learning)**, hệ thống phân tách nghiêm ngặt giữa quá trình huấn luyện mô hình cơ sở (base models), lưu trữ xác suất dự đoán (prediction caching) và thuật toán kết hợp (voting/stacking).

#### Nguyên lý Kỹ thuật Cốt lõi
- **Bộ điều khiển Master CLI thống nhất (`main.py`)**: Giao diện dòng lệnh tập trung tại root điều khiển toàn bộ các thao tác train, evaluate, benchmark, ensemble và report với tên câu lệnh tự giải thích tác vụ.
- **Thư mục Scripts mô-đun (`scripts/`)**: Tất cả các script chạy đơn nhiệm (`train.py`, `evaluate.py`, `run_experiments.py`, `run_ensemble_eval.py`, `generate_comparison.py`) được gom gọn gàng trong `scripts/`.
- **Kiến trúc Quản lý bằng Config**: Tất cả các tham số (kiến trúc mô hình, tốc độ học, mức độ tăng cường dữ liệu, nhãn làm mịn, thiết bị phần cứng) được định nghĩa qua tệp YAML trong `configs/`. Không có tham số cứng (magic numbers) trong mã nguồn Python.
- **Nguyên lý Đóng/Mở (Open/Closed Principle - OCP)**: Thêm mô hình backbone mới (ConvNeXt, ViT...) hay thuật toán ensemble mới mà không cần sửa bất kỳ dòng code huấn luyện hay đánh giá sẵn có nào.
- **Lớp Ensemble Không Phụ thuộc PyTorch**: Các thuật toán Ensemble (`Voting`, `Stacking`) hoạt động hoàn toàn trên các mảng xác suất NumPy (`.npy`), loại bỏ chi phí tính toán GPU PyTorch và cho phép thử nghiệm kết hợp cực nhanh.
- **Xác minh Đồng bộ Thứ tự Lớp (Class Mapping Validation)**: Phương thức `EnsembleBase.validate_class_mappings()` tự động kiểm tra tính thống nhất của `class_to_idx` giữa tất cả các mô hình cơ sở trước khi ensemble, ngăn chặn lỗi lệch cột nhãn.
- **Stacking Không Rò rỉ Dữ liệu (Leakage-Free)**: Bộ tạo Out-of-Fold (OOF) sử dụng `StratifiedKFold` với `split_seed` dùng chung để đảm bảo meta-learner (Logistic Regression, Random Forest, XGBoost) chỉ được huấn luyện trên dự đoán held-out.
- **Hai Chế độ Đánh giá Stacking**: Hỗ trợ cả `mode="val"` (Thử nghiệm nhanh ~30 giây) và `mode="oof"` (Đánh giá chuẩn báo cáo luận văn/bài báo 5-Fold ~10–13 giờ) có menu chọn tương tác.

---

### 2. Cấu trúc Thư mục & Tệp tin

```
pipeline/
├── main.py                   # Bộ điều khiển Master CLI duy nhất
├── configs/                  # Các tệp cấu hình YAML
│   ├── resnet50.yaml         # Cấu hình ResNet-50 baseline
│   ├── densenet121.yaml      # Cấu hình DenseNet-121
│   ├── efficientnet_b0.yaml  # Cấu hình EfficientNet-B0
│   └── swin_tiny.yaml        # Cấu hình Swin Transformer Tiny
│
├── scripts/                  # Các script entrypoint mô-đun gọn gàng
│   ├── train.py              # Script huấn luyện đơn mô hình
│   ├── evaluate.py           # Script đánh giá mô hình độc lập
│   ├── run_experiments.py   # Bộ chạy thử nghiệm tự động nhiều mô hình
│   ├── run_ensemble_eval.py  # Script đánh giá Ensemble
│   └── generate_comparison.py# Script xuất báo cáo & vẽ đồ thị so sánh
│
├── src/                      # Package Python chính
│   ├── __init__.py           # Khởi tạo package chính
│   │
│   ├── datasets/             # Package tải & biến đổi dữ liệu
│   │   ├── __init__.py
│   │   ├── dataset.py        # Wrapper ImageFolder & DataLoader factory
│   │   └── transforms.py     # Pipeline biến đổi dữ liệu Albumentations
│   │
│   ├── models/               # Model backbones & factory
│   │   ├── __init__.py
│   │   └── factory.py        # ModelFactory dạng Registry (@register_model)
│   │
│   ├── engine/               # Bộ huấn luyện, đánh giá & lưu checkpoint
│   │   ├── __init__.py
│   │   ├── trainer.py        # Trainer dùng chung (AMP, Early Stopping, Warmup)
│   │   ├── evaluator.py      # Bộ chạy dự đoán & xuất báo cáo độc lập
│   │   └── checkpoint.py     # CheckpointManager (Lưu trọng số best/last)
│   │
│   ├── ensemble/             # Các thuật toán Ensemble Learning
│   │   ├── __init__.py
│   │   ├── base.py           # Lớp cơ sở EnsembleBase & xác minh class_to_idx
│   │   ├── voting.py         # Hard Voting, Soft Voting & Weighted Voting
│   │   ├── stacking.py       # Stacking Ensemble (Meta-learner wrapper)
│   │   └── oof.py            # Bộ tạo Out-of-Fold (OOF) seed đồng bộ
│   │
│   └── utils/                # Các tiện ích bổ trợ
│       ├── __init__.py
│       ├── config.py         # Bộ đọc file YAML dạng Dataclass
│       ├── logger.py         # Logger giao diện Rich & CSV logger
│       ├── metrics.py        # Tính chỉ số phân loại (Accuracy, F1, CM)
│       ├── report.py         # Tiện ích đo đạc mô hình & trích xuất chỉ số
│       ├── visualization.py  # Vẽ đồ thị Loss/Accuracy & Confusion Matrix
│       └── reproducibility.py# Cố định seed toàn cục & GPU CUDA
│
├── requirements.txt          # Danh sách thư viện khóa phiên bản
├── README.md                 # Tài liệu song ngữ (English & Tiếng Việt)
└── docs/
    └── codebase_documentation.md # Tài liệu kỹ thuật chi tiết toàn bộ mã nguồn
```

---

### 3. Mô tả Chi tiết Tất cả các Tệp tin trong Mã nguồn

#### 3.1 Tệp tin Root & Script Điều khiển CLI

- **`main.py`**
  - **Mục đích**: Bộ điều khiển Master CLI tập trung cho toàn bộ hệ thống.
  - **Hàm/Lớp chính**: `build_parser()`, `prompt_ensemble_mode()`, `resolve_config_paths()`, `main()`.
  - **Lệnh hỗ trợ**: `all-in-one` (`pipeline`), `train-all` (`benchmark`), `train` (`train-single`), `evaluate`, `ensemble`, `report`.
  - **Mô tả**: Phân tích lệnh dòng lệnh, hiển thị menu tương tác chọn `val` hay `oof` mode, và gọi các script thực thi tương ứng.

- **`requirements.txt`**
  - **Mục đích**: Khóa phiên bản thư viện phụ thuộc.
  - **Mô tả**: Khai báo phiên bản PyTorch, torchvision, timm, albumentations, scikit-learn, xgboost, pandas, numpy, matplotlib, seaborn, pyyaml, rich, tabulate.

- **`README.md`**
  - **Mục đích**: Trang hướng dẫn sử dụng nhanh song ngữ English & Tiếng Việt.
  - **Mô tả**: Mô tả tính năng, hướng dẫn cài đặt, cấu trúc dữ liệu, các lệnh CLI và hướng dẫn mở rộng mô hình.

#### 3.2 Các tệp cấu hình (`configs/`)

- **`configs/resnet50.yaml`**
  - **Mục đích**: Tệp cấu hình cho mô hình ResNet-50.
  - **Nội dung**: Khai báo `name: resnet50`, `num_classes: 6`, tốc độ học `lr: 0.0001`, `epochs: 50`, kích thước batch và đường dẫn lưu checkpoint.

- **`configs/densenet121.yaml`**
  - **Mục đích**: Tệp cấu hình cho mô hình DenseNet-121.
  - **Nội dung**: Định nghĩa tham số huấn luyện, lịch trình giảm LR, và đường dẫn đầu ra cho DenseNet-121.

- **`configs/efficientnet_b0.yaml`**
  - **Mục đích**: Tệp cấu hình cho mô hình EfficientNet-B0.
  - **Nội dung**: Định nghĩa độ phân giải ảnh (224x224), bộ tối ưu AdamW, và đường dẫn đầu ra cho EfficientNet-B0.

- **`configs/swin_tiny.yaml`**
  - **Mục đích**: Tệp cấu hình cho mô hình Swin Transformer Tiny.
  - **Nội dung**: Định nghĩa các tham số đặc thù của Vision Transformer (patch size, warmup lr schedule) và đường dẫn đầu ra.

#### 3.3 Các script thực thi mô-đun (`scripts/`)

- **`scripts/train.py`**
  - **Mục đích**: Script huấn luyện một kiến trúc mô hình đơn lẻ.
  - **Hàm chính**: `train(config_path, resume)`.
  - **Mô tả**: Tải file YAML, khởi tạo `Trainer`, chạy vòng lặp train, tự động đánh giá `best_model.pth` sau khi kết thúc và lưu cache xác suất dự đoán (`.npy`).

- **`scripts/evaluate.py`**
  - **Mục đích**: Script đánh giá độc lập một checkpoint mô hình đã huấn luyện.
  - **Hàm chính**: `evaluate(config_path, checkpoint_path, split)`.
  - **Mô tả**: Tải checkpoint (`best_model.pth` hoặc đường dẫn chỉ định như `last_model.pth`), chạy suy luận trên tập `test` hoặc `val`, tính toán metrics và xuất báo cáo.

- **`scripts/run_experiments.py`**
  - **Mục đích**: Script chạy thử nghiệm so sánh hàng loạt tất cả các mô hình base.
  - **Hàm chính**: `run_experiments(config_paths, mode)`, `get_experiment_status()`, `display_status_table()`.
  - **Mô tả**: Quét thư mục đầu ra, kiểm tra trạng thái mô hình, bỏ qua/tiếp tục/train mới theo cờ `--mode`, và tự động xuất bảng so sánh cuối cùng.

- **`scripts/run_ensemble_eval.py`**
  - **Mục đích**: Script chạy đánh giá pipeline các phương pháp Ensemble.
  - **Hàm chính**: `run_ensemble_evaluation(mode, outputs_dir)`, `get_latest_model_dirs()`, `generate_val_predictions_if_missing()`.
  - **Mô tả**: Thực thi Hard Voting, Soft Voting, Weighted Voting và các Meta-learner Stacking (Logistic Regression, Random Forest, XGBoost) theo 2 chế độ `val` hoặc `oof`.

- **`scripts/generate_comparison.py`**
  - **Mục đích**: Script xuất báo cáo so sánh & vẽ đồ thị so sánh chỉ số mô hình base.
  - **Hàm chính**: `generate_base_comparison_report(outputs_dir, config_paths)`.
  - **Mô tả**: Trích xuất chỉ số Validation & Test (`Val_Accuracy`, `Val_Precision`, `Val_Recall`, `Val_F1_Score`, `Test_Accuracy`, `Test_Precision`, `Test_Recall`, `Test_F1_Score`), định dạng %, xuất `comparison_table.csv` & `comparison_table.md`, và vẽ đồ thị thanh so sánh.

#### 3.4 Package Tải & Biến đổi Dữ liệu (`src/datasets/`)

- **`src/datasets/__init__.py`**: Khởi tạo package dataset, export các hàm tạo dataset & loader.
- **`src/datasets/dataset.py`**
  - **Mục đích**: Wrapper PyTorch Dataset và DataLoader factory.
  - **Lớp/Hàm chính**: `TeaLeafDataset`, `create_dataloaders(config)`.
  - **Mô tả**: Bao bọc `ImageFolder` của torchvision, áp dụng transform Albumentations, và tạo các DataLoader train/val/test có cố định seed.
- **`src/datasets/transforms.py`**
  - **Mục đích**: Pipeline biến đổi & tăng cường dữ liệu Albumentations theo config.
  - **Hàm chính**: `get_train_transforms(config)`, `get_val_transforms(config)`.
  - **Mô tả**: Áp dụng xoay ngẫu nhiên, lật ngang/dọc, biến đổi màu sắc, shift-scale-rotate, resize và chuẩn hóa ImageNet cho tập train; resize/chuẩn hóa cố định cho tập val/test.

#### 3.5 Package Kiến trúc Mô hình (`src/models/`)

- **`src/models/__init__.py`**: Khởi tạo package models, export hàm tạo mô hình.
- **`src/models/factory.py`**
  - **Mục đích**: Model Factory dạng Registry giúp dễ dàng mở rộng kiến trúc mô hình.
  - **Lớp/Hàm chính**: `ModelFactory`, `@register_model(name)`, `create_model(model_name, pretrained, num_classes)`.
  - **Mô tả**: Quản lý registry toàn cục `_MODEL_REGISTRY`. Sử dụng decorator để đăng ký các hàm khởi tạo mô hình (`_resnet50`, `_densenet121`, `_efficientnet_b0`, `_swin_tiny`) mà không cần sửa code Trainer.

#### 3.6 Package Bộ Huấn luyện & Đánh giá (`src/engine/`)

- **`src/engine/__init__.py`**: Khởi tạo package engine.
- **`src/engine/trainer.py`**
  - **Mục đích**: Bộ điều khiển huấn luyện dùng chung (Generic PyTorch Trainer).
  - **Lớp chính**: `Trainer`.
  - **Mô tả**: Quản lý toàn bộ vòng đời huấn luyện: Tự động ép kiểu hỗn hợp (AMP `GradScaler`), Cắt tầng đạo hàm (Gradient Clipping), Tích lũy đạo hàm (Gradient Accumulation), Lịch trình giảm LR với Warmup, Dừng sớm (Early Stopping) và ghi nhật ký CSV theo từng epoch.
- **`src/engine/evaluator.py`**
  - **Mục đích**: Bộ suy luận và lưu trữ kết quả đánh giá độc lập.
  - **Hàm chính**: `evaluate_model()`, `run_inference()`, `save_predictions()`.
  - **Mô tả**: Chạy suy luận qua DataLoader, tính toán các chỉ số phân loại, vẽ ma trận nhầm lẫn, và tự động lưu các mảng `.npy` (`test_probabilities.npy`, `test_labels.npy`, `val_probabilities.npy`, `val_labels.npy`) cùng `class_to_idx.json`.
- **`src/engine/checkpoint.py`**
  - **Mục đích**: Quản lý lưu trữ & phục hồi trọng số checkpoint.
  - **Lớp chính**: `CheckpointManager`.
  - **Mô tả**: Quản lý việc lưu/tải file `best_model.pth` (val accuracy cao nhất) và `last_model.pth` (epoch mới nhất), lưu trạng thái model, optimizer, epoch và lịch sử chỉ số.

#### 3.7 Package Thuật toán Ensemble (`src/ensemble/`)

- **`src/ensemble/__init__.py`**: Khởi tạo package ensemble, export các lớp thuật toán.
- **`src/ensemble/base.py`**
  - **Mục đích**: Lớp cơ sở trừu tượng và xác minh tính thống nhất của nhãn lớp.
  - **Lớp/Hàm chính**: `EnsembleBase`, `validate_class_mappings(class_mappings)`.
  - **Mô tả**: Ép buộc các lớp con cài đặt `fit()`, `predict()`, `evaluate()`, và tự động kiểm tra xem `class_to_idx` giữa tất cả các mô hình cơ sở có khớp nhau 100% không.
- **`src/ensemble/voting.py`**
  - **Mục đích**: Cài đặt các thuật toán Ensemble dạng Bầu chọn (Voting).
  - **Lớp chính**: `HardVoting`, `SoftVoting`, `WeightedVoting`.
  - **Mô tả**: Thực thi bầu chọn đa số (`HardVoting`), trung bình cộng mảng xác suất (`SoftVoting`), và tối ưu hóa trọng số mô hình ($\sum w_i = 1$) bằng Grid search (`WeightedVoting`).
- **`src/ensemble/stacking.py`**
  - **Mục đích**: Wrapper huấn luyện các Meta-Learner trong Stacking Ensemble.
  - **Lớp chính**: `StackingEnsemble`.
  - **Mô tả**: Ghép nối các mảng xác suất dự đoán thành ma trận đặc trưng $(N, M \cdot C)$ và huấn luyện các mô hình meta-learner (`logistic_regression`, `random_forest`, `xgboost`) với seed cố định.
- **`src/ensemble/oof.py`**
  - **Mục đích**: Bộ tạo dự đoán Out-of-Fold (OOF) chống rò rỉ dữ liệu.
  - **Lớp chính**: `OOFGenerator`.
  - **Mô tả**: Thực hiện 5-Fold Stratified K-Fold cross-validation (`split_seed=42`) để tạo mảng xác suất dự đoán held-out sạch 100% dùng làm dữ liệu huấn luyện cho Meta-learner Stacking.

#### 3.8 Package Tiện ích Bổ trợ (`src/utils/`)

- **`src/utils/__init__.py`**: Khởi tạo package utils.
- **`src/utils/config.py`**
  - **Mục đích**: Bộ đọc & chuyển đổi tệp cấu hình YAML dạng Dataclass.
  - **Hàm/Dataclass chính**: `load_config(config_path)`, `ExperimentConfig`, `ModelConfig`, `DatasetConfig`, `TrainingConfig`, `CheckpointConfig`.
  - **Mô tả**: Đọc file YAML thành các object Dataclass có định kiểu dữ liệu rõ ràng và tự động nhận diện phần cứng (`cuda` hoặc `cpu`).
- **`src/utils/logger.py`**
  - **Mục đích**: Tiện ích ghi nhật ký console và ghi file CSV.
  - **Lớp/Hàm chính**: `get_logger()`, `CSVLogger`.
  - **Mô tả**: Cấu hình Rich logger để in thông điệp định dạng đẹp mắt trên Terminal và ghi nhật ký chỉ số per-epoch vào file `history.csv`.
- **`src/utils/metrics.py`**
  - **Mục đích**: Tính toán các chỉ số phân loại.
  - **Hàm chính**: `compute_metrics(y_true, y_pred, class_names)`.
  - **Mô tả**: Tính Accuracy, Macro/Weighted Precision, Recall, F1-Score, Confusion Matrix và trả về dictionary chỉ số chi tiết từng lớp.
- **`src/utils/report.py`**
  - **Mục đích**: Tiện ích đo đạc mô hình, đếm tham số và trích xuất chỉ số.
  - **Hàm chính**: `count_parameters()`, `get_model_size_mb()`, `extract_model_history_info()`, `extract_model_val_metrics()`.
  - **Mô tả**: Đo đạc số lượng tham số (M triệu), dung lượng file mô hình trên ổ đĩa (MB), epoch đạt kết quả tốt nhất, và trích xuất đầy đủ bộ chỉ số Validation (`Val_Accuracy`, `Val_Precision`, `Val_Recall`, `Val_F1_Score`).
- **`src/utils/visualization.py`**
  - **Mục đích**: Vẽ đồ thị & trực quan hóa dữ liệu.
  - **Hàm chính**: `plot_history()`, `plot_confusion_matrix()`, `plot_comparison_bar()`.
  - **Mô tả**: Tự động vẽ đồ thị đường Loss/Accuracy theo epoch, đồ thị heatmap Ma trận nhầm lẫn, và đồ thị cột so sánh các mô hình base.
- **`src/utils/reproducibility.py`**
  - **Mục đích**: Cấu hình cố định seed toàn cục và GPU CUDA determinism.
  - **Hàm chính**: `set_seed(seed)`.
  - **Mô tả**: Cố định seed đồng nhất trên `random`, `numpy`, `torch`, `torch.cuda`, và cấu hình `torch.backends.cudnn.deterministic = True` để đảm bảo tính tái lập 100%.

---

### 4. Cấu trúc Đầu ra Kết quả Thử nghiệm

Kết quả được tự động lưu trữ trong thư mục tương ứng:

```
outputs/<experiment_name>/
├── best_model.pth              # Trọng số tốt nhất theo val_accuracy
├── last_model.pth              # Trọng số ở epoch cuối
├── history.csv                 # Nhật ký chỉ số theo từng epoch dạng CSV
├── class_to_idx.json           # Tệp ánh số lớp -> chỉ số index
├── test_probabilities.npy      # Xác suất dự đoán tập Test (N, C)
├── test_labels.npy             # Nhãn thực tế tập Test (N,)
├── val_probabilities.npy       # Xác suất dự đoán tập Val (N, C)
├── val_labels.npy              # Nhãn thực tế tập Val (N,)
├── metrics.json                # Chỉ số đánh giá tổng hợp JSON
├── classification_report.txt   # Báo cáo chi tiết dạng văn bản
└── confusion_matrix.png        # Đồ thị Ma trận nhầm lẫn
```
