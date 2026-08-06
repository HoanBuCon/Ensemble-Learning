# Comprehensive Codebase Documentation / Tài liệu Chi tiết mã nguồn

**Project:** Tea Leaf Ensemble Research Framework  
**Target:** System Architecture, Data Pipeline, Ensemble Engine & Module Specifications  

[English](#english) | [Tiếng Việt](#tiếng-việt)

---

<a name="english"></a>
## English Documentation

### 1. System Overview & Architectural Goals

The **Tea Leaf Ensemble Research Framework** is an enterprise-grade, modular, and reproducible Computer Vision research pipeline built using PyTorch. Designed specifically for academic thesis research and comparative studies in **Ensemble Learning**, the framework enforces strict separation between base model training, prediction caching, and meta-learning/voting ensembles.

#### Core Engineering Principles
- **Config-Driven Architecture**: Every experiment parameter (model selection, learning rate, augmentation intensity, label smoothing, hardware device) is managed via YAML configuration files in `configs/`. No hardcoded magic numbers exist in the python code.
- **Open/Closed Principle (OCP)**: Adding new model backbones (e.g., ConvNeXt, ViT) or custom ensemble techniques requires zero modifications to existing training or evaluation scripts.
- **Zero-PyTorch Ensemble Layer**: Ensemble algorithms (`Voting`, `Stacking`) operate purely on cached NumPy probability arrays (`.npy`), eliminating PyTorch CUDA overhead and allowing rapid post-hoc experiment iterations.
- **Class Index Alignment Verification**: `EnsembleBase.validate_class_mappings()` ensures all base models share identical `class_to_idx` mappings prior to ensembling, preventing silent column misalignment bugs.
- **Leakage-Free Stacking**: Out-of-Fold (OOF) prediction generation uses a unified `split_seed` with `StratifiedKFold` to ensure second-stage meta-learners (Logistic Regression, Random Forest, XGBoost) are trained exclusively on held-out predictions.
- **Dual Stacking Execution Modes**: Supports both `mode="val"` (Fast Prototyping ~30 sec) and `mode="oof"` (Gold Standard 5-Fold Evaluation ~10-13 hrs).

---

### 2. Directory Layout & File Overview

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
│       ├── visualization.py  # Loss/Accuracy curves & Confusion Matrix plotting
│       └── reproducibility.py# Seed setting & deterministic CUDA configuration
│
├── train.py                  # Entry point for single-model training
├── evaluate.py               # Entry point for standalone evaluation
├── run_experiments.py        # Multi-experiment runner & automated summary generator
├── run_ensemble_eval.py      # Ensemble evaluation runner (--mode val / --mode oof)
├── requirements.txt          # Frozen dependencies
└── README.md                 # Bilingual documentation (English & Tiếng Việt)
```

---

### 3. Module Specifications & Technical Details

#### 3.1 `src/utils/` — Utility Package
- **`config.py`**: Parses YAML files into typed dataclasses (`ExperimentConfig`, `ModelConfig`, etc.). Resolves `auto` device (`cuda` or `cpu`).
- **`reproducibility.py`**: `set_seed(seed)` locks seeds across Python `random`, `numpy`, and `torch` (CPU & CUDA), enforcing CuDNN deterministic operations.
- **`metrics.py`**: `compute_metrics(y_true, y_pred, class_names)` computes Accuracy, Macro/Weighted Precision, Recall, F1-Score, Confusion Matrix, and Per-class breakdowns.

#### 3.2 `src/models/` — Model Factory & Backbones
- **`factory.py`**: Global registry `_MODEL_REGISTRY` with `@register_model(name)`. Supported backbones:
  1. `resnet50`: Torchvision ResNet-50.
  2. `densenet121`: Torchvision DenseNet-121.
  3. `efficientnet_b0`: `timm` EfficientNet-B0.
  4. `swin_tiny`: `timm` Swin Transformer Tiny.

#### 3.3 `src/engine/` — Core Training & Evaluation Engine
- **`trainer.py`**: Generic training controller with AMP, Warmup, LR Schedulers, Gradient Clipping, Gradient Accumulation, and Early Stopping.
- **`evaluator.py`**: `save_predictions()` serializes `test_probabilities.npy`, `test_labels.npy`, `val_probabilities.npy`, `val_labels.npy`, and `class_to_idx.json`.

#### 3.4 `src/ensemble/` — Ensemble Package
- **`base.py`**: Defines `EnsembleBase` and `validate_class_mappings(class_mappings)` to assert class mapping consistency across models.
- **`voting.py`**:
  - `HardVoting`: Computes majority vote with flat `.ravel()` 1D array formatting.
  - `SoftVoting`: Computes average softmax probability vectors across models.
  - `WeightedVoting`: Optimizes normalized model weights ($\sum w_i = 1$) via grid search on validation/OOF predictions.
- **`stacking.py`**: `StackingEnsemble` trains meta-learners (`logistic_regression`, `random_forest`, `xgboost`) on stacked probability feature matrices $(N, M \cdot C)$ using explicit seeds (`random_state=42`).
- **`oof.py`**: `OOFGenerator` uses `StratifiedKFold(n_splits=5, random_state=split_seed)` with a unified `split_seed=42` across all backbones, performing direct index assignments `oof_probabilities[val_indices] = fold_probs`.

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
- **Kiến trúc Quản lý bằng Config**: Tất cả các tham số (kiến trúc mô hình, tốc độ học, mức độ tăng cường dữ liệu, nhãn làm mịn, thiết bị phần cứng) được định nghĩa qua tệp YAML trong `configs/`. Không có tham số cứng (magic numbers) trong mã nguồn Python.
- **Nguyên lý Đóng/Mở (Open/Closed Principle - OCP)**: Thêm mô hình backbone mới (ConvNeXt, ViT...) hay thuật toán ensemble mới mà không cần sửa bất kỳ dòng code huấn luyện hay đánh giá sẵn có nào.
- **Lớp Ensemble Không Phụ thuộc PyTorch**: Các thuật toán Ensemble (`Voting`, `Stacking`) hoạt động hoàn toàn trên các mảng xác suất NumPy (`.npy`), loại bỏ chi phí tính toán GPU PyTorch và cho phép thử nghiệm kết hợp cực nhanh.
- **Xác minh Đồng bộ Thứ tự Lớp (Class Mapping Validation)**: Phương thức `EnsembleBase.validate_class_mappings()` tự động kiểm tra tính thống nhất của `class_to_idx` giữa tất cả các mô hình cơ sở trước khi ensemble, ngăn chặn lỗi lệch cột nhãn.
- **Stacking Không Rò rỉ Dữ liệu (Leakage-Free)**: Bộ tạo Out-of-Fold (OOF) sử dụng `StratifiedKFold` với `split_seed` dùng chung để đảm bảo meta-learner (Logistic Regression, Random Forest, XGBoost) chỉ được huấn luyện trên dự đoán held-out.
- **Hai Chế độ Đánh giá Stacking**: Hỗ trợ cả `mode="val"` (Thử nghiệm nhanh ~30 giây) và `mode="oof"` (Đánh giá chuẩn báo cáo luận văn/bài báo 5-Fold ~10–13 giờ).

---

### 2. Cấu trúc Thư mục & Tệp tin

```
pipeline/
├── configs/                  # Các tệp cấu hình YAML
│   ├── resnet50.yaml         # Cấu hình ResNet-50 baseline
│   ├── densenet121.yaml      # Cấu hình DenseNet-121
│   ├── efficientnet_b0.yaml  # Cấu hình EfficientNet-B0
│   └── swin_tiny.yaml        # Cấu hình Swin Transformer Tiny
│
├── src/                      # Package Python chính
│   ├── __init__.py
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
│       ├── visualization.py  # Vẽ đồ thị Loss/Accuracy & Confusion Matrix
│       └── reproducibility.py# Cố định seed toàn cục & GPU CUDA
│
├── train.py                  # Điểm chạy huấn luyện đơn mô hình
├── evaluate.py               # Điểm chạy đánh giá mô hình độc lập
├── run_experiments.py        # Bộ chạy thử nghiệm tự động nhiều mô hình
├── run_ensemble_eval.py      # Bộ chạy thử nghiệm Ensemble (--mode val / --mode oof)
├── requirements.txt          # Danh sách thư viện khóa phiên bản
└── README.md                 # Tài liệu song ngữ (English & Tiếng Việt)
```

---

### 3. Mô tả Chi tiết Mô-đun & Đặc tả Kỹ thuật

#### 3.1 `src/utils/` — Tiện ích Bổ trợ
- **`config.py`**: Chuyển đổi tệp YAML thành các dataclass (`ExperimentConfig`, `ModelConfig`...). Tự động nhận diện thiết bị `cuda` hoặc `cpu`.
- **`reproducibility.py`**: `set_seed(seed)` cố định seed trên Python `random`, `numpy` và `torch`, ép tính toán CuDNN deterministic.
- **`metrics.py`**: `compute_metrics()` tính toán Accuracy, Macro/Weighted Precision, Recall, F1-Score, Confusion Matrix và Báo cáo chi tiết từng lớp.

#### 3.2 `src/models/` — Model Factory & Backbones
- **`factory.py`**: Registry toàn cục `_MODEL_REGISTRY` với `@register_model(name)`. Đã tích hợp sẵn:
  1. `resnet50`: ResNet-50 từ Torchvision.
  2. `densenet121`: DenseNet-121 từ Torchvision.
  3. `efficientnet_b0`: EfficientNet-B0 từ `timm`.
  4. `swin_tiny`: Swin Transformer Tiny từ `timm`.

#### 3.3 `src/engine/` — Bộ Huấn luyện & Đánh giá
- **`trainer.py`**: Trainer đa năng hỗ trợ AMP, Warmup, LR Scheduler, Gradient Clipping, Accumulation và Early Stopping.
- **`evaluator.py`**: `save_predictions()` tự động lưu các file `test_probabilities.npy`, `test_labels.npy`, `val_probabilities.npy`, `val_labels.npy` và `class_to_idx.json`.

#### 3.4 `src/ensemble/` — Package Ensemble Learning
- **`base.py`**: Định nghĩa lớp `EnsembleBase` và phương thức tĩnh `validate_class_mappings(class_mappings)` để phát hiện lỗi lệch nhãn giữa các mô hình.
- **`voting.py`**:
  - `HardVoting`: Bầu chọn đa số với mảng đầu ra chuẩn 1D dạng `.ravel()`.
  - `SoftVoting`: Trung bình cộng các mảng xác suất Softmax.
  - `WeightedVoting`: Tối ưu hóa trọng số mô hình ($\sum w_i = 1$) bằng Grid-search trên tập Val/OOF.
- **`stacking.py`**: `StackingEnsemble` huấn luyện các Meta-learner (`logistic_regression`, `random_forest`, `xgboost`) trên mảng đặc trưng xác suất $(N, M \cdot C)$ với `random_state=42`.
- **`oof.py`**: `OOFGenerator` sử dụng `StratifiedKFold(n_splits=5, random_state=split_seed)` với `split_seed=42` cố định giữa tất cả các backbone, thực hiện gán trực tiếp theo chỉ số `oof_probabilities[val_indices] = fold_probs`.

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
