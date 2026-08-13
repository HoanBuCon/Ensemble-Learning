# Comprehensive Codebase Documentation / Tài liệu Chi tiết mã nguồn

**Project:** Tea Leaf Ensemble Research Framework  
**Target:** System Architecture, Data Pipeline, Master CLI, FastAPI REST API, Ensemble Engine & Exhaustive File Specifications  

[English](#english) | [Tiếng Việt](#tiếng-việt)

---

<a name="english"></a>
## English Documentation

### 1. System Overview & Architectural Goals

The **Tea Leaf Ensemble Research Framework** is an enterprise-grade, modular, and reproducible Computer Vision research pipeline built using PyTorch. Designed specifically for academic thesis research, comparative studies in **Ensemble Learning**, and real-time Web API deployment, the framework enforces strict separation between base model training, prediction caching, meta-learning/voting ensembles, and clean decoupled Web Architecture.

#### Core Engineering Principles
- **Unified Master CLI (`main.py`)**: Centralized command-line entrypoint for orchestrating training, evaluation, benchmarking, ensemble, web serving, and reporting with self-descriptive subcommand names.
- **FastAPI REST API & Web Server (`server.py`)**: High-performance backend inference server supporting multi-image uploads, dynamic model weight caching, and 5 benchmarking modes.
- **Centralized Dataset Registry (`configs/dataset.yaml`)**: Single Source of Truth for dataset metadata, data directory paths (`train/val/test`), class names, and display labels. Dynamic auto-discovery in `src/utils/config.py` (`load_dataset_config()`) falls back to `outputs/**/class_to_idx.json` or subdirectories inside `./data/`. Zero hardcoded class names exist in python code.
- **Unified Interactive Master CLI (`main.py`)**: Centralized command-line entrypoint featuring an Angular CLI-style interactive terminal menu (`↑/↓` arrow key & number key navigation) or direct subcommand execution.
- **FastAPI REST API & Web Server (`server.py`)**: High-performance backend inference server supporting multi-image uploads, dynamic dataset info serving (`GET /api/v1/analytics`), and lazy model weight caching.
- **Decoupled Web Architecture & Analytics Dashboard (`web/`)**: Clean separation between server logic and frontend presentation assets (`web/index.html`, `web/styles.css`, `web/app.js`), featuring dual-tab analytics (`Base Models` & `Ensemble Methods`), per-class disease breakdown charts with interactive metric pills (`F1-Score`, `Precision`, `Recall`), and glowing active button states.
- **Config-Driven Architecture**: Every experiment parameter (model selection, learning rate, augmentation intensity, label smoothing, hardware device) is managed via YAML configuration files in `configs/`. Model YAMLs dynamically inherit `num_classes` from `dataset.yaml`.

---

### 2. Directory Layout & File Overview

```
pipeline/
├── main.py                   # Master CLI controller (Interactive menu & subcommands)
├── server.py                 # FastAPI REST API & Inference Server
├── web/                      # Clean Decoupled Frontend Web Assets
│   ├── index.html            # Dashboard HTML structure & dual analytics tabs
│   ├── styles.css            # Dark mode glassmorphism stylesheet & active glowing UI
│   └── app.js                # Frontend JS logic (Upload, Ctrl+V Paste, Charts, API Fetch)
│
├── configs/                  # YAML experiment configuration files
│   ├── dataset.yaml          # Centralized Dataset Registry (classes, display names, paths)
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

#### 3.1 Root, Web & Server Specifications

- **`main.py`**
  - **Purpose**: Unified Master CLI Controller.
  - **Key Functions**: `build_parser()`, `prompt_ensemble_mode()`, `resolve_config_paths()`, `main()`.
  - **Commands**: `all-in-one`, `train-all`, `train`, `evaluate`, `ensemble`, `serve`, `report`.
  - **Description**: Parses command-line subcommands and routes execution to training scripts, ensemble evaluators, or starts the FastAPI server (`main.py serve`).

- **`server.py`**
  - **Purpose**: FastAPI Web Server & REST API backend.
  - **Key Endpoints**: `GET /`, `GET /api/v1/config`, `POST /api/v1/predict`.
  - **Description**: Manages PyTorch base model weight loading, runs multi-image batch preprocessing with Albumentations, executes Voting/Stacking ensemble predictions, and serves static files from `web/`.

- **`web/index.html`**
  - **Purpose**: Frontend Dashboard HTML5 structure.
  - **Description**: Clean template defining sidebar mode selector, drag & drop zone, file count badges, action buttons (`Run`, `Clear`), and results card grid.

- **`web/styles.css`**
  - **Purpose**: Frontend CSS stylesheet.
  - **Description**: Implements responsive grid layout, dark mode colors, glassmorphism backdrop blur, sticky left control panel (`position: sticky; top: 1.5rem`), and W3C compliant `background-clip: text` styling.

- **`web/app.js`**
  - **Purpose**: Frontend JavaScript client application logic.
  - **Description**: Manages mode switching, drag & drop file upload, global Clipboard Paste (`Ctrl + V`) listener, Clear Button (`🗑️ Clear`) handler, FormData API submission, and dynamic DOM rendering of inference cards and consensus badges.

#### 3.2 Configuration Files (`configs/`)

- **`configs/resnet50.yaml`**: ResNet-50 experiment config.
- **`configs/densenet121.yaml`**: DenseNet-121 experiment config.
- **`configs/efficientnet_b0.yaml`**: EfficientNet-B0 experiment config.
- **`configs/swin_tiny.yaml`**: Swin Transformer Tiny experiment config.

#### 3.3 Execution Scripts (`scripts/`)

- **`scripts/train.py`**: Trains single model, evaluates `best_model.pth`, and saves probability caches (`.npy`).
- **`scripts/evaluate.py`**: Evaluates model checkpoint on `test` or `val` split.
- **`scripts/run_experiments.py`**: Runs benchmark across all base models.
- **`scripts/run_ensemble_eval.py`**: Runs Voting and Stacking ensemble evaluations in `val` or `oof` modes.
- **`scripts/generate_comparison.py`**: Exports comparison tables (`csv`/`md`) and metric bar plots.

#### 3.4 Data & Transforms Package (`src/datasets/`)

- **`src/dataset.py`**: `TeaLeafDataset` wrapper & seeded DataLoaders.
- **`src/transforms.py`**: Config-driven Albumentations image transformation pipelines.

#### 3.5 Models Package (`src/models/`)

- **`src/factory.py`**: Registry-based `ModelFactory` with `@register_model` decorator.

#### 3.6 Engine Package (`src/engine/`)

- **`src/trainer.py`**: Generic unified PyTorch Trainer with AMP, Warmup, LR scheduling, Early Stopping, and CSV logging.
- **`src/evaluator.py`**: Model evaluation serializer creating `test_probabilities.npy`, `val_probabilities.npy`, `class_to_idx.json`.
- **`src/checkpoint.py`**: Saves and loads `best_model.pth` and `last_model.pth`.

#### 3.7 Ensemble Package (`src/ensemble/`)

- **`src/base.py`**: Abstract base class and `validate_class_mappings()` validator.
- **`src/voting.py`**: `HardVoting`, `SoftVoting`, and `WeightedVoting` (grid-search weights).
- **`src/stacking.py`**: `StackingEnsemble` wrapper for Logistic Regression, Random Forest, XGBoost.
- **`src/oof.py`**: `OOFGenerator` performing 5-Fold Stratified K-Fold cross-validation (`split_seed=42`).

#### 3.8 Utilities Package (`src/utils/`)

- **`src/utils/config.py`**: Dataclass-based YAML parser.
- **`src/utils/logger.py`**: Rich console logger and CSVLogger.
- **`src/utils/metrics.py`**: Calculates Accuracy, Precision, Recall, F1, and Confusion Matrix.
- **`src/utils/report.py`**: Parameter counter, model sizing, and validation metric extractor.
- **`src/utils/visualization.py`**: Plotting per-epoch loss/accuracy curves and confusion matrices.
- **`src/utils/reproducibility.py`**: Seed setting across Python, NumPy, PyTorch, and CUDA determinism.

---

### 4. Output Artifacts Directory Structure

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

**Tea Leaf Ensemble Research Framework** là khung nghiên cứu Thị giác máy tính (Computer Vision) cấp doanh nghiệp, dạng mô-đun và tái lập cao được xây dựng trên PyTorch. Được thiết kế chuyên biệt cho luận văn và các bài báo nghiên cứu về **Học kết hợp (Ensemble Learning)** cùng khả năng triển khai Web API thời gian thực, hệ thống phân tách nghiêm ngặt giữa quá trình huấn luyện mô hình cơ sở (base models), lưu trữ xác suất dự đoán (prediction caching), thuật toán kết hợp (voting/stacking) và kiến trúc Web phân tách Frontend/Backend sạch sẽ.

#### Nguyên lý Kỹ thuật Cốt lõi
- **Quản lý Tập dữ liệu Tập trung (`configs/dataset.yaml`)**: Nguồn sự thật duy nhất (Single Source of Truth) khai báo số lượng lớp bệnh, nhãn class, tên hiển thị và đường dẫn dữ liệu. Tiện ích `load_dataset_config()` tự động dò tìm nhãn class mà không hardcode trong code Python.
- **Bộ điều khiển Master CLI tương tác (`main.py`)**: Giao diện dòng lệnh tập trung tích hợp Menu điều khiển tương tác dạng Angular CLI (dùng phím mũi tên `↑/↓` + `Enter` hoặc phím số `1-7`).
- **Máy chủ REST API FastAPI (`server.py`)**: Máy chủ inference hiệu năng cao hỗ trợ xử lý batch nhiều ảnh cùng lúc, trả về thông tin `dataset_info` và cache trọng số mô hình.
- **Kiến trúc Web Dashboard Analytics (`web/`)**: Giao diện Web 2 tab phân tích (`Base Models` & `Ensemble Methods`), trực quan hóa chỉ số phân rã từng loại bệnh lá trà (*F1-Score*, *Precision*, *Recall*) và trạng thái phát sáng Glow UI/UX cho nút đóng/mở.
- **Kiến trúc Quản lý bằng Config**: Tất cả các tham số (kiến trúc mô hình, tốc độ học, mức độ tăng cường dữ liệu, nhãn làm mịn, thiết bị phần cứng) được định nghĩa qua tệp YAML trong `configs/`. Các tệp config mô hình tự động kế thừa `num_classes` từ `dataset.yaml`.

---

### 2. Cấu trúc Thư mục & Tệp tin

```
pipeline/
├── main.py                   # Bộ điều khiển Master CLI (train, evaluate, ensemble, serve, report)
├── server.py                 # Máy chủ FastAPI REST API & Inference Server
├── web/                      # Thư mục Frontend tách biệt độc lập
│   ├── index.html            # Cấu trúc HTML Dashboard
│   ├── styles.css            # Tệp CSS Dark Mode Glassmorphism (Chuẩn W3C background-clip)
│   └── app.js                # Tệp JavaScript xử lý Upload, Ctrl+V Paste, Clear & Fetch API
│
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

#### 3.1 Tệp tin Root, Web & Server

- **`main.py`**: Bộ điều khiển Master CLI tập trung (`train`, `evaluate`, `ensemble`, `serve`, `report`).
- **`server.py`**: Máy chủ FastAPI Backend REST API (`GET /`, `GET /api/v1/config`, `POST /api/v1/predict`).
- **`web/index.html`**: Cấu trúc giao diện HTML5 Dashboard.
- **`web/styles.css`**: Tệp CSS quy định màu sắc Dark mode, Glassmorphism, khung dính cố định (`position: sticky; top: 1.5rem`) và chuẩn `background-clip: text`.
- **`web/app.js`**: Logic JavaScript xử lý chọn mode, Kéo & Thả ảnh, Dán ảnh từ Clipboard (`Ctrl + V`), Nút xóa (`🗑️ Clear`) và render động dữ liệu dự đoán.

#### 3.2 Các tệp cấu hình (`configs/`)
- `configs/resnet50.yaml`, `configs/densenet121.yaml`, `configs/efficientnet_b0.yaml`, `configs/swin_tiny.yaml`.

#### 3.3 Các script thực thi mô-đun (`scripts/`)
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/run_experiments.py`, `scripts/run_ensemble_eval.py`, `scripts/generate_comparison.py`.

#### 3.4 Packages trong `src/`
- `src/datasets/`: Tải dữ liệu & Augmentations Albumentations.
- `src/models/`: ModelFactory dạng Registry với `@register_model`.
- `src/engine/`: Trainer (AMP, Early Stop, Warmup), Evaluator & CheckpointManager.
- `src/ensemble/`: Hard Voting, Soft Voting, Weighted Voting, Stacking & OOF Generator.
- `src/utils/`: Config, Logger, Metrics, Report, Visualization & Reproducibility.

---

### 4. Cấu trúc Đầu ra Kết quả Thử nghiệm

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
