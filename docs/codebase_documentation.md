# Comprehensive Codebase Documentation / Tài liệu Chi tiết Toàn bộ Mã nguồn

**Project:** Tea Leaf Ensemble Research Framework  
**Target:** Architecture, Data Pipeline, Master CLI, Output Hierarchy, REST API, Ensemble Engine, Scientific Verification Suite & Exhaustive Component Specifications  

[English Documentation](#english-documentation) | [Tài liệu Tiếng Việt](#tài-liệu-tiếng-việt)

---

<a name="english-documentation"></a>
# Part I: English Documentation

## 1. System Overview & Engineering Principles

The **Tea Leaf Ensemble Research Framework** is an enterprise-grade, modular, and reproducible Computer Vision research pipeline built on top of PyTorch. Designed specifically for academic thesis research, comparative studies in **Ensemble Learning** (Voting & Stacking), probability calibration & uncertainty quantification, and real-time Web API deployment, the framework enforces clean decoupled architecture across all components.

### Core Engineering Principles
1. **Unified Master CLI (`main.py`)**: Centralized command-line entrypoint featuring an interactive terminal menu (`↑/↓` arrow key & number key navigation) and direct CLI subcommand execution for training, evaluating, 5-Fold cross-validating, ensembling, and scientific verification auditing.
2. **Hierarchical Output Architecture (Zero-Collision)**: Single-split training outputs reside in `outputs/<model_name>/` (with ensemble in `outputs/val/`), while 5-Fold OOF models reside in `outputs/<model_name>/kfold/` (with ensemble in `outputs/oof/`), completely preventing file overwriting.
3. **Scientific Verification & Calibration Suite (`scripts/verification/` & `python main.py verify`)**: Automated benchmarking tools for Expected Calibration Error (ECE, 15 bins), Multi-class Brier Score, Negative Log-Likelihood (NLL), model diversity (Pairwise Disagreement, Yule's Q, Cohen's Kappa), continuous vector ambiguity decomposition ($\\bar{A}_{\\text{prob}}$), cross-protocol paired hypothesis testing matrix (McNemar tests across all 6 ensembles with Edwards' correction, exact binomial $p$-values, Bonferroni adjustment $\\alpha=0.0083$, Risk Difference $95\\%$ CI, Cohen's $h$ effect sizes, and statistical power), latency/throughput profiling, and t-SNE latent manifold clustering metrics.
4. **FastAPI REST API & Web Dashboard (`server.py` + `web/`)**: High-performance backend inference server with dynamic `dataset_info` serving, batch inference, model weight caching, dual-protocol switcher (`5-Fold OOF Mode` vs. `Single-Split Mode`), Grad-CAM visual attention heatmaps, and 3 analytics tabs (`Base Models`, `Ensemble Methods`, `Scientific Verification & Calibration`).
5. **Centralized Dataset Registry (`configs/dataset.yaml`)**: Single Source of Truth for dataset metadata, data directory paths (`train/val/test`), class names, and display labels. Zero hardcoded class names exist in python code.
6. **Config-Driven Architecture**: Every experiment parameter (model selection, learning rate, augmentation intensity, label smoothing, hardware device) is managed via YAML configuration files in `configs/`.

---

## 2. Directory Layout & File Overview

```
pipeline/
├── main.py                   # Master CLI controller (Interactive menu & subcommands)
├── server.py                 # FastAPI REST API & Inference Server
├── web/                      # Clean Decoupled Frontend Web Assets
│   ├── index.html            # Dashboard HTML structure & 3 analytics tabs
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
│   ├── run_kfold.py          # 5-Fold Cross Validation runner
│   ├── generate_comparison.py# Comparison report generator
│   ├── generate_all_plots.py # Diagnostic plot & heatmap generator
│   └── verification/         # Scientific Verification & Benchmarking Suite
│       ├── __init__.py
│       ├── common_utils.py             # Dynamic auto-discovery & directory resolver
│       ├── verify_all_metrics.py       # Master automated verification suite
│       ├── eval_calibration.py         # Probability calibration (ECE, Brier, NLL)
│       ├── eval_diversity_ambiguity.py # Ambiguity decomposition & Yule's Q
│       ├── eval_mcnemar_test.py        # Cross-protocol McNemar hypothesis testing matrix
│       ├── eval_advanced_metrics.py    # ROC-AUC, MCC, Cohen's Kappa, Weighted F1
│       ├── eval_latency_throughput.py  # GPU/CPU latency & FPS throughput benchmark
│       └── eval_tsne.py                # 2D t-SNE latent feature space clustering
│
├── outputs/                  # Default output directory (Hierarchical & collision-free)
│   ├── <model_name>/         # Single-Split parent directory (best_model.pth, val_probabilities.npy)
│   │   └── kfold/            # 5-Fold OOF subdirectory (fold_1..5 checkpoints, oof_probabilities.npy)
│   ├── val/                  # Single-Split Ensemble comparison reports & metrics
│   ├── oof/                  # 5-Fold OOF Ensemble comparison reports & metrics
│   └── verification/         # Master verification CSV/JSON/MD audit reports
│
├── src/                      # Main Python package
│   ├── datasets/             # Data loading & Albumentations transforms
│   ├── models/               # ModelFactory with backbone registrations
│   ├── engine/               # Generic Trainer, Evaluator, & CheckpointManager
│   ├── ensemble/             # Voting, Stacking, and OOF Generators
│   └── utils/                # Config parser, Logger, Metrics, Report, & Visualization
│
├── requirements.txt          # Frozen dependencies
├── README.md                 # Bilingual documentation (English & Tiếng Việt)
├── reports/                  # Academic Research Papers (TEA_LEAF_ENSEMBLE_RESEARCH_REPORT.md)
└── docs/
    └── codebase_documentation.md # Exhaustive technical reference document
```

---

## 3. Output Directory Hierarchy & Zero-Collision Architecture

To ensure that running Single-Split training and 5-Fold OOF cross-validation does not overwrite any model weights or predictions, the framework utilizes a parent-child subdirectory structure:

| Protocol | Model Outputs Location | Ensemble Outputs Location | Key Files Generated |
| :--- | :--- | :--- | :--- |
| **Single-Split (Default)** | `outputs/<model_name>/` | `outputs/val/` | `best_model.pth`, `val_probabilities.npy`, `test_probabilities.npy`, `ensemble_comparison.csv` |
| **5-Fold OOF** | `outputs/<model_name>/kfold/` | `outputs/oof/` | `fold_1/..fold_5/best_model.pth`, `oof_probabilities.npy`, `oof_metrics.json`, `ensemble_comparison.csv` |
| **Scientific Verification**| — | `outputs/verification/` | `FULL_SCIENTIFIC_VERIFICATION_REPORT.md`, `mcnemar_cross_protocol_matrix.csv/.md`, `calibration_benchmark.csv`, `diversity_benchmark.csv`, `latency_benchmark.csv`, `tsne_latent_space.png` |

---

## 4. Exhaustive File-by-File & Module Specifications

### 4.1. Core Application & Server (`main.py`, `server.py`, `web/`)
- **`main.py`**: Central CLI entrypoint. Supports both interactive menu and subcommands (`all-in-one`, `train`, `evaluate`, `ensemble`, `kfold`, `kfold-all`, `verify`, `report`, `plot`, `serve`).
- **`server.py`**: FastAPI backend server exposing REST endpoints:
  - `GET /`: Serves web dashboard frontend (`web/index.html`).
  - `GET /api/v1/health`: Health status, CUDA availability, and loaded model count.
  - `GET /api/v1/config`: Active dataset classes and display labels.
  - `GET /api/v1/analytics`: Full JSON metrics for Base Models, Ensembles, and Cross-Protocol Verification matrix.
  - `POST /api/v1/predict`: Single/multi-image upload inference supporting all base models and 6 ensemble methods with Grad-CAM heatmaps.
- **`web/`**: Decoupled frontend assets (`index.html`, `styles.css`, `app.js`) featuring dual-protocol switching (`5-Fold OOF` vs. `Single-Split`), responsive drag-and-drop image uploads, clipboard pasting (`Ctrl + V`), dynamic Chart.js analytics, and complete Verification contingency matrices.

### 4.2. Configuration Package (`configs/`)
- **`configs/dataset.yaml`**: Centralized Dataset Registry specifying classes (`Brown_Blight`, `Gray_Blight`, `Green_mirid_bug`, `Healthy_leaf`, `Helopeltis`, `Tea_algal_leaf_spot`), display names, and paths.
- **`configs/*.yaml`**: Individual backbone model configurations (`resnet50.yaml`, `densenet121.yaml`, `efficientnet_b0.yaml`, `swin_tiny.yaml`).

### 4.3. Pipeline Execution Scripts (`scripts/`)
- **`scripts/train.py`**: Trains a single model backbone with config loading, early stopping, and automatic test evaluation upon completion.
- **`scripts/evaluate.py`**: Evaluates a trained checkpoint on either `test` or `val` split.
- **`scripts/run_experiments.py`**: Sequentially trains and evaluates all 4 base models with auto-resume capability.
- **`scripts/run_kfold.py`**: Executes Stratified 5-Fold Cross Validation per model or across all models.
- **`scripts/run_ensemble_eval.py`**: Evaluates Hard Voting, Soft Voting, Weighted Voting (SLSQP), and Stacking Meta-Learners (Logistic Regression, Random Forest, XGBoost) in both `val` and `oof` modes.
- **`scripts/generate_comparison.py`**: Generates comparison tables (`comparison_table.csv`, `.md`) and Pareto trade-off charts.
- **`scripts/generate_all_plots.py`**: Generates the complete diagnostic visualization suite (ROC curves, PR curves, normalized Confusion Matrices, Per-class metrics bar charts, and Radar plots).

### 4.4. Scientific Verification Suite (`scripts/verification/`)
- **`scripts/verification/common_utils.py`**: Provides dynamic auto-discovery and path resolution across `RESULTS/DEFAULT_TRAINING/`, `RESULTS/OOF_TRAINING/`, `Default_Result_V2/`, `OOF_Results/`, and `outputs/`.
- **`scripts/verification/eval_calibration.py`**: Computes Expected Calibration Error (ECE, 15 bins), Multi-class Brier Score, and Negative Log-Likelihood (NLL). Exports `calibration_benchmark.csv` and `.md`.
- **`scripts/verification/eval_diversity_ambiguity.py`**: Computes Pairwise Disagreement, Yule's Q-Statistic, Cohen's Kappa, Global Disagreement, and Continuous Probability Vector Ambiguity ($\\bar{A}_{\\text{prob}}$) as per Krogh & Vedelsby (1995). Exports `diversity_benchmark.csv` and `.md`.
- **`scripts/verification/eval_mcnemar_test.py`**: Executes exhaustive cross-protocol pairwise McNemar tests across all 6 Ensemble methods (Single-Split vs. 5-Fold OOF) with Risk Difference ($95\\%$ CI), Cohen's $h$ effect sizes, Edwards' Chi-Square, Exact Two-Sided Binomial $p$-values, Bonferroni correction ($\\alpha = 0.0083$), and statistical power. Exports `mcnemar_cross_protocol_matrix.csv`, `mcnemar_cross_protocol_matrix.md`, and `mcnemar_test_results.json`.
- **`scripts/verification/eval_advanced_metrics.py`**: Computes Macro ROC-AUC (One-vs-Rest), Matthews Correlation Coefficient (MCC), Cohen's Kappa, and Weighted F1-score across all models and ensembles.
- **`scripts/verification/eval_latency_throughput.py`**: Measures GPU/CPU inference latency (ms), throughput (FPS), model parameter count, and memory footprint.
- **`scripts/verification/eval_tsne.py`**: Generates 2D t-SNE latent manifold embeddings and computes cluster quality metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz).
- **`scripts/verification/verify_all_metrics.py`**: Master runner executing all verification modules in sequence and compiling the consolidated `FULL_SCIENTIFIC_VERIFICATION_REPORT.md`.

### 4.5. Source Code Package (`src/`)
- **`src/datasets/`**:
  - `dataset.py`: `ImageFolderDataset` wrapper with validation check for class indices and PyTorch `DataLoader` factory.
  - `transforms.py`: Albumentations augmentation pipeline (RandomHorizontalFlip, RandomVerticalFlip, ShiftScaleRotate, ColorJitter, ImageNet normalization).
- **`src/models/`**:
  - `factory.py`: Registry-based `ModelFactory` with `@register_model` decorator. Supports ResNet-50, DenseNet-121, EfficientNet-B0, and Swin Transformer Tiny with dynamically adjusted classification heads.
- **`src/engine/`**:
  - `trainer.py`: Generic Trainer supporting Automatic Mixed Precision (AMP), gradient clipping, Cosine Annealing with warmup, and Early Stopping.
  - `evaluator.py`: High-throughput batch inference engine computing classification metrics and caching `.npy` probability arrays.
  - `checkpoint.py`: `CheckpointManager` tracking `best_model.pth` and `last_model.pth`.
- **`src/ensemble/`**:
  - `base.py`: Abstract `EnsembleBase` class with validation checks.
  - `voting.py`: `HardVoting` (vote proportions), `SoftVoting` (probability mean), and `WeightedVoting` (SLSQP optimization).
  - `stacking.py`: `StackingEnsemble` wrapping meta-classifiers (Logistic Regression, Random Forest, XGBoost) fitted on 24-dimensional concatenated probability vectors.
  - `oof.py`: `OOFGenerator` building unified 5-fold cross-validation splits and out-of-fold feature matrices.
- **`src/utils/`**:
  - `config.py`: Dataclass YAML parser and centralized dataset config loader (`load_dataset_config()`).
  - `logger.py`: Rich console logger and structured CSV history logger.
  - `metrics.py`: Computes Accuracy, Macro/Weighted Precision, Recall, F1-Score, and Confusion Matrix.
  - `report.py`: Model parameter counter, memory sizing in MB, and benchmark table formatter.
  - `visualization.py`: Matplotlib/Seaborn plotting functions for training curves and confusion matrices.
  - `reproducibility.py`: Deterministic seed setting across Python, NumPy, PyTorch CPU, and CUDA.

---

## 5. Master CLI Command Reference (`main.py`)

```bash
# 1. Full End-to-End Pipeline
python main.py all-in-one --ensemble-mode val    # Fast validation mode (~30s)
python main.py all-in-one --ensemble-mode oof    # Full 5-Fold OOF mode (~10-13 hrs)

# 2. Train Base Models
python main.py train configs/resnet50.yaml       # Train single model
python main.py train-all                         # Train all 4 models sequentially

# 3. Ensemble Evaluation Only
python main.py ensemble --mode val               # Evaluate Single-Split ensemble
python main.py ensemble --mode oof               # Evaluate 5-Fold OOF ensemble

# 4. 5-Fold Cross Validation Only
python main.py kfold configs/resnet50.yaml       # 5-Fold CV for single model
python main.py kfold-all                         # 5-Fold CV for all 4 models

# 5. Scientific Verification & Calibration Suite
python main.py verify                            # Run full verification suite (auto-discovers folders)
python main.py verify --task calibration         # ECE, Brier Score, NLL
python main.py verify --task diversity           # Disagreement Rate, Yule's Q, Ambiguity Decomposition
python main.py verify --task mcnemar             # Cross-Protocol McNemar Hypothesis Testing Matrix & Effect Size
python main.py verify --task advanced            # Macro ROC-AUC, MCC, Cohen's Kappa, Weighted F1
python main.py verify --task latency             # GPU/CPU Latency & Throughput (FPS)
python main.py verify --task tsne                # 2D t-SNE Latent Feature Space Clustering
python main.py verify --save-dir custom_reports/ # Custom export directory

# 6. Web Dashboard Server
python main.py serve --host 127.0.0.1 --port 8000
```

---

## 6. Advanced Academic Benchmarks Reference Table

| Task / Evaluation Metric | Source Script | CLI Subcommand | Generated Output Files |
| :--- | :--- | :--- | :--- |
| **Probability Calibration** | `scripts/verification/eval_calibration.py` | `python main.py verify --task calibration` | `calibration_benchmark.csv/.md` |
| **Model Diversity & Ambiguity** | `scripts/verification/eval_diversity_ambiguity.py` | `python main.py verify --task diversity` | `diversity_benchmark.csv/.md` |
| **Cross-Protocol McNemar Matrix** | `scripts/verification/eval_mcnemar_test.py` | `python main.py verify --task mcnemar` | `mcnemar_cross_protocol_matrix.csv/.md`, `mcnemar_test_results.json` |
| **ROC-AUC & Matthews Corr (MCC)** | `scripts/verification/eval_advanced_metrics.py` | `python main.py verify --task advanced` | `advanced_metrics_benchmark.csv/.md` |
| **Hardware Latency & Throughput** | `scripts/verification/eval_latency_throughput.py` | `python main.py verify --task latency` | `latency_benchmark.csv/.md` |
| **t-SNE Latent Space 2D** | `scripts/verification/eval_tsne.py` | `python main.py verify --task tsne` | `tsne_latent_space.png`, `tsne_clustering_metrics.csv/.md` |
| **Consolidated 6-Step Master Report** | `scripts/verification/verify_all_metrics.py` | `python main.py verify` | `FULL_SCIENTIFIC_VERIFICATION_REPORT.md` |

---

<a name="tài-liệu-tiếng-việt"></a>
# Phần II: Tài liệu Tiếng Việt

## 7. Tổng quan Hệ thống & Nguyên lý Thiết kế

**Tea Leaf Ensemble Research Framework** là khung nghiên cứu Thị giác Máy tính chuẩn doanh nghiệp được xây dựng trên nền tảng PyTorch. Được thiết kế chuyên biệt cho nghiên cứu Luận văn Thạc sĩ/Tiến sĩ, các bài báo so sánh chuyên sâu về **Học kết hợp Đa tầng (Ensemble Learning: Voting & Stacking)**, đánh giá độ hiệu chuẩn xác suất & độ bất định, và triển khai hệ thống Web API thời gian thực.

### Nguyên lý Kỹ thuật Cốt lõi
1. **Master CLI Hợp nhất (`main.py`)**: Điểm điều khiển trung tâm tích hợp Menu tương tác Terminal dạng Angular CLI (`↑/↓` + `Enter` hoặc phím số `1-10`) và các lệnh subcommand trực tiếp phục vụ toàn bộ quy trình: Huấn luyện, Đánh giá, Kiểm định 5-Fold, Ensemble, Web API, Báo cáo và Kiểm định Khoa học.
2. **Kiến trúc Thư mục Phân cấp (Không ghi đè dữ liệu)**: Mô hình Single-Split lưu tại `outputs/<model_name>/` (Ensemble tại `outputs/val/`), mô hình 5-Fold OOF lưu tại `outputs/<model_name>/kfold/` (Ensemble tại `outputs/oof/`), hoàn toàn loại bỏ nguy cơ ghi đè file.
3. **Bộ Công cụ Kiểm định Khoa học Toàn diện (`scripts/verification/` & `python main.py verify`)**: Tự động hóa kiểm định độ hiệu chuẩn xác suất (ECE 15 bins, Brier Score, NLL), đo lường độ đa dạng và phân rã Ambiguity trên vector xác suất ($\\bar{A}_{\\text{prob}}$), Ma trận kiểm định chéo McNemar toàn bộ 6 phương pháp Ensemble (Risk Difference $95\\%$ CI, Effect Size Cohen's $h$, hiệu chỉnh Bonferroni $\\alpha=0.0083$, Statistical Power), đo đạc hiệu năng phần cứng (Latency ms, Throughput FPS) và phân tích cụm không gian ẩn t-SNE.
4. **FastAPI REST API & Web Dashboard (`server.py` + `web/`)**: Máy chủ suy luận hiệu năng cao, hỗ trợ batch inference nhiều ảnh đồng thời, tự động cache trọng số mô hình, chuyển đổi giao thức `5-Fold OOF` / `Single-Split` tức thì, bản đồ chú ý nhiệt Grad-CAM và giao diện Web Analytics & Verification 3 tab trực quan.
5. **Quản lý Dữ liệu Tập trung (`configs/dataset.yaml`)**: Nguồn sự thật duy nhất (Single Source of Truth) định nghĩa số lượng lớp bệnh, nhãn ground-truth và đường dẫn dữ liệu. Không hardcode tên lớp trong code Python.
6. **Kiến trúc Hướng Cấu hình (Config-Driven)**: Toàn bộ siêu tham số (kiến trúc mô hình, tốc độ học, tăng cường ảnh, label smoothing, thiết bị GPU) được quản lý qua các file YAML trong `configs/`.

---

## 8. Cấu trúc Thư mục & Phân nhánh Đầu ra

```
pipeline/
├── main.py                   # Bộ điều khiển Master CLI (Menu tương tác & subcommands)
├── server.py                 # Máy chủ FastAPI REST API & Inference Server
├── web/                      # Giao diện Frontend Web Dashboard độc lập
│   ├── index.html            # Cấu trúc HTML5 Dashboard & 3 tab Analytics & Verification
│   ├── styles.css            # Tệp CSS Dark Mode Glassmorphism & hiệu ứng Glow Active
│   └── app.js                # Tệp JavaScript xử lý Upload, Ctrl+V Paste, Charts & Fetch API
│
├── configs/                  # Các tệp cấu hình YAML
│   ├── dataset.yaml          # Quản lý Tập dữ liệu Tập trung (classes, display names, paths)
│   ├── resnet50.yaml         # Cấu hình ResNet-50 baseline
│   ├── densenet121.yaml      # Cấu hình DenseNet-121
│   ├── efficientnet_b0.yaml  # Cấu hình EfficientNet-B0
│   └── swin_tiny.yaml        # Cấu hình Swin Transformer Tiny
│
├── scripts/                  # Các script entrypoint mô-đun
│   ├── train.py              # Script huấn luyện đơn mô hình
│   ├── evaluate.py           # Script đánh giá mô hình độc lập
│   ├── run_experiments.py   # Bộ chạy thử nghiệm tự động 4 mô hình cơ sở
│   ├── run_ensemble_eval.py  # Script đánh giá 6 phương pháp Ensemble
│   ├── run_kfold.py          # Bộ chạy kiểm định chéo 5-Fold Cross Validation
│   ├── generate_comparison.py# Script xuất bảng so sánh Markdown/CSV
│   ├── generate_all_plots.py # Bộ sinh toàn bộ đồ thị chẩn đoán (ROC, PR, CM, Heatmaps)
│   └── verification/         # Bộ Công cụ Kiểm định & Thẩm định Khoa học
│       ├── __init__.py
│       ├── common_utils.py             # Tiện ích tự động nhận diện thư mục kết quả
│       ├── verify_all_metrics.py       # Script kiểm định tự động toàn diện Master
│       ├── eval_calibration.py         # Kiểm định hiệu chuẩn xác suất (ECE, Brier, NLL)
│       ├── eval_diversity_ambiguity.py # Phân tích độ đa dạng & phân rã Ambiguity
│       ├── eval_mcnemar_test.py        # Ma trận kiểm định chéo McNemar & Effect size
│       ├── eval_advanced_metrics.py    # Đo lường ROC-AUC, MCC, Cohen's Kappa, Weighted F1
│       ├── eval_latency_throughput.py  # Đo đạc độ trễ & FPS trên GPU/CPU
│       └── eval_tsne.py                # Phân tích cụm không gian ẩn 2D t-SNE
│
├── outputs/                  # Thư mục kết quả mặc định (Phân cấp & Không ghi đè)
│   ├── <tên_mô_hình>/        # Thư mục Cha: Kết quả Single-Split (best_model.pth, val_probabilities.npy)
│   │   └── kfold/            # Thư mục Con: Kết quả 5-Fold OOF (fold_1..5 checkpoints, oof_probabilities.npy)
│   ├── val/                  # Bảng so sánh 6 Ensemble Single-Split (Fit trên 1,540 mẫu Val)
│   ├── oof/                  # Bảng so sánh 6 Ensemble 5-Fold OOF (Fit trên 7,192 mẫu OOF)
│   └── verification/         # Báo cáo Kiểm định Khoa học xuất ra (Master report & CSV/JSON)
│
├── src/                      # Gói mã nguồn Python chính
│   ├── datasets/             # Dataset loaders & biến đổi ảnh Albumentations
│   ├── models/               # ModelFactory dạng Registry (@register_model)
│   ├── engine/               # Trainer (AMP, Early Stopping), Evaluator & CheckpointManager
│   ├── ensemble/             # Voting (Hard/Soft/Weighted), Stacking & OOF Generator
│   └── utils/                # Đọc Config, Logger, Metrics, Báo cáo & Trực quan hóa
│
├── requirements.txt          # Danh sách thư viện khóa phiên bản
├── README.md                 # Tài liệu song ngữ (English & Tiếng Việt)
├── reports/                  # Báo cáo Khoa học Toàn văn (TEA_LEAF_ENSEMBLE_RESEARCH_REPORT.md)
└── docs/
    └── codebase_documentation.md # Tài liệu kỹ thuật chi tiết toàn bộ mã nguồn
```

---

## 9. Đặc tả Cấu trúc Phân nhánh Output Không Ghi đè (Zero-Collision)

| Chế độ Huấn luyện | Vị trí Lưu Mô hình Cơ sở | Vị trí Lưu Kết quả Ensemble | Các File Quan trọng Sinh ra |
| :--- | :--- | :--- | :--- |
| **Single-Split (Default)** | `outputs/<tên_mô_hình>/` | `outputs/val/` | `best_model.pth`, `val_probabilities.npy`, `test_probabilities.npy`, `ensemble_comparison.csv` |
| **5-Fold OOF** | `outputs/<tên_mô_hình>/kfold/` | `outputs/oof/` | `fold_1/..fold_5/best_model.pth`, `oof_probabilities.npy`, `oof_metrics.json`, `ensemble_comparison.csv` |
| **Kiểm định Khoa học** | — | `outputs/verification/` | `FULL_SCIENTIFIC_VERIFICATION_REPORT.md`, `mcnemar_cross_protocol_matrix.csv/.md`, `calibration_benchmark.csv`, `diversity_benchmark.csv`, `latency_benchmark.csv`, `tsne_latent_space.png` |

---

## 10. Đặc tả Chi tiết Từng File & Module Mã nguồn

### 10.1. Bộ điều khiển & Web Server (`main.py`, `server.py`, `web/`)
- **`main.py`**: Điểm điều khiển Master CLI. Hỗ trợ menu tương tác và các subcommands (`all-in-one`, `train`, `evaluate`, `ensemble`, `kfold`, `kfold-all`, `verify`, `report`, `plot`, `serve`).
- **`server.py`**: Máy chủ FastAPI Backend REST API:
  - `GET /`: Phục vụ giao diện Web Dashboard (`web/index.html`).
  - `GET /api/v1/health`: Trạng thái máy chủ, CUDA GPU và số mô hình đã nạp.
  - `GET /api/v1/config`: Danh sách lớp bệnh và tên hiển thị.
  - `GET /api/v1/analytics`: Dữ liệu JSON toàn bộ chỉ số của Base Models, Ensembles và Ma trận kiểm định chéo.
  - `POST /api/v1/predict`: Dự đoán phân loại đơn ảnh / đa ảnh cho tất cả 4 mô hình cơ sở và 6 phương pháp Ensemble kèm bản đồ nhiệt Grad-CAM.
- **`web/`**: Giao diện người dùng độc lập (`index.html`, `styles.css`, `app.js`), hỗ trợ chuyển đổi dual-protocol (`5-Fold OOF` vs. `Single-Split`), kéo thả ảnh, dán ảnh từ clipboard (`Ctrl + V`), vẽ biểu đồ động qua Chart.js và xem ma trận kiểm định thống kê.

### 10.2. Gói Cấu hình (`configs/`)
- **`configs/dataset.yaml`**: Khai báo tập trung danh mục 6 lớp bệnh lá chè, đường dẫn thư mục `data/` và tên hiển thị.
- **`configs/*.yaml`**: Cấu hình siêu tham số riêng cho từng mô hình (`resnet50.yaml`, `densenet121.yaml`, `efficientnet_b0.yaml`, `swin_tiny.yaml`).

### 10.3. Các Script Thực thi Mô-đun (`scripts/`)
- **`scripts/train.py`**: Huấn luyện 1 mô hình đơn lẻ với Early Stopping và tự động đánh giá trên tập Test sau khi hoàn thành.
- **`scripts/evaluate.py`**: Đánh giá checkpoint đã huấn luyện trên tập `test` hoặc `val`.
- **`scripts/run_experiments.py`**: Huấn luyện tự động lần lượt cả 4 mô hình cơ sở với tính năng auto-resume.
- **`scripts/run_kfold.py`**: Chạy kiểm định chéo phân tầng 5 lượt (5-Fold Cross Validation) cho 1 mô hình hoặc cả 4 mô hình.
- **`scripts/run_ensemble_eval.py`**: Đánh giá 6 phương pháp Ensemble (Hard/Soft/Weighted Voting và Stacking LR/RF/XGB) trên cả 2 chế độ `val` và `oof`.
- **`scripts/generate_comparison.py`**: Xuất bảng so sánh tổng hợp Markdown/CSV và đồ thị tối ưu Pareto.
- **`scripts/generate_all_plots.py`**: Sinh toàn bộ biểu đồ chẩn đoán (ROC, Precision-Recall, Confusion Matrix chuẩn hóa, Biểu đồ cột Per-class, Radar chart).

### 10.4. Bộ Công cụ Kiểm định Khoa học (`scripts/verification/`)
- **`scripts/verification/common_utils.py`**: Tiện ích tự động nhận diện thư mục kết quả qua 3 tầng dấu vân tay tệp tin.
- **`scripts/verification/eval_calibration.py`**: Tính toán ECE (15 bins), Brier Score và NLL (Log-Loss). Xuất file `calibration_benchmark.csv` và `.md`.
- **`scripts/verification/eval_diversity_ambiguity.py`**: Đo lường tỷ lệ bất đồng, Yule's Q, Cohen's Kappa và độ đa dạng xác suất Ambiguity ($\\bar{A}_{\\text{prob}}$). Xuất file `diversity_benchmark.csv` và `.md`.
- **`scripts/verification/eval_mcnemar_test.py`**: Thực hiện Ma trận kiểm định chéo McNemar toàn diện cho 6 phương pháp Ensemble giữa Single-Split và 5-Fold OOF kèm Risk Difference ($95\\%$ CI), Effect Size Cohen's $h$, hiệu chỉnh Bonferroni ($\\alpha = 0.0083$) và Statistical Power. Xuất file `mcnemar_cross_protocol_matrix.csv`, `mcnemar_cross_protocol_matrix.md` và `mcnemar_test_results.json`.
- **`scripts/verification/eval_advanced_metrics.py`**: Đo lường Macro ROC-AUC (One-vs-Rest), Matthews Correlation Coefficient (MCC), Cohen's Kappa và Weighted F1 cho toàn bộ mô hình đơn và ensemble.
- **`scripts/verification/eval_latency_throughput.py`**: Đo đạc độ trễ suy luận GPU/CPU (ms), tốc độ khung hình (FPS), kích thước mô hình MB và số lượng tham số.
- **`scripts/verification/eval_tsne.py`**: Phân tích biểu diễn không gian ẩn 2D t-SNE và đánh giá chất lượng phân cụm (Silhouette, Davies-Bouldin, Calinski-Harabasz).
- **`scripts/verification/verify_all_metrics.py`**: Chạy toàn bộ quy trình kiểm định và tự động xuất báo cáo Master: `FULL_SCIENTIFIC_VERIFICATION_REPORT.md`.

### 10.5. Gói Mã nguồn Chính (`src/`)
- **`src/datasets/`**: `ImageFolderDataset` wrapper và pipeline tăng cường ảnh Albumentations (`transforms.py`).
- **`src/models/`**: `ModelFactory` dạng Registry hỗ trợ ResNet-50, DenseNet-121, EfficientNet-B0, Swin Transformer Tiny qua `@register_model`.
- **`src/engine/`**: `Trainer` (AMP, Early Stopping, Cosine Annealing Warmup), `Evaluator` (chạy batch inference và lưu `.npy`), và `CheckpointManager`.
- **`src/ensemble/`**: `HardVoting`, `SoftVoting`, `WeightedVoting` (tối ưu SLSQP), `StackingEnsemble` (LR, RF, XGBoost) và `OOFGenerator` (tạo ma trận đặc trưng meta $7,192 \\times 24$).
- **`src/utils/`**: Bộ đọc YAML Dataclass (`config.py`), Rich logger (`logger.py`), tính toán chỉ số (`metrics.py`), đo đạc tham số & bộ nhớ (`report.py`), vẽ đồ thị (`visualization.py`), và cố định seed ngẫu nhiên (`reproducibility.py`).

---

## 11. Hướng dẫn Lệnh CLI Master (`main.py`)

```bash
# 1. Chạy Toàn bộ Pipeline từ đầu đến cuối
python main.py all-in-one --ensemble-mode val    # Chế độ Fast Validation (~30 giây)
python main.py all-in-one --ensemble-mode oof    # Chế độ 5-Fold OOF (~10-13 giờ)

# 2. Huấn luyện Mô hình Cơ sở
python main.py train configs/resnet50.yaml       # Huấn luyện 1 mô hình
python main.py train-all                         # Huấn luyện cả 4 mô hình

# 3. Đánh giá Ensemble
python main.py ensemble --mode val               # Đánh giá Ensemble Single-Split
python main.py ensemble --mode oof               # Đánh giá Ensemble 5-Fold OOF

# 4. Kiểm định Chéo 5-Fold
python main.py kfold configs/resnet50.yaml       # 5-Fold CV cho 1 mô hình
python main.py kfold-all                         # 5-Fold CV cho cả 4 mô hình

# 5. Kiểm định Khoa học & Hiệu chuẩn Xác suất
python main.py verify                            # Chạy toàn bộ bộ kiểm định (tự động nhận diện thư mục)
python main.py verify --task calibration         # ECE, Brier, NLL
python main.py verify --task diversity           # Bất đồng, Yule's Q, Ambiguity
python main.py verify --task mcnemar             # Ma trận kiểm định chéo McNemar & Effect size
python main.py verify --task advanced            # ROC-AUC, MCC, Cohen's Kappa, Weighted F1
python main.py verify --task latency             # Độ trễ Latency & Throughput FPS
python main.py verify --task tsne                # Phân cụm không gian ẩn t-SNE 2D
python main.py verify --save-dir outputs/my_ver/ # Xuất báo cáo vào thư mục tùy chọn

# 6. Khởi chạy Web Server Dashboard
python main.py serve --host 127.0.0.1 --port 8000
```

---

## 12. Bảng Tra cứu Kiểm định Chuẩn Quốc tế (Advanced Academic Benchmarks)

| Tác vụ / Chỉ số | Script tương ứng | CLI Command | File kết quả xuất |
| :--- | :--- | :--- | :--- |
| **Probability Calibration** | `scripts/verification/eval_calibration.py` | `python main.py verify --task calibration` | `calibration_benchmark.csv/.md` |
| **Model Diversity & Ambiguity** | `scripts/verification/eval_diversity_ambiguity.py` | `python main.py verify --task diversity` | `diversity_benchmark.csv/.md` |
| **Cross-Protocol McNemar Matrix** | `scripts/verification/eval_mcnemar_test.py` | `python main.py verify --task mcnemar` | `mcnemar_cross_protocol_matrix.csv/.md`, `mcnemar_test_results.json` |
| **ROC-AUC & Matthews Corr (MCC)** | `scripts/verification/eval_advanced_metrics.py` | `python main.py verify --task advanced` | `advanced_metrics_benchmark.csv/.md` |
| **Hardware Latency & Throughput** | `scripts/verification/eval_latency_throughput.py` | `python main.py verify --task latency` | `latency_benchmark.csv/.md` |
| **t-SNE Latent Space 2D** | `scripts/verification/eval_tsne.py` | `python main.py verify --task tsne` | `tsne_latent_space.png`, `tsne_clustering_metrics.csv/.md` |
| **Toàn bộ 6 Quy trình & Master Report** | `scripts/verification/verify_all_metrics.py` | `python main.py verify` | `FULL_SCIENTIFIC_VERIFICATION_REPORT.md` |
