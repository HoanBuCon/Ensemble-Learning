# Tea Leaf Ensemble — Research Framework

[English](#english) | [Tiếng Việt](#tiếng-việt) | [📖 Full Documentation / Tài liệu chi tiết](docs/codebase_documentation.md)

---

<a name="english"></a>
## English

PyTorch research framework for image classification and ensemble evaluation. Results are shown only when provenance-linked run artifacts are available; exact runtime and reproducibility claims depend on those artifacts.

> 📖 **Detailed Architecture & Codebase Documentation**: See [docs/codebase_documentation.md](docs/codebase_documentation.md) for full module breakdowns, data flows, REST API endpoints, and technical specifications.

<p align="center">
  <img src="assets/potential_too_high.jpg" alt="Framework Overview" width="100%"/>
</p>

### Key Features

- **Centralized Dataset Registry (`configs/dataset.yaml`)**: Single Source of Truth for dataset metadata, class names, directory paths, and display names with zero hardcoded class names in python code.
- **Config-Driven Architecture**: YAML files control every parameter (model, dataset, training, augmentations, checkpoints). Backbone model YAMLs dynamically inherit `num_classes` from `dataset.yaml`.
- **Model Factory**: Registry pattern allows adding backbones (ResNet, DenseNet, EfficientNet, Swin Transformer, ConvNeXt, etc.) via a simple decorator (`@register_model`).
- **Generic Trainer**: Unified trainer supporting Mixed Precision (AMP), gradient clipping, gradient accumulation, learning rate scheduling with warmup, early stopping, and automatic checkpointing.
- **Ensemble Learning Module**: Independent numpy-based module supporting:
  - Hard Voting (Majority Vote)
  - Soft Voting (Probability Averaging)
  - Weighted Voting (Automated grid-search weight optimization)
  - Stacking Ensemble (Logistic Regression, Random Forest, XGBoost) with Out-of-Fold (OOF) prediction generation to eliminate data leakage.
- **Scientific Verification & Calibration Suite (`scripts/verification/` & `python main.py verify`)**:
  - Probability Calibration: Expected Calibration Error (ECE, 15 bins), Multi-class Brier Score, and Negative Log-Likelihood (NLL).
  - Model Diversity & Continuous Probability Vector Ambiguity ($\bar{A}_{\text{prob}}$) as per Krogh & Vedelsby (1995).
  - Cross-Protocol Paired Hypothesis Testing Matrix: six same-method comparisons with exact paired p-values and a dynamically derived Bonferroni threshold; failure to reject is not equivalence.
  - Hardware Efficiency & Inference Latency Benchmark (GPU/CPU Latency, FPS, Model Memory & Params).
  - Latent Feature Space t-SNE 2D Manifold Quality (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz).
  - Automated report generation exporting comprehensive CSV, JSON, and Markdown summary files (`FULL_SCIENTIFIC_VERIFICATION_REPORT.md`).
- **FastAPI Web Server & REST API (`server.py`)**: High-performance backend inference server with batch processing, dynamic `dataset_info` serving, and lazy model weight caching.
- **Interactive Web Dashboard Analytics**:
  - Dual-Protocol Switcher (`5-Fold OOF Mode` vs. `Single-Split Mode`) dynamically refreshing all charts, tables, and verification metrics.
  - Tab 1: `Base Models (4)` & Tab 2: `Ensemble Methods (6)` with Per-Class Disease Breakdown Charts and interactive metric pills (`F1-Score`, `Precision`, `Recall`).
  - Tab 3: `Scientific Verification & Calibration` with interactive 2x2 Contingency Matrix, Cross-Protocol Hypothesis Testing Matrix, Reliability diagrams, and Ambiguity decomposition.
  - Explainable AI: Multi-model Grad-CAM visual attention heatmaps.
- **Unified Interactive Master CLI (`main.py`)**: Terminal controller with Angular CLI-style menu (`↑/↓` arrow key & number key navigation) or direct subcommand flags (`all-in-one`, `train`, `ensemble`, `serve`, `verify`, `report`).

### Project Structure

```
pipeline/
├── main.py                   # Master CLI controller (Interactive menu & subcommands)
├── server.py                 # FastAPI REST API & Inference Server
├── web/                      # Clean Decoupled Frontend Web Assets
│   ├── index.html            # Dashboard HTML structure & Analytics tabs
│   ├── styles.css            # Dark mode glassmorphism stylesheet & active button glow
│   └── app.js                # Frontend JS logic (Upload, Ctrl+V Paste, Charts, API Fetch)
├── configs/                  # YAML configuration files
│   ├── dataset.yaml          # Centralized Dataset Registry (classes, display names, paths)
│   ├── resnet50.yaml
│   ├── densenet121.yaml
│   ├── efficientnet_b0.yaml
│   └── swin_tiny.yaml
├── scripts/                  # Modular entrypoint scripts
│   ├── train.py              # Single-model training script
│   ├── evaluate.py           # Standalone evaluation script
│   ├── run_experiments.py   # Multi-model benchmark runner
│   ├── run_ensemble_eval.py  # Ensemble evaluation pipeline
│   ├── run_kfold.py          # 5-Fold Cross Validation runner
│   ├── generate_comparison.py# Comparison report & plot generator
│   └── verification/         # Scientific Verification & Calibration Suite
│       ├── __init__.py
│       ├── common_utils.py             # Dynamic auto-discovery & directory resolver
│       ├── verify_all_metrics.py       # Master automated verification suite
│       ├── eval_calibration.py         # Probability calibration (ECE, Brier, NLL)
│       ├── eval_diversity_ambiguity.py # Ambiguity decomposition & Yule's Q
│       ├── eval_mcnemar_test.py        # Cross-protocol McNemar hypothesis testing matrix
│       ├── eval_advanced_metrics.py    # ROC-AUC, MCC, Cohen's Kappa, Weighted F1
│       ├── eval_latency_throughput.py  # GPU/CPU latency & FPS benchmark
│       └── eval_tsne.py                # 2D t-SNE latent feature space clustering
├── outputs/                  # Hierarchical output directory (Zero-collision)
│   ├── <model_name>/         # Single-Split parent directory
│   │   └── kfold/            # 5-Fold OOF child directory
│   ├── val/                  # Single-Split Ensemble results
│   ├── oof/                  # 5-Fold OOF Ensemble results
│   └── verification/         # Verification audit reports (Master report & CSV/JSON matrices)
├── src/
│   ├── datasets/             # Dataset loaders & Albumentations transforms
│   ├── models/               # ModelFactory with backbone registrations
│   ├── engine/               # Generic Trainer, Evaluator, & CheckpointManager
│   ├── ensemble/             # Voting, Stacking, and OOF Generators
│   └── utils/                # Config parser, Logger, Metrics, Report, & Visualization
├── requirements.txt          # Frozen dependencies
├── README.md                 # Bilingual documentation (English & Tiếng Việt)
└── docs/
    └── codebase_documentation.md # Exhaustive technical reference document
```

### Supported Models

- **ResNet-50** (`resnet50`)
- **DenseNet-121** (`densenet121`)
- **EfficientNet-B0** (`efficientnet_b0`)
- **Swin Transformer Tiny** (`swin_tiny`)

---

### Quick Start Guide

#### 1. Setup Environment & Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install PyTorch with CUDA support (Recommended for NVIDIA GPUs, e.g. CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other requirements (including FastAPI & Uvicorn)
pip install -r requirements.txt

# Verify CUDA GPU acceleration
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### 2. Prepare Dataset

Structure your dataset in standard `ImageFolder` format inside `data/`:

```
data/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── ...
├── val/
│   ├── class_0/
│   └── ...
└── test/ (optional)
    ├── class_0/
    └── ...
```

#### 3. Master CLI Controller (`main.py`) & Web Server

Use `python main.py` as the primary interface for all commands:

```bash
# 🌐 0. Start FastAPI REST API & Localhost Web Dashboard Server
python main.py serve                            # Starts server on http://127.0.0.1:8000
python server.py                                # Alternative direct entrypoint

# ⭐️ 1. Full End-to-End Pipeline (Train All Base -> Report -> Ensemble)
python main.py all-in-one                       # Interactive prompt for val vs oof mode
python main.py all-in-one --ensemble-mode val   # Fast validation mode (~30s)
python main.py all-in-one --ensemble-mode oof   # Full 5-Fold OOF mode (~10-13 hrs)

# 2. Train & Evaluate ALL Base Models Only (No Ensemble)
python main.py train-all                 # Default: auto-detect & skip completed models
python main.py train-all --mode scratch  # Re-train all base models from scratch (Epoch 1)
python main.py train-all --mode resume   # Resume training from last saved checkpoint

# 3. Train a Single Base Model
python main.py train configs/resnet50.yaml
python main.py train configs/resnet50.yaml --resume

# 4. Evaluate a Trained Model Checkpoint
python main.py evaluate configs/resnet50.yaml                                    # Evaluate best_model.pth on Test
python main.py evaluate configs/resnet50.yaml --split val                       # Evaluate best_model.pth on Val

# 5. Perform Ensemble Evaluation Only (Voting & Stacking)
python main.py ensemble                          # Interactive prompt for val vs oof mode
python main.py ensemble --mode val               # Fast Validation Mode (~30s)
python main.py ensemble --mode oof               # Full 5-Fold OOF Mode (~10-13 hrs)

# 6. 5-Fold Cross-Validation Across All Backbones
python main.py kfold-all

# 7. Scientific Verification & Calibration Suite
python main.py verify                            # Run complete verification suite (auto-discovers folders)
python main.py verify --task calibration         # ECE, Brier Score, NLL
python main.py verify --task diversity           # Disagreement Rate, Yule's Q, Ambiguity Decomposition
python main.py verify --task mcnemar             # Cross-Protocol McNemar Hypothesis Testing Matrix & Effect Size
python main.py verify --task advanced            # Macro ROC-AUC, MCC, Cohen's Kappa, Weighted F1
python main.py verify --task latency             # GPU/CPU Latency & Throughput (FPS)
python main.py verify --task tsne                # 2D t-SNE Latent Feature Space Clustering
python main.py verify --save-dir custom_folder/  # Custom export directory

# 8. Re-generate Comparison Tables & Plots Only
python main.py report
```

#### 4. Web Dashboard & REST API Endpoints

Once the server is running (`python main.py serve`):
- **Interactive Web Dashboard UI**: `http://127.0.0.1:8000/` (Drag & drop images, press `Ctrl + V` to paste screenshots, or click Clear).
- **Swagger OpenAPI Documentation**: `http://127.0.0.1:8000/docs`.
- **Multi-Image Prediction API**: `POST /api/v1/predict` (Accepts `files` and `mode_type`).

---

<a name="tiếng-việt"></a>
## Tiếng Việt

Khung nghiên cứu PyTorch cấp doanh nghiệp (Enterprise-grade) phục vụ nghiên cứu Phân loại ảnh, Học kết hợp (Ensemble Learning - Hard Voting, Soft Voting, Weighted Voting, Stacking) và triển khai Web API phân loại ảnh thời gian thực. Khung làm việc được thiết kế theo kiến trúc mô-đun, đảm bảo tính tái lập (reproducibility) cao, điều khiển hoàn toàn bằng cấu hình (config-driven) và tách biệt kiến trúc Frontend/Backend chuẩn Clean Code dành cho khóa luận tốt nghiệp, bài báo nghiên cứu và ứng dụng Web.

> 📖 **Tài liệu Chi tiết về Mã nguồn & Kiến trúc**: Xem tệp [docs/codebase_documentation.md](docs/codebase_documentation.md) để đọc phân tích chi tiết từng mô-đun, luồng thực thi dữ liệu, REST API endpoints và đặc tả kỹ thuật.

<p align="center">
  <img src="assets/tiem_nang_qua_lon.jpg" alt="Tổng quan Hệ thống" width="100%"/>
</p>

### Tính năng chính

- **Quản lý bằng Config**: Mọi tham số (mô hình, dữ liệu, huấn luyện, tăng cường dữ liệu, điểm kiểm tra checkpoint) được định nghĩa hoàn toàn qua tệp YAML.
- **Model Factory**: Sử dụng mẫu thiết kế Registry giúp dễ dàng thêm kiến trúc mạng mới (ResNet, DenseNet, EfficientNet, Swin Transformer, ConvNeXt...) chỉ với một decorator duy nhất (`@register_model`).
- **Bộ huấn luyện dùng chung (Generic Trainer)**: Trainer hỗ trợ Tự động ép kiểu hỗn hợp (AMP), Cắt tầng đạo hàm (Gradient clipping), Tích lũy đạo hàm (Gradient accumulation), Lịch trình học tập có khởi động (Warmup LR), Dừng sớm (Early stopping) và lưu trữ checkpoint.
- **Quản lý Tập dữ liệu Tập trung (`configs/dataset.yaml`)**: Nguồn sự thật duy nhất (Single Source of Truth) khai báo số lớp, nhãn class, đường dẫn dữ liệu và tên tiếng Việt hiển thị, không hardcode tên class trong code Python.
- **Tự động nhận diện (Auto-Discovery)**: `src/utils/config.py` tự động đọc `dataset.yaml` hoặc tự quét tên thư mục `data/` và tệp `outputs/**/class_to_idx.json` để tự cấu hình `num_classes` cho tất cả các mô hình.
- **Bộ điều khiển Master CLI tương tác (`main.py`)**: Hỗ trợ phím mũi tên `↑` / `↓` + `Enter` hoặc gõ số `1`-`10` để chọn tác vụ dạng Angular CLI (`all-in-one`, `train`, `ensemble`, `kfold`, `verify`, `serve`, `report`).
- **Giao diện Web Dashboard & Phân tích Đa Protocol (`server.py` + `web/`)**:
  - Chuyển đổi linh hoạt giữa `5-Fold OOF Mode` và `Single-Split Mode` với cơ chế làm mới toàn bộ biểu đồ, bảng dữ liệu động.
  - Tab 1: `Base Models (4)` & Tab 2: `Ensemble Methods (6)` với biểu đồ phân rã chỉ số từng loại bệnh (*F1-Score*, *Precision*, *Recall*).
  - Tab 3: `Scientific Verification & Calibration` hiển thị Ma trận kiểm định chéo McNemar, bảng Contingency $2 \times 2$, phân rã Ambiguity và biểu đồ Calibration.
  - Thị giác giải thích Explainable AI: Trực quan hóa bản đồ chú ý nhiệt đa mô hình qua Grad-CAM.
- **Mô-đun Ensemble độc lập**: Xử lý trực tiếp trên các mảng xác suất NumPy (không phụ thuộc vào PyTorch) bao gồm:
  - Hard Voting (Bầu chọn theo đa số)
  - Soft Voting (Trung bình cộng xác suất)
  - Weighted Voting (Tối ưu hóa trọng số tự động bằng SLSQP)
  - Stacking Ensemble (Logistic Regression, Random Forest, XGBoost) dùng ma trận OOF $N_{train} \times 24$ với outer holdout chỉ dành cho inference.
- **Bộ Công cụ Kiểm định Khoa học Toàn diện (`scripts/verification/` & `python main.py verify`)**:
  - Đo lường Hiệu chuẩn Xác suất: Expected Calibration Error (ECE 15 bins), Brier Score, Negative Log-Likelihood (NLL).
  - Đo lường Độ đa dạng Mô hình & Phân rã Ambiguity ($\bar{A}_{\text{prob}}$) theo định lý Krogh & Vedelsby (1995).
  - Ma trận McNemar liên giao thức: sáu so sánh cùng phương pháp với ngưỡng Bonferroni được tính từ số phép kiểm định; không diễn giải không bác bỏ như bằng chứng tương đương.
  - Đánh giá Phân tách Không gian Ẩn t-SNE 2D (Silhouette Score, Davies-Bouldin, Calinski-Harabasz).
  - Đo đạc Hiệu năng Phần cứng: Độ trễ Latency (ms), Tốc độ Throughput (FPS) trên CPU/GPU, Dung lượng MB và Số lượng Tham số.
  - Tự động xuất báo cáo tổng hợp Master: `FULL_SCIENTIFIC_VERIFICATION_REPORT.md`.
- **FastAPI Web Server & REST API (`server.py`)**: Máy chủ inference hiệu năng cao hỗ trợ xử lý batch nhiều ảnh, trả về metadata `dataset_info` và cache trọng số mô hình tối ưu.

### Cấu trúc thư mục

```
pipeline/
├── main.py                   # Bộ điều khiển Master CLI (Menu tương tác & các câu lệnh)
├── server.py                 # Máy chủ FastAPI REST API & Inference Server
├── web/                      # Thư mục Frontend tách biệt độc lập
│   ├── index.html            # Cấu trúc HTML Dashboard & 3 tab Analytics & Verification
│   ├── styles.css            # Tệp CSS Dark Mode Glassmorphism & hiệu ứng Glow Active
│   └── app.js                # Tệp JavaScript xử lý Upload, Ctrl+V Paste, Charts & Fetch API
├── configs/                  # Các tệp cấu hình YAML
│   ├── dataset.yaml          # Registry Tập dữ liệu tập trung (nhãn class, tên tiếng Việt)
│   ├── resnet50.yaml
│   ├── densenet121.yaml
│   ├── efficientnet_b0.yaml
│   └── swin_tiny.yaml
├── scripts/                  # Các script entrypoint mô-đun gọn gàng
│   ├── train.py              # Script huấn luyện đơn mô hình
│   ├── evaluate.py           # Script đánh giá mô hình độc lập
│   ├── run_experiments.py   # Bộ chạy thử nghiệm tự động nhiều mô hình
│   ├── run_ensemble_eval.py  # Script đánh giá Ensemble
│   ├── run_kfold.py          # Bộ chạy kiểm định chéo 5-Fold Cross Validation
│   ├── generate_comparison.py# Script xuất báo cáo & vẽ đồ thị so sánh
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
├── outputs/                  # Thư mục kết quả mặc định (Phân cấp & Không ghi đè)
│   ├── <tên_mô_hình>/        # Thư mục Cha: Kết quả Single-Split (best_model.pth)
│   │   └── kfold/            # Thư mục Con: Kết quả 5-Fold OOF (fold_1..5 checkpoints)
│   ├── val/                  # Bảng so sánh 6 Ensemble Single-Split
│   ├── oof/                  # Bảng so sánh 6 Ensemble 5-Fold OOF
│   └── verification/         # Báo cáo Kiểm định Khoa học xuất ra (Master report & CSV/JSON)
├── src/
│   ├── datasets/             # Tải dữ liệu & biến đổi dữ liệu (Albumentations)
│   ├── models/               # ModelFactory & Đăng ký mô hình
│   ├── engine/               # Generic Trainer, Evaluator & CheckpointManager
│   ├── ensemble/             # Mô-đun Ensemble (Voting, Stacking, OOF)
│   └── utils/                # Đọc Config, Logger, Metrics, Báo cáo & Trực quan hóa
├── requirements.txt          # Danh sách thư viện phụ thuộc đã khóa phiên bản
├── README.md                 # Tài liệu song ngữ (English & Tiếng Việt)
└── docs/
    └── codebase_documentation.md # Tài liệu kỹ thuật chi tiết toàn bộ mã nguồn
```

### Các mô hình hỗ trợ ban đầu

- **ResNet-50** (`resnet50`)
- **DenseNet-121** (`densenet121`)
- **EfficientNet-B0** (`efficientnet_b0`)
- **Swin Transformer Tiny** (`swin_tiny`)

---

### Hướng dẫn sử dụng nhanh

#### 1. Cài đặt môi trường & Thư viện

```bash
# Tạo và kích hoạt môi trường ảo
python -m venv venv
# Trên Windows:
.\venv\Scripts\activate
# Trên Linux/macOS:
source venv/bin/activate

# Cài đặt PyTorch hỗ trợ GPU CUDA (Khuyên dùng cho card NVIDIA, VD: CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Cài đặt các thư viện cần thiết còn lại (bao gồm FastAPI & Uvicorn)
pip install -r requirements.txt

# Kiểm tra GPU đã nhận diện chưa
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### 2. Chuẩn bị dữ liệu

Sắp xếp dữ liệu theo định dạng chuẩn `ImageFolder` trong thư mục `data/`:

```
data/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── ...
├── val/
│   ├── class_0/
│   └── ...
└── test/ (tùy chọn)
    ├── class_0/
    └── ...
```

#### 3. Bộ điều khiển Master CLI (`main.py` - Khuyên dùng)

Sử dụng `python main.py` làm giao diện chính duy nhất cho mọi lệnh:

```bash
# 🌐 0. Khởi chạy Server FastAPI REST API & Web Dashboard Localhost
python main.py serve                            # Chạy server tại http://127.0.0.1:8000
python server.py                                # Hoặc chạy trực tiếp file server.py

# ⭐️ 1. Lệnh All-in-One chạy trọn gói Pipeline (Train Base -> Báo cáo Base -> Đánh giá Ensemble)
python main.py all-in-one                       # Chọn mode val/oof qua giao diện menu tương tác
python main.py all-in-one --ensemble-mode val   # Fast validation mode (~30s)
python main.py all-in-one --ensemble-mode oof   # Full 5-Fold OOF mode (~10-13 hrs)

# 2. Train & Đánh giá TẤT CẢ mô hình Base (Không chạy Ensemble)
python main.py train-all                 # Mặc định: tự động phát hiện & skip mô hình đã train xong
python main.py train-all --mode scratch  # Train lại tất cả mô hình base từ đầu (Epoch 1)
python main.py train-all --mode resume   # Tiếp tục train các mô hình bị dở dang từ checkpoint

# 3. Huấn luyện 1 mô hình đơn
python main.py train configs/resnet50.yaml
python main.py train configs/resnet50.yaml --resume

# 4. Đánh giá độc lập một Checkpoint mô hình đã train
python main.py evaluate configs/resnet50.yaml                                    # Đánh giá best_model.pth trên tập Test
python main.py evaluate configs/resnet50.yaml --split val                       # Đánh giá best_model.pth trên tập Val

# 5. Thực hiện thử nghiệm Ensemble (Voting & Stacking)
python main.py ensemble                          # Chọn mode val/oof qua giao diện menu tương tác
python main.py ensemble --mode val               # Chế độ Validation nhanh (~30s)
python main.py ensemble --mode oof               # Chế độ Full 5-Fold OOF (~10-13 hrs)

# 6. Chạy Kiểm định Chéo 5-Fold cho tất cả mô hình
python main.py kfold-all

# 7. Chạy Bộ Kiểm định Khoa học & Hiệu chuẩn Xác suất
python main.py verify                            # Chạy toàn bộ bộ kiểm định (tự động nhận diện thư mục)
python main.py verify --task calibration         # Đo ECE, Brier Score, NLL
python main.py verify --task diversity           # Đo Tỷ lệ Bất đồng, Yule's Q, Phân rã Ambiguity
python main.py verify --task mcnemar             # Ma trận kiểm định chéo McNemar & Effect size
python main.py verify --task advanced            # Đo lường Macro ROC-AUC, MCC, Cohen's Kappa, Weighted F1
python main.py verify --task latency             # Đo đạc độ trễ Latency & Tốc độ Throughput (FPS)
python main.py verify --task tsne                # Phân tích cụm không gian ẩn 2D t-SNE
python main.py verify --save-dir my_reports/     # Xuất báo cáo vào thư mục tùy chọn

# 8. Tạo lại các bảng báo cáo & vẽ đồ thị so sánh mô hình base
python main.py report
```

#### 4. Giao diện Web Dashboard & Các Endpoint REST API

Khi máy chủ đang hoạt động (`python main.py serve`):
- **Giao diện Web Dashboard Trực quan**: `http://127.0.0.1:8000/` (Kéo thả ảnh, nhấn `Ctrl + V` để dán ảnh chụp màn hình, hoặc nhấn nút Xóa).
- **Tài liệu Swagger OpenAPI Tương tác**: `http://127.0.0.1:8000/docs`.
- **API Dự đoán Phân loại Đa ảnh**: `POST /api/v1/predict` (Nhận danh sách `files` và tham số `mode_type`).

---

## Giấy phép (License)
Dành cho Nghiên cứu Học thuật & Khóa luận Tốt nghiệp (Academic Research & Thesis Use).
