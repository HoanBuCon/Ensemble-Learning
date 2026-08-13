# Tea Leaf Ensemble — Research Framework

[English](#english) | [Tiếng Việt](#tiếng-việt) | [📖 Full Documentation / Tài liệu chi tiết](docs/codebase_documentation.md)

---

<a name="english"></a>
## English

Enterprise-grade PyTorch research framework designed for image classification, Ensemble Learning research (Hard Voting, Soft Voting, Weighted Voting, Stacking), and real-time multi-image REST API deployment. Built with modular architecture, strict reproducibility, full configuration drive, and clean decoupled Web Architecture for academic research, thesis work, and web deployment.

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
- **FastAPI Web Server & REST API (`server.py`)**: High-performance backend inference server with batch processing, dynamic `dataset_info` serving, and lazy model weight caching.
- **Interactive Web Dashboard Analytics**:
  - `Base Models (4)` Tab & `Ensemble Methods (6)` Tab.
  - Per-Class Disease Breakdown Charts for both Base Models and Ensemble Methods with interactive metric pills (`F1-Score`, `Precision`, `Recall`).
  - Active glowing state UI/UX for Analytics toggle button.
- **Unified Interactive Master CLI (`main.py`)**: Terminal controller with Angular CLI-style menu (`↑/↓` arrow key & number key navigation) or direct subcommand flags (`all-in-one`, `train`, `ensemble`, `serve`, `report`).

### Project Structure

```
pipeline/
├── main.py                   # Master CLI controller (Interactive menu & subcommands)
├── server.py                 # FastAPI REST API & Inference Server
├── web/                      # Clean Decoupled Frontend Web Assets
│   ├── index.html            # Dashboard HTML structure
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
│   └── generate_comparison.py# Comparison report & plot generator
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

# 6. Re-generate Comparison Tables & Plots Only
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
- **Bộ điều khiển Master CLI tương tác (`main.py`)**: Hỗ trợ phím mũi tên `↑` / `↓` + `Enter` hoặc gõ số `1`-`7` để chọn tác vụ dạng Angular CLI.
- **Biểu đồ Phân tích Web Analytics**:
  - Tích hợp 2 tab phân tích riêng biệt: `Base Models (4)` & `Ensemble Methods (6)`.
  - Biểu đồ phân rã chỉ số từng loại bệnh lá trà (*F1-Score*, *Precision*, *Recall*) cho cả Mô hình đơn và Phương pháp Ensemble.
  - Trạng thái Active phát sáng Glow UI/UX cho nút đóng/mở Analytics.
- **Mô-đun Ensemble độc lập**: Xử lý trực tiếp trên các mảng xác suất NumPy (không phụ thuộc vào PyTorch) bao gồm:
  - Hard Voting (Bầu chọn theo đa số)
  - Soft Voting (Trung bình cộng xác suất)
  - Weighted Voting (Tối ưu hóa trọng số tự động bằng Grid-search)
  - Stacking Ensemble (Logistic Regression, Random Forest, XGBoost) kết hợp với bộ tạo Out-of-Fold (OOF) để chống rò rỉ dữ liệu.
- **FastAPI Web Server & REST API (`server.py`)**: Máy chủ inference hiệu năng cao hỗ trợ xử lý batch nhiều ảnh, trả về metadata `dataset_info` và cache trọng số mô hình tối ưu.

### Cấu trúc thư mục

```
pipeline/
├── main.py                   # Bộ điều khiển Master CLI (Menu tương tác & các câu lệnh)
├── server.py                 # Máy chủ FastAPI REST API & Inference Server
├── web/                      # Thư mục Frontend tách biệt độc lập
│   ├── index.html            # Cấu trúc HTML Dashboard & 2 tab Analytics
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
│   └── generate_comparison.py# Script xuất báo cáo & vẽ đồ thị so sánh
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

# 6. Tạo lại các bảng báo cáo & vẽ đồ thị so sánh mô hình base
python main.py report
```

---

## License
Academic Research & Thesis Use.
