# Tea Leaf Ensemble — Research Framework

[English](#english) | [Tiếng Việt](#tiếng-việt) | [📖 Full Documentation / Tài liệu chi tiết](docs/codebase_documentation.md)

---

<a name="english"></a>
## English

Enterprise-grade PyTorch research framework designed for image classification and Ensemble Learning research (Hard Voting, Soft Voting, Weighted Voting, Stacking). Built with modular architecture, strict reproducibility, and full configuration drive for academic research and thesis work.

> 📖 **Detailed Architecture & Codebase Documentation**: See [docs/codebase_documentation.md](docs/codebase_documentation.md) for full module breakdowns, data flows, and technical specifications.

<p align="center">
  <img src="assets/potential_too_high.jpg" alt="Framework Overview" width="100%"/>
</p>

### Key Features

- **Config-Driven**: YAML files control every parameter (model, dataset, training, augmentations, checkpoints).
- **Model Factory**: Registry pattern allows adding backbones (ResNet, DenseNet, EfficientNet, Swin Transformer, ConvNeXt, etc.) via a simple decorator.
- **Generic Trainer**: A single unified trainer supporting Mixed Precision (AMP), gradient clipping, gradient accumulation, learning rate scheduling with warmup, early stopping, and automatic checkpointing.
- **Ensemble Learning Module**: Independent numpy-based module supporting:
  - Hard Voting (Majority Vote)
  - Soft Voting (Probability Averaging)
  - Weighted Voting (Automated grid-search optimization)
  - Stacking Ensemble (Logistic Regression, Random Forest, XGBoost) with Out-of-Fold (OOF) prediction generation to prevent data leakage.
- **Unified Master CLI (`main.py`)**: Centralized command-line entrypoint for orchestrating training, evaluation, benchmarking, ensemble, and reporting.
- **Reproducibility**: Global seed management, deterministic CUDA operations, and seeded DataLoaders.
- **Platform Agnostic**: Runs seamlessly on local machines (Windows/Linux/macOS), Kaggle, and Google Colab.

### Project Structure

```
pipeline/
├── main.py                   # Master CLI controller
├── configs/                  # YAML configuration files
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
│   └── utils/                # Config parser, Logger, Metrics, Visualization, & Reports
├── requirements.txt          # Frozen dependencies
└── README.md
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

# Install other requirements
pip install -r requirements.txt

# Verify CUDA GPU acceleration
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

> ⚠️ **Note on PyTorch CUDA**: Ensure you install the PyTorch build matching your installed CUDA toolkit version (e.g. `cu124`, `cu121`). Running on CPU will slow down training significantly.

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

#### 3. Master CLI Controller (`main.py`)

Use `python main.py` as the primary interface for all commands:

```bash
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

#### 4. Direct Modular Script Execution (`scripts/`)

Alternatively, run specific entry scripts directly from the `scripts/` folder:

```bash
python scripts/train.py configs/resnet50.yaml
python scripts/evaluate.py configs/resnet50.yaml --split test
python scripts/run_experiments.py
python scripts/run_ensemble_eval.py --mode val
python scripts/generate_comparison.py
```

*Or via Python API:*

```python
import numpy as np
from src.ensemble import HardVoting, SoftVoting, WeightedVoting, StackingEnsemble

# Load probability predictions saved during evaluation
p_resnet = np.load("outputs/resnet50/test_probabilities.npy")
p_densenet = np.load("outputs/densenet121/test_probabilities.npy")
p_swin = np.load("outputs/swin_tiny/test_probabilities.npy")
labels = np.load("outputs/resnet50/test_labels.npy")

probs = [p_resnet, p_densenet, p_swin]

# Hard Voting
hv = HardVoting()
print("Hard Voting:", hv.evaluate(probs, labels))

# Soft Voting
sv = SoftVoting()
print("Soft Voting:", sv.evaluate(probs, labels))

# Weighted Voting
wv = WeightedVoting()
wv.fit(probs, labels)
print("Weighted Voting:", wv.evaluate(probs, labels))

# Stacking Ensemble
stacker = StackingEnsemble(meta_learner="logistic_regression")
stacker.fit(probs, labels) # Fit on OOF probabilities for leak-free evaluation
print("Stacking:", stacker.evaluate(probs, labels))
```

#### 5. Adding a New Model Architecture

Register the model in `src/models/factory.py`:

```python
@register_model("convnext_tiny")
def _convnext_tiny(pretrained: bool = True, num_classes: int = 6, **kwargs):
    import timm
    return timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)
```

Create `configs/convnext_tiny.yaml` and run `python main.py train configs/convnext_tiny.yaml` without changing any training pipeline logic!

---

<a name="tiếng-việt"></a>
## Tiếng Việt

Khung nghiên cứu PyTorch cấp doanh nghiệp (Enterprise-grade) phục vụ nghiên cứu Phân loại ảnh và Học kết hợp (Ensemble Learning - Hard Voting, Soft Voting, Weighted Voting, Stacking). Khung làm việc được thiết kế theo kiến trúc mô-đun, đảm bảo tính tái lập (reproducibility) cao và điều khiển hoàn toàn bằng cấu hình (config-driven) dành cho khóa luận tốt nghiệp và các bài báo nghiên cứu.

> 📖 **Tài liệu Chi tiết về Mã nguồn & Kiến trúc**: Xem tệp [docs/codebase_documentation.md](docs/codebase_documentation.md) để đọc phân tích chi tiết từng mô-đun, luồng thực thi dữ liệu và đặc tả kỹ thuật.

<p align="center">
  <img src="assets/tiem_nang_qua_lon.jpg" alt="Tổng quan Hệ thống" width="100%"/>
</p>

### Tính năng chính

- **Quản lý bằng Config**: Mọi tham số (mô hình, dữ liệu, huấn luyện, tăng cường dữ liệu, điểm kiểm tra checkpoint) được định nghĩa hoàn toàn qua tệp YAML.
- **Model Factory**: Sử dụng mẫu thiết kế Registry giúp dễ dàng thêm kiến trúc mạng mới (ResNet, DenseNet, EfficientNet, Swin Transformer, ConvNeXt...) chỉ với một decorator duy nhất.
- **Bộ huấn luyện dùng chung (Generic Trainer)**: Một Trainer duy nhất hỗ trợ Tự động ép kiểu hỗn hợp (AMP), Cắt tầng đạo hàm (Gradient clipping), Tích lũy đạo hàm (Gradient accumulation), Lịch trình học tập có khởi động (Warmup LR), Dừng sớm (Early stopping) và lưu trữ checkpoint.
- **Mô-đun Ensemble độc lập**: Xử lý trực tiếp trên các mảng xác suất NumPy (không phụ thuộc vào PyTorch) bao gồm:
  - Hard Voting (Bầu chọn theo đa số)
  - Soft Voting (Trung bình cộng xác suất)
  - Weighted Voting (Tối ưu hóa trọng số tự động bằng Grid-search)
  - Stacking Ensemble (Logistic Regression, Random Forest, XGBoost) kết hợp với bộ tạo Out-of-Fold (OOF) để chống rò rỉ dữ liệu (data leakage).
- **Bộ điều khiển Master CLI thống nhất (`main.py`)**: Giao diện dòng lệnh tập trung tại root điều khiển toàn bộ các thao tác train, evaluate, benchmark, ensemble và report.
- **Tính tái lập (Reproducibility)**: Cố định seed toàn cục, cấu hình tính toán CUDA nhất quán và DataLoader theo seed.
- **Tương thích đa nền tảng**: Chạy tốt trên máy cục bộ (Windows, Linux, macOS), Kaggle Notebooks và Google Colab.

### Cấu trúc thư mục

```
pipeline/
├── main.py                   # Bộ điều khiển Master CLI duy nhất
├── configs/                  # Các tệp cấu hình YAML
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
│   └── utils/                # Đọc Config, Logger, Metrics, Đồ thị & Báo cáo
├── requirements.txt          # Danh sách thư viện phụ thuộc đã khóa phiên bản
└── README.md
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

# Cài đặt các thư viện cần thiết còn lại
pip install -r requirements.txt

# Kiểm tra GPU đã nhận diện chưa
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

> ⚠️ **LƯU Ý VỀ CUDA**: Hãy đảm bảo bạn cài đặt phiên bản PyTorch khớp với phiên bản CUDA Toolkit trên máy của bạn (VD: `cu124`, `cu121`). Nếu chạy trên CPU (`CUDA available: False`), tốc độ huấn luyện sẽ chậm hơn hàng chục lần.

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

#### 4. Chạy trực tiếp từ thư mục `scripts/`

Bạn cũng có thể chạy trực tiếp các script mô-đun riêng lẻ trong thư mục `scripts/`:

```bash
python scripts/train.py configs/resnet50.yaml
python scripts/evaluate.py configs/resnet50.yaml --split test
python scripts/run_experiments.py
python scripts/run_ensemble_eval.py --mode val
python scripts/generate_comparison.py
```

*Hoặc qua Python API:*

```python
import numpy as np
from src.ensemble import HardVoting, SoftVoting, WeightedVoting, StackingEnsemble

# Tải xác suất dự đoán đã lưu sau quá trình đánh giá (sử dụng tên tệp chuẩn hóa)
p_resnet = np.load("outputs/resnet50/test_probabilities.npy")
p_densenet = np.load("outputs/densenet121/test_probabilities.npy")
p_swin = np.load("outputs/swin_tiny/test_probabilities.npy")
labels = np.load("outputs/resnet50/test_labels.npy")

probs = [p_resnet, p_densenet, p_swin]

# Hard Voting
hv = HardVoting()
print("Hard Voting:", hv.evaluate(probs, labels))

# Soft Voting
sv = SoftVoting()
print("Soft Voting:", sv.evaluate(probs, labels))

# Weighted Voting
wv = WeightedVoting()
wv.fit(probs, labels)
print("Weighted Voting:", wv.evaluate(probs, labels))

# Stacking Ensemble
stacker = StackingEnsemble(meta_learner="logistic_regression")
stacker.fit(probs, labels) # Fit trên xác suất OOF để chống rò rỉ dữ liệu
print("Stacking:", stacker.evaluate(probs, labels))
```

#### 5. Thêm kiến trúc mô hình mới

Đăng ký mô hình mới trong `src/models/factory.py`:

```python
@register_model("convnext_tiny")
def _convnext_tiny(pretrained: bool = True, num_classes: int = 6, **kwargs):
    import timm
    return timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)
```

Tạo tệp cấu hình `configs/convnext_tiny.yaml` và thực thi `python main.py train configs/convnext_tiny.yaml` mà không cần thay đổi bất kỳ dòng mã nguồn huấn luyện nào!

---
## License
Academic Research & Thesis Use.
