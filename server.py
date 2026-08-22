"""
FastAPI Image Classification Server & REST API Pipeline
========================================================

FastAPI server supporting 5 flexible classification modes:
1. Single Base Model (ResNet-50, DenseNet-121, EfficientNet-B0, Swin-Tiny)
2. All Base Models (Simultaneous comparison of 4 backbones)
3. Single Ensemble Method (Hard Voting, Soft Voting, Weighted Voting, Stacking LR/RF/XGB)
4. All Ensemble Methods (Simultaneous comparison of 6 ensemble methods)
5. Full Benchmark (All 4 Base Models + All 6 Ensemble Methods = 10 methods per image)

Supports both:
- Single-Split standard training outputs (outputs/<model_name>/best_model.pth)
- 5-Fold Cross-Validation / OOF outputs (outputs/<model_name>/kfold/fold_*/best_model.pth)

Run server:
    python server.py
    # or: python main.py serve
"""

from __future__ import annotations

import glob
import io
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Any, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.factory import create_model, list_models
from src.ensemble import HardVoting, SoftVoting, WeightedVoting, StackingEnsemble
from src.utils.config import load_config, load_dataset_config

DATASET_CONFIG = load_dataset_config()
DEFAULT_CLASS_NAMES = DATASET_CONFIG.get("classes", [
    "Brown_Blight",
    "Gray_Blight",
    "Green_mirid_bug",
    "Healthy_leaf",
    "Helopeltis",
    "Tea_algal_leaf_spot",
])

BASE_MODEL_KEYS = ["resnet50", "densenet121", "efficientnet_b0", "swin_tiny"]
ENSEMBLE_KEYS = [
    "hard_voting",
    "soft_voting",
    "weighted_voting",
    "stacking_logistic",
    "stacking_rf",
    "stacking_xgb",
]

ENSEMBLE_DISPLAY_NAMES = {
    "hard_voting": "Hard Voting (Majority Vote)",
    "soft_voting": "Soft Voting (Probability Averaging)",
    "weighted_voting": "Weighted Voting Ensemble",
    "stacking_logistic": "Stacking (Logistic Regression)",
    "stacking_rf": "Stacking (Random Forest)",
    "stacking_xgb": "Stacking (XGBoost)",
}

BASE_DISPLAY_NAMES = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "swin_tiny": "Swin Transformer Tiny",
}

app = FastAPI(
    title="Tea Leaf Disease Classification API",
    description="Multi-Image Classification Pipeline & Multi-Model Ensemble Benchmark Server",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Cache
BASE_MODELS: Dict[str, torch.nn.Module] = {}
CLASS_NAMES: List[str] = DEFAULT_CLASS_NAMES
ENSEMBLE_MODELS: Dict[str, Any] = {}
LOADED_MODEL_INFO: Dict[str, str] = {}


def get_possible_output_dirs() -> List[str]:
    """Return prioritized list of output directories."""
    candidates = [
        os.path.join(PROJECT_ROOT, "outputs"),
        os.path.join(PROJECT_ROOT, "OOF_Results", "outputs"),
        os.path.join(PROJECT_ROOT, "Default_Results", "outputs"),
    ]
    return [d for d in candidates if os.path.isdir(d)]


def get_preprocess_transform(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def find_model_checkpoint(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Find best checkpoint path, class mapping path, and description info for a base model.
    Supports both Single Split ('best_model.pth') and 5-Fold OOF ('kfold/fold_*/best_model.pth').
    """
    for out_dir in get_possible_output_dirs():
        possible_dirs = sorted(glob.glob(os.path.join(out_dir, f"{name}*")))
        for d in possible_dirs:
            # 1. Check direct single-split checkpoint
            direct_pth = os.path.join(d, "best_model.pth")
            cmap_path = os.path.join(d, "class_to_idx.json")
            if not os.path.exists(cmap_path):
                cmap_path = os.path.join(d, "kfold", "class_to_idx.json")
            cmap = cmap_path if os.path.exists(cmap_path) else None

            if os.path.exists(direct_pth):
                return direct_pth, cmap, "Single Split Checkpoint"

            # 2. Check 5-Fold OOF directory
            kfold_dir = os.path.join(d, "kfold")
            if os.path.isdir(kfold_dir):
                if not cmap:
                    k_cmap = os.path.join(kfold_dir, "class_to_idx.json")
                    if os.path.exists(k_cmap):
                        cmap = k_cmap

                folds = sorted(glob.glob(os.path.join(kfold_dir, "fold_*")))
                best_fold_pth = None
                best_acc = -1.0
                best_fold_name = None

                for f in folds:
                    pth = os.path.join(f, "best_model.pth")
                    if not os.path.exists(pth):
                        continue
                    m_json = os.path.join(f, "metrics.json")
                    acc = 0.0
                    if os.path.exists(m_json):
                        try:
                            with open(m_json, "r", encoding="utf-8") as fp:
                                acc = json.load(fp).get("accuracy", 0.0)
                        except Exception:
                            pass
                    if acc > best_acc or best_fold_pth is None:
                        best_acc = acc
                        best_fold_pth = pth
                        best_fold_name = os.path.basename(f)

                if best_fold_pth:
                    info = f"5-Fold OOF ({best_fold_name}" + (f", Acc: {best_acc*100:.2f}%)" if best_acc > 0 else ")")
                    return best_fold_pth, cmap, info

    return None, None, None


def load_all_base_models():
    """Load and cache all 4 base models on startup, supporting both standard and OOF checkpoints."""
    global CLASS_NAMES, BASE_MODELS, LOADED_MODEL_INFO
    BASE_MODELS.clear()
    LOADED_MODEL_INFO.clear()

    print(f"\n{'='*70}")
    print(f"   LOADING BASE COMPUTER VISION MODELS (Device: {DEVICE})")
    print(f"{'='*70}")

    for name in BASE_MODEL_KEYS:
        weights_path, cmap_path, info_str = find_model_checkpoint(name)

        if not weights_path or not os.path.exists(weights_path):
            print(f"  [!] {BASE_DISPLAY_NAMES.get(name, name)}: No checkpoint found. Skipping.")
            continue

        if cmap_path and os.path.exists(cmap_path):
            try:
                with open(cmap_path, "r", encoding="utf-8") as f:
                    cmap = json.load(f)
                    CLASS_NAMES = [k for k, _ in sorted(cmap.items(), key=lambda item: item[1])]
            except Exception:
                pass

        try:
            model = create_model(model_name=name, pretrained=False, num_classes=len(CLASS_NAMES))
            checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()

            BASE_MODELS[name] = model
            LOADED_MODEL_INFO[name] = info_str or "Loaded"
            print(f"  [+] {BASE_DISPLAY_NAMES.get(name, name):<24} -> {weights_path} [{info_str}]")
        except Exception as e:
            print(f"  [X] Failed to load {name} from {weights_path}: {e}")

    print(f"Loaded {len(BASE_MODELS)}/{len(BASE_MODEL_KEYS)} base models successfully.\n")


def init_ensemble_models():
    """Fit and cache ensemble models on stored val/oof probability arrays."""
    fit_probs_list = []
    fit_labels = None

    available_models = [m for m in BASE_MODEL_KEYS if m in BASE_MODELS]
    if len(available_models) < 2:
        print(f"Ensemble initialization skipped (requires at least 2 loaded base models).")
        return

    output_dirs = get_possible_output_dirs()

    # Search for training probabilities for each available model
    for m_name in available_models:
        found_p, found_l = None, None
        for out_dir in output_dirs:
            possible_dirs = sorted(glob.glob(os.path.join(out_dir, f"{m_name}*")))
            for d in possible_dirs:
                # Priority 1: 5-Fold OOF probabilities
                candidates = [
                    (os.path.join(d, "kfold", "oof_probabilities.npy"), os.path.join(d, "kfold", "oof_labels.npy")),
                    (os.path.join(d, "oof", "oof_probabilities.npy"), os.path.join(d, "oof", "oof_labels.npy")),
                    (os.path.join(d, "val_probabilities.npy"), os.path.join(d, "val_labels.npy")),
                    (os.path.join(d, "kfold", "test_probabilities.npy"), os.path.join(d, "kfold", "oof_labels.npy")),
                    (os.path.join(d, "probabilities.npy"), os.path.join(d, "labels.npy")),
                ]
                for p_cand, l_cand in candidates:
                    if os.path.exists(p_cand) and os.path.exists(l_cand):
                        found_p = np.load(p_cand)
                        found_l = np.load(l_cand)
                        break
                if found_p is not None:
                    break
            if found_p is not None:
                break

        if found_p is not None and found_l is not None:
            fit_probs_list.append(found_p)
            if fit_labels is None:
                fit_labels = found_l

    if len(fit_probs_list) == len(available_models) and fit_labels is not None:
        print(f"--> Initializing and fitting Ensemble models on {len(fit_labels)} validation/OOF samples...")

        # 1. Hard Voting
        ENSEMBLE_MODELS["hard_voting"] = HardVoting()

        # 2. Soft Voting
        ENSEMBLE_MODELS["soft_voting"] = SoftVoting()

        # 3. Weighted Voting
        opt_weights = None
        for out_dir in output_dirs:
            for sub in ["oof", "val", ""]:
                comp_csv = os.path.join(out_dir, sub, "ensemble_comparison.csv")
                if os.path.exists(comp_csv):
                    try:
                        import pandas as pd
                        df = pd.read_csv(comp_csv)
                        for _, row in df.iterrows():
                            if "Weighted Voting" in str(row.get("Method", "")):
                                details = str(row.get("Details", ""))
                                parsed_w = []
                                for m in available_models:
                                    match = re.search(rf"{m}:\s*([\d\.]+)", details)
                                    if match:
                                        parsed_w.append(float(match.group(1)))
                                if len(parsed_w) == len(available_models):
                                    opt_weights = parsed_w
                                break
                    except Exception:
                        pass
                if opt_weights:
                    break
            if opt_weights:
                break

        if opt_weights:
            print(f"  [Weighted Voting] Loaded precomputed optimal weights: {opt_weights}")
            wv = WeightedVoting(weights=opt_weights)
        else:
            wv = WeightedVoting(grid_resolution=6)
            wv.fit(fit_probs_list, fit_labels)
            print(f"  [Weighted Voting] Optimized weights via grid search: {wv.weights}")
        ENSEMBLE_MODELS["weighted_voting"] = wv

        # 4. Stacking Logistic Regression
        st_lr = StackingEnsemble(meta_learner="logistic_regression")
        st_lr.fit(fit_probs_list, fit_labels)
        ENSEMBLE_MODELS["stacking_logistic"] = st_lr
        print(f"  [Stacking LR] Meta-Learner fitted.")

        # 5. Stacking Random Forest
        st_rf = StackingEnsemble(meta_learner="random_forest")
        st_rf.fit(fit_probs_list, fit_labels)
        ENSEMBLE_MODELS["stacking_rf"] = st_rf
        print(f"  [Stacking RF] Meta-Learner fitted.")

        # 6. Stacking XGBoost
        try:
            st_xgb = StackingEnsemble(meta_learner="xgboost")
            st_xgb.fit(fit_probs_list, fit_labels)
            ENSEMBLE_MODELS["stacking_xgb"] = st_xgb
            print(f"  [Stacking XGBoost] Meta-Learner fitted.")
        except Exception as e:
            print(f"  [Stacking XGBoost] Skipped: {e}")

        print(f"Ensemble models ready: {list(ENSEMBLE_MODELS.keys())}\n")
    else:
        # Fallback to zero-training voting methods
        ENSEMBLE_MODELS["hard_voting"] = HardVoting()
        ENSEMBLE_MODELS["soft_voting"] = SoftVoting()
        print(f"Fallback: Initialized HardVoting and SoftVoting (zero-fit).")


@app.on_event("startup")
def startup_event():
    load_all_base_models()
    init_ensemble_models()


def process_image_bytes(image_bytes: bytes, transform: A.Compose) -> torch.Tensor:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file format.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image_rgb)
    return augmented["image"]


def compute_base_probabilities(batch_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
    """Run inference across all available base models and return softmax probabilities."""
    base_probs = {}
    with torch.no_grad():
        for name, model in BASE_MODELS.items():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            base_probs[name] = probs
    return base_probs


def compute_ensemble_predictions(
    base_probs_map: Dict[str, np.ndarray], ensemble_key: str
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute ensemble predictions/probabilities for a specific ensemble method.
    
    Returns:
        Tuple of (predicted_class_indices, probability_matrix_or_None)
    """
    probs_list = [base_probs_map[m] for m in BASE_MODEL_KEYS if m in base_probs_map]

    if not probs_list:
        raise ValueError("No base model probabilities available to compute ensemble.")

    if ensemble_key == "hard_voting":
        hv = ENSEMBLE_MODELS.get("hard_voting", HardVoting())
        preds = hv.predict(probs_list)
        return preds, None

    elif ensemble_key == "soft_voting":
        sv = ENSEMBLE_MODELS.get("soft_voting", SoftVoting())
        probs = sv.predict_proba(probs_list)
        preds = np.argmax(probs, axis=1)
        return preds, probs

    elif ensemble_key == "weighted_voting":
        wv = ENSEMBLE_MODELS.get("weighted_voting")
        if wv is not None:
            probs = wv.predict_proba(probs_list)
            preds = np.argmax(probs, axis=1)
            return preds, probs
        else:
            sv = SoftVoting()
            probs = sv.predict_proba(probs_list)
            return np.argmax(probs, axis=1), probs

    elif ensemble_key in ["stacking_logistic", "stacking_rf", "stacking_xgb"]:
        st = ENSEMBLE_MODELS.get(ensemble_key)
        if st is not None:
            probs = st.predict_proba(probs_list)
            preds = np.argmax(probs, axis=1)
            return preds, probs
        else:
            sv = SoftVoting()
            probs = sv.predict_proba(probs_list)
            return np.argmax(probs, axis=1), probs

    raise ValueError(f"Unknown ensemble key: {ensemble_key}")


@app.get("/api/v1/config")
def get_pipeline_config():
    """Return available base models, ensemble models, and system status."""
    return {
        "base_models": [
            {
                "id": k,
                "name": BASE_DISPLAY_NAMES.get(k, k),
                "loaded": k in BASE_MODELS,
                "info": LOADED_MODEL_INFO.get(k, "Not loaded"),
            }
            for k in BASE_MODEL_KEYS
        ],
        "ensemble_methods": [
            {"id": k, "name": ENSEMBLE_DISPLAY_NAMES.get(k, k), "loaded": k in ENSEMBLE_MODELS}
            for k in ENSEMBLE_KEYS
        ],
        "class_names": CLASS_NAMES,
        "device": str(DEVICE),
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Return empty 204 response for browser favicon requests."""
    return Response(status_code=204)


@app.post("/api/v1/predict")
async def predict(
    files: List[UploadFile] = File(..., description="Upload multiple image files"),
    mode_type: str = Form("single_base", description="Mode: single_base, all_base, single_ensemble, all_ensemble, all_all"),
    selected_model: Optional[str] = Form("resnet50", description="Selected model or ensemble key"),
):
    """
    Unified Endpoint Supporting 5 Classification Modes:
    1. single_base: Individual Base Model
    2. all_base: All 4 Base Models simultaneously
    3. single_ensemble: Individual Ensemble Method
    4. all_ensemble: All 6 Ensemble Methods simultaneously
    5. all_all: Complete Benchmark (All Base Models + All Ensemble Methods)
    """
    if not files:
        raise HTTPException(status_code=400, detail="No image files uploaded.")

    if not BASE_MODELS:
        raise HTTPException(
            status_code=500,
            detail="No base models loaded. Please train models first or verify outputs/ directory."
        )

    transform = get_preprocess_transform(image_size=224)
    start_total_time = time.time()

    image_tensors = []
    valid_filenames = []
    errors = []

    for file in files:
        try:
            contents = await file.read()
            tensor = process_image_bytes(contents, transform)
            image_tensors.append(tensor)
            valid_filenames.append(file.filename)
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})

    if not image_tensors:
        raise HTTPException(status_code=400, detail="Failed to process any uploaded images.")

    batch_tensor = torch.stack(image_tensors).to(DEVICE)
    num_samples = len(valid_filenames)

    # 1. Compute Base Probabilities
    base_probs = compute_base_probabilities(batch_tensor)

    predictions_by_file = []

    for img_idx, filename in enumerate(valid_filenames):
        img_result: Dict[str, Any] = {
            "filename": filename,
            "predictions": {},
        }

        # Determine which methods to run based on mode_type
        run_base_keys = []
        run_ensemble_keys = []

        if mode_type == "single_base":
            target = selected_model or "resnet50"
            if target in BASE_MODELS:
                run_base_keys.append(target)
            else:
                run_base_keys.append(list(BASE_MODELS.keys())[0])

        elif mode_type == "all_base":
            run_base_keys = [k for k in BASE_MODEL_KEYS if k in BASE_MODELS]

        elif mode_type == "single_ensemble":
            target = selected_model or "soft_voting"
            run_ensemble_keys.append(target)

        elif mode_type == "all_ensemble":
            run_ensemble_keys = [k for k in ENSEMBLE_KEYS if k in ENSEMBLE_MODELS or k in ["hard_voting", "soft_voting"]]

        elif mode_type == "all_all":
            run_base_keys = [k for k in BASE_MODEL_KEYS if k in BASE_MODELS]
            run_ensemble_keys = [k for k in ENSEMBLE_KEYS if k in ENSEMBLE_MODELS or k in ["hard_voting", "soft_voting"]]

        # Extract Base Model Predictions for this image
        for b_key in run_base_keys:
            probs = base_probs[b_key][img_idx]
            pred_class_idx = int(np.argmax(probs))
            conf = float(probs[pred_class_idx] * 100)
            prob_dict = {cls: float(probs[i] * 100) for i, cls in enumerate(CLASS_NAMES)}

            img_result["predictions"][b_key] = {
                "name": BASE_DISPLAY_NAMES.get(b_key, b_key),
                "category": "Base Model",
                "predicted_class": CLASS_NAMES[pred_class_idx],
                "confidence_percent": round(conf, 2),
                "probabilities": prob_dict,
            }

        # Extract Ensemble Predictions for this image
        for e_key in run_ensemble_keys:
            single_base_map = {k: v[img_idx:img_idx+1] for k, v in base_probs.items()}
            preds, probs_mat = compute_ensemble_predictions(single_base_map, e_key)

            pred_class_idx = int(preds[0])
            prob_dict = {}
            conf = 0.0

            if probs_mat is not None:
                probs = probs_mat[0]
                conf = float(probs[pred_class_idx] * 100)
                prob_dict = {cls: float(probs[i] * 100) for i, cls in enumerate(CLASS_NAMES)}
            else:
                conf = 100.0  # Majority vote placeholder

            img_result["predictions"][e_key] = {
                "name": ENSEMBLE_DISPLAY_NAMES.get(e_key, e_key),
                "category": "Ensemble Method",
                "predicted_class": CLASS_NAMES[pred_class_idx],
                "confidence_percent": round(conf, 2) if probs_mat is not None else "N/A (Majority Vote)",
                "probabilities": prob_dict,
            }

        # Consensus summary class
        all_predicted_classes = [v["predicted_class"] for v in img_result["predictions"].values()]
        if all_predicted_classes:
            from collections import Counter
            consensus_class = Counter(all_predicted_classes).most_common(1)[0][0]
            img_result["consensus_class"] = consensus_class

        predictions_by_file.append(img_result)

    total_time_ms = (time.time() - start_total_time) * 1000

    return {
        "status": "success",
        "mode_type": mode_type,
        "selected_target": selected_model,
        "total_images": num_samples,
        "execution_time_ms": round(total_time_ms, 2),
        "results": predictions_by_file,
    }


@app.get("/api/v1/analytics")
def get_analytics():
    """Retrieve full model evaluation metrics, per-class metrics, and epoch history for interactive charts."""
    ds_cfg = load_dataset_config()
    analytics_data = {
        "status": "success",
        "dataset_info": ds_cfg,
        "class_names": ds_cfg.get("classes", DEFAULT_CLASS_NAMES),
        "class_display_names": ds_cfg.get("display_names", {}),
        "base_models": {},
        "ensemble_models": {},
    }

    output_dirs = get_possible_output_dirs()

    for model_key in BASE_MODEL_KEYS:
        model_info = {
            "name": BASE_DISPLAY_NAMES.get(model_key, model_key),
            "key": model_key,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "per_class": {},
            "history": [],
        }

        found_metrics = None
        found_history = None

        for out_dir in output_dirs:
            possible_dirs = sorted(glob.glob(os.path.join(out_dir, f"{model_key}*")))
            for d in possible_dirs:
                # Metrics candidates (OOF, Root, Folds)
                m_cands = [
                    os.path.join(d, "kfold", "oof_metrics.json"),
                    os.path.join(d, "metrics.json"),
                    os.path.join(d, "kfold", "fold_1", "metrics.json"),
                    os.path.join(d, "val_evaluation", "metrics.json"),
                ]
                for mc in m_cands:
                    if os.path.exists(mc) and found_metrics is None:
                        try:
                            with open(mc, "r", encoding="utf-8") as f:
                                found_metrics = json.load(f)
                                break
                        except Exception:
                            pass

                # History candidates
                h_cands = [
                    os.path.join(d, "kfold", "fold_1", "history.csv"),
                    os.path.join(d, "history.csv"),
                ]
                for hc in h_cands:
                    if os.path.exists(hc) and found_history is None:
                        found_history = hc
                        break

            if found_metrics and found_history:
                break

        if found_metrics:
            model_info["accuracy"] = round(found_metrics.get("accuracy", 0) * 100, 2)
            model_info["precision"] = round(found_metrics.get("precision", 0) * 100, 2)
            model_info["recall"] = round(found_metrics.get("recall", 0) * 100, 2)
            model_info["f1_score"] = round(found_metrics.get("f1_score", 0) * 100, 2)

            per_class_raw = found_metrics.get("per_class", {})
            for c_name, c_metrics in per_class_raw.items():
                model_info["per_class"][c_name] = {
                    "precision": round(c_metrics.get("precision", 0) * 100, 2),
                    "recall": round(c_metrics.get("recall", 0) * 100, 2),
                    "f1_score": round(c_metrics.get("f1_score", 0) * 100, 2),
                }

        if found_history:
            try:
                import pandas as pd
                df = pd.read_csv(found_history)
                for _, row in df.iterrows():
                    val_acc_raw = float(row.get("val_acc", 0))
                    val_acc_val = round(val_acc_raw * 100, 2) if val_acc_raw <= 1.0 else round(val_acc_raw, 2)
                    model_info["history"].append({
                        "epoch": int(row.get("epoch", 0)),
                        "train_loss": round(float(row.get("train_loss", 0)), 4),
                        "val_loss": round(float(row.get("val_loss", 0)), 4),
                        "val_acc": val_acc_val,
                    })
            except Exception as e:
                print(f"Error reading history for {model_key}: {e}")

        analytics_data["base_models"][model_key] = model_info

    # Read Ensemble Metrics from ensemble_comparison.csv (check oof -> val -> root)
    for out_dir in output_dirs:
        for sub in ["oof", "val", ""]:
            ensemble_csv = os.path.join(out_dir, sub, "ensemble_comparison.csv")
            if os.path.exists(ensemble_csv):
                try:
                    import pandas as pd
                    df_ens = pd.read_csv(ensemble_csv)
                    for _, row in df_ens.iterrows():
                        method_name = str(row.get("Method", ""))
                        method_type = str(row.get("Type", ""))
                        if "Ensemble" in method_type or "Stacking" in method_type or "Voting" in method_type:
                            acc_str = str(row.get("Accuracy", "0")).replace("%", "").strip()
                            prec_str = str(row.get("Precision", "0")).replace("%", "").strip()
                            rec_str = str(row.get("Recall", "0")).replace("%", "").strip()
                            f1_str = str(row.get("F1_Score", "0")).replace("%", "").strip()
                            imp_str = str(row.get("Improvement", "0")).replace("%", "").replace("+", "").strip()

                            try:
                                imp_val = float(imp_str) if "Base" not in imp_str else 0.0
                            except ValueError:
                                imp_val = 0.0

                            analytics_data["ensemble_models"][method_name] = {
                                "name": method_name,
                                "type": method_type,
                                "accuracy": float(acc_str),
                                "precision": float(prec_str),
                                "recall": float(rec_str),
                                "f1_score": float(f1_str),
                                "improvement": imp_val,
                                "details": str(row.get("Details", "")),
                                "per_class": {},
                            }
                    if analytics_data["ensemble_models"]:
                        break
                except Exception as e:
                    print(f"Error reading ensemble_comparison.csv: {e}")
        if analytics_data["ensemble_models"]:
            break

    # Read exact per-class metrics from ensemble_full_metrics.json if available
    for out_dir in output_dirs:
        for sub in ["oof", "val", ""]:
            full_json_path = os.path.join(out_dir, sub, "ensemble_full_metrics.json")
            if os.path.exists(full_json_path):
                try:
                    with open(full_json_path, "r", encoding="utf-8") as f:
                        full_json = json.load(f)
                        for ens_name in analytics_data["ensemble_models"]:
                            if ens_name in full_json:
                                per_cls_raw = full_json[ens_name].get("per_class", {})
                                per_cls_dict = {}
                                for c_name, c_metrics in per_cls_raw.items():
                                    per_cls_dict[c_name] = {
                                        "precision": round(c_metrics.get("precision", 0) * 100, 2),
                                        "recall": round(c_metrics.get("recall", 0) * 100, 2),
                                        "f1_score": round(c_metrics.get("f1_score", 0) * 100, 2),
                                    }
                                analytics_data["ensemble_models"][ens_name]["per_class"] = per_cls_dict
                        if any(v.get("per_class") for v in analytics_data["ensemble_models"].values()):
                            break
                except Exception as e:
                    print(f"Error reading ensemble_full_metrics.json: {e}")
        if any(v.get("per_class") for v in analytics_data["ensemble_models"].values()):
            break

    return analytics_data


# Mount Static Files from web/ directory
web_dir = os.path.join(PROJECT_ROOT, "web")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/")
def serve_dashboard():
    """Serve localhost Web Dashboard UI from web/index.html."""
    index_path = os.path.join(PROJECT_ROOT, "web", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend web/index.html file not found.")
    return FileResponse(index_path)


if __name__ == "__main__":
    print(f"Starting FastAPI Enterprise Inference Server on http://127.0.0.1:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
