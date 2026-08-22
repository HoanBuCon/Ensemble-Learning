"""
Report Utilities
================

Centralized helper functions for metric extraction, checkpoint history parsing,
parameter counting, and file size measurements used across experiments and report generation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np

from src.models.factory import create_model
from src.utils.metrics import compute_metrics


def count_parameters(model_name: str, num_classes: int = 6) -> int:
    """Count trainable parameters for a registered model."""
    import torch

    try:
        model = create_model(model_name, pretrained=False, num_classes=num_classes)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return n_params
    except Exception:
        return 0


def get_model_size_mb(save_dir: str) -> float:
    """Get the size of the best model checkpoint in MB."""
    path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def extract_model_history_info(save_dir: str) -> Dict[str, Any]:
    """Extract best_epoch and best_val_accuracy from best_model.pth or training_history.json."""
    import torch

    best_pth = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_pth):
        try:
            ckpt = torch.load(best_pth, map_location="cpu", weights_only=False)
            epoch = ckpt.get("epoch", 0)
            best_val = ckpt.get("best_value", 0.0)
            if epoch > 0 and best_val > 0.0:
                return {
                    "best_epoch": int(epoch),
                    "best_val_accuracy": float(best_val),
                }
        except Exception:
            pass

    history_path = os.path.join(save_dir, "training_history.json")
    best_epoch = 0
    best_val_acc = 0.0

    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            val_accs = data.get("val_accuracy", [])
            if val_accs:
                best_idx = int(np.argmax(val_accs))
                best_epoch = best_idx + 1  # 1-indexed epoch
                best_val_acc = float(val_accs[best_idx])
        except Exception:
            pass

    return {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
    }


def extract_model_val_metrics(save_dir: str) -> Dict[str, float]:
    """Compute full validation metrics (Acc, Precision, Recall, F1) from val_probabilities.npy."""
    val_prob_path = os.path.join(save_dir, "val_probabilities.npy")
    val_label_path = os.path.join(save_dir, "val_labels.npy")

    if os.path.exists(val_prob_path) and os.path.exists(val_label_path):
        try:
            val_probs = np.load(val_prob_path)
            val_labels = np.load(val_label_path)
            preds = np.argmax(val_probs, axis=1)
            m = compute_metrics(val_labels, preds)
            return {
                "Val_Accuracy": m["accuracy"] * 100,
                "Val_Precision": m["precision"] * 100,
                "Val_Recall": m["recall"] * 100,
                "Val_F1_Score": m["f1_score"] * 100,
            }
        except Exception:
            pass

    hist_info = extract_model_history_info(save_dir)
    best_val = hist_info["best_val_accuracy"]
    return {
        "Val_Accuracy": best_val,
        "Val_Precision": best_val,
        "Val_Recall": best_val,
        "Val_F1_Score": best_val,
    }


def extract_model_training_time(save_dir: str) -> str:
    """Extract total elapsed training time string from history.csv if available."""
    history_csv = os.path.join(save_dir, "history.csv")
    if os.path.exists(history_csv):
        try:
            import pandas as pd
            df_hist = pd.read_csv(history_csv)
            if "elapsed" in df_hist.columns and len(df_hist) > 0:
                last_elapsed = str(df_hist["elapsed"].iloc[-1]).strip()
                return last_elapsed
        except Exception:
            pass
    return "N/A"
