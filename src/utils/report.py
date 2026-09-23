"""
Report Utilities
================

Centralized helper functions for metric extraction, checkpoint history parsing,
parameter counting, and file size measurements used across experiments and report generation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np

from src.models.factory import create_model
from src.utils.metrics import compute_metrics


def count_parameters(model_name: str, num_classes: int = 6) -> int:
    """Count trainable parameters for a registered model."""
    import torch

    model = create_model(model_name, pretrained=False, num_classes=num_classes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n_params


def get_model_size_mb(save_dir: str) -> float:
    """Get the size of the best model checkpoint in MB (supports root and kfold)."""
    path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    
    import glob
    kfold_ckpts = sorted(glob.glob(os.path.join(save_dir, "kfold", "fold_*", "best_model.pth")))
    if kfold_ckpts:
        return os.path.getsize(kfold_ckpts[0]) / (1024 * 1024)
    return 0.0


def extract_model_history_info(save_dir: str) -> Dict[str, Any]:
    """Extract best_epoch and best_val_accuracy from best_model.pth or training_history.json (supports kfold)."""
    import torch
    import glob

    candidates = [
        os.path.join(save_dir, "best_model.pth"),
    ] + sorted(glob.glob(os.path.join(save_dir, "kfold", "fold_*", "best_model.pth")))

    for best_pth in candidates:
        if os.path.exists(best_pth):
            try:
                ckpt = torch.load(best_pth, map_location="cpu", weights_only=False)
                epoch = ckpt.get("epoch", 0)
                best_val = ckpt.get("best_value", 0.0)
                if epoch > 0 and best_val > 0.0:
                    return {
                        "accepted_checkpoint_epoch": int(
                            ckpt.get("best_epoch", epoch)
                        ),
                        "accepted_checkpoint_val_accuracy": float(
                            ckpt.get("best_metric", best_val)
                        ),
                    }
            except Exception:
                pass

    history_candidates = [
        os.path.join(save_dir, "training_history.json"),
    ] + sorted(glob.glob(os.path.join(save_dir, "kfold", "fold_*", "training_history.json")))

    for history_path in history_candidates:
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                val_accs = data.get("val_accuracy", [])
                if val_accs:
                    best_idx = int(np.argmax(val_accs))
                    return {
                        "accepted_checkpoint_epoch": None,
                        "accepted_checkpoint_val_accuracy": None,
                        "raw_max_val_accuracy": float(val_accs[best_idx]),
                    }
            except Exception:
                pass

    return {
        "accepted_checkpoint_epoch": None,
        "accepted_checkpoint_val_accuracy": None,
        "raw_max_val_accuracy": None,
    }


def extract_model_val_metrics(save_dir: str) -> Dict[str, Optional[float]]:
    """Compute full validation metrics (Acc, Precision, Recall, F1) from val_probabilities or OOF."""
    # 1. Single split val probabilities
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

    # 2. K-Fold OOF probabilities
    oof_prob_path = os.path.join(save_dir, "kfold", "oof_probabilities.npy")
    oof_label_path = os.path.join(save_dir, "kfold", "oof_labels.npy")
    if not os.path.exists(oof_prob_path):
        oof_prob_path = os.path.join(save_dir, "oof_probabilities.npy")
        oof_label_path = os.path.join(save_dir, "oof_labels.npy")

    if os.path.exists(oof_prob_path) and os.path.exists(oof_label_path):
        try:
            oof_probs = np.load(oof_prob_path)
            oof_labels = np.load(oof_label_path)
            preds = np.argmax(oof_probs, axis=1)
            m = compute_metrics(oof_labels, preds)
            return {
                "Val_Accuracy": m["accuracy"] * 100,
                "Val_Precision": m["precision"] * 100,
                "Val_Recall": m["recall"] * 100,
                "Val_F1_Score": m["f1_score"] * 100,
            }
        except Exception:
            pass

    # Missing prediction evidence is reported as unavailable; no metric fabrication.
    return {
        "Val_Accuracy": None,
        "Val_Precision": None,
        "Val_Recall": None,
        "Val_F1_Score": None,
    }


def extract_model_training_time(save_dir: str) -> str:
    """Extract total elapsed training time string from history.csv or kfold logs."""
    import glob
    import pandas as pd

    # Single split history.csv
    history_csv = os.path.join(save_dir, "history.csv")
    if os.path.exists(history_csv):
        try:
            df_hist = pd.read_csv(history_csv)
            if "elapsed" in df_hist.columns and len(df_hist) > 0:
                last_elapsed = str(df_hist["elapsed"].iloc[-1]).strip()
                return last_elapsed
        except Exception:
            pass

    # K-Fold summary or fold history
    fold_histories = sorted(glob.glob(os.path.join(save_dir, "kfold", "fold_*", "history.csv")))
    if fold_histories:
        try:
            total_seconds = 0
            for fh in fold_histories:
                df_f = pd.read_csv(fh)
                if "epoch_time_s" in df_f.columns:
                    total_seconds += int(df_f["epoch_time_s"].sum())
            if total_seconds > 0:
                mins, secs = divmod(total_seconds, 60)
                hours, mins = divmod(mins, 60)
                if hours > 0:
                    return f"{hours}h {mins}m {secs}s"
                return f"{mins}m {secs}s"
        except Exception:
            pass

    return "N/A"
