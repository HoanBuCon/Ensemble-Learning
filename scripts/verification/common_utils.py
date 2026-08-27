"""
Verification Utilities & Path Resolver
======================================
Provides dynamic auto-discovery of output directories across multiple directory layouts.
"""

from __future__ import annotations
import os
from typing import Optional, Tuple


def resolve_outputs_dirs(
    default_dir: Optional[str] = None,
    oof_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Dynamically resolve directories for Default (Single-Split) and OOF (5-Fold) results.
    
    Priority Order for Default:
      1. Explicit parameter / CLI argument
      2. RESULTS/DEFAULT_TRAINING/outputs
      3. RESULTS/DEFAULT_TRAINING
      4. Default_Result_V2/outputs
      5. Default_Result_V2
      6. outputs
      
    Priority Order for OOF:
      1. Explicit parameter / CLI argument
      2. RESULTS/OOF_TRAINING/outputs
      3. RESULTS/OOF_TRAINING
      4. OOF_Results/outputs
      5. OOF_Results
      6. outputs
    """
    # 1. Resolve Default dir
    def_candidates = [
        default_dir,
        "RESULTS/DEFAULT_TRAINING/outputs",
        "RESULTS/DEFAULT_TRAINING",
        "outputs",
    ]
    resolved_default = None
    for p in def_candidates:
        if p and os.path.exists(p) and (
            os.path.exists(os.path.join(p, "densenet121")) or 
            os.path.exists(os.path.join(p, "val"))
        ):
            resolved_default = p
            break
    if not resolved_default:
        resolved_default = "RESULTS/DEFAULT_TRAINING/outputs"

    # 2. Resolve OOF dir
    oof_candidates = [
        oof_dir,
        "RESULTS/OOF_TRAINING/outputs",
        "RESULTS/OOF_TRAINING",
        "outputs",
    ]
    resolved_oof = None
    for p in oof_candidates:
        if p and os.path.exists(p) and (
            os.path.exists(os.path.join(p, "densenet121")) or 
            os.path.exists(os.path.join(p, "oof"))
        ):
            resolved_oof = p
            break
    if not resolved_oof:
        resolved_oof = "RESULTS/OOF_TRAINING/outputs"

    return resolved_default, resolved_oof
