"""Strict final-run identity and backbone configuration registry."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

import yaml


BACKBONE_CONFIG_PATHS = (
    "configs/resnet50.yaml",
    "configs/densenet121.yaml",
    "configs/efficientnet_b0.yaml",
    "configs/swin_tiny.yaml",
)
SUPPORTED_PROTOCOLS = {"single_split", "oof"}
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_run_id(run_id: str) -> str:
    value = str(run_id or "").strip()
    if not _RUN_ID.fullmatch(value):
        raise ValueError(
            "run_id is required and must contain only letters, digits, '.', '_' or '-'"
        )
    return value


def validate_protocol(protocol: str) -> str:
    value = str(protocol)
    if value not in SUPPORTED_PROTOCOLS:
        raise ValueError("protocol must be exactly 'single_split' or 'oof'")
    return value


def final_run_root(results_root: str, run_id: str) -> Path:
    """Return the only canonical root for a scientific run."""
    return Path(results_root) / "runs" / validate_run_id(run_id)


def protocol_run_root(results_root: str, run_id: str, protocol: str) -> Path:
    return final_run_root(results_root, run_id) / validate_protocol(protocol)


def model_run_root(
    results_root: str,
    run_id: str,
    protocol: str,
    model_name: str,
) -> Path:
    root = protocol_run_root(results_root, run_id, protocol) / str(model_name)
    return root / "kfold" if protocol == "oof" else root


def resolve_backbone_config_paths(paths: Iterable[str] | None = None) -> List[str]:
    """Resolve only executable backbone configs; reject scientific config lookalikes."""
    selected = list(paths) if paths is not None else list(BACKBONE_CONFIG_PATHS)
    allowed = {Path(path).as_posix() for path in BACKBONE_CONFIG_PATHS}
    resolved: List[str] = []
    for value in selected:
        normalized = Path(value).as_posix()
        if normalized not in allowed:
            raise ValueError(f"Not a registered backbone config: {value}")
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Backbone config not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        model = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(model, dict) or not model.get("name"):
            raise ValueError(f"Backbone config has no model.name: {path}")
        resolved.append(str(path))
    return resolved


def prepare_exact_run_directory(path: Path, *, resume: bool) -> Path:
    """Separate scratch and resume without suffixing or latest-directory discovery."""
    target = Path(path)
    if resume:
        checkpoint = target / "last_model.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        return target
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            f"Fresh run target is not empty: {target}. Use a new run_id or explicit resume."
        )
    target.mkdir(parents=True, exist_ok=True)
    return target
