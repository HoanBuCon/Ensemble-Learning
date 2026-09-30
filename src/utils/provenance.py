"""Immutable dataset and experiment provenance helpers."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
}


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: os.PathLike[str] | str) -> str:
    """Hash a file without modifying it."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: os.PathLike[str] | str, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _resolve_validation_dir(data_root: Path) -> Tuple[str, Path]:
    candidates = [("val", data_root / "val"), ("valid", data_root / "valid")]
    existing = [(name, path) for name, path in candidates if path.is_dir()]
    if len(existing) != 1:
        names = ", ".join(str(path) for _, path in existing) or "none"
        raise RuntimeError(
            "Dataset snapshot requires exactly one validation directory named "
            f"'val' or 'valid'; found: {names}"
        )
    return existing[0]


def collect_dataset_records(
    data_root: os.PathLike[str] | str,
    project_root: os.PathLike[str] | str = ".",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read the frozen dataset and return manifest records plus summary metadata."""
    root = Path(data_root).resolve()
    project = Path(project_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    _, validation_dir = _resolve_validation_dir(root)
    split_dirs = [
        ("train", root / "train"),
        ("validation", validation_dir),
        ("test", root / "test"),
    ]
    for split, split_dir in split_dirs:
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Required dataset split '{split}' not found: {split_dir}")

    reference_classes = sorted(path.name for path in split_dirs[0][1].iterdir() if path.is_dir())
    if not reference_classes:
        raise RuntimeError("No class directories found in the training split")
    class_to_idx = {name: index for index, name in enumerate(reference_classes)}

    records: List[Dict[str, Any]] = []
    class_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    split_counts: Dict[str, int] = {}

    for split, split_dir in split_dirs:
        split_classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        if split_classes != reference_classes:
            raise RuntimeError(
                f"Class mapping mismatch in split '{split}': "
                f"expected {reference_classes}, found {split_classes}"
            )

        split_count = 0
        for class_name in reference_classes:
            class_dir = split_dir / class_name
            files = sorted(
                path for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            for path in files:
                try:
                    relative_path = path.resolve().relative_to(project).as_posix()
                except ValueError:
                    relative_path = path.resolve().relative_to(root).as_posix()
                records.append({
                    "sample_id": sha256_text(relative_path),
                    "relative_path": relative_path,
                    "split": split,
                    "class_name": class_name,
                    "class_index": class_to_idx[class_name],
                    "sha256": sha256_file(path),
                })
                split_count += 1
                class_counts[split][class_name] += 1
        split_counts[split] = split_count

    if len({record["sample_id"] for record in records}) != len(records):
        raise RuntimeError("Dataset manifest sample_id collision detected")

    summary = {
        "total_samples": len(records),
        "train_samples": split_counts["train"],
        "validation_samples": split_counts["validation"],
        "test_samples": split_counts["test"],
        "class_order": reference_classes,
        "class_to_idx": class_to_idx,
        "class_counts": {
            split: {name: class_counts[split][name] for name in reference_classes}
            for split, _ in split_dirs
        },
        "validation_directory": validation_dir.name,
    }
    return records, summary


def create_dataset_snapshot(
    data_root: os.PathLike[str] | str = "data",
    manifest_path: os.PathLike[str] | str = "artifacts/manifests/dataset_manifest.csv",
    snapshot_path: os.PathLike[str] | str = "artifacts/manifests/dataset_snapshot.json",
    project_root: os.PathLike[str] | str = ".",
) -> Dict[str, Any]:
    """Create a read-only inventory of the current human-approved dataset."""
    records, summary = collect_dataset_records(data_root, project_root=project_root)
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "relative_path", "split", "class_name", "class_index", "sha256",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    payload = dict(summary)
    payload.update({
        "manifest_sha256": sha256_file(manifest),
        "created_at": utc_now_iso(),
        "dataset_policy": "HUMAN-APPROVED FROZEN INPUT DATASET",
    })
    write_json(snapshot_path, payload)
    return payload


def verify_dataset_snapshot(
    data_root: os.PathLike[str] | str = "data",
    manifest_path: os.PathLike[str] | str = "artifacts/manifests/dataset_manifest.csv",
    snapshot_path: os.PathLike[str] | str = "artifacts/manifests/dataset_snapshot.json",
    project_root: os.PathLike[str] | str = ".",
) -> Dict[str, Any]:
    """Verify the frozen dataset against the stored manifest without writing files."""
    manifest = Path(manifest_path)
    snapshot = Path(snapshot_path)
    if not manifest.is_file() or not snapshot.is_file():
        raise FileNotFoundError("Dataset manifest/snapshot is missing")

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        stored_records = list(csv.DictReader(handle))
    current_records, current_summary = collect_dataset_records(data_root, project_root=project_root)
    normalized_current = [
        {key: str(value) for key, value in record.items()} for record in current_records
    ]
    if stored_records != normalized_current:
        raise RuntimeError("Current dataset does not match the frozen dataset manifest")

    with snapshot.open("r", encoding="utf-8") as handle:
        stored_snapshot = json.load(handle)
    manifest_hash = sha256_file(manifest)
    if stored_snapshot.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("dataset_snapshot.json manifest hash does not match dataset_manifest.csv")
    for key in ("total_samples", "train_samples", "validation_samples", "test_samples"):
        if int(stored_snapshot[key]) != int(current_summary[key]):
            raise RuntimeError(f"Dataset snapshot count mismatch for {key}")
    return stored_snapshot


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_identity(project_root: os.PathLike[str] | str = ".") -> Dict[str, Any]:
    def run(*args: str, preserve_leading_whitespace: bool = False) -> Optional[str]:
        try:
            output = subprocess.check_output(
                ["git", *args], cwd=project_root, text=True, stderr=subprocess.DEVNULL,
            )
            # Porcelain's first two columns are data.  Calling ``strip()`` on
            # the whole output turns a first-line `` M path`` into ``M path``
            # and shifts the pathname.  Only remove record terminators here.
            return (
                output.rstrip("\r\n")
                if preserve_leading_whitespace
                else output.strip()
            )
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run(
        "status", "--porcelain", "--untracked-files=all",
        preserve_leading_whitespace=True,
    )
    scientific_changes = _scientific_changes_from_porcelain(status or "")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("branch", "--show-current"),
        "scientific_worktree_changes": scientific_changes,
    }


def _scientific_changes_from_porcelain(status: str) -> List[str]:
    """Extract dirty scientific paths without altering porcelain status columns."""
    scientific_changes: List[str] = []
    for line in status.splitlines():
        if not line:
            continue
        if len(line) < 4 or line[2] != " ":
            raise RuntimeError(f"Malformed git status --porcelain record: {line!r}")
        # For renames/copies, both the source and destination can be relevant.
        candidates = line[3:].split(" -> ")
        for candidate in candidates:
            normalized = candidate.strip('"').replace("\\", "/")
            if normalized in {"main.py", "server.py", "requirements.txt"} or normalized.startswith(
                ("configs/", "scripts/", "src/")
            ):
                scientific_changes.append(normalized)
    return list(dict.fromkeys(scientific_changes))


def require_git_commit(project_root: os.PathLike[str] | str = ".") -> str:
    """Return a full source commit or fail rather than inventing provenance."""
    identity = git_identity(project_root)
    commit = identity.get("git_commit")
    if commit is None or len(commit) != 40:
        raise RuntimeError("A full Git source commit is required for scientific artifacts")
    changes = identity.get("scientific_worktree_changes") or []
    if changes:
        raise RuntimeError(
            "Scientific source/config must be committed before artifact generation: "
            + ", ".join(changes)
        )
    return commit


def runtime_identity() -> Dict[str, Any]:
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    driver_version = None
    if torch.cuda.is_available():
        try:
            driver_version = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL,
            ).splitlines()[0].strip()
        except (OSError, subprocess.CalledProcessError, IndexError):
            driver_version = None
    return {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "ram_bytes": _physical_memory_bytes(),
        "torch_version": _package_version("torch"),
        "torchvision_version": _package_version("torchvision"),
        "timm_version": _package_version("timm"),
        "albumentations_version": _package_version("albumentations"),
        "sklearn_version": _package_version("scikit-learn"),
        "scipy_version": _package_version("scipy"),
        "xgboost_version": _package_version("xgboost"),
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "gpu_name": gpu_name,
        "driver_version": driver_version,
    }


def _physical_memory_bytes() -> Optional[int]:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except (ImportError, AttributeError):
        return None


def load_dataset_manifest_sha256(
    snapshot_path: os.PathLike[str] | str = "artifacts/manifests/dataset_snapshot.json",
) -> str:
    with open(snapshot_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    value = payload.get("manifest_sha256")
    if not value:
        raise RuntimeError(f"manifest_sha256 missing from {snapshot_path}")
    return str(value)


def write_experiment_manifest(
    output_path: os.PathLike[str] | str,
    *,
    protocol: str,
    model: str,
    config_path: os.PathLike[str] | str,
    seed: int,
    fold: Optional[int] = None,
    checkpoint_path: Optional[os.PathLike[str] | str] = None,
    prediction_path: Optional[os.PathLike[str] | str] = None,
    dataset_snapshot_path: os.PathLike[str] | str = "artifacts/manifests/dataset_snapshot.json",
    arguments: Optional[Dict[str, Any]] = None,
    project_root: os.PathLike[str] | str = ".",
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist a complete provenance record for a new scientific run."""
    config = Path(config_path)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    prediction = Path(prediction_path) if prediction_path else None
    payload: Dict[str, Any] = {
        "run_id": str(run_id) if run_id else f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        "created_at": utc_now_iso(),
        "protocol": protocol,
        "model": model,
        "fold": fold,
        **git_identity(project_root),
        "config_path": config.as_posix(),
        "config_sha256": sha256_file(config),
        "dataset_manifest_sha256": load_dataset_manifest_sha256(dataset_snapshot_path),
        "seed": int(seed),
        "command": " ".join(sys.argv),
        "arguments": arguments or {},
        **runtime_identity(),
        "checkpoint_path": checkpoint.as_posix() if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint and checkpoint.is_file() else None,
        "prediction_path": prediction.as_posix() if prediction else None,
        "prediction_sha256": sha256_file(prediction) if prediction and prediction.is_file() else None,
    }
    write_json(output_path, payload)
    return payload


def validate_run_start_manifest(
    path: os.PathLike[str] | str,
    *,
    run_id: str,
    protocol: str,
    model: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    source_commit: str,
) -> Dict[str, Any]:
    """Validate immutable run-start identity before resuming optimization."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Scientific resume requires run-start manifest: {target}")
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "run_id": run_id,
        "protocol": protocol,
        "model": model,
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "git_commit": source_commit,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"Run-start provenance mismatch at {field}: "
                f"actual={payload.get(field)!r}, expected={value!r}"
            )
    return payload
