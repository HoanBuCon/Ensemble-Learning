"""Benchmark real FINAL_V2 checkpoint and fitted-ensemble pipelines."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.verification.common_utils import base_model_order, verification_output_dir
from src.ensemble.stacking import StackingEnsemble
from src.ensemble.voting import WeightedVoting
from src.models.factory import create_model
from src.utils.config import load_config
from src.utils.provenance import runtime_identity, write_json


class FoldAveragedModel(torch.nn.Module):
    """Execute every saved fold model and average its probabilities."""

    def __init__(self, models: Sequence[torch.nn.Module]) -> None:
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probabilities = [torch.softmax(model(inputs), dim=1) for model in self.models]
        return torch.stack(probabilities, dim=0).mean(dim=0)


def _checkpoint_paths(results_root: str, protocol: str, model_name: str) -> List[Path]:
    root = Path(results_root) / protocol / model_name
    if protocol == "single_split":
        paths = [root / "best_model.pth"]
    elif protocol == "oof":
        paths = [root / "kfold" / f"fold_{index}" / "best_model.pth" for index in range(1, 6)]
    else:
        raise ValueError("protocol must be single_split or oof")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Final checkpoint(s) missing: {missing}")
    return paths


def _load_family(
    results_root: str,
    protocol: str,
    model_name: str,
    device: torch.device,
) -> Tuple[FoldAveragedModel, List[Path]]:
    config = load_config(f"configs/{model_name}.yaml")
    paths = _checkpoint_paths(results_root, protocol, model_name)
    models = []
    for path in paths:
        model = create_model(
            model_name,
            pretrained=False,
            num_classes=config.model.num_classes,
            drop_rate=config.model.drop_rate,
        )
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        models.append(model)
    return FoldAveragedModel(models).to(device).eval(), paths


def _benchmark_callable(
    operation: Callable[[], object],
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> Dict[str, object]:
    with torch.no_grad():
        for _ in range(warmup):
            operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    repeat_wall: List[float] = []
    repeat_cuda: List[float] = []
    with torch.no_grad():
        for _ in range(repeats):
            wall_times = []
            cuda_times = []
            for _ in range(iterations):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                started = time.perf_counter()
                operation()
                if device.type == "cuda":
                    end_event.record()
                    torch.cuda.synchronize(device)
                    cuda_times.append(float(start_event.elapsed_time(end_event)))
                wall_times.append((time.perf_counter() - started) * 1000.0)
            repeat_wall.append(float(np.mean(wall_times)))
            if cuda_times:
                repeat_cuda.append(float(np.mean(cuda_times)))
    mean_wall = float(np.mean(repeat_wall))
    return {
        "latency_ms": mean_wall,
        "throughput_per_second": 1000.0 / mean_wall,
        "repeat_latency_ms": repeat_wall,
        "cuda_event_latency_ms": (
            float(np.mean(repeat_cuda)) if repeat_cuda else None
        ),
        "cuda_event_repeat_latency_ms": repeat_cuda,
    }


def _ensemble_operation(
    method: str,
    families: Dict[str, FoldAveragedModel],
    model_order: Sequence[str],
    dummy_input: torch.Tensor,
    artifact_dir: Path,
) -> Callable[[], object]:
    weighted = None
    stacker = None
    if method == "weighted_voting":
        weighted = WeightedVoting.load(str(artifact_dir / "weighted_voting.json"))
        weights = torch.tensor(weighted.weights, device=dummy_input.device, dtype=dummy_input.dtype)
    elif method.startswith("stacking_"):
        learner_map = {
            "stacking_logistic_regression": ("logistic_regression", "stacking_lr.joblib"),
            "stacking_random_forest": ("random_forest", "stacking_rf.joblib"),
            "stacking_xgboost": ("xgboost", "stacking_xgb.json"),
        }
        learner, filename = learner_map[method]
        stacker = StackingEnsemble(learner).load(str(artifact_dir / filename))

    def operation() -> object:
        probabilities = [families[name](dummy_input) for name in model_order]
        stacked = torch.stack(probabilities, dim=0)
        if method == "soft_voting":
            return stacked.mean(dim=0)
        if method == "hard_voting":
            predictions = torch.argmax(stacked, dim=2)
            return torch.stack(
                [(predictions == index).float().mean(dim=0) for index in range(stacked.shape[2])],
                dim=1,
            )
        if method == "weighted_voting":
            return torch.sum(stacked * weights[:, None, None], dim=0)
        features = torch.cat(probabilities, dim=1).detach().cpu().numpy()
        return stacker._model.predict_proba(features)

    return operation


def run_hardware_benchmark(
    protocol: str,
    results_root: str = "RESULTS/FINAL_V2",
    benchmark_config: str = "configs/final_experiment.yaml",
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    with open(benchmark_config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["latency"]
    order = base_model_order()
    artifact_dir = Path(results_root) / protocol / "ensembles" / "ensemble_artifacts"
    required_artifacts = [
        artifact_dir / "weighted_voting.json",
        artifact_dir / "stacking_lr.joblib",
        artifact_dir / "stacking_rf.joblib",
        artifact_dir / "stacking_xgb.json",
    ]
    missing = [str(path) for path in required_artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Fitted ensemble artifact(s) missing: {missing}")

    rows = []
    raw_results: Dict[str, object] = {}
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.insert(0, torch.device("cuda"))

    for device in devices:
        if device.type == "cuda":
            warmup, iterations = int(config["gpu_warmup"]), int(config["gpu_iterations"])
        else:
            warmup, iterations = int(config["cpu_warmup"]), int(config["cpu_iterations"])
        repeats = int(config["repeats"])
        dummy_input = torch.randn(*config["input_shape"], device=device)
        families: Dict[str, FoldAveragedModel] = {}
        checkpoint_paths: Dict[str, List[Path]] = {}
        for model_name in order:
            family, paths = _load_family(results_root, protocol, model_name, device)
            families[model_name] = family
            checkpoint_paths[model_name] = paths

        for model_name in order:
            measurement = _benchmark_callable(
                lambda name=model_name: families[name](dummy_input),
                device, warmup, iterations, repeats,
            )
            parameter_count = sum(parameter.numel() for parameter in families[model_name].parameters())
            storage_bytes = sum(path.stat().st_size for path in checkpoint_paths[model_name])
            rows.append(_row(model_name, "base", device, measurement, parameter_count, storage_bytes, warmup, iterations, repeats))
            raw_results[f"{device.type}:{model_name}"] = measurement

        total_parameters = sum(
            parameter.numel() for family in families.values() for parameter in family.parameters()
        )
        base_storage = sum(
            path.stat().st_size for paths in checkpoint_paths.values() for path in paths
        )
        methods = [
            "soft_voting", "weighted_voting", "stacking_logistic_regression",
            "stacking_random_forest", "stacking_xgboost",
        ]
        for method in methods:
            operation = _ensemble_operation(
                method, families, order, dummy_input, artifact_dir
            )
            measurement = _benchmark_callable(
                operation, device, warmup, iterations, repeats
            )
            artifact_bytes = 0
            if method == "weighted_voting":
                artifact_bytes = (artifact_dir / "weighted_voting.json").stat().st_size
            elif method == "stacking_logistic_regression":
                artifact_bytes = (artifact_dir / "stacking_lr.joblib").stat().st_size
            elif method == "stacking_random_forest":
                artifact_bytes = (artifact_dir / "stacking_rf.joblib").stat().st_size
            elif method == "stacking_xgboost":
                artifact_bytes = (artifact_dir / "stacking_xgb.json").stat().st_size
            rows.append(_row(method, "ensemble", device, measurement, total_parameters, base_storage + artifact_bytes, warmup, iterations, repeats))
            raw_results[f"{device.type}:{method}"] = measurement

        del families
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    output = Path(save_dir) if save_dir else verification_output_dir(results_root)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"latency_{protocol}.csv", index=False)
    environment = runtime_identity()
    environment.update({
        "protocol": protocol,
        "dtype": "float32",
        "batch_size": int(config["batch_size"]),
        "input_shape": config["input_shape"],
        "torch_thread_count": torch.get_num_threads(),
        "gpu_warmup": int(config["gpu_warmup"]),
        "gpu_iterations": int(config["gpu_iterations"]),
        "cpu_warmup": int(config["cpu_warmup"]),
        "cpu_iterations": int(config["cpu_iterations"]),
        "repeats": int(config["repeats"]),
        "timing": "end_to_end synchronized wall clock; CUDA event timing also recorded",
        "raw_measurements": raw_results,
    })
    write_json(output / f"latency_{protocol}_environment.json", environment)
    return frame


def _row(
    method: str,
    method_type: str,
    device: torch.device,
    measurement: Dict[str, object],
    parameters: int,
    storage_bytes: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> Dict[str, object]:
    return {
        "method": method,
        "type": method_type,
        "device": device.type,
        "latency_ms": measurement["latency_ms"],
        "cuda_event_latency_ms": measurement["cuda_event_latency_ms"],
        "throughput_per_second": measurement["throughput_per_second"],
        "actual_parameters": parameters,
        "actual_artifact_storage_bytes": storage_bytes,
        "batch_size": 1,
        "warmup": warmup,
        "iterations": iterations,
        "repeats": repeats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark real FINAL_V2 pipelines")
    parser.add_argument("--protocol", required=True, choices=["single_split", "oof"])
    parser.add_argument("--results-root", default="RESULTS/FINAL_V2")
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()
    run_hardware_benchmark(args.protocol, args.results_root, save_dir=args.save_dir)
