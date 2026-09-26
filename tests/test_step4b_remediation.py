from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import main
import server
from scripts.evaluate import select_evaluation_loader
from scripts.generate_all_plots import _generate_canonical_plots
from scripts.generate_comparison import generate_base_comparison_report
from scripts.run_ensemble_eval import _protocol_paths
from scripts.verification.eval_latency_throughput import _ensemble_operation
from src.engine.checkpoint import CheckpointManager
from src.engine.trainer import Trainer
from src.ensemble.artifacts import (
    PredictionArtifact,
    load_prediction_artifact,
    save_prediction_artifact,
    validate_prediction_alignment,
)
from src.ensemble.oof import OOFGenerator
from src.utils.run_identity import (
    model_run_root,
    prepare_exact_run_directory,
    resolve_backbone_config_paths,
    validate_protocol,
)
from src.utils.provenance import require_git_commit


def probs(rows: int = 4, classes: int = 2) -> np.ndarray:
    values = np.arange(1, rows * classes + 1, dtype=np.float64).reshape(rows, classes)
    return values / values.sum(axis=1, keepdims=True)


def artifact(method: str = "resnet50", protocol: str = "single_split") -> PredictionArtifact:
    matrix = probs()
    return PredictionArtifact(
        sample_ids=np.asarray([f"sample-{i}" for i in range(4)]),
        y_true=np.asarray([0, 1, 0, 1]),
        probabilities=matrix,
        predictions=matrix.argmax(1),
        class_order=np.asarray(["a", "b"]),
        protocol=protocol,
        method=method,
        split="test",
        run_id="run-a",
        dataset_manifest_sha256="a" * 64,
        config_sha256="b" * 64,
        source_commit="c" * 40,
        artifact_hashes={"checkpoint": "d" * 64},
        aggregation_semantics="single_checkpoint_inference",
    )


class Step4BRemediationTests(unittest.TestCase):
    def test_final_run_identity_is_exact_and_shared(self) -> None:
        expected = Path("RESULTS/runs/run-a/single_split/resnet50")
        self.assertEqual(
            model_run_root("RESULTS", "run-a", "single_split", "resnet50"),
            expected,
        )
        fit, _, _ = _protocol_paths(
            "single_split", "RESULTS", ["resnet50"], "run-a"
        )
        self.assertEqual(Path(fit["resnet50"]).parent, expected)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "runs" / "run-a" / "single_split" / "resnet50"
            self.assertEqual(prepare_exact_run_directory(target, resume=False), target)
            (target / "sentinel").write_text("occupied", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_exact_run_directory(target, resume=False)

    def test_oof_cache_identity_accept_and_reject_cases(self) -> None:
        expected = {
            "protocol": "oof", "run_id": "run-a", "dataset_manifest_sha256": "a" * 64,
            "ordered_sample_ids_sha256": "b" * 64, "class_order": ["a", "b"],
            "n_splits": 5, "fold_index_set": [1, 2, 3, 4, 5], "split_seed": 42,
            "config_sha256": "c" * 64, "backbone": "resnet50",
            "folds": {"1": {"train_indices_sha256": "d" * 64,
                              "holdout_indices_sha256": "e" * 64,
                              "checkpoint_sha256": "f" * 64}},
        }
        OOFGenerator.validate_cache_identity(expected, expected)
        for key, changed in (
            ("dataset_manifest_sha256", "0" * 64),
            ("split_seed", 7),
            ("config_sha256", "1" * 64),
            ("ordered_sample_ids_sha256", "2" * 64),
        ):
            actual = dict(expected)
            actual[key] = changed
            with self.assertRaisesRegex(RuntimeError, key):
                OOFGenerator.validate_cache_identity(actual, expected)

    def test_backbone_registry_excludes_non_models(self) -> None:
        paths = resolve_backbone_config_paths()
        self.assertEqual(len(paths), 4)
        for forbidden in ("configs/dataset.yaml", "configs/ensemble.yaml", "configs/final_experiment.yaml"):
            self.assertNotIn(forbidden, paths)
            with self.assertRaises(ValueError):
                resolve_backbone_config_paths([forbidden])

    def test_master_cli_uses_canonical_results_root_and_current_contracts(self) -> None:
        parser = main.build_parser()
        ensemble = parser.parse_args([
            "ensemble", "--mode", "val", "--run-id", "run-a",
        ])
        self.assertEqual(ensemble.results_root, "RESULTS")
        self.assertFalse(hasattr(ensemble, "outputs_dir"))
        report = parser.parse_args(["report", "--run-id", "run-a"])
        self.assertEqual(report.results_root, "RESULTS")
        self.assertEqual(main.canonical_protocol_from_mode("val"), "single_split")
        with self.assertRaisesRegex(ValueError, "Unknown ensemble mode"):
            main.canonical_protocol_from_mode("latest")

        pipeline = parser.parse_args([
            "all-in-one", "--run-id", "run-a", "--train-mode", "scratch",
        ])
        self.assertEqual(pipeline.results_root, "RESULTS")
        self.assertEqual(pipeline.start_at, "single-train")
        self.assertEqual(pipeline.stop_after, "verification")
        self.assertFalse(hasattr(pipeline, "ensemble_mode"))

    def test_all_in_one_runs_complete_pipeline_in_canonical_order(self) -> None:
        configs = [
            "configs/resnet50.yaml",
            "configs/densenet121.yaml",
            "configs/efficientnet_b0.yaml",
            "configs/swin_tiny.yaml",
        ]
        calls = []

        with patch.object(main, "resolve_config_paths", return_value=configs), patch.object(
            main, "run_experiments", side_effect=lambda *a, **k: calls.append(("single-train", a, k))
        ) as train, patch.object(
            main, "run_ensemble_evaluation", side_effect=lambda *a, **k: calls.append(("single-ensemble", a, k))
        ) as single_ensemble, patch.object(
            main, "run_all_kfold_experiments", side_effect=lambda *a, **k: calls.append(("oof", a, k))
        ) as oof, patch.object(
            main, "run_verification_suite", side_effect=lambda *a, **k: calls.append(("verification", a, k))
        ) as verify:
            main.run_end_to_end_pipeline(
                config_paths=configs,
                train_mode="scratch",
                run_id="run-a",
                results_root="RESULTS",
                skip_tsne=True,
            )

        self.assertEqual([entry[0] for entry in calls], list(main.END_TO_END_PHASES))
        train.assert_called_once_with(
            configs, mode="scratch", run_id="run-a", results_root="RESULTS"
        )
        single_ensemble.assert_called_once_with(
            protocol="single_split", results_root="RESULTS", run_id="run-a"
        )
        oof.assert_called_once_with(
            config_paths=configs,
            n_splits=5,
            split_seed=42,
            force_retrain=False,
            eval_ensemble=True,
            run_id="run-a",
            results_root="RESULTS",
        )
        verify.assert_called_once_with("RESULTS", None, True, run_id="run-a")

    def test_all_in_one_phase_window_does_not_repeat_completed_training(self) -> None:
        configs = [
            "configs/resnet50.yaml",
            "configs/densenet121.yaml",
            "configs/efficientnet_b0.yaml",
            "configs/swin_tiny.yaml",
        ]
        with patch.object(main, "resolve_config_paths", return_value=configs), patch.object(
            main, "run_experiments"
        ) as train, patch.object(main, "run_ensemble_evaluation") as ensemble, patch.object(
            main, "run_all_kfold_experiments"
        ) as oof, patch.object(main, "run_verification_suite") as verify:
            main.run_end_to_end_pipeline(
                config_paths=configs,
                train_mode="resume",
                run_id="run-a",
                start_at="oof",
                stop_after="verification",
            )

        train.assert_not_called()
        ensemble.assert_not_called()
        oof.assert_called_once()
        verify.assert_called_once()

        with self.assertRaisesRegex(ValueError, "start-at"):
            main.run_end_to_end_pipeline(
                config_paths=configs,
                train_mode="resume",
                run_id="run-a",
                start_at="verification",
                stop_after="single-train",
            )

    def test_provenance_rejects_uncommitted_scientific_source(self) -> None:
        with patch(
            "src.utils.provenance.git_identity",
            return_value={
                "git_commit": "a" * 40,
                "git_branch": "fix/scientific-pipeline-v2",
                "scientific_worktree_changes": ["src/engine/trainer.py"],
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "must be committed"):
                require_git_commit()
        with patch(
            "src.utils.provenance.git_identity",
            return_value={
                "git_commit": "a" * 40,
                "git_branch": "fix/scientific-pipeline-v2",
                "scientific_worktree_changes": [],
            },
        ):
            self.assertEqual(require_git_commit(), "a" * 40)

    def test_master_verification_dispatch_uses_current_latency_contract(self) -> None:
        with patch.object(main, "evaluate_hardware_latency", return_value="ok") as mocked:
            result = main.dispatch_verification_task(
                "latency", results_root="R", run_id="run-a", save_dir="S",
                protocol="oof",
            )
        self.assertEqual(result, "ok")
        mocked.assert_called_once_with("oof", "R", save_dir="S", run_id="run-a")
        with self.assertRaisesRegex(ValueError, "protocol"):
            main.dispatch_verification_task(
                "latency", results_root="R", run_id="run-a", save_dir=None
            )

    def test_server_oof_request_never_falls_back_to_single(self) -> None:
        with patch.dict(server.BASE_MODELS_OOF, {}, clear=True), patch.dict(
            server.BASE_MODELS_SINGLE, {"resnet50": object()}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "OOF protocol requested"):
                server.compute_base_probabilities(torch.zeros(1, 3, 4, 4), "oof")
        with patch.dict(server.ENSEMBLE_MODELS_OOF, {}, clear=True), patch.dict(
            server.ENSEMBLE_MODELS_SINGLE, {"soft_voting": object()}, clear=True
        ):
            fake = {
                name: np.asarray([[0.6, 0.4]]) for name in server.BASE_MODEL_KEYS
            }
            with self.assertRaisesRegex(RuntimeError, "unavailable for oof"):
                server.compute_ensemble_predictions(fake, "soft_voting", "oof")

    def test_prediction_identity_roundtrip_and_fail_closed_validation(self) -> None:
        record = artifact()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.npz"
            save_prediction_artifact(
                str(path), sample_ids=record.sample_ids, y_true=record.y_true,
                probabilities=record.probabilities, predictions=record.predictions,
                class_order=record.class_order, protocol=record.protocol,
                method=record.method, split=record.split, run_id=record.run_id,
                dataset_manifest_sha256=record.dataset_manifest_sha256,
                config_sha256=record.config_sha256, source_commit=record.source_commit,
                artifact_hashes=record.artifact_hashes,
                aggregation_semantics=record.aggregation_semantics,
            )
            loaded = load_prediction_artifact(str(path))
        self.assertEqual(loaded.run_id, "run-a")
        validate_prediction_alignment(
            [loaded], expected_protocol="single_split", expected_split="test",
            expected_methods=["resnet50"], expected_run_id="run-a",
            expected_dataset_manifest_sha256="a" * 64,
            expected_config_sha256={"resnet50": "b" * 64},
            expected_artifact_hashes={"resnet50": {"checkpoint": "d" * 64}},
        )
        bad = artifact()
        object.__setattr__(bad, "probabilities", np.asarray([[1.1, -0.1]] * 4))
        with self.assertRaisesRegex(ValueError, "below zero"):
            bad.validate()

    def test_row_reordering_is_rejected_by_policy(self) -> None:
        first = artifact("resnet50")
        second = artifact("densenet121")
        order = np.asarray([1, 0, 2, 3])
        object.__setattr__(second, "sample_ids", second.sample_ids[order])
        object.__setattr__(second, "y_true", second.y_true[order])
        object.__setattr__(second, "probabilities", second.probabilities[order])
        object.__setattr__(second, "predictions", second.predictions[order])
        with self.assertRaisesRegex(ValueError, "sample_id ordering mismatch"):
            validate_prediction_alignment(
                [first, second], expected_protocol="single_split", expected_split="test"
            )

    def test_final_plot_path_has_no_estimator_fit(self) -> None:
        self.assertNotIn(".fit(", inspect.getsource(_generate_canonical_plots))
        with self.assertRaisesRegex(ValueError, "rejects canonical RESULTS"):
            generate_base_comparison_report("RESULTS")

    def test_split_selection_never_falls_back(self) -> None:
        val = object()
        test = object()
        self.assertIs(select_evaluation_loader("val", val, test), val)
        self.assertIs(select_evaluation_loader("test", val, test), test)
        with self.assertRaises(FileNotFoundError):
            select_evaluation_loader("test", val, None)

    def test_unknown_protocol_and_oof_optimizer_fail(self) -> None:
        with self.assertRaises(ValueError):
            validate_protocol("latest")
        generator = OOFGenerator.__new__(OOFGenerator)
        generator.config = SimpleNamespace(
            train=SimpleNamespace(optimizer="invented", lr=1e-3, weight_decay=0.0, momentum=0.9)
        )
        with self.assertRaisesRegex(ValueError, "Unknown optimizer"):
            generator._build_optimizer(torch.nn.Linear(2, 2))

    def test_latency_hard_voting_executes_real_dispatch(self) -> None:
        class Family(torch.nn.Module):
            def forward(self, inputs):
                return torch.tensor([[0.7, 0.3]], dtype=inputs.dtype)
        families = {name: Family() for name in ("a", "b")}
        operation = _ensemble_operation(
            "hard_voting", families, ["a", "b"], torch.zeros(1, 3, 4, 4), Path(".")
        )
        np.testing.assert_allclose(operation().numpy(), np.asarray([[1.0, 0.0]]))

    def test_resume_checkpoint_persists_logical_continuation_state(self) -> None:
        class Scaler:
            def state_dict(self):
                return {"scale": 16.0}
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            manager = CheckpointManager(directory)
            manager.save_if_best(
                model, 90.0, epoch=2,
                metrics={"val_loss": 0.4, "val_accuracy": 90.0},
                scaler=Scaler(), early_stop_counter=0,
            )
            manager.save_last(
                model, epoch=5,
                metrics={"val_loss": 0.5, "val_accuracy": 89.0},
                scaler=Scaler(), early_stop_counter=3,
            )
            state = manager.load_last()
        self.assertEqual(state["early_stop_counter"], 3)
        self.assertEqual(state["scaler_state_dict"], {"scale": 16.0})
        self.assertEqual(state["best_epoch"], 2)
        self.assertEqual(state["resume_semantics"], "logical_state_continuation_rng_not_persisted")

    def test_trainer_restores_last_continuation_state_without_best_state_mix(self) -> None:
        class Scaler:
            def __init__(self):
                self.loaded = None

            def state_dict(self):
                return {"scale": 8.0}

            def load_state_dict(self, state):
                self.loaded = state

        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            manager = CheckpointManager(directory)
            manager.save_if_best(
                model, 92.0, optimizer, scheduler, epoch=2,
                metrics={"val_loss": 0.3, "val_accuracy": 92.0},
                history={"val_accuracy": [91.0, 92.0]},
                scaler=Scaler(), early_stop_counter=0,
            )
            manager.save_last(
                model, optimizer, scheduler, epoch=5,
                metrics={"val_loss": 0.4, "val_accuracy": 90.0},
                history={"val_accuracy": [91.0, 92.0, 91.0, 90.5, 90.0]},
                scaler=Scaler(), early_stop_counter=3,
            )
            trainer = Trainer.__new__(Trainer)
            trainer.config = SimpleNamespace(
                checkpoint=SimpleNamespace(save_dir=directory, monitor="val_accuracy"),
                train=SimpleNamespace(epochs=30),
                experiment_name="resume-test",
            )
            trainer.device = torch.device("cpu")
            trainer.model = torch.nn.Linear(2, 2)
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
            trainer.scheduler = torch.optim.lr_scheduler.StepLR(
                trainer.optimizer, step_size=1
            )
            trainer.scaler = Scaler()
            trainer.ckpt_manager = CheckpointManager(directory)
            trainer.history = {}
            trainer._best_metric = -float("inf")
            trainer._early_stop_counter = 0
            trainer.logger = SimpleNamespace(info=lambda *_: None, warning=lambda *_: None)
            start_epoch = trainer._restore_checkpoint()
        self.assertEqual(start_epoch, 6)
        self.assertEqual(trainer._early_stop_counter, 3)
        self.assertEqual(trainer.ckpt_manager.best_epoch, 2)
        self.assertEqual(trainer.ckpt_manager.best_value, 92.0)
        self.assertEqual(trainer.ckpt_manager.min_val_loss, 0.3)
        self.assertEqual(trainer.scaler.loaded, {"scale": 8.0})


if __name__ == "__main__":
    unittest.main()
