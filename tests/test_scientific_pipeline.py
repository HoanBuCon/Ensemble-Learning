from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.run_ensemble_eval import _protocol_paths, run_ensemble_evaluation
from scripts.verification.eval_mcnemar_test import compute_mcnemar_metrics
from scripts.verification.eval_tsne import feature_label
from src.engine.checkpoint import CheckpointManager
from src.engine.evaluator import evaluate_model
from src.ensemble.artifacts import (
    PredictionArtifact,
    load_prediction_artifact,
    save_prediction_artifact,
    validate_prediction_alignment,
)
from src.ensemble.oof import OOFGenerator
from src.ensemble.stacking import StackingEnsemble, load_stacking_config
from src.ensemble.voting import WeightedVoting
from src.utils.report import extract_model_val_metrics


def probabilities(rows: int = 12, classes: int = 3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.random((rows, classes))
    return values / values.sum(axis=1, keepdims=True)


class ToyDataset(Dataset):
    class_to_idx = {"a": 0, "b": 1}

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return (
            torch.full((3, 4, 4), float(index)),
            index,
            f"data/test/a/scientific-test-{index}.jpg",
        )


class ToyModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = inputs.mean(dim=(1, 2, 3))
        return torch.stack((1.0 - score, score), dim=1)


class ScientificPipelineTests(unittest.TestCase):
    def test_01_oof_boundary_and_placement(self) -> None:
        train_indices = np.array([0, 1, 2, 3])
        holdout_indices = np.array([4, 5])
        OOFGenerator._assert_fold_boundary(train_indices, holdout_indices)
        with self.assertRaises(AssertionError):
            OOFGenerator._assert_fold_boundary(train_indices, np.array([3, 4]))

        signature = inspect.signature(OOFGenerator._train_fold)
        self.assertIn("external_val_loader", signature.parameters)
        self.assertNotIn("outer_holdout_loader", signature.parameters)
        target = np.zeros((6, 2))
        counts = np.zeros(6, dtype=np.int8)
        predicted = np.array([[0.8, 0.2], [0.1, 0.9]])
        OOFGenerator._place_oof_rows(target, counts, holdout_indices, predicted)
        np.testing.assert_allclose(target[holdout_indices], predicted)
        np.testing.assert_array_equal(counts[holdout_indices], np.ones(2))
        with self.assertRaises(AssertionError):
            OOFGenerator._place_oof_rows(target, counts, holdout_indices, predicted)

    def test_02_weighted_voting_slsqp(self) -> None:
        labels = np.arange(18) % 3
        matrices = [probabilities(18, 3, seed) for seed in range(4)]
        model = WeightedVoting().fit(matrices, labels)
        self.assertTrue(model.optimizer_result["success"])
        self.assertTrue(np.isfinite(model.optimizer_result["objective_value"]))
        self.assertTrue(np.all(np.asarray(model.weights) >= 0.0))
        self.assertAlmostEqual(sum(model.weights), 1.0, places=8)

    def test_03_ensemble_serialization_round_trip(self) -> None:
        labels = np.arange(24) % 3
        matrices = [probabilities(24, 3, seed) for seed in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weighted = WeightedVoting().fit(matrices, labels)
            weighted_path = root / "weighted.json"
            weighted.save(
                str(weighted_path),
                base_model_order=["a", "b", "c", "d"],
                fit_split="val",
                input_probability_hashes={name: name for name in "abcd"},
            )
            replayed = WeightedVoting.load(str(weighted_path))
            np.testing.assert_allclose(
                weighted.predict_proba(matrices), replayed.predict_proba(matrices)
            )

            stacker = StackingEnsemble("logistic_regression").fit(matrices, labels)
            before = stacker.predict_proba(matrices)
            stack_path = root / "stacking_lr.joblib"
            stacker.save(str(stack_path))
            after = StackingEnsemble("logistic_regression").load(
                str(stack_path)
            ).predict_proba(matrices)
            np.testing.assert_allclose(before, after)

    def test_04_protocol_isolation(self) -> None:
        order = ["a", "b"]
        single_fit, _, single_split = _protocol_paths("single_split", "R", order)
        oof_fit, _, oof_split = _protocol_paths("oof", "R", order)
        self.assertTrue(all("single_split" in path for path in single_fit.values()))
        self.assertTrue(all("/oof/" in path.replace("\\", "/") for path in oof_fit.values()))
        self.assertEqual(single_split, "val")
        self.assertEqual(oof_split, "oof_train")
        with self.assertRaises(ValueError):
            run_ensemble_evaluation(
                mode="oof", outputs_dir="RESULTS/OOF_TRAINING"
            )

    def test_05_sample_alignment_rejects_reordered_rows(self) -> None:
        probs = probabilities(4, 2)
        first = PredictionArtifact(
            np.array(["a", "b", "c", "d"]), np.array([0, 1, 0, 1]),
            probs, probs.argmax(1), np.array(["a", "b"]),
            "single_split", "m1", "test",
        )
        second = PredictionArtifact(
            np.array(["b", "a", "c", "d"]), np.array([1, 0, 0, 1]),
            probs[[1, 0, 2, 3]], probs[[1, 0, 2, 3]].argmax(1),
            np.array(["a", "b"]), "single_split", "m2", "test",
        )
        with self.assertRaisesRegex(ValueError, "sample_id ordering mismatch"):
            validate_prediction_alignment(
                [first, second], expected_protocol="single_split", expected_split="test"
            )

    def test_06_missing_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "stacking_lr.joblib")
            with self.assertRaises(FileNotFoundError):
                StackingEnsemble("logistic_regression").load(missing)
            with self.assertRaises(FileNotFoundError):
                load_prediction_artifact(str(Path(directory) / "missing.npz"))

    @patch("src.engine.evaluator.plot_confusion_matrix")
    @patch("src.engine.evaluator.plot_per_class_metrics")
    @patch("src.engine.evaluator.plot_roc_curves")
    @patch("src.engine.evaluator.plot_precision_recall_curves")
    def test_07_split_naming(
        self, _pr, _roc, _per_class, _cm
    ) -> None:
        with tempfile.TemporaryDirectory(dir=".") as directory:
            evaluate_model(
                ToyModel(), DataLoader(ToyDataset(), batch_size=2), ["a", "b"],
                directory, device="cpu", split="val", protocol="single_split",
            )
            root = Path(directory)
            self.assertTrue((root / "val_predictions.npz").is_file())
            self.assertTrue((root / "val_metrics.json").is_file())
            self.assertFalse((root / "test_predictions.npz").exists())
            self.assertFalse((root / "metrics.json").exists())

    def test_08_report_missing_metrics_are_not_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = extract_model_val_metrics(directory)
        self.assertEqual(
            metrics,
            {
                "Val_Accuracy": None,
                "Val_Precision": None,
                "Val_Recall": None,
                "Val_F1_Score": None,
            },
        )

    def test_09_checkpoint_resume_state_has_best_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            manager = CheckpointManager(directory)
            accepted = manager.save_if_best(
                model, 91.0, epoch=3, metrics={"val_loss": 0.2, "val_accuracy": 91.0}
            )
            self.assertTrue(accepted)
            state = manager.load_best()
            restored = CheckpointManager(directory)
            restored.restore_tracking(state)
            self.assertEqual(restored.best_epoch, 3)
            self.assertEqual(state["monitor"], "val_accuracy")
            self.assertEqual(state["mode"], "max")
            self.assertAlmostEqual(state["loss_gate_tolerance"], 0.05)

    def test_10_tsne_metadata_and_statistical_decision(self) -> None:
        matrix = np.zeros((5, 24))
        self.assertEqual(
            feature_label("ensemble", "probability_vector", matrix),
            "ensemble: probability_vector (24-D)",
        )
        result = compute_mcnemar_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 0]),
            np.array([0, 0, 1, 1]),
            alpha=0.01,
        )
        self.assertIn(result["decision"], {"REJECT", "FAIL_TO_REJECT"})
        self.assertEqual(result["alpha"], 0.01)
        self.assertFalse(result["power_used_for_decision"])
        self.assertIn("marginal_accuracy_cohens_h", result)

    def test_canonical_stacking_parameters(self) -> None:
        config = load_stacking_config()
        self.assertEqual(config["logistic_regression"]["solver"], "lbfgs")
        self.assertEqual(config["random_forest"]["n_estimators"], 200)
        self.assertEqual(config["xgboost"]["objective"], "multi:softprob")


if __name__ == "__main__":
    unittest.main()
