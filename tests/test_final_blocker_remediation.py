from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from scripts.run_ensemble_eval import (
    run_ensemble_evaluation,
    validate_ensemble_fit_test_identity,
)
from src.datasets.transforms import build_transforms
from src.engine.checkpoint import (
    CheckpointManager,
    scientific_checkpoint_provenance,
)
from src.engine.trainer import Trainer
from src.ensemble.artifacts import PredictionArtifact
from src.utils.config import load_config
from src.utils.provenance import (
    _scientific_changes_from_porcelain,
    git_identity,
    sha256_file,
    validate_run_start_manifest,
)
from src.utils.reproducibility import (
    fold_seed,
    seed_dataset_transform,
    seed_worker,
    worker_seed_from_torch_initial_seed,
)


class FinalBlockerRemediationTests(unittest.TestCase):
    def test_f01_porcelain_status_columns_and_order(self) -> None:
        cases = {
            " M src/engine/trainer.py\n": ["src/engine/trainer.py"],
            " M configs/resnet50.yaml\n": ["configs/resnet50.yaml"],
            " M main.py\n": ["main.py"],
            " D docs/readme.md\n M src/engine/trainer.py\n": [
                "src/engine/trainer.py"
            ],
            "M  src/engine/trainer.py\n": ["src/engine/trainer.py"],
            "?? src/new_scientific_module.py\n": ["src/new_scientific_module.py"],
            " D docs/readme.md\n?? audit/report.md\n": [],
            "": [],
        }
        for status, expected in cases.items():
            with self.subTest(status=status):
                self.assertEqual(_scientific_changes_from_porcelain(status), expected)

        def fake_git(command, **_kwargs):
            if command[1] == "status":
                return " M src/engine/trainer.py\n"
            if command[1:3] == ["rev-parse", "HEAD"]:
                return "a" * 40 + "\n"
            if command[1:3] == ["branch", "--show-current"]:
                return "fix/scientific-pipeline-v2\n"
            raise AssertionError(command)

        with patch("src.utils.provenance.subprocess.check_output", side_effect=fake_git):
            identity = git_identity()
        self.assertEqual(
            identity["scientific_worktree_changes"], ["src/engine/trainer.py"]
        )

    @staticmethod
    def _write_manifest(path: Path) -> str:
        rows = [
            ("train-a", "data/train/a/1.jpg", "train", "a", 0),
            ("train-b", "data/train/b/1.jpg", "train", "b", 1),
            ("val-a", "data/valid/a/1.jpg", "validation", "a", 0),
            ("val-b", "data/valid/b/1.jpg", "validation", "b", 1),
            ("test-a", "data/test/a/1.jpg", "test", "a", 0),
            ("test-b", "data/test/b/1.jpg", "test", "b", 1),
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                ["sample_id", "relative_path", "split", "class_name", "class_index", "sha256"]
            )
            for sample_id, relative, split, name, index in rows:
                writer.writerow([sample_id, relative, split, name, index, "f" * 64])
        return sha256_file(path)

    @staticmethod
    def _prediction(
        *,
        ids,
        split: str,
        protocol: str,
        dataset_hash: str,
        source: str = "c" * 40,
        config_hash: str = "b" * 64,
        hashes=None,
        semantics: str = "single_checkpoint_inference",
    ) -> PredictionArtifact:
        probabilities = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64)
        return PredictionArtifact(
            sample_ids=np.asarray(ids, dtype=str),
            y_true=np.asarray([0, 1], dtype=np.int64),
            probabilities=probabilities,
            predictions=np.asarray([0, 1], dtype=np.int64),
            class_order=np.asarray(["a", "b"], dtype=str),
            protocol=protocol,
            method="resnet50",
            split=split,
            run_id="run-a",
            dataset_manifest_sha256=dataset_hash,
            config_sha256=config_hash,
            source_commit=source,
            artifact_hashes=hashes or {"checkpoint": "d" * 64},
            aggregation_semantics=semantics,
        )

    def test_f02_single_fit_test_lineage_and_membership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            dataset_hash = self._write_manifest(manifest)
            fit = self._prediction(
                ids=["val-a", "val-b"], split="val",
                protocol="single_split", dataset_hash=dataset_hash,
            )
            test = self._prediction(
                ids=["test-a", "test-b"], split="test",
                protocol="single_split", dataset_hash=dataset_hash,
            )
            kwargs = dict(
                model_order=["resnet50"], protocol="single_split", run_id="run-a",
                current_source_commit="c" * 40,
                expected_config_sha256={"resnet50": "b" * 64},
                manifest_path=str(manifest),
            )
            validate_ensemble_fit_test_identity([fit], [test], **kwargs)

            failures = (
                (replace(fit, source_commit="e" * 40), test, "source"),
                (replace(fit, config_sha256="e" * 64), test, "config"),
                (
                    fit,
                    replace(test, artifact_hashes={"checkpoint": "e" * 64}),
                    "checkpoint lineage",
                ),
                (replace(fit, sample_ids=np.asarray(["fake-a", "fake-b"])), test, "sample IDs"),
                (replace(fit, sample_ids=test.sample_ids.copy()), test, "sample IDs"),
            )
            for bad_fit, bad_test, message in failures:
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    validate_ensemble_fit_test_identity([bad_fit], [bad_test], **kwargs)

    def test_f02_oof_complete_five_fold_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            dataset_hash = self._write_manifest(manifest)
            lineage = {f"fold_{index}": str(index) * 64 for index in range(1, 6)}
            fit = self._prediction(
                ids=["train-a", "train-b"], split="oof_train", protocol="oof",
                dataset_hash=dataset_hash, hashes=lineage,
                semantics="one_outer_holdout_checkpoint_per_training_row",
            )
            test = self._prediction(
                ids=["test-a", "test-b"], split="test", protocol="oof",
                dataset_hash=dataset_hash, hashes=lineage,
                semantics="mean_fold_probabilities",
            )
            kwargs = dict(
                model_order=["resnet50"], protocol="oof", run_id="run-a",
                current_source_commit="c" * 40,
                expected_config_sha256={"resnet50": "b" * 64},
                manifest_path=str(manifest), n_splits=5,
            )
            validate_ensemble_fit_test_identity([fit], [test], **kwargs)
            changed = dict(lineage)
            changed["fold_5"] = "e" * 64
            with self.assertRaisesRegex(ValueError, "checkpoint lineage"):
                validate_ensemble_fit_test_identity(
                    [fit], [replace(test, artifact_hashes=changed)], **kwargs
                )

        source = __import__("inspect").getsource(run_ensemble_evaluation)
        self.assertLess(
            source.index("validate_ensemble_fit_test_identity"), source.index(".fit(")
        )

    @staticmethod
    def _checkpoint_origin(**changes):
        values = dict(
            run_id="run-a",
            protocol="single_split",
            backbone="resnet50",
            dataset_manifest_sha256="a" * 64,
            source_commit="c" * 40,
            effective_config_sha256="b" * 64,
            fold_identity=None,
        )
        values.update(changes)
        return scientific_checkpoint_provenance(**values)

    def test_f03_checkpoint_origin_accepts_exact_and_rejects_foreign(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            origin = self._checkpoint_origin()
            manager = CheckpointManager(directory, provenance=origin)
            self.assertTrue(
                manager.save_if_best(
                    model, 90.0, epoch=2,
                    metrics={"val_loss": 0.4, "val_accuracy": 90.0},
                )
            )
            path = str(Path(directory) / "best_model.pth")
            CheckpointManager.load_scientific(
                path, origin, expected_role="best", device="cpu"
            )
            with self.assertRaisesRegex(ValueError, "checkpoint_role"):
                CheckpointManager.load_scientific(
                    path, origin, expected_role="last", device="cpu"
                )

            altered = (
                self._checkpoint_origin(run_id="run-b"),
                self._checkpoint_origin(protocol="oof"),
                self._checkpoint_origin(effective_config_sha256="e" * 64),
                self._checkpoint_origin(dataset_manifest_sha256="e" * 64),
                self._checkpoint_origin(source_commit="e" * 40),
                self._checkpoint_origin(
                    protocol="oof",
                    fold_identity={"fold": 2},
                ),
            )
            for expected in altered:
                with self.subTest(expected=expected), self.assertRaisesRegex(
                    ValueError, "Checkpoint provenance mismatch"
                ):
                    CheckpointManager.load_scientific(
                        path, expected, expected_role="best", device="cpu"
                    )

            oof_dir = Path(directory) / "oof-fold"
            fold_one = self._checkpoint_origin(
                protocol="oof", fold_identity={"fold": 1}
            )
            oof_manager = CheckpointManager(str(oof_dir), provenance=fold_one)
            oof_manager.save_if_best(
                model, 90.0, epoch=1,
                metrics={"val_loss": 0.4, "val_accuracy": 90.0},
            )
            fold_two = self._checkpoint_origin(
                protocol="oof", fold_identity={"fold": 2}
            )
            with self.assertRaisesRegex(ValueError, "fold_identity"):
                CheckpointManager.load_scientific(
                    str(oof_dir / "best_model.pth"), fold_two,
                    expected_role="best", device="cpu",
                )

            legacy_dir = Path(directory) / "legacy"
            legacy = CheckpointManager(str(legacy_dir))
            legacy.save_if_best(
                model, 90.0, epoch=1,
                metrics={"val_loss": 0.4, "val_accuracy": 90.0},
            )
            with self.assertRaisesRegex(ValueError, "missing required"):
                CheckpointManager.load_scientific(
                    str(legacy_dir / "best_model.pth"), origin,
                    expected_role="best", device="cpu",
                )

    def test_f03_run_start_identity_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_start_manifest.json"
            payload = {
                "run_id": "run-a", "protocol": "single_split", "model": "resnet50",
                "config_sha256": "b" * 64,
                "dataset_manifest_sha256": "a" * 64,
                "git_commit": "c" * 40,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            validate_run_start_manifest(
                path, run_id="run-a", protocol="single_split", model="resnet50",
                config_sha256="b" * 64, dataset_manifest_sha256="a" * 64,
                source_commit="c" * 40,
            )
            with self.assertRaisesRegex(ValueError, "run_id"):
                validate_run_start_manifest(
                    path, run_id="run-b", protocol="single_split", model="resnet50",
                    config_sha256="b" * 64, dataset_manifest_sha256="a" * 64,
                    source_commit="c" * 40,
                )

    def test_f04_albumentations_seed_ownership(self) -> None:
        config = load_config("configs/resnet50.yaml")
        image = np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3)

        def sequence(seed, stage="train"):
            transform = build_transforms(
                getattr(config.augmentation, stage), image_size=32,
                stage=stage, seed=seed,
            )
            return [transform(image=image.copy())["image"].numpy() for _ in range(16)]

        first = sequence(42)
        second = sequence(42)
        different = sequence(43)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second)))
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(first, different)))
        for stage in ("val", "test"):
            outputs = sequence(42, stage=stage)
            self.assertTrue(
                all(np.array_equal(outputs[0], output) for output in outputs[1:])
            )

        self.assertEqual(fold_seed(42, 0), 42)
        self.assertEqual(fold_seed(42, 4), 46)
        self.assertEqual(fold_seed(42, 4), fold_seed(42, 4))
        self.assertNotEqual(fold_seed(42, 0), fold_seed(42, 1))
        self.assertEqual(worker_seed_from_torch_initial_seed(2**32 + 17), 17)

        seeded = Mock()
        nested_dataset = SimpleNamespace(dataset=SimpleNamespace(transform=seeded))
        self.assertTrue(seed_dataset_transform(nested_dataset, 123))
        seeded.set_random_seed.assert_called_once_with(123)

        worker_transform = Mock()
        worker_dataset = SimpleNamespace(transform=worker_transform)
        with patch(
            "src.utils.reproducibility.get_worker_info",
            return_value=SimpleNamespace(dataset=worker_dataset),
        ), patch(
            "src.utils.reproducibility.torch.initial_seed", return_value=2**32 + 17
        ):
            seed_worker(0)
        worker_transform.set_random_seed.assert_called_once_with(17)
        worker_transform.reset_mock()
        with patch(
            "src.utils.reproducibility.get_worker_info",
            return_value=SimpleNamespace(dataset=worker_dataset),
        ), patch(
            "src.utils.reproducibility.torch.initial_seed", return_value=2**32 + 18
        ):
            seed_worker(1)
        worker_transform.set_random_seed.assert_called_once_with(18)

    @staticmethod
    def _resume_trainer(directory: str, counter: int, patience: int) -> Trainer:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        manager = CheckpointManager(directory)
        manager.save_if_best(
            model, 92.0, optimizer, epoch=2,
            metrics={"val_loss": 0.3, "val_accuracy": 92.0},
            history={"val_accuracy": [91.0, 92.0]},
        )
        manager.save_last(
            model, optimizer, epoch=8,
            metrics={"val_loss": 0.4, "val_accuracy": 90.0},
            history={"val_accuracy": [91.0, 92.0, 91.0]},
            early_stop_counter=counter,
        )
        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleNamespace(
            checkpoint=SimpleNamespace(save_dir=directory, monitor="val_accuracy"),
            train=SimpleNamespace(epochs=30, early_stopping_patience=patience),
            model=SimpleNamespace(name="resnet50"),
            experiment_name="resume-terminal-test",
        )
        trainer.device = torch.device("cpu")
        trainer.model = torch.nn.Linear(2, 2)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        trainer.scheduler = None
        trainer.scaler = None
        trainer.ckpt_manager = CheckpointManager(directory)
        trainer.history = {}
        trainer._best_metric = -float("inf")
        trainer._early_stop_counter = 0
        trainer.num_params = 6
        trainer.is_resume = True
        trainer.logger = SimpleNamespace(info=lambda *_: None, warning=lambda *_: None)
        return trainer

    def test_f05_terminal_resume_never_enters_epoch_nine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._resume_trainer(directory, counter=6, patience=6)
            trainer._train_one_epoch = Mock(side_effect=AssertionError("epoch 9 executed"))
            result = trainer.train()
            trainer._train_one_epoch.assert_not_called()
            self.assertEqual(result["accepted_checkpoint_epoch"], 2)

        with tempfile.TemporaryDirectory() as directory:
            trainer = self._resume_trainer(directory, counter=5, patience=6)
            self.assertEqual(trainer._restore_checkpoint(), 9)


if __name__ == "__main__":
    unittest.main()
