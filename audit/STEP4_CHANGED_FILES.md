# Step 4 Changed Files

This inventory excludes the pre-existing deletion of
`docs/codebase_documentation.md`, which Step 4 did not cause or modify.

## Configuration

- `configs/ensemble.yaml`
- `configs/final_experiment.yaml`
- `configs/resnet50.yaml`
- `configs/densenet121.yaml`
- `configs/efficientnet_b0.yaml`
- `configs/swin_tiny.yaml`

## Core pipeline

- `src/engine/checkpoint.py`
- `src/engine/evaluator.py`
- `src/engine/trainer.py`
- `src/ensemble/artifacts.py`
- `src/ensemble/oof.py`
- `src/ensemble/stacking.py`
- `src/ensemble/voting.py`
- `src/utils/provenance.py`
- `src/utils/report.py`

## Runners and verification

- `scripts/create_dataset_snapshot.py`
- `scripts/evaluate.py`
- `scripts/run_ensemble_eval.py`
- `scripts/run_kfold.py`
- `scripts/train.py`
- `scripts/verification/__init__.py`
- `scripts/verification/common_utils.py`
- `scripts/verification/eval_advanced_metrics.py`
- `scripts/verification/eval_calibration.py`
- `scripts/verification/eval_diversity_ambiguity.py`
- `scripts/verification/eval_latency_throughput.py`
- `scripts/verification/eval_mcnemar_test.py`
- `scripts/verification/eval_tsne.py`
- `scripts/verification/verify_all_metrics.py`
- `server.py`

## Tests, manifests, and audit records

- `tests/__init__.py`
- `tests/test_scientific_pipeline.py`
- `artifacts/manifests/dataset_manifest.csv`
- `artifacts/manifests/dataset_snapshot.json`
- `audit/STEP4_PHASE_LOG.md`
- `audit/STEP4_CODE_REMEDIATION_REPORT.md`
- `audit/STEP4_TEST_RESULTS.md`
- `audit/STEP4_CHANGED_FILES.md`
