# Step 4B Changed Files

## Configuration and user-facing contract

- `README.md`
- `configs/dataset.yaml`
- `configs/densenet121.yaml`
- `configs/efficientnet_b0.yaml`
- `configs/ensemble.yaml`
- `configs/final_experiment.yaml`
- `configs/resnet50.yaml`
- `configs/swin_tiny.yaml`
- `main.py`
- `server.py`
- `web/app.js`
- `web/index.html`

## Scientific runners and verification

- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/run_experiments.py`
- `scripts/run_kfold.py`
- `scripts/run_ensemble_eval.py`
- `scripts/generate_all_plots.py`
- `scripts/generate_comparison.py`
- `scripts/verification/common_utils.py`
- `scripts/verification/eval_advanced_metrics.py`
- `scripts/verification/eval_calibration.py`
- `scripts/verification/eval_diversity_ambiguity.py`
- `scripts/verification/eval_latency_throughput.py`
- `scripts/verification/eval_mcnemar_test.py`
- `scripts/verification/eval_tsne.py`
- `scripts/verification/verify_all_metrics.py`

## Core implementation

- `src/datasets/dataset.py`
- `src/engine/checkpoint.py`
- `src/engine/evaluator.py`
- `src/engine/trainer.py`
- `src/ensemble/artifacts.py`
- `src/ensemble/oof.py`
- `src/utils/config.py`
- `src/utils/provenance.py`
- `src/utils/run_identity.py` (new)

## Tests

- `tests/test_scientific_pipeline.py`
- `tests/test_step4b_remediation.py` (new)

## Step 4B reports

- `audit/STEP4B_REMEDIATION_REPORT.md`
- `audit/STEP4B_REGRESSION_TESTS.md`
- `audit/STEP4B_CHANGED_FILES.md`
- `audit/STEP4B_READY_FOR_TRAINING.md`

## Explicit exclusions

- No path under `data/` was modified.
- No checkpoint, prediction array, TensorBoard run, or result artifact was created or modified.
- The pre-existing tracked deletion `docs/codebase_documentation.md` is not a Step 4B change and was not restored or intentionally altered.
- Pre-existing untracked `audit/REAUDIT_*` inputs are not claimed as Step 4B outputs.
