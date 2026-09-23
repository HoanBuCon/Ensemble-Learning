# Step 4 Phase Log

Starting branch: `fix/scientific-pipeline-v2`  
Starting commit: `a65290a7c52151c5904ac9accb1798fd6dc788f5`  
Starting tree: `7b99aaa3f09ddbd9ee1250034a93d057b7d157ba`

Pre-existing working-tree state: tracked `docs/codebase_documentation.md` was already
deleted. The Step-1/Step-2 `audit/` files and local untracked manuscript files referenced
by the request were no longer present when Step 4 began. Their evidence content had been
read in the preceding audit turns and remains the evidence basis; Step 4 did not restore
or modify the missing user documentation.

## PHASE A — Configuration freeze

Files changed:

- `configs/ensemble.yaml`
- `configs/final_experiment.yaml`

Behavior before: ensemble parameters and final protocol/output identity were distributed
across source defaults and historical paths.

Behavior after: canonical SLSQP, LR/RF/XGB, cross-fitting, output-root, and latency
settings are explicitly recorded. Existing per-backbone LR, weight decay, clipping,
dropout, batch, epochs, warmup, patience, label smoothing, optimizer, and scheduler values
remain unchanged.

Tests: YAML parse/config-invariant tests will run at Phase Q.

Scientific finding addressed: CLAIM-03, CLAIM-04, CLAIM-06, CLAIM-09, CLAIM-12.

New assumptions introduced: none; values are those explicitly required for Step 4.

## PHASE B — Read-only dataset snapshot

Files changed:

- `src/utils/provenance.py`
- `scripts/create_dataset_snapshot.py`
- `artifacts/manifests/dataset_manifest.csv`
- `artifacts/manifests/dataset_snapshot.json`

Behavior before: no immutable manifest identified the human-approved local dataset used
by new experiments.

Behavior after: all 10,278 current images are locked by sample ID, repository-relative
path, split, class name/index, and SHA-256. Counts are train 7,192, validation 1,540, test
1,546. Manifest SHA-256 is
`a94426015d83a3b8e9f66a30bddee9e4b376eaf509f166aa522d7a204fe22dc5`.

Tests: creation completed; read-only verification is included in Phase Q and the final
consistency gate.

Scientific finding addressed: dataset/run identity and experiment provenance evidence
gap.

New assumptions introduced: none. Existing `data/valid` is recorded as the fixed external
validation directory. No duplicate/near-duplicate analysis was performed.

## PHASE C — OOF scientific protocol

Files changed:

- `src/ensemble/oof.py`

Behavior before: each outer held-out training fold was passed to the per-epoch validation
path; its metrics selected the checkpoint and the same fold was then used for OOF inference.

Behavior after: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` still splits
only `data/train`; one fixed `data/valid` or `data/val` loader now drives validation,
scheduling, checkpoint acceptance, best epoch, and early stopping. The disjoint outer
holdout is absent from `_train_fold()` and is consumed only by
`_evaluate_outer_holdout()` after the selected state is frozen. OOF row placement and mean
fold-test aggregation are retained.

Tests: source syntax parsed successfully; regression tests are added in Phase Q.

Scientific finding addressed: CLAIM-01 (outer-fold model-selection reuse).

New assumptions introduced: none; the current frozen dataset has exactly one supported
external-validation directory (`data/valid`).

## PHASE D — Checkpoint semantics

Files changed:

- `src/engine/checkpoint.py`
- `src/engine/trainer.py`
- `src/ensemble/oof.py`

Behavior before: checkpoint files persisted only the overloaded `best_value`; the manager
did not own `best_epoch`, completed-run resume referenced that undefined attribute, and
trainer output called the raw maximum validation accuracy the best checkpoint accuracy.

Behavior after: accepted checkpoints persist `best_epoch`, `best_metric`,
`best_val_loss`, `monitor`, `mode`, `loss_gate_tolerance`, and minimum observed validation
loss. Resume restores this state. Trainer results separately expose
`raw_max_val_accuracy`, `accepted_checkpoint_val_accuracy`, and
`accepted_checkpoint_epoch`. OOF uses the same checkpoint manager and loss gate.

Tests: source inspection and syntax gate; checkpoint/resume regressions run in Phase Q.

Scientific finding addressed: CLAIM-05 and NEW-FINDING-09.

New assumptions introduced: none; monitor, mode, tolerance, and patience remain configured
as before.

## PHASE E — Fold identity and checkpoint failure

Files changed:

- `src/engine/checkpoint.py`
- `src/ensemble/oof.py`

Behavior before: checkpoint loading searched historical roots and included a `fold_0`
fallback although the OOF generator wrote `fold_1` through `fold_5`.

Behavior after: checkpoint loading is exact-path only and raises `FileNotFoundError` when
the requested checkpoint is absent. The only generated fold convention is `fold_1` through
`fold_5`; no random or cross-root fallback remains.

Tests: repository search found no remaining `fold_0` or checkpoint fallback candidates;
regression tests run in Phase Q.

Scientific finding addressed: audited fold-numbering fallback and cross-protocol identity
risk.

New assumptions introduced: none.

## PHASE F — Canonical weighted voting

Files changed: `src/ensemble/voting.py`, `configs/ensemble.yaml`.

Behavior before: an 11-point Cartesian grid normalized nonzero tuples and selected the
first strict accuracy maximum.

Behavior after: constrained SciPy SLSQP minimizes epsilon-clipped multiclass log-loss
from uniform weights with bounds `[0,1]` and `sum(weights)=1`. Failure is explicit and the
optimizer result, fit split, base order, hashes, objective, weights, status, message, and
iterations are serialized in `weighted_voting.json`.

Tests: non-negativity, unit sum, optimizer success, finite objective, and save/load replay
passed.

Scientific finding addressed: CLAIM-02 and hard-coded-weight findings.

New assumptions introduced: none; the algorithm is explicitly required by Step 4.

## PHASE G — Canonical stacking configuration

Files changed: `configs/ensemble.yaml`, `src/ensemble/stacking.py`.

Behavior before: constructor defaults were distributed in source and verification used an
independent XGBoost specification.

Behavior after: LR, RF, and XGBoost constructors load one explicit YAML definition. The
canonical XGBoost objective, class count, evaluation metric, seed, depth, learning rate,
and estimator count are pinned.

Tests: canonical parameter assertions and module import passed.

Scientific finding addressed: CLAIM-06.

New assumptions introduced: none.

## PHASE H — Fitted ensemble serialization

Files changed: `src/ensemble/artifacts.py`, `src/ensemble/stacking.py`,
`scripts/run_ensemble_eval.py`, `src/engine/evaluator.py`, `src/ensemble/oof.py`.

Behavior before: base arrays lacked sample/protocol identity and fitted ensemble objects
were not the single persisted source used downstream.

Behavior after: strict NPZ artifacts include sample IDs, labels, probabilities,
predictions, class order, protocol, method, and split. Weighted voting, LR, RF, and XGB
are serialized; the runner reloads those artifacts before producing final test predictions.
`ensemble_manifest.json` records fit identity and file hashes.

Tests: fit/save/load/predict numerical replay passed.

Scientific finding addressed: missing fitted-object serialization and prediction identity.

New assumptions introduced: none.

## PHASE I — Verification replay

Files changed: `scripts/verification/common_utils.py`, `eval_calibration.py`,
`eval_advanced_metrics.py`, `eval_diversity_ambiguity.py`, `eval_mcnemar_test.py`,
`verify_all_metrics.py`, `scripts/verification/__init__.py`, `server.py`.

Behavior before: verification and the server independently refit meta-models, including a
different XGBoost specification.

Behavior after: verification reads the exact saved base/ensemble prediction artifacts;
the server loads fitted FINAL_V2 objects. No verification path calls estimator `fit()`.

Tests: module imports and missing-artifact failure tests passed.

Scientific finding addressed: CLAIM-06 and verification reconstruction/refit findings.

New assumptions introduced: none.

## PHASE J — Fail-closed scientific evaluation

Files changed: `scripts/run_ensemble_eval.py`, `scripts/verification/*`, `server.py`,
`src/utils/report.py`, `src/engine/checkpoint.py`.

Behavior before: roots were auto-discovered, missing stacking could be relabeled voting,
and missing validation metrics could copy accuracy into precision/recall/F1.

Behavior after: protocol roots and exact artifacts are mandatory; row/class/protocol
identity mismatch raises; missing algorithms raise; unavailable metrics are `None`; no
historical root or algorithm substitution is performed.

Tests: protocol isolation, row reordering rejection, missing-artifact failure, and report
fallback regression tests passed.

Scientific finding addressed: NEW-FINDING-02 and audited silent fallbacks.

New assumptions introduced: none.

## PHASE K — Split-specific artifact naming

Files changed: `src/engine/evaluator.py`, `src/engine/trainer.py`, `scripts/evaluate.py`.

Behavior before: `--split val` was not propagated and ambiguous test/unsuffixed copies
could be produced.

Behavior after: val/test identity is propagated into distinct prediction, metric,
classification, and plot names; ambiguous unsuffixed prediction copies are not created.

Tests: val-output regression passed and confirmed no test or unsuffixed metrics output.

Scientific finding addressed: NEW-FINDING-03.

New assumptions introduced: none.

## PHASE L — Statistical verification cleanup

Files changed: `scripts/verification/eval_mcnemar_test.py`,
`scripts/verification/verify_all_metrics.py`.

Behavior before: rounded p-values fed decisions, alpha was not consistently applied,
weights were hard-coded, and non-rejection was described as equivalence.

Behavior after: raw exact-binomial p-values feed `REJECT`/`FAIL_TO_REJECT`; alpha drives
both decision and CI; Bonferroni remains `0.05/6`; paired RD/CI and Edwards correction are
retained; Cohen's h is explicitly marginal-accuracy; post-hoc power is supplementary only.

Tests: alpha/decision/effect-label regression passed.

Scientific finding addressed: CLAIM-07 and CLAIM-08.

New assumptions introduced: none.

## PHASE M — t-SNE semantics

Files changed: `scripts/verification/eval_tsne.py`.

Behavior before: missing deep checkpoints silently switched to probabilities while plot
titles still asserted a deep latent dimension.

Behavior after: feature source is mandatory; probability and penultimate modes are
separate; deep mode fails when a checkpoint is absent; titles, filenames, CSV, and JSON
metadata derive from the actual matrix dimension and feature source.

Tests: actual-source/actual-dimension label regression passed.

Scientific finding addressed: audited t-SNE fallback and semantic mismatch.

New assumptions introduced: none.

## PHASE N — Real pipeline latency benchmark

Files changed: `scripts/verification/eval_latency_throughput.py`,
`configs/final_experiment.yaml`.

Behavior before: random untrained models, fixed parameter/storage tables, and a 0.45 ms
stacking constant represented final performance.

Behavior after: the benchmark requires final saved checkpoints and fitted objects, executes
base inference through the requested voting/stacking operation, uses synchronized CUDA
Events plus end-to-end wall timing, uses 50/200 GPU and 20/100 CPU warmup/measured runs
over 5 repeats, counts loaded parameters, measures artifact file bytes, and stores hardware
and runtime identity.

Tests: syntax/import and configuration checks passed; numerical benchmark awaits final
artifacts in Phase S.

Scientific finding addressed: CLAIM-09.

New assumptions introduced: none.

## PHASE O — Experiment provenance

Files changed: `src/utils/provenance.py`, `scripts/train.py`, `src/ensemble/oof.py`,
`scripts/run_ensemble_eval.py`.

Behavior before: new runs had no immutable source/config/dataset/runtime/artifact linkage.

Behavior after: single-model, each OOF fold, and ensemble runs write manifests with run ID,
Git identity, config/dataset/artifact hashes, seed, command/arguments, dependency versions,
CUDA/cuDNN/driver/GPU, and checkpoint/prediction identity.

Tests: frozen dataset verification and provenance helper imports passed.

Scientific finding addressed: CLAIM-12 and `UNVERIFIABLE PROVENANCE LINKAGE`.

New assumptions introduced: none.

## PHASE P — Immutable historical results and FINAL_V2 root

Files changed: four backbone YAMLs, `configs/final_experiment.yaml`,
`scripts/run_kfold.py`, `scripts/run_ensemble_eval.py`.

Behavior before: configs/runners targeted historical result roots or auto-discovered them.

Behavior after: all new runs target `RESULTS/FINAL_V2/{single_split,oof,verification}`;
historical `RESULTS/DEFAULT_TRAINING` and `RESULTS/OOF_TRAINING` are read by neither final
runner nor verifier and are not overwritten.

Tests: exact protocol-path checks passed.

Scientific finding addressed: result auto-discovery/protocol mixing and immutable-history
requirements.

New assumptions introduced: none.

## PHASE Q — Pre-training regression tests

Files changed: `tests/__init__.py`, `tests/test_scientific_pipeline.py`.

Behavior before: no regression gate encoded the audited scientific boundaries.

Behavior after: 11 unit tests cover all ten required gate categories plus canonical
stacking configuration.

Tests: `python -m unittest discover -s tests -v` — 11 passed, 0 failed.

Scientific finding addressed: regression protection for all remediated findings.

New assumptions introduced: `unittest` is used because this environment does not have
pytest installed; it is the repository-appropriate dependency-free test command.

## PHASE R — Code-remediation gate

Files changed: audit reports listed in the Step-4 artifact index.

Behavior before: expensive training was prohibited until code and environment gates passed.

Behavior after: 11/11 unit tests pass; `pip check` reports no broken requirements; all 43
source/script Python files parse; key modules import; four backbone configs match frozen
hyperparameters; dataset snapshot verification passes unchanged.

Tests: PASS. Environment warning: pip reports a stray invalid `-illow` distribution entry,
while also reporting `No broken requirements found`; this did not alter the environment.

Scientific finding addressed: mandatory gate before Phase S.

New assumptions introduced: none.

## Out of scope for Step 4

- dataset deduplication
- dataset cleaning
- dataset resplitting
- near-duplicate detection
- class relabeling
- species relabeling
- new data collection
- new backbone architecture
- hyperparameter search
- nested CV
- UI redesign
- unrelated refactor

## PHASE S — Final training status at session close

Files changed: `audit/STEP4_TRAINING_PAUSE_STATE.md` and this phase log; generated
DenseNet partial artifacts were preserved and were not staged.

Behavior before: DenseNet-121 single-split training was active.

Behavior after: the process is stopped. Epoch 9 is the last durable checkpoint boundary;
epoch 10 briefly began in memory before the interrupt and left no checkpoint/history row.
The user superseded resume intent and directed a fresh restart in the next session.

Tests: process inspection confirmed no `scripts/train.py` process and no Python process
associated with this workspace remains active.

Scientific finding addressed: operational training cancellation and truthful partial-run
provenance; no scientific result is asserted.

New assumptions introduced: none. A fresh non-conflicting result identity/root remains a
precondition for restarting without overwriting the preserved partial run.
