# Targeted final-training blocker remediation report

Date: 2026-09-30 (Asia/Saigon)

Branch: `fix/scientific-pipeline-v2`

Starting HEAD: `a2b19a2e8f66ed2dc3e8e244be8040cf930d605d`

Scope: **F01–F05 only**. No CNN training, OOF execution, final ensemble fit, final test evaluation, benchmark, dataset mutation, metric change, frozen-plan change, augmentation-policy change, checkpoint-selection-rule change, or model/method change was performed.

## Outcome

| Finding | Result | Blocking mechanism after remediation |
|---|---|---|
| F01 | **PASS** | Dirty scientific paths are detected without losing porcelain status columns |
| F02 | **PASS** | Approved split membership and fit/test model lineage are validated before any fitted ensemble `.fit()` |
| F03 | **PASS** | New scientific checkpoints persist origin identity; resume/evaluation fail closed on missing or foreign provenance |
| F04 | **PASS** | Albumentations RNG is explicitly owned by run/fold/worker seed state while augmentation definitions remain unchanged |
| F05 | **PASS** | A restored counter already at patience cannot execute an additional optimization epoch |

The source is intentionally uncommitted at report time. The corrected F01 gate detects these scientific changes and therefore final training must not start until the remediation is reviewed and committed. This task's completion criterion is readiness for a targeted GO verification, not permission to train from a dirty tree.

There is currently no final result to retrain. Historical checkpoints lacking the new mandatory scientific provenance are not promoted to final scientific checkpoints.

## F01 — Dirty scientific source detection

Original counterexample:

```text
git status porcelain first record = " M src/engine/trainer.py"
whole-output .strip()
→ first status-space removed
→ line[3:] becomes "rc/engine/trainer.py"
→ scientific change missed
```

Source change:

- `src/utils/provenance.py:git_identity()` now removes only record terminators from porcelain output and preserves leading status columns.
- `_scientific_changes_from_porcelain()` validates the `XY<space>path` record layout, handles both sides of rename/copy records, normalizes paths and retains the existing scientific-root policy.
- Scientific roots remain `src/`, `scripts/`, `configs/`, plus `main.py`, `server.py`, `requirements.txt`. Audit/docs-only changes remain non-blocking.

New expected contract: unstaged, staged, untracked and rename/copy scientific paths are detected independently of line order. Malformed porcelain input fails rather than being silently treated as clean.

Regression test: `tests/test_final_blocker_remediation.py:test_f01_porcelain_status_columns_and_order` covers all seven required cases and invokes actual `git_identity()` with mocked raw Git output. The current real working tree also returned all modified scientific paths.

Scientific behavior changed: provenance rejection behavior only; no optimizer/data/model behavior changed.

Existing final result retraining: **NOT APPLICABLE — no final result exists**. If a future run were executed from an undetectably dirty historical source, its source identity would need independent recovery or rerun.

## F02 — Fit/test identity and frozen-manifest membership

Original counterexample: internally aligned fit and test groups with the same shape/run/dataset string but different source/config/checkpoint lineage—and even reused test IDs marked `val`—were accepted before ensemble fitting.

Source change:

- `scripts/run_ensemble_eval.py:validate_ensemble_fit_test_identity()` loads the frozen CSV records and checks ordered IDs, labels and canonical class order against the approved split.
- Single-Split requires exact validation rows for fit, exact test rows for inference, disjoint populations, the current source/config and identical `{"checkpoint": hash}` lineage per backbone.
- OOF requires exact train rows for fit, exact test rows for inference, disjoint populations, the current source/config and the same complete `fold_1`…`fold_5` checkpoint-hash mapping per backbone.
- Protocol-specific aggregation semantics are checked: `single_checkpoint_inference`; or OOF `one_outer_holdout_checkpoint_per_training_row` and `mean_fold_probabilities`.
- The validator is called before output creation, `WeightedVoting.fit()` and every stacking `.fit()`.

New expected contract: a dataset-hash string, matching array shape or matching within-group identity cannot substitute for approved sample membership and cross-group lineage proof.

Regression tests:

- `test_f02_single_fit_test_lineage_and_membership`
- `test_f02_oof_complete_five_fold_lineage`

They reject different source, config, checkpoint lineage, fake fit IDs and test IDs reused as fit rows; correct Single and OOF lineage passes. Source-order assertion proves the gate precedes `.fit()`.

Scientific behavior changed: input rejection only. Weighted Voting objective, fit split, six methods and stacking feature construction are unchanged.

Existing final result retraining: **NOT APPLICABLE — no final result exists**. With valid base artifacts, a rejected ensemble needs correct ensemble regeneration, not automatic CNN retraining.

## F03 — Scientific checkpoint origin

Original counterexample: a shape-compatible OOF/fold checkpoint could be loaded by standalone Single-Split evaluation and assigned current evaluation-time provenance. Resume also trusted the expected filename without proving checkpoint origin.

Source change:

- `src/engine/checkpoint.py` defines a required scientific origin contract: run ID, protocol, backbone, frozen dataset-manifest SHA-256, source commit, effective config SHA-256, fold identity and checkpoint role.
- Best and last checkpoints persist distinct roles. A scientific load validates every required field before returning state.
- `Trainer` creates its checkpoint manager with Single-Split provenance. Resume therefore rejects missing/foreign provenance before loading model/optimizer/scheduler state.
- OOF fold checkpoints include fold number, n_splits, split seed and train/holdout index hashes.
- `scripts/evaluate.py` verifies the frozen dataset and loads only a best checkpoint with exact Single-Split origin. Output provenance comes from the validated checkpoint.
- `scripts/train.py` writes a run-start manifest before the epoch loop; resume validates the existing start manifest rather than overwriting it.
- Each OOF fold writes its run-start identity before `_train_fold()`.

New expected contract: filename and tensor compatibility are insufficient scientific identity. Newly generated final checkpoints without required provenance, or with any requested identity mismatch, fail closed.

Regression tests:

- `test_f03_checkpoint_origin_accepts_exact_and_rejects_foreign`
- `test_f03_run_start_identity_is_checked`

They accept exact origin and reject wrong run, protocol, config, dataset, source, OOF fold, checkpoint role/missing required provenance. Generic non-scientific manager use remains possible, but final Trainer/evaluation paths always supply expected provenance.

Scientific behavior changed: checkpoint/run provenance and failure behavior. Model architecture, optimizer and checkpoint-selection rule are unchanged.

Existing final result retraining: **NOT APPLICABLE — no final result exists**. A future historical checkpoint without provenance cannot be silently declared final; its origin must be independently established or it must be rerun.

## F04 — Albumentations RNG ownership

Original counterexample: two fresh train transforms created after the same global seed produced different augmentation sequences because Albumentations 2 owns an independent RNG.

Source change:

- `build_transforms(..., seed=...)` passes the declared owner seed to `A.Compose`.
- Single-Split train/valid/test transforms receive the experiment seed.
- OOF defines `fold_seed = experiment_seed + zero_based_fold_index`, uses it for model/global RNG, DataLoader generator and the fold's main-process transform.
- In multiprocessing, `seed_worker()` derives a 32-bit seed from PyTorch's deterministic DataLoader worker initial seed and applies it to the worker's dataset transform. Worker streams therefore inherit run/fold generator ownership and do not remain identical cloned Compose states.
- Validation/test transformation definitions remain deterministic and unchanged.

Exact ownership:

```text
Single process: experiment seed (or OOF fold seed) → Albumentations Compose
Worker process: DataLoader generator seed → torch worker initial seed
               → 32-bit worker seed → Python/NumPy/Albumentations worker state
OOF fold seed: experiment seed + zero-based fold index
```

New expected contract: equal declared seed with a fresh transform reproduces the sequence; intended different fold/worker seeds own distinct streams. This is logical augmentation reproducibility, not universal bitwise CUDA reproducibility.

Regression test: `test_f04_albumentations_seed_ownership` checks same-seed sequence equality, different-seed sequence difference, deterministic valid/test, fold seed derivation, nested dataset transform seeding and distinct worker-derived seeds.

Scientific behavior changed: stochastic train augmentation is now controlled by declared identity. Probabilities, transforms, limits and train-only scope are unchanged.

Existing final result retraining: **NOT APPLICABLE — no final result exists**. A future result claiming the repaired seed contract must be generated after this change.

## F05 — Terminal early-stop resume

Original counterexample:

```text
last completed epoch = 8
restored early_stop_counter = patience = 6
no final test_metrics.json
old resume path returns start_epoch = 9
→ optimizer can execute another epoch
```

Source change: `Trainer._restore_checkpoint()` now restores the complete last state, then checks the already-persisted counter. If it meets/exceeds configured patience, it returns the existing completed/terminal path and does not enter the epoch loop. For a counter below patience, ordinary `last_epoch + 1` continuation remains unchanged.

New expected contract: interruption between terminal last-checkpoint write and final evaluation cannot create an extra optimization epoch. This does not add RNG-state/bitwise resume claims.

Regression test: `test_f05_terminal_resume_never_enters_epoch_nine` uses the audited epoch-8/counter-6 fixture and a sentinel `_train_one_epoch`; the sentinel is never called. A counter of 5 returns epoch 9 normally. The pre-existing last-state-versus-best-state regression remains PASS.

Scientific behavior changed: terminal recovery only; ordinary mid-run continuation and the accuracy/loss-gate/patience definitions are unchanged.

Existing final result retraining: **NOT APPLICABLE — no final result exists**.

## Scope preservation

- Dataset manifest SHA-256 remains `a94426015d83a3b8e9f66a30bddee9e4b376eaf509f166aa522d7a204fe22dc5`.
- Dataset snapshot and frozen metric-plan files were not edited.
- F06–F13 remain outside this remediation. This report makes no claim that they are fixed. In particular, future result/report generation still needs the previously documented F06–F13 review.
- The pre-existing deletion of `docs/codebase_documentation.md` was not modified or staged.

## Verification summary

| Gate | Result |
|---|---|
| Targeted F01–F05 tests | PASS: 7/7 |
| Full safe unittest discovery | PASS: 35/35, zero failures/errors/skips |
| `python -m pip check` | PASS: `No broken requirements found` (the environment emitted a non-failing stale `-illow` distribution warning) |
| `git diff --check` | PASS: exit 0 (line-ending conversion warnings only) |
| CNN/OOF/final ensemble/final test/benchmark | NOT RUN |

Decision: **all five targeted blocker contracts pass their regression tests**. Review/commit and an independent targeted GO verification are still required before invoking training.

`FINAL TRAINING BLOCKERS REMEDIATED — READY FOR TARGETED GO VERIFICATION.`
