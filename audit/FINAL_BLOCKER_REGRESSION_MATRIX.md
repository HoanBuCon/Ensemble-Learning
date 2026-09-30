# Final blocker regression matrix

Snapshot basis: starting HEAD `a2b19a2e8f66ed2dc3e8e244be8040cf930d605d`; changes are uncommitted and scientific dirty-tree detection intentionally blocks training until commit.

| Finding | Original adversarial input | Expected after remediation | Regression test | Result | Scientific behavior changed? | Existing final result retraining? |
|---|---|---|---|---|---|---|
| F01 | First line ` M src/engine/trainer.py` (also config/main; staged/untracked; docs preceding) | Every scientific path detected regardless of status/order; docs/audit-only clean under policy | `test_f01_porcelain_status_columns_and_order` | **PASS** | Provenance gate only | N/A — no final result |
| F02 | Same-shape fit/test with different source/config/checkpoints or fake/reused IDs | Reject before `.fit()`; accept exact approved Single lineage and five-fold OOF lineage | `test_f02_single_fit_test_lineage_and_membership`; `test_f02_oof_complete_five_fold_lineage` | **PASS** | Input acceptance only; ensemble math unchanged | N/A — no final result |
| F03 | Foreign compatible checkpoint; missing provenance; wrong run/protocol/config/dataset/source/fold | Exact best/last role and full origin required in final paths; start identity exists before epoch loop | `test_f03_checkpoint_origin_accepts_exact_and_rejects_foreign`; `test_f03_run_start_identity_is_checked` | **PASS** | Provenance/schema/failure behavior only | N/A — no final result |
| F04 | Same global seed + fresh Compose gives different sequence; cloned workers retain identical RNG | Same owner seed repeats; fold and worker ownership deterministic/distinct; val/test unchanged | `test_f04_albumentations_seed_ownership` | **PASS** | Train augmentation RNG ownership; definitions unchanged | N/A — no final result |
| F05 | Epoch 8, counter 6, patience 6, no final metrics → epoch 9 | No optimization call; counter 5 still resumes at epoch 9 | `test_f05_terminal_resume_never_enters_epoch_nine`; existing last-state test | **PASS** | Terminal recovery only | N/A — no final result |

## Required-case coverage

| Required case | Result |
|---|---|
| F01 first unstaged `src/` line | PASS |
| F01 first unstaged `configs/` line | PASS |
| F01 first unstaged `main.py` line | PASS |
| F01 preceding documentation then scientific line | PASS |
| F01 staged modification | PASS |
| F01 untracked scientific file | PASS |
| F01 clean scientific tree | PASS |
| F02 different source | REJECTED |
| F02 different config | REJECTED |
| F02 different checkpoint lineage | REJECTED |
| F02 fake fit IDs | REJECTED |
| F02 test IDs reused in fit | REJECTED |
| F02 correct Single lineage | ACCEPTED |
| F02 correct OOF five-fold lineage | ACCEPTED |
| F03 correct checkpoint origin | ACCEPTED |
| F03 wrong run/protocol/config/dataset/source | REJECTED |
| F03 wrong OOF fold | REJECTED |
| F03 missing final-scientific provenance | REJECTED |
| F04 same seed + fresh train transform | SAME SEQUENCE |
| F04 different seed | DIFFERENT SEQUENCE IN FIXTURE |
| F04 deterministic fold seed derivation | PASS |
| F04 distinct worker seed ownership | PASS |
| F04 repeated val/test transform | DETERMINISTIC |
| F05 exhausted counter | NO NEXT EPOCH |
| F05 non-exhausted counter | NORMAL NEXT EPOCH |

## Safe gate record

| Command | Interpreter / result |
|---|---|
| `python -m unittest tests.test_final_blocker_remediation -v` | Repository venv; PASS 7/7 |
| `python -m unittest discover -s tests -v` | Repository venv; PASS 35/35 |
| `python -m pip check` | Repository venv; PASS: `No broken requirements found` (non-failing stale `-illow` distribution warning emitted) |
| `git diff --check` | PASS: exit 0; line-ending conversion warnings only |

Test phase banners printed by mocked E2E dispatch tests do not represent training execution. No official result root was created.

## Deferred findings

F06, F07, F08, F09, F10, F11, F12 and F13 were not remediated or reclassified by this task. Their status remains governed by `FINAL_SCIENTIFIC_AUDIT_REPORT.md` until independently addressed.
