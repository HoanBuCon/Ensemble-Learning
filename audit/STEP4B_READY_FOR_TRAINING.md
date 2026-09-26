# Step 4B Readiness Gate

Canonical result layout: `RESULTS/runs/<run_id>/<protocol>/...`

| # | Gate question | Status | Evidence summary |
|---:|---|---|---|
| 1 | Is fresh Single-Split output consumed by the exact same run identity? | PASS | Training and ensemble paths share `results_root` + validated `run_id`; scratch cannot suffix the directory. |
| 2 | Is OOF cache identity-safe? | PASS | Complete cache reuse validates all required dataset, row, fold, seed, config, checkpoint, source, run, and class identities. |
| 3 | Can Single/OOF artifacts cross-load? | PASS | Protocol roots and prediction metadata are validated; cross-protocol records are rejected. |
| 4 | Can verification/plots refit canonical ensemble models? | PASS | Canonical verification and run-ID plotting replay saved predictions/fitted objects only. |
| 5 | Can any scientific evaluation silently change split? | PASS | Requested split loader is selected exactly; absence is an error. |
| 6 | Can server silently change protocol? | PASS | Protocol-specific base and ensemble artifacts are mandatory; missing OOF cannot fall back to Single-Split. |
| 7 | Are prediction artifacts provenance-linked? | PASS | Canonical metadata includes run/dataset/config/source/artifact identities and aggregation semantics; dirty scientific source is rejected before artifact generation. |
| 8 | Are post-remediation CLI contracts valid? | PASS | Current parser/dispatch/runner signatures are smoke-tested; result root defaults to `RESULTS`. |
| 9 | Are all safe regression tests passing? | PASS | 26 unit tests, package consistency, compilation, and whitespace gate pass. |
| 10 | Was dataset untouched? | PASS | `git diff -- data artifacts/manifests` is empty; no dataset mutation command was run. |

## P0 decision

| P0 item | Status |
|---|---|
| P0-1 Run identity | PASS |
| P0-2 OOF cache identity | PASS |
| P0-3 API/config/aggregate contracts | PASS |
| P0-4 Server protocol isolation | PASS |

**CODE READINESS: PASS.**

Operational precondition: commit the scientific source/config changes before any final run. The pipeline enforces this and refuses artifact generation from an uncommitted scientific working tree. No training was started in Step 4B.

`NO DATASET IMAGE, CLASS, OR SPLIT WAS MODIFIED BY STEP 4B.`
