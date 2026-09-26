# Step 4B Targeted Final-Pipeline Remediation Report

## Scope and source identity

- Branch: `fix/scientific-pipeline-v2`
- Starting HEAD: `7e4796b863d30239eeca42b306baf614bdee6e9d`
- Canonical result root: `RESULTS`
- Canonical run layout: `RESULTS/runs/<run_id>/<protocol>/...`
- Model training executed in Step 4B: **NO**
- Dataset mutation executed in Step 4B: **NO**

The user-requested root rename from `RESULTS/FINAL_V2` to `RESULTS` is applied to current configuration, CLIs, runners, verification, server, tests, and UI. No existing result directory was moved or deleted.

## Remediation status

| Item | Status | Implemented behavior |
|---|---|---|
| P0-1 Final run identity | PASS | `run_id` and `RESULTS/runs/<run_id>` are mandatory in scientific entrypoints. `Trainer` no longer exposes suffix/latest-directory resolution. Scratch refuses a non-empty exact model root; resume requires that root's `last_model.pth`. Ensemble input paths are derived from the same run identity. |
| P0-2 OOF cache identity | PASS | Complete-cache reuse validates protocol, run ID, dataset hash, ordered sample-ID hash, class order, fold count/set, fold train/holdout index hashes, split seed, config hash, backbone, source commit, checkpoint hashes, and cache-artifact hashes. Missing, partial, or mismatched caches fail with the identity path/reason. |
| P0-3 API regressions | PASS | Master verification dispatch uses current signatures; latency/t-SNE require protocol; backbone registry contains exactly four model configs; aggregate runner consumes canonical return fields and no longer overwrites canonical model-size paths. Main ensemble calls use `protocol`, `results_root`, and `run_id`. |
| P0-4 Server protocol isolation | PASS | Server uses `RESULTS_RUN_ID`, exact run roots, canonical method IDs, and protocol-specific model/ensemble dictionaries. OOF requests fail when OOF base or ensemble artifacts are absent; Single-Split objects are not substituted. |
| P1-1 Prediction identity | PASS | Canonical NPZ records persist row identity, labels, probabilities, predictions, class order, protocol, method, split, run ID, dataset/config hashes, source commit, checkpoint/artifact hashes, and aggregation semantics. |
| P1-2 Prediction validation | PASS | Validation checks shapes, unique IDs/classes, finite probabilities, bounds, row sums, label/prediction ranges, argmax consistency, method/protocol/split/run/dataset/config/artifact identity. Row reordering is rejected explicitly. |
| P1-3 Plot refit removal | PASS | Canonical plotting is selected by explicit `run_id`, consumes saved prediction records, and does not call estimator `fit`. Missing canonical records fail. Historical plotting remains isolated behind the no-`run_id` legacy path. |
| P1-4 Legacy isolation | PASS | Canonical plotting requires run identity; the legacy comparison generator rejects the canonical `RESULTS` root/config roots; server latest/historical discovery helpers were removed; canonical evaluator does not emit ambiguous compatibility NPY files by default. |
| P1-5 Split identity | PASS | Standalone and Trainer evaluation select the requested loader exactly. Missing test/validation loaders raise instead of cross-split fallback. Output filenames carry the actual split. |
| P1-6 Unknown configuration | PASS | Unknown optimizer, scheduler, ensemble mode/method, or protocol raises a configuration/identity error. Dataset identity must come from the explicit YAML and is no longer auto-discovered from outputs or folders. |
| P2-1 Statistics/reporting | PASS | Active server/UI removes historical fixed N, p-values, thresholds, calibration conclusions, and result values. Missing data is shown as unavailable. McNemar displays saved raw values and neutral `REJECT`/`FAIL_TO_REJECT` decisions. |
| P2-2 Latency completeness | PASS | The benchmark dispatches all six ensemble methods through real loaded base/fitted artifacts. No estimated stacking overhead is used in the canonical benchmark. Step 4B did not execute the benchmark. |
| P2-3 Resume semantics | PASS | Last checkpoints persist optimizer, scheduler, history, scaler, early-stop counter, and checkpoint-selection tracking. Restore uses the last checkpoint consistently. Exact RNG continuation is explicitly not claimed. |
| P2-4 Validation path | PASS | Effective validation path is explicitly `./data/valid` in dataset/final configuration and is consumed directly; no dataset directory was renamed or moved. |

## Provenance gate

Scientific artifact generation requires a full Git commit and rejects uncommitted changes under `configs/`, `scripts/`, `src/`, `main.py`, `server.py`, or `requirements.txt`. This prevents a future artifact from identifying only the previous HEAD while executing changed scientific source. Audit/report-only working-tree changes do not alter the scientific source identity check.

## Resume boundary

Resume provides logical training-state continuation. It restores model, optimizer, scheduler, history, checkpoint-selection fields, AMP scaler state, and early-stop counter. RNG state is not persisted; bitwise-identical continuation is not claimed.

## Explicitly not performed

- No CNN or meta-learner training was executed.
- No final verification or latency benchmark was executed.
- No final result artifacts were generated.
- No dataset image, class directory, file name, or split was changed.
- No Soft+T method, new backbone, hyperparameter search, deduplication, or resplitting was introduced.

`NO DATASET IMAGE, CLASS, OR SPLIT WAS MODIFIED BY STEP 4B.`
