# Step 4 Test Results

## Regression suite

Command: `python -m unittest discover -s tests -v`

Result: **PASS — 11 passed, 0 failed**.

Covered gates:

1. OOF optimization/outer-holdout boundary and exact row placement.
2. SLSQP Weighted Voting constraints and finite objective.
3. Fitted ensemble save/load/predict replay.
4. Single-split/OOF protocol isolation.
5. Sample-order mismatch rejection.
6. Missing stacking/prediction artifact fail-closed behavior.
7. Validation/test output naming.
8. No fabricated precision/recall/F1 when predictions are absent.
9. Persisted checkpoint resume state including `best_epoch`.
10. t-SNE source/dimension metadata and neutral statistical decisions.
11. Canonical stacking parameter source.

## Other gates

- `python -m pip check`: PASS (`No broken requirements found`).
- Frozen dataset snapshot verification: PASS.
- Backbone invariant check: PASS (4/4).
- Python syntax parse: PASS (43 files).
- Critical module import smoke test: PASS.
- `git diff --check`: PASS; only Git CRLF conversion warnings were emitted.

No CNN training was performed before these gates passed.
