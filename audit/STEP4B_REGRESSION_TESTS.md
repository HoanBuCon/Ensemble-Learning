# Step 4B Regression Tests

## Safe gate commands

| Command | Result |
|---|---|
| `python -m unittest discover -s tests -v` | PASS — 26 tests |
| `python -m pip check` | PASS — no broken requirements; environment emitted the pre-existing `-illow` distribution warning |
| `git diff --check` | PASS — no whitespace errors; Git emitted line-ending conversion warnings only |
| `python -m compileall -q main.py server.py src scripts tests` | PASS |

No command above starts CNN training, K-fold training, ensemble fitting for scientific results, final verification, or final latency benchmarking.

## Step 4B coverage map

| Contract | Regression evidence |
|---|---|
| Exact run root and scratch collision | `test_final_run_identity_is_exact_and_shared` |
| OOF same-identity cache accepted | `test_oof_cache_identity_accept_and_reject_cases` |
| Dataset hash mismatch rejected | `test_oof_cache_identity_accept_and_reject_cases` |
| Split seed mismatch rejected | `test_oof_cache_identity_accept_and_reject_cases` |
| Config hash mismatch rejected | `test_oof_cache_identity_accept_and_reject_cases` |
| Same-shape/different-sample identity rejected | `test_oof_cache_identity_accept_and_reject_cases`, `test_row_reordering_is_rejected_by_policy` |
| Backbone-only config discovery | `test_backbone_registry_excludes_non_models` |
| Current master verification dispatch | `test_master_verification_dispatch_uses_current_latency_contract` |
| Current CLI contracts and `RESULTS` root | `test_master_cli_uses_canonical_results_root_and_current_contracts` |
| OOF server fail-closed for base and ensemble | `test_server_oof_request_never_falls_back_to_single` |
| Full canonical prediction metadata/validation | `test_prediction_identity_roundtrip_and_fail_closed_validation` |
| Canonical plots do not refit; legacy comparison rejects `RESULTS` | `test_final_plot_path_has_no_estimator_fit` |
| Val/test loader isolation | `test_split_selection_never_falls_back`, `test_07_split_naming` |
| Unknown protocol/optimizer rejection | `test_unknown_protocol_and_oof_optimizer_fail` |
| Hard Voting real latency dispatch | `test_latency_hard_voting_executes_real_dispatch` |
| Checkpoint persisted resume state | `test_resume_checkpoint_persists_logical_continuation_state` |
| Trainer restores the last continuation state consistently | `test_trainer_restores_last_continuation_state_without_best_state_mix` |
| Dirty scientific source rejected by provenance gate | `test_provenance_rejects_uncommitted_scientific_source` |
| Missing report values are not fabricated | `test_08_report_missing_metrics_are_not_fabricated` |
| OOF boundary/placement and external-validation signature | `test_01_oof_boundary_and_placement` |
| Weighted Voting SLSQP constraints | `test_02_weighted_voting_slsqp` |
| Fitted ensemble save/load replay | `test_03_ensemble_serialization_round_trip` |

Unit-test success is evidence for these contracts, not evidence for final numerical scientific results. No final experiment artifacts existed or were required by this gate.
