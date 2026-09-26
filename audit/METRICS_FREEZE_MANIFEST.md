# Final Evaluation Plan Freeze Manifest

## 1. Freeze identity

| Field | Frozen value |
|---|---|
| Freeze timestamp | `2026-09-27T02:25:01+07:00` |
| Branch | `fix/scientific-pipeline-v2` |
| Pre-freeze HEAD | `d9cdaa8aa79a35f5d5e3b71f5f3b141d96cee9ab` |
| Pre-freeze tree SHA | `c14a86133cfa4020f7dc7ed51f0303d50e2a929e` |
| Final-training scientific source commit | `d9cdaa8aa79a35f5d5e3b71f5f3b141d96cee9ab` |
| Dataset manifest SHA-256 | `a94426015d83a3b8e9f66a30bddee9e4b376eaf509f166aa522d7a204fe22dc5` |
| Dataset policy | `HUMAN-APPROVED FROZEN INPUT DATASET` |

The scientific source/config tree had no uncommitted changes at freeze inspection. The pre-existing unstaged deletion of `docs/codebase_documentation.md` and unrelated untracked audit files are not part of this freeze and are not evidence of scientific-source modification.

## 2. Frozen dataset identity

| Split | Samples |
|---|---:|
| Train | 7,192 |
| Validation (`data/valid`) | 1,540 |
| Test | 1,546 |
| Total | 10,278 |

Frozen class order:

```text
0 Brown_Blight
1 Gray_Blight
2 Green_mirid_bug
3 Healthy_leaf
4 Helopeltis
5 Tea_algal_leaf_spot
```

## 3. Accepted endpoint assumption

Within this study, the six classes are assigned equal importance at the overall evaluation-endpoint level.

This is a study-specific methodological decision. It does not assert that the six classes have equal agricultural, economic, biological, clinical, or safety costs, and it is not a literature-mandated requirement.

## 4. Frozen metric hierarchy

### PRIMARY

- **Arithmetic classwise Macro-F1** as implemented by `sklearn.metrics.f1_score(..., average="macro", zero_division=0)`.

Exact definition:

```text
F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
Macro-F1 = (1/6) * sum_c F1_c
```

This is not the harmonic mean of Macro Precision and Macro Recall.

### MAIN SECONDARY

- Accuracy.
- Multiclass Matthews correlation coefficient (MCC).
- Negative log-likelihood (NLL) for continuous softmax/mixed/meta-model probability outputs.

### SUPPORTING

- Multiclass Brier score.
- Top-label ECE with 15 equal-width bins.
- Reliability diagram.
- Per-class Precision, Recall, F1, and Support.
- Raw and row-normalized confusion matrices.
- Planned paired metric differences and confidence intervals.
- Computational cost metrics with raw repeat evidence and environment/protocol metadata.

### SUPPLEMENTARY

- Macro Precision.
- Macro Recall / Balanced Accuracy, reported as one value/name only.
- Weighted F1.
- Cohen's Kappa.
- Macro one-vs-rest ROC-AUC.
- Per-class one-vs-rest ROC-AUC.
- Optional AP/PR analysis with exact aggregation labels.
- Ensemble diversity metrics.

### EXPLORATORY

- Probability ambiguity.
- t-SNE.
- Silhouette score.
- Davies–Bouldin index.
- Calinski–Harabasz index.
- Any champion/best-method analysis selected after viewing test results.

### NOT FOR SCIENTIFIC CONCLUSIONS

- Post-hoc power.

## 5. Frozen uncertainty plan

### Macro-F1

```text
design: paired class-stratified nonparametric bootstrap
resampling unit: test sample_id within y_true class
resamples: 10,000
seed: 42
interval: two-sided 95% percentile CI
class composition: fixed at the six observed test class counts
estimand boundary: conditional on the observed class composition
```

### Accuracy, MCC, NLL, and Brier

```text
design: paired ordinary nonparametric bootstrap
resampling unit: test sample_id across the complete test set
draw size: N = 1,546 with replacement
resamples: 10,000
seed: 42
interval: two-sided 95% percentile CI
class composition: allowed to vary around empirical test proportions
estimand boundary: empirical joint distribution represented by the test sample
```

For every paired comparison, the same bootstrap sample indices must be reused for both methods. Full-precision values are used internally; rounding occurs only for display.

These intervals are conditional on the fitted one-seed experiment and the declared empirical test-data model. They do not represent training-seed uncertainty, external-site uncertainty, specimen/plant clustering, or annotation uncertainty.

## 6. Frozen confirmatory hypothesis-testing family

Exactly six tests are confirmatory:

1. Hard Voting: Single-Split versus OOF.
2. Soft Voting: Single-Split versus OOF.
3. Weighted Voting: Single-Split versus OOF.
4. Stacking LR: Single-Split versus OOF.
5. Stacking RF: Single-Split versus OOF.
6. Stacking XGB: Single-Split versus OOF.

Frozen test and correction:

```text
test: exact two-sided McNemar
family-wise alpha: 0.05
family size: 6
Bonferroni threshold: 0.05 / 6 = 0.008333333333333333
decision labels: REJECT or FAIL_TO_REJECT
```

`FAIL_TO_REJECT` must never be interpreted as equivalence. No additional confirmatory pairwise tests may be generated after observing final test results.

## 7. Hard Voting probability treatment

- Label-performance metrics: included in the same principal comparison tables.
- Probability-quality ranking against continuous probability methods: excluded.
- Brier, NLL, and ECE: vote-confidence diagnostics only, in a separate labeled section.
- Probability semantics: empirical distribution of four deterministic base-model votes.
- Attainable top confidence levels: `0.25`, `0.50`, `0.75`, `1.00`.
- Zero true-class votes: unclipped NLL is infinite; current reporting clip gives `-log(10^-15) ≈ 34.5388` for that sample.
- Reliability output: report count and empirical correctness for each attainable confidence level.

## 8. Frozen inference boundaries

The final experiment does not establish:

```text
external-site generalization
protocol equivalence
seed robustness
causal superiority
biological/species validity
clinical/agricultural safety
state-of-the-art status
```

The current experiment uses one configured seed (`42`). Five OOF folds are cross-fitting components, not five independent seeds.

## 9. Authoritative frozen methodology

Precedence order:

1. `audit/METRICS_VERIFICATION_CORRECTIONS.md`
2. `audit/METRICS_PROPOSED_EVALUATION_PLAN.md`
3. `audit/METRICS_STATISTICAL_ANALYSIS_PLAN.md`
4. `audit/METRICS_FREEZE_MANIFEST.md`

Historical wording in `METRICS_SCIENTIFIC_RESEARCH_REPORT.md` cannot override these files.

### SHA-256 integrity table

| Frozen methodology/support file | SHA-256 |
|---|---|
| `audit/METRICS_VERIFICATION_CORRECTIONS.md` | `55af77ae89f140754c43cab29d25482f28e52fd6d0281a4224d02711af423990` |
| `audit/METRICS_PROPOSED_EVALUATION_PLAN.md` | `20c03c3d5cc1f56afba97ae47f3657688644ca4d0be3db292b2cc447c7a7cef5` |
| `audit/METRICS_STATISTICAL_ANALYSIS_PLAN.md` | `c120fed9dbbf16ce8d2ececda34213d990289ddaddb397ef55b777d0d5b36407` |
| `audit/METRICS_REFERENCES.md` | `f75060178824694ec350e4043d549352a38ef7a634f0a7287b31223b782b9bae` |
| `audit/METRICS_SCIENTIFIC_RESEARCH_REPORT.md` with freeze notice | `e516a7de251eaef888b493aa96cdfd6b4e00fb66b5b657b4f16a7bccd54d4604` |
| `audit/METRICS_FREEZE_MANIFEST.md` | Self-referential file hash recorded post-commit in `audit/METRICS_FREEZE_REPORT.md`; immutable Git identity is the final evaluation-plan commit/tree. |

An ordinary file SHA-256 cannot be embedded as a literal inside the same file without changing that hash. The post-commit report therefore records this manifest's final SHA-256, while the Git commit/tree pins the complete frozen byte content.

## 10. Post-freeze change rule

> Any metric, hypothesis test, decision rule, comparison family, or primary/secondary-role change made after inspection of FINAL test results must be explicitly labeled POST-HOC/EXPLORATORY and must not silently replace this frozen plan.

No final test result was inspected or generated during this freeze.
