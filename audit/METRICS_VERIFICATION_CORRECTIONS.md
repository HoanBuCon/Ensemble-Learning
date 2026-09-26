# Metrics Plan Verification Corrections

## 1. Scope and repository snapshot

```text
Branch: fix/scientific-pipeline-v2
HEAD: d9cdaa8aa79a35f5d5e3b71f5f3b141d96cee9ab
Verification date: 2026-09-27
```

This pass did not change the selected metric plan by inspecting results: no final result exists. It verifies definitions, estimands, and reference-to-claim mappings. Current executable source remains authoritative. The following corrections supersede the affected wording in `METRICS_SCIENTIFIC_RESEARCH_REPORT.md`; that report was not modified because the allowed update list was limited to the proposal, statistical plan, gap matrix, references, and this correction record.

No source code, config, dataset, result, manuscript, checkpoint, or training artifact was changed.

## 2. Correction summary

| Issue | Previous wording/assumption | Verified correction | Files updated |
|---|---|---|---|
| Macro-F1 source | Sokolova & Lapalme cited as if it directly supported current formula | sklearn uses arithmetic mean of per-class F1; Sokolova is associated with the competing harmonic-of-macro-P/R definition | proposal, statistical plan, references |
| Macro-F1 role | “minimum defensible primary endpoint” sounded mandatory | conditional methodological proposal only, requiring pre-specified equal endpoint-level importance for all six classes | proposal, statistical plan, gap matrix |
| Bootstrap | one class-stratified design for five endpoints; frozen test called population | test is an observed evaluation sample; Macro-F1 gets fixed-composition stratified bootstrap, while Accuracy/MCC/NLL/Brier get ordinary paired bootstrap | statistical plan, gap matrix |
| Hard Voting | could share probability-quality table with annotation | separate vote-confidence diagnostic section; no probability-quality ranking against continuous outputs | proposal, gap matrix |
| ROC-AUC | Hand & Till cited for current macro OvR | current semantics come from sklearn; Hand & Till is pairwise/one-vs-one and not the exact source | proposal, references |
| Latency | median [IQR] across five repeat means presented as preferred headline | mean ± SD retained as current descriptive headline; median [IQR] optional; neither is an MLPerf mandate | proposal, statistical plan, gap matrix, references |
| GAP-26 | missing results marked required before final run | results are produced by the run; absence is not a readiness prerequisite | gap matrix |

## 3. Macro-F1 definition and scientific assumption

### Current executable definition

`src/utils/metrics.py:L54-L57` and `scripts/verification/eval_advanced_metrics.py:L50-L54` call:

```python
f1_score(y_true, y_pred, average="macro", zero_division=0)
```

Official sklearn semantics are:

```text
F1_c = 2 Precision_c Recall_c / (Precision_c + Recall_c)
Macro-F1 = (1/C) sum_c F1_c
```

That is the arithmetic mean of six classwise harmonic means. It is not:

```text
2 MacroPrecision MacroRecall / (MacroPrecision + MacroRecall)
```

Direct inspection of Sokolova & Lapalme's multiclass metric table confirms that it defines macro precision and macro recall first and then takes their harmonic mean. Opitz & Burst explicitly distinguishes that formula from arithmetic-mean classwise F1 and shows that the formulas can give different values and rankings. Takahashi et al. directly defines the arithmetic-mean classwise variant and its uncertainty problem. Exact implementation semantics are tied to [sklearn `f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html), with formula disambiguation from [Opitz & Burst](https://arxiv.org/abs/1911.03347) and [Takahashi et al.](https://doi.org/10.1007/s10489-021-02635-5).

### Endpoint decision boundary

Macro-F1 is not mandated by literature. It is a defensible candidate primary endpoint **if and only if** investigators decide before test inspection that all six classes should receive equal endpoint-level weight. The repository's class names and counts do not establish equal agricultural loss, intervention cost, biological severity, or safety consequence. If domain costs differ, a domain-defined endpoint is required; this audit does not invent one.

The corrected plan therefore uses “proposed PRIMARY, conditional” rather than “minimum defensible primary endpoint.”

## 4. Bootstrap estimand correction

### Why “fixed test population” and bootstrap uncertainty conflicted

For a completely fixed set of 1,546 images, each metric is a deterministic descriptive number. Bootstrap uncertainty arises only after adopting a working sampling model in which the observed rows approximate repeat draws. The repository does not establish a probability sampling frame for a wider agricultural population. Therefore the intervals are empirical-distribution uncertainty summaries, not proof of generalization to all future tea leaves.

### A. Fixed-composition conditional estimand

For proposed Macro-F1:

- resample image IDs with replacement separately inside each true-class stratum;
- preserve the six observed class counts;
- use identical bootstrap indices for methods being compared;
- 10,000 replicates, seed 42, percentile 95% interval.

The estimand is performance conditional on the current class composition. This excludes prevalence uncertainty. Despite equal arithmetic weight across classwise F1 values, Macro-F1 is not entirely prevalence-invariant because each class's precision depends on false positives from other classes.

### B. Empirical joint-distribution estimand

For Accuracy, MCC, NLL, and Brier:

- resample all 1,546 image IDs with replacement without fixing class counts;
- reuse identical indices across paired methods;
- 10,000 replicates, seed 42, percentile 95% interval.

This allows class composition to vary around observed empirical proportions. That is preferable for these prevalence/marginal-dependent endpoints:

- Accuracy, NLL, and Brier are sample-weighted averages;
- MCC depends on the full joint confusion structure and its marginals.

A fixed-count stratified interval for these metrics would be a narrower conditional sensitivity analysis and must be labeled as such. One bootstrap design is therefore not retained for all five endpoints.

[Efron & Tibshirani](https://doi.org/10.1214/ss/1177013815) and [Davison & Hinkley](https://doi.org/10.1017/CBO9780511802843) support the general need to make resampling reflect the assumed data structure. They do not mandate 10,000 replicates, percentile intervals, or this endpoint split; those are transparent proposal choices. Takahashi et al. addresses arithmetic-mean Macro-F1 uncertainty but does not mandate the proposed bootstrap.

## 5. Hard Voting probability-quality correction

### Current source behavior

`src/ensemble/voting.py:L48-L98` forms a six-class vector from four hard labels:

```text
p_c = number of base-model votes for class c / 4
```

Top-label confidence can therefore be only `0.25`, `0.50`, `0.75`, or `1.00`. `scripts/verification/eval_calibration.py:L24-L47` applies the same ECE-15, Brier, and clipped NLL formulas used for other methods.

### Exact diagnostic estimands

- **Brier:** squared error of the empirical four-vote class distribution against the one-hot observed class. It measures the issued vote distribution, not a smooth conditional posterior.
- **NLL:** logarithmic loss of the vote fraction assigned to the observed class after clipping. With zero true-class votes, the unclipped log loss is infinite; current code reports `-log(10^-15) = 34.538776...` for that sample. Hence the finite result and any ranking depend directly on the chosen epsilon.
- **ECE-15:** calibration of top-vote fraction versus empirical correctness. Because only four confidence values are attainable, at most four of 15 bins are occupied. Empty bins provide no additional calibration resolution, and a reliability diagram should show exact confidence levels plus sample counts.

### Reporting decision

Use **separate vote-confidence diagnostics (option B)**:

- keep Hard Voting in label-performance tables;
- exclude it from the main continuous probability-quality ranking;
- report its Brier/NLL/ECE in a clearly labeled vote-confidence section;
- show exact-confidence reliability rows at 0.25/0.50/0.75/1.00;
- state the NLL epsilon and zero-vote count.

Proper-scoring-rule theory supports evaluating an issued distribution, but finite ensemble distributions have ensemble-size effects. [Ferro (2014)](https://doi.org/10.1002/qj.2270) is scope-limited background because its exchangeable ensemble-forecast setting is not established for four heterogeneous deterministic CNNs.

## 6. ROC-AUC reference correction

Current code calls:

```python
roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
```

The exact estimand is six one-vs-rest AUCs, each class compared against the pooled remaining classes, followed by their unweighted mean. Official [sklearn API documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) and its [multiclass ROC example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) are the exact implementation references.

[Hand & Till (2001)](https://doi.org/10.1023/A:1010920819831) averages pairwise class comparisons. It is retained only as background for a different multiclass AUC generalization and removed from claims about the current OvR metric.

## 7. Latency-summary correction

Current code stores five repeat-level mean latencies. It does not store an MLPerf query stream or claim MLPerf compliance.

[Reddi et al.](https://doi.org/10.1109/ISCA45697.2020.00045) supports scenario-specific workloads, environment control, and transparent latency/throughput reporting. MLPerf Single-Stream uses 90th-percentile query latency under its own load generator; it does not support “median [IQR] across five repeat means” as a standard.

Corrected reporting:

- retain every raw repeat mean;
- headline the currently natural mean ± SD across five repeat means;
- optionally show median [IQR] as a robustness description, explicitly noting the coarse `n=5` distribution;
- retain environment metadata, warmup/timed iterations, repeats, batch size, device/protocol identity, and derived-FPS labeling.

These are transparent descriptive choices, not literature-mandated summaries. No benchmark was executed.

## 8. GAP-26 workflow correction

The absence of final numerical evidence before the final run is expected. It cannot be a prerequisite that must be satisfied before the run that produces it. `GAP-26` now states:

```text
required_before_final_run = NO
required_before_final_reporting = YES
```

This does not declare readiness or generate evidence; it only corrects workflow logic.

## 9. Reference audit result

Every reference in `METRICS_REFERENCES.md` now has exactly one audit classification:

```text
DIRECT SUPPORT
BACKGROUND ONLY
IMPLEMENTATION SEMANTICS
SCOPE-LIMITED
REMOVE/REPLACE
```

Most material changes are:

- Sokolova & Lapalme: `REMOVE/REPLACE` for exact current Macro-F1.
- Hand & Till: `REMOVE/REPLACE` for exact current macro OvR AUC.
- Saito & Rehmsmeier: `SCOPE-LIMITED` to binary imbalance.
- Feinstein & Cicchetti: `SCOPE-LIMITED` to its rater/medical context.
- Efron & Tibshirani: `BACKGROUND ONLY` for general bootstrap, not the exact proposed design.
- Reddi et al.: `SCOPE-LIMITED`; it does not mandate median/IQR.
- sklearn F1 and ROC docs: `IMPLEMENTATION SEMANTICS`.
- Opitz & Burst and Takahashi et al.: `DIRECT SUPPORT` for the current arithmetic-mean Macro-F1 formula.

## 10. Verification disposition

The hierarchy of metrics was not changed based on outcomes. Macro-F1 remains the candidate primary endpoint only under an explicit domain assumption of equal endpoint-level class importance. Bootstrap inference, Hard Voting probability reporting, ROC references, and latency summaries are now bounded to the exact estimands and source support available.

No scientific result was generated.
