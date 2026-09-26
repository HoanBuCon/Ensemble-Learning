# Proposed Evaluation Plan

## 1. Decision principles

This proposal evaluates the design the repository actually implements: six-class, single-label classification; four base backbones; six ensembles; two protocols evaluated on the same frozen test rows. It does not add Temperature Scaling, new models, new data, or new training protocols.

Metric roles are deliberately narrow:

- **PRIMARY**: one pre-specified endpoint used for the main classifier-performance conclusion.
- **MAIN SECONDARY**: a small set answering distinct, important questions.
- **SUPPORTING**: interpretation of the main endpoints.
- **SUPPLEMENTARY**: useful but not necessary for the main claim.
- **EXPLORATORY**: hypothesis-generating only.
- **NOT RECOMMENDED**: redundant or liable to unsupported interpretation in the main paper.

## 2. Research questions supported by the current design

| Research question | Data/artifact available after final run | Paired? | Probability needed? | Test split involved? | Intended inference |
|---|---|---:|---:|---:|---|
| Q1. How do the four base backbones perform within each protocol? | canonical per-sample base artifacts | yes, within a protocol | only for probability metrics | yes | descriptive estimates and uncertainty; not architecture causality |
| Q2. Do ensemble predictions improve on base predictions? | canonical base and ensemble artifacts | yes | depends on endpoint | yes | descriptive paired differences unless a comparison is pre-specified before test inspection |
| Q3. How does each of the six ensemble methods differ between Single-Split and OOF? | same-method artifacts from both protocols on identical test IDs | yes | depends on endpoint | yes | confirmatory accuracy family via McNemar; estimation for other endpoints |
| Q4. How do probability quality and calibration diagnostics differ? | canonical probability vectors for all ten methods/protocol | yes | yes | yes | proper-score estimates plus calibration diagnostics |
| Q5. How diverse are the four base learners within each protocol? | aligned base labels/probabilities | yes | ambiguity needs probabilities | yes | descriptive diversity only; no causal claim about ensemble gain |
| Q6. What is the inference-cost trade-off? | benchmark source supports saved pipelines | repeated timings, not test-row paired | no | no | hardware-conditional descriptive performance |
| Q7. Does the method generalize to new locations/devices/populations? | no external dataset/site metadata | no | no | no | **NOT SUPPORTED BY CURRENT DESIGN** |
| Q8. Is Single-Split equivalent to OOF? | no equivalence margin/design | paired rows exist but equivalence design absent | no | yes | **NOT SUPPORTED BY CURRENT DESIGN** |
| Q9. Are results stable across training seeds? | one configured seed; folds are not seeds | no | no | yes | **NOT SUPPORTED BY CURRENT DESIGN** |

## 3. Minimum Final Paper Set

### 3.1 Candidate primary endpoint

**The arithmetic mean of per-class F1 is the proposed PRIMARY classifier-performance endpoint, conditional on a pre-specified scientific choice that all six classes receive equal endpoint-level importance.** In the remainder of this plan, `Macro-F1` means exactly `sklearn.metrics.f1_score(..., average="macro", zero_division=0)`:

`Macro-F1 = (1/C) Σ_c F1_c`.

It does **not** mean the competing formula `2·MacroPrecision·MacroRecall/(MacroPrecision+MacroRecall)`. The two definitions can differ and can rank classifiers differently (Opitz & Burst, 2019). The sklearn documentation directly defines `average="macro"` as the unweighted mean of label-wise F1 scores.

Reasoning:

1. If domain investigators pre-specify equal endpoint-level importance for all six classes, Macro-F1 operationalizes that choice by assigning equal arithmetic weight to the six classwise F1 values. Class names alone do not establish equal agricultural, economic, or clinical error costs.
2. It complements rather than merely reproduces sample-weighted accuracy.
3. It is valid for all ten methods because each produces a single hard label per sample.
4. It does not depend on treating Hard Voting’s vote fractions as continuous posterior probabilities.

Boundary: this is a methodological proposal matched to an explicit equal-class endpoint, not a literature-mandated requirement. If the six classes have unequal decision costs, the primary endpoint or a cost-sensitive analysis must be specified by domain knowledge before inspecting test results. Macro-F1 is a nonlinear average and does not have a “percentage correctly classified” interpretation; Accuracy and per-class results therefore remain visible.

### 3.2 Main secondary endpoints

1. **Accuracy** — direct sample-level correctness; all methods; allows exact paired McNemar analysis.
2. **Multiclass MCC** — a single confusion-matrix summary that reflects all cells and remains informative under unequal class frequencies.
3. **NLL** — primary probability-quality loss for continuous softmax/mixed/meta-model probability outputs, with explicit epsilon. Hard Voting is excluded from this ranking and reported separately.

NLL is not allowed to replace Macro-F1 as the classifier-label endpoint. It answers a different question: quality of probability mass assigned to the observed class.

### 3.3 Supporting diagnostics

- raw and row-normalized confusion matrices;
- per-class precision, recall, F1, and support;
- multiclass Brier score;
- top-label ECE-15 plus reliability diagram when implemented;
- paired differences and 95% confidence intervals for planned comparisons.

### 3.4 Minimum main-paper table

For base softmax, Soft Voting, Weighted Voting, and the three stacking methods × protocol:

```text
Macro-F1 [95% CI]
Accuracy [95% CI]
MCC [95% CI]
NLL [95% CI]
Brier [95% CI]
```

Hard Voting remains in the same label-performance table for Macro-F1, Accuracy, and MCC. Its NLL/Brier/ECE must be placed in a **separate vote-confidence diagnostic section**, not ranked in the same probability-quality table as continuous softmax/mixed/meta-model outputs. ECE is better accompanied by a reliability table/figure because a single ECE value conceals confidence-level behavior.

## 4. Optional Extended/Supplementary Set

- Macro precision and macro recall (the latter is balanced accuracy).
- Weighted F1, primarily to show prevalence-weighted behavior; not a headline metric.
- Cohen's kappa versus truth.
- Macro OvR ROC-AUC and per-class OvR AUC.
- Per-class average precision/PR curves; macro AP only if explicitly implemented and defined.
- ECE sensitivity to bin count and/or classwise calibration, if a calibration-focused analysis is intended.
- Diversity: pairwise disagreement, Yule's Q, pairwise kappa, global disagreement, probability ambiguity.
- t-SNE plots and original-feature cluster indices.
- computational metrics: GPU/CPU latency, derived FPS, parameter count, artifact storage.

## 5. Final metric architecture

| Category | Metric | Current implementation | Proposed role | Keep/Add/Remove/Modify | Scientific reason | Citation |
|---|---|---|---|---|---|---|
| Label performance | Macro-F1 = arithmetic mean of classwise F1 | yes, sklearn macro, zero division 0 | PRIMARY, conditional | Keep and define explicitly | implements equal classwise endpoint weighting only if that scientific assumption is pre-specified | sklearn `f1_score`; Opitz & Burst (2019); Takahashi et al. (2022) |
| Label performance | Accuracy | yes | MAIN SECONDARY | Keep | transparent fraction correct and McNemar-compatible | McNemar (1947); Dietterich (1998) |
| Label performance | Multiclass MCC | yes | MAIN SECONDARY | Keep | uses full multiclass contingency structure | Gorodkin (2004) |
| Probability quality | NLL | yes, clip `1e-15` | MAIN SECONDARY for continuous probability methods | Keep; separate Hard Voting | strictly proper log-score loss for issued probability vectors; clipped Hard-Vote zeros create an arbitrary finite surrogate for infinite log loss | Gneiting & Raftery (2007); Ferro (2014, scope-limited) |
| Probability quality | Multiclass Brier | yes, class-summed 0–2 convention | SUPPORTING | Keep | proper quadratic score, less singular than NLL | Brier (1950); Gneiting & Raftery (2007) |
| Calibration | ECE-15 top-label | yes | SUPPORTING | Keep but never alone | interpretable diagnostic but bin-dependent and not a proper score | Vaicenavicius et al. (2019); Naeini et al. (2015) |
| Calibration | Reliability diagram | not currently canonical | SUPPORTING | Add analysis before final reporting | reveals direction and location hidden by scalar ECE | Vaicenavicius et al. (2019) |
| Class detail | Per-class P/R/F1/support | yes | SUPPORTING | Keep | exposes class-specific failure hidden by aggregate summaries | sklearn metric semantics; Sokolova & Lapalme (background only) |
| Class detail | Confusion matrix | yes | SUPPORTING | Keep | exact error pattern; normalized plot handles support differences | current implementation; Sokolova & Lapalme (background only) |
| Label performance | Macro precision | yes | SUPPORTING | Keep | separates false-positive behavior; not a ranking endpoint | sklearn metric semantics |
| Label performance | Macro recall / balanced accuracy | yes as Macro_recall | SUPPORTING | Keep one name only | equal-class recall; adding balanced accuracy duplicates it | sklearn metric semantics |
| Label performance | Weighted F1 | yes | SUPPLEMENTARY | Keep out of headline | prevalence weighted; partially redundant with accuracy in this setting | sklearn metric semantics |
| Agreement | Cohen's kappa vs truth | yes | SUPPLEMENTARY | Keep | chance-adjusted agreement, but sensitive to marginals/prevalence | Feinstein & Cicchetti (1990) |
| Discrimination | Macro ROC-AUC OvR | yes | SUPPLEMENTARY | Keep | unweighted mean of six class-vs-rest AUCs; does not measure calibration or deployed argmax performance | official sklearn `roc_auc_score` documentation |
| Discrimination | Per-class ROC-AUC | plot only | SUPPLEMENTARY | Retain plots; add table only if used in claims | shows class-vs-rest ranking heterogeneity | official sklearn multiclass ROC example |
| Discrimination | Per-class AP/PR | plot only | SUPPLEMENTARY | Optional; do not add solely for metric count | useful minority-class ranking view, but imbalance is moderate and it adds no core endpoint | Saito & Rehmsmeier (2015) |
| Diversity | Disagreement | yes | SUPPLEMENTARY | Keep | error/prediction diversity among bases, not classifier performance | Kuncheva & Whitaker (2003) |
| Diversity | Yule's Q | yes | SUPPLEMENTARY | Keep with zero-denominator rule disclosed | correctness association, not a final ranking metric | Kuncheva & Whitaker (2003) |
| Diversity | Pairwise kappa | yes | SUPPLEMENTARY | Keep with marginal caveat | label agreement between base learners | Feinstein & Cicchetti (1990) |
| Diversity | Probability ambiguity | yes, project-specific dispersion | EXPLORATORY | Keep only with exact formula | no universal calibrated scale; descriptive probability diversity | Kuncheva & Whitaker (2003) |
| Representation | t-SNE visualization | yes | EXPLORATORY | Keep | neighborhood visualization only; sensitive to parameters and interpretation | van der Maaten & Hinton (2008); Wattenberg et al. (2016) |
| Representation | Silhouette/DB/CH | yes on original features | EXPLORATORY | Keep out of classifier ranking | cluster geometry depends on feature source/dimension and does not measure classification generalization | van der Maaten & Hinton (2008) |
| Statistics | Exact McNemar | yes | MAIN SECONDARY inference for six protocol accuracy contrasts | Keep | correct paired binary correctness comparison | McNemar (1947) |
| Statistics | Marginal Cohen's h | yes | SUPPLEMENTARY | Keep only under its current explicit label | marginal effect, not paired effect size | current formula; no stronger interpretation |
| Statistics | Post-hoc power | yes | NOT RECOMMENDED for conclusions | Retain only as supplementary compatibility output | contributes no valid equivalence/significance decision after results | current source already excludes it from decisions |
| Compute | CPU/GPU latency | supported, no final result | SUPPORTING | Keep; report raw repeats and descriptive summaries | deployment cost is hardware/protocol conditional; current benchmark is not MLPerf | Reddi et al. (2020, scope-limited) |
| Compute | FPS/throughput | derived from batch-1 latency | SUPPLEMENTARY | Keep but label derived | not independent saturated-throughput measurement | current formula; Reddi et al. only supplies benchmarking context |
| Compute | parameters/storage | actual loaded/files | SUPPORTING | Keep | capacity and deployable artifact cost | current implementation |
| Redundant | Separate Balanced Accuracy | absent | NOT RECOMMENDED | Do not add alongside Macro_recall | numerically identical to macro recall here | metric identity |
| Absent method | Soft+T | absent | NOT RECOMMENDED in this study | Do not add | outside current pre-specified methods | scope rule |

## 6. Probability and calibration interpretation by method

| Method type | NLL/Brier valid to compute? | ECE valid to compute? | Required interpretation |
|---|---:|---:|---|
| Base softmax | yes | yes | uncalibrated softmax predictive scores |
| Soft/Weighted Voting | yes | yes | convex mixtures of uncalibrated base scores |
| Stacking LR/RF/XGB | yes | yes | meta-model probabilities; no post-hoc calibration was applied |
| Hard Voting | scoreable as an issued four-member empirical distribution, but not included in the continuous-probability ranking | diagnostic only | coarse four-vote fractions; use a separate vote-confidence section |

No method is declared calibrated merely because its ECE is low on one test set. NLL and Brier are proper-score losses when applied to an issued probability distribution; ECE and reliability diagrams diagnose empirical calibration from different angles.

### Hard Voting decision

Use reporting structure **B: a separate vote-confidence diagnostic section**.

- The current vector estimates the empirical distribution of four deterministic votes, so Brier/NLL evaluate that issued four-vote distribution against the observed class. They do not estimate the same smooth conditional probability object as softmax, convex mixtures, or meta-model `predict_proba`.
- If no base learner votes for the true class, the un-clipped logarithmic score is infinite. Current clipping replaces it with `-log(10^-15) ≈ 34.5388` for that sample. Therefore Hard Voting NLL is materially controlled by an arbitrary numerical epsilon and must not participate in the main probability-quality ranking.
- Top-label confidence can attain only `0.25`, `0.50`, `0.75`, or `1.00`. ECE-15 therefore has at most four occupied bins; eleven or more empty bins add no resolution. A reliability display should aggregate by the exact attainable confidence levels and show counts/accuracy at each level, while preserving the current ECE value only as a diagnostic.
- Brier remains finite and can summarize the squared error of the vote distribution, but finite-ensemble scoring has ensemble-size effects. Ferro (2014) is relevant background for finite ensemble forecast scores, not a direct validation of this deterministic classifier ensemble.

## 7. Discrimination metrics decision

Macro OvR ROC-AUC is retained as supplementary because it answers ranking/discrimination rather than argmax classification or calibration. Current semantics are established by official sklearn documentation: calculate each class against the pooled remainder, then take their unweighted mean. Hand & Till (2001) instead averages pairwise class comparisons and is **not** the exact source for the current metric. Per-class AUC is useful to reveal heterogeneity. PR/AP remains optional: the minority/majority ratio is about 1:1.79, so the data are not so extremely imbalanced that PR-AUC must become a main endpoint. If reported, AP must be per-class or have a precisely defined macro aggregation; the current micro AP plot cannot be silently renamed macro PR-AUC.

## 8. Diversity and representation decision

Diversity metrics apply only to the four bases within a protocol. They must not be mixed with Single-vs-OOF agreement and must not rank final classifiers. A relationship between diversity and ensemble improvement cannot be inferred causally from two protocols and one dataset.

t-SNE is exploratory. Current cluster indices are correctly computed in original 6-D or 24-D feature space, not on the 2-D projection, but comparing their absolute values across different feature dimensions/sources remains exploratory. Titles and metadata must always state the actual source and dimension.

## 9. Computational reporting

Minimum defensible report after artifacts exist:

- device-specific batch-1 latency for every base and all six ensembles;
- all five repeat means as raw evidence;
- mean ± SD across the five repeat means as the current headline descriptive summary;
- median [IQR] only as an optional robustness description, explicitly noting that `n=5` repeats makes quartiles coarse;
- derived FPS clearly labeled as `1000/mean latency`, not saturated throughput;
- actual parameter count and artifact storage;
- warmup, timed iterations, repeats, dtype, device, CPU threads, OS, Python, PyTorch, CUDA/cuDNN, and GPU.

Neither Reddi et al. nor MLPerf mandates median [IQR] for this repository’s protocol. MLPerf Single-Stream uses a 90th-percentile query latency under its own load-generation rules; the current code stores five repeat-level means and is not an MLPerf implementation. Mean ± SD and optional median [IQR] are therefore transparent descriptive choices, not literature standards. Raw repeat values, environment, warmup/iteration counts, protocol, and derived-FPS labeling are mandatory for reproducibility.

## 10. Training versus final evaluation

Model selection currently maximizes validation accuracy subject to a validation-loss gate. The proposed primary test endpoint is Macro-F1. They are not mathematically required to be identical: selection and reporting can target different operational summaries, provided the selection rule is fixed before test inspection and disclosed. The consequence is interpretive: these checkpoints were not directly optimized or selected for Macro-F1, so the paper must not claim that they are validation-optimal for Macro-F1.

## 11. Seed design

Current design uses one seed (`42`). The five OOF folds are not five independent random seeds.

- **Minimum defensible design:** present results as fixed-pipeline, fixed-seed estimates; use sample-level paired uncertainty; make no claim of training-seed robustness.
- **Stronger optional design:** repeat the complete experiment for 3 or 5 pre-specified seeds and summarize between-seed variation. One seed entails 4 Single + 20 OOF CNN runs = 24 runs. Three total seeds entail 72 runs (48 additional); five total seeds entail 120 runs (96 additional). This is optional, not silently required by the current research question.

## 12. Claims that the proposed suite does not support

The suite does not establish equivalence, external validity, clinical safety, biological species validity, causal superiority, state of the art, real-time suitability on unspecified hardware, or seed-invariant performance.
