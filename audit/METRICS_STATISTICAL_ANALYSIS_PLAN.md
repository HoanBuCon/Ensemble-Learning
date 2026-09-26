# Statistical Analysis Plan

## 1. Evaluation sample, estimand, and inference boundary

The frozen `data/test` set identified by manifest SHA-256 `a94426015d83a3b8e9f66a30bddee9e4b376eaf509f166aa522d7a204fe22dc5` is the **observed evaluation sample**, not a documented probability sample from a defined agricultural target population. Its point estimates describe those 1,546 images exactly. If those rows are treated as a completely fixed finite set, there is no sampling uncertainty to estimate.

Bootstrap intervals require an additional working model: the observed rows are treated as an empirical proxy for repeat draws from a hypothetical distribution represented by this test set. The repository contains no sampling frame, location-level design, specimen clustering identity, or population weights; consequently, these intervals must not be described as design-based inference to all tea leaves, farms, devices, or future deployments.

The resampling unit is one image/sample ID. All within-protocol method comparisons and all same-method Single-Split versus OOF comparisons use the same ordered test IDs and are therefore paired.

Inference is conditional on:

- the empirical distribution represented by the frozen test sample, under the endpoint-specific resampling model below;
- the fitted checkpoints and ensemble artifacts from one configured training seed;
- the current preprocessing and decision rules.

Sample-level intervals do not measure training-seed variability, external-site variation, plant/specimen clustering, or annotation uncertainty.

## 2. Pre-specified endpoints

1. **Proposed primary endpoint:** arithmetic mean of classwise F1, conditional on a pre-specified choice of equal endpoint-level class importance.
2. **Main secondary label endpoints:** Accuracy and multiclass MCC.
3. **Main secondary probability endpoint:** NLL for continuous softmax/mixed/meta-model probability outputs.
4. **Supporting probability/calibration endpoints:** multiclass Brier and top-label ECE-15 for continuous probability outputs; Hard Voting is reported separately as vote-confidence diagnostics.
5. **Supporting class diagnostics:** confusion matrix and per-class precision/recall/F1.

Only Accuracy has a current exact paired hypothesis test. Other endpoints use estimates and paired confidence intervals, not an improvised collection of p-values.

## 3. Bootstrap uncertainty specification

One class-stratified design is **not** used for all five endpoints because it fixes prevalence by construction and therefore targets a different estimand from an ordinary sample bootstrap.

### 3.1 Conditional, fixed-composition endpoint

Use a paired class-stratified nonparametric bootstrap for the proposed Macro-F1 endpoint:

```text
estimand: performance conditional on the six observed class counts/composition
resampling unit: test sample_id within its y_true class
draw size: the original count in every class, sampled with replacement
pairing: identical sampled IDs/indices for every compared method
resamples: 10,000
random seed: 42
interval: two-sided 95% percentile interval
```

This design varies examples within each true-class stratum while fixing all six class counts. It does not include uncertainty in class prevalence. Macro-F1 gives equal arithmetic weight to classwise F1 values, but each classwise precision still depends on false positives arriving from the other classes; its value is therefore not wholly prevalence-invariant. The interval must be labeled **conditional on the observed class composition**.

### 3.2 Empirical joint-distribution endpoints

Use a paired ordinary nonparametric bootstrap over all 1,546 sample IDs for Accuracy, MCC, NLL, and Brier:

```text
estimand: performance under the empirical joint distribution of (y, prediction)
resampling unit: test sample_id across the full test set
draw size: N=1,546 sampled with replacement
class counts: allowed to vary around the observed empirical proportions
pairing: identical sampled IDs/indices for every compared method
resamples: 10,000
random seed: 42
interval: two-sided 95% percentile interval
```

Consequences:

- Accuracy, NLL, and Brier are sample/prevalence-weighted means; ordinary resampling includes empirical class-mix variation that fixed-count stratification would remove.
- MCC depends on the full confusion matrix and both true/predicted marginals; ordinary resampling retains their empirical joint variation.
- A stratified version of these four metrics would answer a narrower conditional question with fixed class counts. It may be shown only as a sensitivity analysis, not silently substituted for the main interval.

Pairing preserves within-image covariance between methods. Ten thousand resamples, seed 42, and percentile intervals are transparent protocol choices, not universal literature mandates. Efron & Tibshirani (1986) and Davison & Hinkley (1997) support nonparametric bootstrap reasoning and the need to match resampling to the data structure; Takahashi et al. (2022) directly treats uncertainty for the arithmetic-mean classwise Macro-F1 definition but does not mandate this exact bootstrap recipe.

For each method, bootstrap the absolute metric under its declared resampling design. For each planned method comparison, calculate the metric for both methods inside each same bootstrap replicate and save `Δ = metric_A − metric_B` (reverse the sign for losses if the table is intended to mean “A improvement”; the sign convention must be declared). The interval for Δ comes directly from the paired Δ distribution.

Do not bootstrap rounded metrics. Do not bootstrap aggregate confusion matrices without resampling their underlying sample rows.

## 4. Confirmatory McNemar family

The only current confirmatory hypothesis family is the six same-method protocol contrasts:

```text
Hard Voting: Single-Split vs OOF
Soft Voting: Single-Split vs OOF
Weighted Voting: Single-Split vs OOF
Stacking LR: Single-Split vs OOF
Stacking RF: Single-Split vs OOF
Stacking XGB: Single-Split vs OOF
```

For each method, McNemar tests equality of marginal error probabilities using only discordant paired correctness:

```text
n10 = Single correct, OOF wrong
n01 = Single wrong, OOF correct
H0: P(n10 event) = P(n01 event), conditional on discordance
exact p = two-sided Binomial(n10 | n10+n01, 0.5)
```

Family-wise alpha is 0.05. Retain the current Bonferroni threshold `0.05/6`; it is conservative but valid and changing correction is unnecessary for this small, explicitly defined family. Report raw p-values and the correction rule. Use full-precision p-values internally and round only in display.

Current code also supplies a Bonferroni-level paired accuracy-difference interval because it passes adjusted alpha into the normal CI. Its confidence level is `1 − 0.05/6 = 99.1667%`, not 95%. Label it accordingly. The proposed paired bootstrap can additionally provide a descriptive 95% ΔAccuracy interval.

### What McNemar can conclude

- Evidence to reject or fail to reject equal marginal error rates for the two paired classifiers.
- Direction and size can be described by paired accuracy difference and its interval.

### What McNemar cannot conclude

- equality or difference of Macro-F1, MCC, ROC-AUC, NLL, Brier, or ECE;
- model equivalence or non-inferiority;
- external generalization;
- independence of predictions;
- why a protocol performed differently.

`p > alpha` must remain `FAIL_TO_REJECT`, never “equivalent.”

## 5. Other comparisons and multiplicity

| Comparison family | Status | Multiplicity handling |
|---|---|---|
| Six same-method Single-vs-OOF accuracy contrasts | confirmatory | Bonferroni FWER, alpha 0.05/6 |
| Absolute performance of all methods | estimation/descriptive | point estimates + 95% CIs; no p-values |
| Macro-F1/MCC/NLL/Brier Single-vs-OOF differences | estimation | paired 95% Δ intervals using each metric's declared stratified/ordinary design; explicitly secondary, no binary significance labels |
| Ensemble versus four bases | descriptive unless pre-specified before test inspection | paired Δ and CI; no test-driven champion hypothesis |
| All pairwise method comparisons | not planned | do not generate |
| Post-hoc “best/champion” comparison chosen from test | exploratory | descriptive only; selection must be disclosed |
| Calibration/diversity/t-SNE panels | supporting/exploratory | no inferential multiplicity claims |

No FDR procedure is needed because the plan does not turn the large descriptive panel into a discovery-testing family. Holm is a defensible alternative to Bonferroni, but this proposal does not silently change the already implemented method.

## 6. Comparison matrix

| Research comparison | Metric | Estimate | Uncertainty | Hypothesis test | Multiple-testing family | Interpretation |
|---|---|---|---|---|---|---|
| Same method, Single vs OOF | Macro-F1 | absolute values and paired Δ | paired class-stratified percentile 95% CI | none | none; secondary estimation | conditional on fixed observed class composition |
| Same method, Single vs OOF | Accuracy | absolute values and paired RD | paired ordinary bootstrap 95% CI; code’s simultaneous RD CI labeled 99.1667% | exact McNemar | six protocol comparisons, Bonferroni | unequal marginal error rates if rejected; no equivalence claim otherwise |
| Same method, Single vs OOF | MCC | absolute values and paired Δ | paired ordinary percentile 95% CI | none | none | empirical joint-distribution association difference |
| Same method, Single vs OOF | NLL/Brier for continuous probability methods | absolute losses and paired Δ | paired ordinary percentile 95% CI | none | none | empirical prevalence-weighted probability-quality difference; lower is better |
| Ensemble vs base, same protocol | Macro-F1/Accuracy/MCC | absolute and paired Δ | Macro-F1 stratified; Accuracy/MCC ordinary paired bootstrap | none under current design | descriptive | quantifies observed gain/loss; not a pre-specified superiority test |
| Continuous probability methods, same protocol | ECE-15 | absolute value | optional bootstrap interval, supplementary | none | none | bin-dependent top-label calibration diagnostic; Hard Voting reported separately by exact vote-confidence level |
| Base pairs within protocol | disagreement/Q/kappa | pairwise values | none required in minimum set | none | none | descriptive diversity, not performance |
| Inference cost | latency | five raw repeat means | mean ± SD headline; optional median [IQR] as descriptive sensitivity | none | none | hardware-conditional cost; no claim of MLPerf compliance |

## 7. Exact metric definitions for manuscript

Let `N` be test samples, `C=6`, `y_i` true class, `ŷ_i` predicted class, and `p_ic` probability assigned to class `c`.

### Macro-F1 — proposed PRIMARY, conditional on equal endpoint-level class importance

For each class `c`, treating it one-vs-rest:

```text
Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 Precision_c Recall_c / (Precision_c + Recall_c)
Macro-F1    = (1/C) Σ_c F1_c
```

Undefined zero divisions are assigned zero, matching sklearn. Range `[0,1]`; higher is better. Implementation: `sklearn.metrics.f1_score(..., average='macro', zero_division=0)`. This is the arithmetic mean of classwise F1 (Opitz & Burst's “averaged F1”), not the harmonic mean of macro precision and macro recall associated with the alternative terminology in Sokolova & Lapalme (2009).

### Accuracy

`Accuracy = (1/N) Σ_i I(ŷ_i = y_i)`. Range `[0,1]`; higher is better. Implementation: sklearn `accuracy_score`.

### Multiclass MCC

Use sklearn’s generalized multiclass Matthews correlation coefficient computed from the multiclass confusion matrix. Range `[-1,1]`; higher is better, with zero representing no better than the correlation baseline. Implementation: `sklearn.metrics.matthews_corrcoef`.

### NLL

`NLL = -(1/N) Σ_i log(clip(p_i,y_i, 10^-15, 1))`. Range `[0,∞)` under clipping; lower is better. This is the reporting definition. Weighted-vote optimization separately uses epsilon `10^-12` and must not be described as numerically identical at zero probability. Hard Voting is excluded from the continuous-probability ranking: zero true-class votes produce an unclipped infinite loss and a clipped per-sample penalty of `-log(10^-15) ≈ 34.5388`.

### Multiclass Brier score

`Brier = (1/N) Σ_i Σ_c (p_ic − I[y_i=c])²`. Range `[0,2]` for the implemented unnormalized multiclass convention; lower is better.

### ECE-15 top-label

Define confidence `q_i=max_c p_ic`, correctness `a_i=I(argmax_c p_ic=y_i)`, and 15 equal-width bins over `[0,1]`. The first bin includes both endpoints; later bins exclude their lower and include their upper endpoint. Then:

`ECE = Σ_b (|B_b|/N) |mean(a_i in B_b) − mean(q_i in B_b)|`.

Range `[0,1]`; lower is better. It is a diagnostic, not a proper score. For Hard Voting, `q_i` is the largest vote fraction and can only be 0.25, 0.50, 0.75, or 1.00; the vote-confidence diagnostic therefore reports exact-level counts/accuracy separately rather than ranking it with continuous outputs.

### Macro ROC-AUC OvR

For each class, calculate the ROC-AUC of `p_ic` against binary truth `I[y_i=c]`, then take the unweighted mean across six classes. Range `[0,1]`; higher is better. Implementation: sklearn `roc_auc_score(one_hot, p, average='macro', multi_class='ovr')`.

### Weighted F1

`Weighted-F1 = Σ_c (n_c/N) F1_c`. Range `[0,1]`; higher is better. It is supplementary because it weights by observed prevalence.

### Macro recall / balanced accuracy

`Macro recall = (1/C) Σ_c TP_c/(TP_c+FN_c)`. In this task it is identical to multiclass balanced accuracy. Range `[0,1]`; higher is better.

### Pairwise diversity

- Disagreement: `(1/N)Σ_i I(ŷ_i^A != ŷ_i^B)`.
- Yule’s Q: `(n11*n00−n10*n01)/(n11*n00+n10*n01)` from the two models’ correctness states; the current code returns 1 when the denominator is zero.
- Pairwise kappa: sklearn Cohen kappa between model A and model B class labels, not between prediction and truth.
- Probability ambiguity: mean squared Euclidean dispersion of each base probability vector around the four-model mean probability vector.

These are descriptive; no direction universally means “better classifier.”

## 8. Rounding and missingness

- Preserve full precision through computation, tests, differences, and corrections.
- Round only final display tables.
- Undefined or unavailable metrics must be `N/A`/missing with a reason, never substituted with zero or another metric.
- If a bootstrap replicate makes a metric undefined, record and report the replicate failure rate; do not silently coerce it to zero.

## 9. One-seed limitation

The primary and secondary intervals above quantify sample uncertainty conditional on fitted artifacts. They do not replace repeated training. The minimum paper must state that only seed 42 was run if that remains true. A stronger optional analysis repeats the entire 24-run protocol across pre-specified seeds and reports across-seed distributions without treating OOF folds as independent repetitions.
