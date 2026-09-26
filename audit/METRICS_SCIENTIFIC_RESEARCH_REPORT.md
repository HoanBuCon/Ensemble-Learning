> **FREEZE NOTICE:**
> Where recommendations or definitions differ, the authoritative frozen methodology is:
> 1. `METRICS_VERIFICATION_CORRECTIONS.md`
> 2. `METRICS_PROPOSED_EVALUATION_PLAN.md`
> 3. `METRICS_STATISTICAL_ANALYSIS_PLAN.md`
> 4. `METRICS_FREEZE_MANIFEST.md`
>
> Historical wording in this report must not override the frozen plan.

# Scientific Evaluation Metrics Research Report

## 1. Scope, snapshot, and evidence boundary

This is a source-of-truth audit of the current executable repository, not a review of historical manuscript claims.

```text
Branch: fix/scientific-pipeline-v2
HEAD: d9cdaa8aa79a35f5d5e3b71f5f3b141d96cee9ab
Tree SHA: c14a86133cfa4020f7dc7ed51f0303d50e2a929e
Audit date: 2026-09-27
```

The working tree was dirty before this audit because of one unstaged documentation deletion and untracked re-audit files. The Step 4B readiness report records all ten readiness questions and all four P0 gates as PASS, but that report was not treated as runtime proof. Current prediction schema, protocol isolation, metric functions, ensemble semantics, and verification paths were independently traced in source.

No current `RESULTS/` directory exists. Therefore this report proposes an evaluation design but makes no numerical claim about a final model. No CNN training, fold training, ensemble fitting, final evaluation, latency benchmark, dataset mutation, source modification, or manuscript modification was performed.

## 2. What the current experiment can answer

The code supports six bounded questions:

1. descriptive comparison of four backbones on a common test set;
2. descriptive comparison of six ensemble methods with the bases;
3. paired same-method comparison of Single-Split and OOF protocols;
4. comparison of uncalibrated predictive probability quality;
5. descriptive diversity among the four base learners;
6. hardware-conditional inference cost after final artifacts exist.

It does not support external-site generalization, biological/species validation, clinical safety, causal explanations, state-of-the-art claims, protocol equivalence, or seed robustness. The exact research-question matrix is in `METRICS_PROPOSED_EVALUATION_PLAN.md`.

## 3. Current prediction semantics

All ten methods yield `y_pred` and a six-column normalized vector on the same canonical sample IDs and class order. That common shape does not make the vectors semantically identical.

- Base models: softmax of logits; predicted label is argmax.
- Hard Voting: predicted label is plurality/mode of four base labels; probability vector is the fraction of the four votes for each class. It is discrete in increments of 0.25 and ties resolve to the lowest class index.
- Soft Voting: unweighted arithmetic mean of the four base softmax vectors.
- Weighted Voting: convex probability mixture, fitted by SLSQP multiclass log-loss on validation (Single) or OOF-train (OOF), never on test.
- Stacking LR/RF/XGB: a fitted meta-model receives 24 concatenated base probabilities and emits `predict_proba`.

Single and OOF use the same test sample IDs. OOF meta-training rows are one outer-held-out prediction per train sample and backbone; OOF test base features are the mean of five fold-checkpoint probabilities. This asymmetry is an implemented protocol characteristic and is not, by itself, evidence of performance harm.

## 4. Current metric implementation

The full implementation table, formulas, inputs, averaging, output locations, and current source line ranges are in `METRICS_SOURCE_OF_TRUTH_INVENTORY.md`. In summary, current code calculates:

- label performance: Accuracy, macro Precision/Recall/F1, weighted F1, per-class P/R/F1, confusion matrix, multiclass MCC, Cohen's kappa;
- discrimination: macro OvR ROC-AUC, per-class/micro/macro ROC plots, per-class and micro PR/AP plots;
- probability quality/calibration: top-label ECE-15, class-summed multiclass Brier, NLL;
- paired statistics: exact McNemar, Edwards statistic, paired accuracy difference and normal CI, marginal-accuracy Cohen's h, supplementary post-hoc power;
- base diversity: disagreement, Yule's Q, pairwise prediction kappa, global disagreement, probability ambiguity;
- representation: t-SNE plus silhouette, Davies–Bouldin, and Calinski–Harabasz in the original feature space;
- efficiency: real saved-pipeline latency, derived FPS, actual parameter count, and actual artifact storage.

Important exact semantics:

- F1/precision/recall default to macro and set zero divisions to zero.
- AUC is macro OvR over one-hot truth and probability columns.
- ECE uses 15 equal-width confidence bins and top-label correctness.
- Brier is `mean(sum_c((p-y_onehot)^2))`, range 0–2.
- reporting NLL clips the true-class probability at `1e-15`; weighted-vote optimization clips at `1e-12`.
- ROC-AUC failures are not converted to zero; the current path raises.
- clustering indices are calculated on original 6-D or 24-D features, not 2-D t-SNE coordinates.

## 5. Class imbalance and endpoint selection

The frozen split is moderately imbalanced. Test counts range from 193 to 345 per class, a ratio of 1.788. Accuracy is therefore meaningful but reflects observed prevalence: an error in a 345-sample class contributes more to accuracy than an error in a 193-sample class.

### Proposed PRIMARY: Macro-F1

Macro-F1 is the minimum defensible primary endpoint because it gives every named class equal weight and combines false-positive and false-negative performance. This follows the actual six-class scientific design rather than a generic “journal-standard” rule. Classification metrics answer different questions, so metric choice must follow the target estimand ([Sokolova & Lapalme, 2009](https://doi.org/10.1016/j.ipm.2009.03.002)).

Macro-F1 is valid for all ten methods and does not require treating Hard Voting votes as posterior probabilities. Its limitation is equally explicit: it is a nonlinear class-average and is not the fraction of correct samples.

### Main secondary metrics

- **Accuracy:** interpretable sample correctness and the target of paired McNemar comparisons.
- **Multiclass MCC:** a complementary full-confusion-matrix correlation measure ([Gorodkin, 2004](https://doi.org/10.1016/j.compbiolchem.2004.09.006)).
- **NLL:** the main probability-quality loss. The logarithmic score is strictly proper for probabilistic forecasts ([Gneiting & Raftery, 2007](https://doi.org/10.1198/016214506000001437)).

### Supporting metrics

Brier, ECE-15, confusion matrices, and per-class P/R/F1 explain why headline metrics differ. The implemented multiclass Brier is the original class-summed convention and is a proper quadratic score ([Brier, 1950](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2)).

Balanced Accuracy should not be added as another number: in this single-label multiclass design it is identical to the already implemented macro recall. Weighted F1 is supplementary because its prevalence weighting overlaps conceptually with the sample weighting behind accuracy.

## 6. Probability quality and calibration

NLL and Brier are proper scoring-rule losses. ECE is a binned empirical diagnostic, not a proper scoring rule, and different calibration notions need not agree in multiclass settings ([Vaicenavicius et al., 2019](https://proceedings.mlr.press/v89/vaicenavicius19a.html)).

Minimum calibration suite:

1. NLL;
2. multiclass Brier;
3. ECE-15 labeled **top-label, equal-width**;
4. a reliability diagram, which is not yet a canonical output.

Classwise ECE is optional rather than mandatory: class supports of 193–345 make a heavily binned classwise analysis noisy, and it does not answer the primary classifier question.

Hard Voting can mathematically enter Brier, ECE, NLL, AUC, and AP because its vote fractions form a simplex. Its interpretation must differ: it measures four-voter consensus, not smooth neural confidence. A zero true-class vote makes NLL an epsilon-defined penalty. Consequently, Hard Voting calibration results should be visibly marked `probability_type=vote_fraction`, not silently pooled as an equivalent posterior.

No Temperature Scaling/Soft+T exists in the current method set and none is proposed.

## 7. Discrimination metrics

Macro OvR ROC-AUC is useful as supplementary threshold-independent ranking information. It does not measure calibration and need not agree with argmax Macro-F1. The code’s exact OvR macro semantics must be retained; multiclass AUC aggregation choices are not interchangeable ([Hand & Till, 2001](https://doi.org/10.1023/A:1010920819831)).

Per-class AUC plots help identify heterogeneous class separability. PR/AP is optional. Precision–recall views can be especially informative under strong imbalance ([Saito & Rehmsmeier, 2015](https://doi.org/10.1371/journal.pone.0118432)), but this dataset’s imbalance is moderate and the current research question does not justify promoting another ranking metric to the main table. If AP is reported, “micro AP,” “per-class AP,” and “macro AP” must not be conflated; current code does not produce a canonical macro AP table.

## 8. Statistical inference

### Paired bootstrap

Report 95% uncertainty for Macro-F1, Accuracy, MCC, NLL, and Brier using 10,000 class-stratified paired bootstrap resamples of test sample IDs with seed 42. The same sampled indices are reused for every method in a comparison. This preserves the paired test-row structure and supports nonlinear endpoints. Bootstrap confidence intervals are a general resampling approach for statistical accuracy and complex estimators ([Efron & Tibshirani, 1986](https://doi.org/10.1214/ss/1177013815)).

These intervals condition on fitted models and the observed class composition. They do not quantify training randomness.

### McNemar

The exact McNemar test operates on paired correctness discordance, not on the magnitude of probability errors ([McNemar, 1947](https://doi.org/10.1007/BF02295996)). It can test unequal marginal error/accuracy rates. It cannot test Macro-F1, AUC, NLL, Brier, ECE, or equivalence.

Keep one confirmatory family: the six same-method Single-vs-OOF ensemble accuracy comparisons. Retain current Bonferroni `0.05/6`. Raw p-values must be used for decisions; display rounding occurs later. `p > alpha` means `FAIL_TO_REJECT`, never equivalent.

No omnibus web of pairwise p-values is proposed. Absolute method tables are estimation/descriptive. Test-selected champion comparisons remain explicitly exploratory. This prevents multiplicity from being disguised across metrics and methods.

The current paired risk-difference CI receives adjusted alpha and is therefore 99.1667%, not 95%; final output must label that exact level. A descriptive paired bootstrap 95% ΔAccuracy interval may also be presented.

## 9. Diversity analysis

Disagreement, Yule's Q, pairwise kappa, global disagreement, and probability ambiguity describe relationships among the four base learners within a protocol. They are not predictive-performance endpoints. Diversity measures can have nontrivial and nonmonotonic relationships to ensemble accuracy ([Kuncheva & Whitaker, 2003](https://doi.org/10.1023/A:1022859003006)).

Pairwise kappa is additionally sensitive to marginal label frequencies; a chance-corrected value should not be over-interpreted as a universal agreement scale ([Feinstein & Cicchetti, 1990](https://doi.org/10.1016/0895-4356(90)90158-L)). Current “probability ambiguity” is a project-specific squared dispersion and must be reported with its formula.

These metrics cannot establish that diversity caused an ensemble gain, and they should never rank final classifiers.

## 10. t-SNE and representation analysis

t-SNE preserves selected local neighborhood relationships for visualization, not global geometry or classifier performance. Apparent cluster size/distance can change with settings ([van der Maaten & Hinton, 2008](https://www.jmlr.org/papers/v9/vandermaaten08a.html); [Wattenberg et al., 2016](https://doi.org/10.23915/distill.00002)).

Recommended roles:

- t-SNE plots: **EXPLORATORY**;
- silhouette/DB/CH on original features: **EXPLORATORY**, at most supplementary;
- none of these may select or rank the final classifier.

Current code correctly records feature source/dimension and computes cluster indices before 2-D projection. Nevertheless, a 6-D Swin probability space and a 24-D concatenated ensemble space differ in dimension and construction, so absolute cluster-index differences are not direct evidence of a superior classifier.

## 11. Computational efficiency

Current benchmark source is capable of executing final saved pipelines for all four bases and all six ensembles. It uses batch size 1, five repeats, GPU 50/200 warmup/timed iterations, CPU 20/100, CUDA synchronization/events, actual parameters, and actual file sizes.

Final reporting should give:

- median [IQR] across the five repeat means as the robust headline;
- mean ± SD as a secondary familiar summary;
- all protocol/environment settings;
- FPS explicitly labeled as reciprocal batch-1 latency, not independently measured saturated throughput.

Benchmark numbers are hardware- and implementation-conditional; standardized workload and environment disclosure are necessary ([Reddi et al., 2020](https://doi.org/10.1109/ISCA45697.2020.00045)). No benchmark was run in this task.

## 12. Training monitoring versus test evaluation

Current checkpoint selection maximizes validation accuracy with a strict-improvement rule and a 5% validation-loss gate. The proposed primary test endpoint is Macro-F1.

It is not mandatory that selection and final reporting use the same metric. It is mandatory that:

- the selection rule is fixed without test input;
- the mismatch is disclosed;
- the selected checkpoint is not described as validation-optimal for Macro-F1;
- the test primary endpoint is pre-specified before inspecting final results.

No checkpoint monitor change is proposed in this audit.

## 13. Training-seed variability

Current configuration uses seed 42. Five folds are dependent cross-fitting components, not five independent training replications. A fixed-seed final experiment plus sample-level CIs is the minimum defensible design if the claim is explicitly conditional on that pipeline realization.

A stronger optional design repeats the complete 24-CNN-run experiment for three or five pre-specified seeds. This addresses algorithmic randomness but costs 72 total runs for three seeds or 120 for five. It is not mandated here. Variance from dataset sampling and training randomness should not be conflated ([Bouthillier et al., 2021](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf)).

## 14. Minimum versus extended suite

### A. Minimum Final Paper Set

- Macro-F1 (PRIMARY), Accuracy, MCC;
- NLL and multiclass Brier;
- per-class P/R/F1/support and confusion matrix;
- ECE-15 plus reliability diagram;
- paired bootstrap 95% CIs and planned Δ intervals;
- exact McNemar only for the six pre-defined protocol/accuracy contrasts;
- CPU/GPU latency, parameters, and storage after final artifacts exist.

### B. Optional Extended/Supplementary Set

- macro precision/recall, weighted F1, kappa;
- macro/per-class ROC-AUC and optional AP/PR curves;
- diversity suite;
- t-SNE and original-feature cluster diagnostics;
- multiple full training seeds.

## 15. Direct answers to the 12 required questions

1. **Current code tính metrics nào?** Label metrics, AUC/AP plots, ECE/Brier/NLL, McNemar/RD/effect/power, diversity, cluster diagnostics, and computational metrics listed in Section 4 and the inventory.
2. **Semantics thực tế?** Exact formulas, averaging, clipping, input population, and failure behavior are recorded in `METRICS_SOURCE_OF_TRUTH_INVENTORY.md` and `METRICS_STATISTICAL_ANALYSIS_PLAN.md`.
3. **Metric phù hợp classification performance?** Macro-F1 primary; Accuracy and multiclass MCC main secondary; per-class metrics/confusion supporting.
4. **Metric phù hợp probability quality/calibration?** NLL and Brier for proper-score quality; ECE-15 and reliability diagram as diagnostics.
5. **Metric chỉ đo diversity?** Disagreement, Yule's Q, pairwise prediction kappa, global disagreement, and probability ambiguity.
6. **Metric chỉ exploratory?** t-SNE, original-feature cluster indices, probability ambiguity, and post-hoc champion panels.
7. **Primary metric?** Macro-F1, because each of six scientifically named classes receives equal weight under moderate class imbalance.
8. **Inference tối thiểu?** Paired class-stratified bootstrap CIs plus exact McNemar for the six pre-specified accuracy contrasts; no broad p-value grid.
9. **McNemar kết luận gì?** Evidence about unequal paired marginal error rates/accuracy only; not Macro-F1/AUC/NLL/calibration/equivalence.
10. **Có cần CI/bootstrap?** Yes. Point estimates alone do not express test-sample uncertainty; use the fully specified paired bootstrap while disclosing it excludes training-seed uncertainty.
11. **Metric nào nên bỏ khỏi main paper?** Weighted F1, kappa, ROC/PR, diversity, t-SNE/cluster metrics, marginal Cohen's h, and post-hoc power should not be headline endpoints; they may be supplementary/exploratory. Do not duplicate macro recall with Balanced Accuracy.
12. **Cần bổ sung gì trước FINAL run?** No new metric is required to start training. Before final reporting, add the pre-specified bootstrap analysis and a canonical reliability diagram; improve latency distribution summaries and Hard Voting probability labeling. These are reporting/analysis gaps, not reasons to alter training.

## 16. Source-of-truth assessment

The current pipeline is capable of producing aligned, provenance-linked label and probability outputs for all ten methods under both protocols. Its present metric coverage is already broad. The defensible scientific improvement is therefore not to maximize metric count, but to pre-specify a small endpoint hierarchy, attach paired uncertainty to estimates, limit hypothesis testing to a coherent family, and enforce semantic labels for probability, diversity, representation, and computational outputs.

No current numerical winner, improvement magnitude, calibration advantage, or latency claim is established because final artifacts do not exist.
