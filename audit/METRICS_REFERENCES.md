# Scientific Metrics References

Audit date: 2026-09-27. These references support the proposal; they are not evidence of current runtime behavior. Current behavior is established only from the executable source cited in the inventory.

## Classification and multiclass performance

1. Sokolova, M., & Lapalme, G. “A systematic analysis of performance measures for classification tasks.” *Information Processing & Management*, 45(4), 427–437, 2009. DOI: [10.1016/j.ipm.2009.03.002](https://doi.org/10.1016/j.ipm.2009.03.002). **Important:** its multiclass table defines macro precision and macro recall first, then defines `Fscore_M` as their harmonic mean. It is not the direct definition source for sklearn's arithmetic mean of classwise F1.
2. Gorodkin, J. “Comparing two K-category assignments by a K-category correlation coefficient.” *Computational Biology and Chemistry*, 28(5–6), 367–374, 2004. DOI: [10.1016/j.compbiolchem.2004.09.006](https://doi.org/10.1016/j.compbiolchem.2004.09.006).
3. Hand, D. J., & Till, R. J. “A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems.” *Machine Learning*, 45, 171–186, 2001. DOI: [10.1023/A:1010920819831](https://doi.org/10.1023/A:1010920819831). Its generalization averages pairwise class comparisons; it is not the exact source for current sklearn macro OvR AUC.
4. Saito, T., & Rehmsmeier, M. “The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.” *PLOS ONE*, 10(3), e0118432, 2015. DOI: [10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432). This paper is binary-focused; it supports caution about PR/ROC under imbalance, not an automatic requirement to add multiclass PR-AUC.

## Probability quality and calibration

5. Gneiting, T., & Raftery, A. E. “Strictly Proper Scoring Rules, Prediction, and Estimation.” *Journal of the American Statistical Association*, 102(477), 359–378, 2007. DOI: [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).
6. Brier, G. W. “Verification of Forecasts Expressed in Terms of Probability.” *Monthly Weather Review*, 78(1), 1–3, 1950. DOI: [10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2).
7. Vaicenavicius, J., Widmann, D., Andersson, C., Lindsten, F., Roll, J., & Schön, T. B. “Evaluating model calibration in classification.” *Proceedings of AISTATS*, PMLR 89, 3459–3467, 2019. Stable URL: [PMLR](https://proceedings.mlr.press/v89/vaicenavicius19a.html). DOI NOT VERIFIED.
8. Naeini, M. P., Cooper, G., & Hauskrecht, M. “Obtaining Well Calibrated Probabilities Using Bayesian Binning.” *Proceedings of AAAI*, 29(1), 2015. DOI: [10.1609/aaai.v29i1.9602](https://doi.org/10.1609/aaai.v29i1.9602).
9. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. “On Calibration of Modern Neural Networks.” *Proceedings of ICML*, PMLR 70, 1321–1330, 2017. Stable URL: [PMLR](https://proceedings.mlr.press/v70/guo17a.html). DOI NOT VERIFIED. Temperature scaling is discussed by this reference but is explicitly outside the current method set.

## Paired inference, uncertainty, and multiplicity

10. McNemar, Q. “Note on the sampling error of the difference between correlated proportions or percentages.” *Psychometrika*, 12, 153–157, 1947. DOI: [10.1007/BF02295996](https://doi.org/10.1007/BF02295996).
11. Dietterich, T. G. “Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.” *Neural Computation*, 10(7), 1895–1923, 1998. DOI: [10.1162/089976698300017197](https://doi.org/10.1162/089976698300017197).
12. Efron, B., & Tibshirani, R. “Bootstrap Methods for Standard Errors, Confidence Intervals, and Other Measures of Statistical Accuracy.” *Statistical Science*, 1(1), 54–75, 1986. DOI: [10.1214/ss/1177013815](https://doi.org/10.1214/ss/1177013815).
13. Holm, S. “A Simple Sequentially Rejective Multiple Test Procedure.” *Scandinavian Journal of Statistics*, 6(2), 65–70, 1979. Stable DOI record: [10.2307/4615733](https://doi.org/10.2307/4615733). The present minimum plan retains the already implemented Bonferroni family; Holm is cited only as an available alternative, not silently substituted.

## Diversity and agreement

14. Kuncheva, L. I., & Whitaker, C. J. “Measures of Diversity in Classifier Ensembles and Their Relationship with the Ensemble Accuracy.” *Machine Learning*, 51, 181–207, 2003. DOI: [10.1023/A:1022859003006](https://doi.org/10.1023/A:1022859003006).
15. Feinstein, A. R., & Cicchetti, D. V. “High agreement but low kappa: I. The problems of two paradoxes.” *Journal of Clinical Epidemiology*, 43(6), 543–549, 1990. DOI: [10.1016/0895-4356(90)90158-L](https://doi.org/10.1016/0895-4356(90)90158-L).

## Representation visualization

16. van der Maaten, L., & Hinton, G. “Visualizing Data using t-SNE.” *Journal of Machine Learning Research*, 9, 2579–2605, 2008. Stable URL: [JMLR](https://www.jmlr.org/papers/v9/vandermaaten08a.html). DOI NOT VERIFIED.
17. Wattenberg, M., Viégas, F., & Johnson, I. “How to Use t-SNE Effectively.” *Distill*, 2016. DOI: [10.23915/distill.00002](https://doi.org/10.23915/distill.00002).
18. Kobak, D., & Berens, P. “The art of using t-SNE for single-cell transcriptomics.” *Nature Communications*, 10, 5416, 2019. DOI: [10.1038/s41467-019-13056-x](https://doi.org/10.1038/s41467-019-13056-x). The domain differs, but the visualization cautions are methodological.

## Computational benchmarking and variability

19. Reddi, V. J., et al. “MLPerf Inference Benchmark.” *Proceedings of ISCA*, 446–459, 2020. DOI: [10.1109/ISCA45697.2020.00045](https://doi.org/10.1109/ISCA45697.2020.00045). MLPerf Single-Stream uses 90th-percentile query latency under its own load-generation protocol; it does not prescribe median [IQR] across five repeat means.
20. Bouthillier, X., et al. “Accounting for Variance in Machine Learning Benchmarks.” *Proceedings of MLSys*, 2021. Stable URL: [MLSys paper](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf). DOI NOT VERIFIED.

## Authoritative implementation references

21. scikit-learn, `roc_auc_score` API and multiclass semantics: [official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
22. scikit-learn, multiclass ROC example: [official documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html).
23. scikit-learn, `matthews_corrcoef`: [official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html).
24. SciPy, `scipy.stats.binomtest`: [official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

25. scikit-learn, `f1_score` API: `average="macro"` calculates each label's F1 and returns their unweighted mean. [Official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).
26. Opitz, J., & Burst, S. “Macro F1 and Macro F1.” arXiv:1911.03347, 2019. Stable URL: [arXiv](https://arxiv.org/abs/1911.03347). DOI: [10.48550/arXiv.1911.03347](https://doi.org/10.48550/arXiv.1911.03347). Directly distinguishes the arithmetic mean of classwise F1 from the harmonic mean of macro precision and macro recall.
27. Takahashi, K., Yamamoto, K., Kuchiba, A., & Koyama, T. “Confidence interval for micro-averaged F1 and macro-averaged F1 scores.” *Applied Intelligence*, 52, 4961–4972, 2022. DOI: [10.1007/s10489-021-02635-5](https://doi.org/10.1007/s10489-021-02635-5). Directly defines arithmetic-mean classwise Macro-F1 and develops frequentist large-sample intervals; it does not mandate the proposed bootstrap recipe.
28. Davison, A. C., & Hinkley, D. V. *Bootstrap Methods and their Application*. Cambridge University Press, 1997. DOI: [10.1017/CBO9780511802843](https://doi.org/10.1017/CBO9780511802843). General methodological support for matching bootstrap resampling to homogeneous/stratified data structures; not a direct prescription for these five endpoints.
29. Ferro, C. A. T. “Fair scores for ensemble forecasts.” *Quarterly Journal of the Royal Meteorological Society*, 140(683), 1917–1923, 2014. DOI: [10.1002/qj.2270](https://doi.org/10.1002/qj.2270). Scope-limited support concerning finite ensemble forecast distributions and ensemble-size effects; the current four deterministic classifiers are not established as exchangeable forecast members.

## Reference-to-claim verification matrix

| Ref. | Classification | Claim mapping after verification |
|---:|---|---|
| 1 | REMOVE/REPLACE | Do not use for the exact current Macro-F1 formula. Retain only as background on classification measures and the competing macro-F formulation. |
| 2 | DIRECT SUPPORT | Supports the generalized multiclass MCC concept; current numeric semantics still come from sklearn. |
| 3 | REMOVE/REPLACE | Do not cite for current macro OvR AUC. It supports a different pairwise/one-vs-one multiclass generalization. |
| 4 | SCOPE-LIMITED | Supports PR-vs-ROC cautions for binary imbalance; does not mandate multiclass PR-AUC here. |
| 5 | DIRECT SUPPORT | Supports NLL/log score and Brier/quadratic score as proper scoring rules. |
| 6 | DIRECT SUPPORT | Original multicategory Brier-score background and formula family. |
| 7 | DIRECT SUPPORT | Supports the need to specify multiclass calibration notion and cautions against treating a scalar calibration metric as exhaustive. |
| 8 | SCOPE-LIMITED | Supports binned ECE/reliability concepts; original treatment does not by itself define current six-class top-label ECE. |
| 9 | SCOPE-LIMITED | Supports modern-network/top-label ECE context; Temperature Scaling is outside this project. |
| 10 | DIRECT SUPPORT | Supports McNemar inference for paired correlated proportions. |
| 11 | BACKGROUND ONLY | Places McNemar among classifier-comparison tests; it is not the implementation specification for current exact binomial code. |
| 12 | BACKGROUND ONLY | Supports general bootstrap uncertainty/percentile methods, not class-stratification or one universal design for all endpoints. |
| 13 | BACKGROUND ONLY | Describes Holm multiplicity correction; the current plan intentionally retains Bonferroni. |
| 14 | DIRECT SUPPORT | Supports diversity measures and the nontrivial relationship between diversity and ensemble accuracy. |
| 15 | SCOPE-LIMITED | Supports marginal/prevalence-related kappa cautions in a different rater/medical setting; not an exact multiclass model-kappa specification. |
| 16 | DIRECT SUPPORT | Defines the t-SNE method. |
| 17 | BACKGROUND ONLY | Practical interpretation cautions for t-SNE; not a formal classifier-performance result. |
| 18 | SCOPE-LIMITED | Methodological t-SNE guidance from a different scientific domain. |
| 19 | SCOPE-LIMITED | Supports workload/environment-specific benchmarking and a p90 Single-Stream metric; does not support median [IQR] across five repeat means. |
| 20 | BACKGROUND ONLY | Supports distinguishing benchmark variance sources; does not prescribe a seed count for this repository. |
| 21 | IMPLEMENTATION SEMANTICS | Exact source for sklearn macro OvR AUC behavior. |
| 22 | IMPLEMENTATION SEMANTICS | Official worked example for macro one-vs-rest ROC construction. |
| 23 | IMPLEMENTATION SEMANTICS | Exact library behavior for current MCC call. |
| 24 | IMPLEMENTATION SEMANTICS | Exact library behavior for current two-sided exact binomial test. |
| 25 | IMPLEMENTATION SEMANTICS | Exact source for current arithmetic mean of label-wise F1. |
| 26 | DIRECT SUPPORT | Establishes the two competing “Macro-F1” formulas and directly matches the sklearn/current-code variant. |
| 27 | DIRECT SUPPORT | Directly defines arithmetic-mean classwise Macro-F1 and its uncertainty problem; does not prescribe the selected bootstrap. |
| 28 | BACKGROUND ONLY | Supports endpoint/data-structure-aware bootstrap construction; exact current resampling designs remain explicit proposal choices. |
| 29 | SCOPE-LIMITED | Supports caution when scoring finite vote distributions; exchangeable meteorological ensemble assumptions do not directly hold for four heterogeneous deterministic backbones. |

## Scope note

No citation above establishes that a metric is automatically appropriate merely because it is common. In particular, no cited paper proves that all six tea-leaf classes have equal agricultural cost or endpoint-level importance. The proposed Macro-F1 role is conditional on that domain choice being explicitly pre-specified. Proposed roles are derived jointly from current code, class distribution, paired prediction identity, probability semantics, and the scientific question each metric answers.
