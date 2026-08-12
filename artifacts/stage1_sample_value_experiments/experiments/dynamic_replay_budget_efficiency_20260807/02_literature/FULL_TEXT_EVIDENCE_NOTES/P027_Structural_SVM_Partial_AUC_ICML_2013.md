# P027 - A Structural SVM Based Approach for Optimizing Partial AUC

## Identity

- Paper ID: P027
- Authors: Harikrishna Narasimhan and Shivani Agarwal
- Venue and year: ICML 2013, PMLR 28(1):516-524
- Published page: https://proceedings.mlr.press/v28/narasimhan13.html
- Main PDF: `source_papers/Structural_SVM_Partial_AUC_ICML_2013.pdf`, SHA256 `3181AD0BD445DEB002F3F3D6C4A415E5FA008F61219A79DE6078856DB84DF51B`
- Supplement: `source_papers/Structural_SVM_Partial_AUC_ICML_2013_supp.pdf`, SHA256 `31CE4C2849B035B9BB424F290E509ED7B039DF547A7BFE2DB880995F1C4C5052`
- Paper-declared code: `http://clweb.csa.iisc.ernet.in/harikrishna/Papers/SVMpAUC/`; no versioned retrievable snapshot was found in this review.

## Reading Coverage

- Main manuscript: 9/9 pages read, including the normalized empirical pAUC definition with fractional boundary terms, structural output formulation, optimization problems OP1-OP6, Theorems 1-2, Algorithms 1, all four application sections, runtime analysis and conclusions.
- Supplement: 2/2 pages read, including cutting-plane Algorithm 2, full proof of Theorem 1, parameter grids, split procedures, preprocessing and hardware.
- Visual verification: all 11 rendered pages under `audit/visual_checks/P027_Structural_SVM_Partial_AUC_ICML_2013/` inspected.
- Code: the paper's legacy code URL was recorded but no immutable versioned source could be obtained. The later 2017 extended article was identified but was not substituted for the 2013 artifact.

## Research Question

The paper optimizes normalized partial AUC over an arbitrary false-positive interval `[alpha, beta]`. Its central technical difficulty is directly relevant to Stage1: the normal examples that occupy an operating tail depend on the current scoring function and can change whenever the model changes.

For a model `f`, sort normal scores in descending order. Empirical pAUC uses only normals at ranks corresponding to `[alpha, beta]`, with fractional weights at non-integer boundaries. Thus the target set is not a frozen list of images:

```text
tail_members(f, alpha, beta) changes when f changes.
```

This means a sample's role is conditional on model state and the surrounding order statistics even before replay dynamics are considered.

## Core Formulation

For `m` positives and `n` negatives, the normalized empirical pAUC sums positive-negative ordering indicators only over negative ranks `j_alpha` through `j_beta`, with fractional endpoint terms when `n*alpha` or `n*beta` is non-integer.

The method encodes each positive-negative ordering as a binary matrix `pi` where `pi_ij=1` means positive `i` is ranked below negative `j`. The structural SVM minimizes

```text
0.5 * ||w||^2 + C * xi
```

subject to one loss-augmented margin constraint for every valid ordering matrix. The slack is stated to upper-bound empirical pAUC risk, although the main paper defers details of that bound to a longer version and the two-page supplement does not supply them.

The cutting-plane solver repeatedly finds the most violated ordering. Unlike full AUC, pAUC does not initially decompose over individual pairs because the relevant normal subset changes with the candidate ordering.

Theorem 1 restricts the optimizer to orderings where normals separated by a positive remain sorted according to current model scores. For `[0,beta]`, the loss-augmented problem then decomposes elementwise. For general `[alpha,beta]`, each positive's row can be optimized separately. The claimed compact complexity is `O((m+n) log(m+n))`, matching the full-AUC routine per most-violated-constraint call.

## Experimental Protocol

- Every experiment learns a linear scoring function.
- Cheminformatics: five targets, each with 50 active compounds and 1,892 inactive compounds; 10 random 10%/90% train/test splits per target; pAUC `[0,0.1]`.
- Information retrieval: TD2004 and TREC10; 10 random query-level 60%/20%/20% train/validation/test splits; target method trained for `[0,0.1]`, with multiple top-of-list evaluation ranges.
- PPI: 2,865 known interacting pairs and 237,384 randomly selected pairs assumed non-interacting; 10 random 1%/9%/90% train/validation/test splits.
- Breast cancer: 102,294 candidate ROIs from four images per patient; 10 random 5%/95% train/test splits; FROC range rescaled to pAUC `[0.2s,0.3s]`.
- SVM regularization is selected by five-fold cross-validation or a validation split. The error tolerance is `1e-4` except IR and runtime experiments, where it is relaxed to `0.1` because smaller values take too long.
- Significance stars use a two-sided Wilcoxon test at 95% confidence, but tables report point means without uncertainty intervals.

All of these model, split and optimization settings are literature context only and cannot change the Stage1 canonical configuration.

## Positive And Negative Evidence

1. Targeting the relevant ROC region can outperform full-AUC training on several datasets, especially PPI and the very top of TD2004/TREC10.
2. Improvements are not universal. On TREC10, SVMpAUC is worse than full-AUC SVM at pAUC `[0,0.1]` and at full AUC; its top-five comparison is not significant. On cheminformatics, the difference from ASVM is not significant. On the breast-cancer task, the difference from full-AUC SVM is not marked significant.
3. The runtime plot shows that one most-violated-constraint call costs roughly the same across `beta`, but the number of calls rises steeply as `beta` shrinks, approaching about 5,000 at `beta=0.01` in the plotted TREC10 run. Extreme-tail focus increases total optimization difficulty even when per-call complexity is unchanged.
4. PPI preprocessing uses statistics from the entire dataset in a declared transductive setting. That result is not an inductive holdout estimate under Stage1's lineage rules.
5. PPI negatives are only assumed non-interacting, so label uncertainty is built into the benchmark.
6. Breast-cancer splits are described as random ROI splits while observations come from multiple images per patient; patient-group separation is not reported. This leaves a possible dependence pathway.
7. The paper assumes iid class samples and mostly assumes no score ties, then uses a modified tie-aware metric in experiments. Stage1 video frames violate simple iid identity assumptions unless grouped explicitly.

## Direct Support For Stage1

1. Persist tail membership at every epoch, not only scores. For each probe identity record entry/exit epochs, rank, boundary distance and consecutive tail residence.
2. Distinguish fixed probe identity from model-defined tail membership. A fixed probe supports longitudinal comparison; a dynamic tail describes the operational state. Both are needed.
3. Store exact finite-sample endpoint weights and tie policy when integrating raw safety-frontier area or pAUC-like summaries.
4. Measure tail-set turnover across epochs and seeds using Jaccard overlap, rank correlation and identity transition counts.
5. Measure effective tail sample size, video/cluster concentration and the number of unique defect-normal pairs. A nominal percentile can contain little independent information when frames are correlated.
6. Keep per-identity pair violations and aggregate set loss separate. The structural objective is non-decomposable before conditioning on current order.
7. Treat very narrow-tail diagnostics as high-variance measurements requiring uncertainty and group-aware sensitivity, not as a precise scalar ranking.

## What It Does Not Support

1. Replacing canonical Stage1 training with a structural SVM or pAUC surrogate.
2. Importing any `alpha`, `beta`, `C`, epsilon, split fraction, preprocessing or linear-model choice.
3. Treating the current normal tail as a permanent high-value sample list.
4. Claiming that optimizing pAUC automatically improves `TN_at_FN95`, `FN_at_TN68253` or the full raw `FN=0-95` frontier.
5. Claiming deep-network, replay-timing or cross-seed stability from linear SVM experiments.
6. Treating a per-call asymptotic complexity result as evidence that extreme-tail training is operationally cheap.

## Stage1 Field Contract

Under the exact canonical hyperparameter lock, add or retain:

- per-epoch rank and dynamic tail-membership flag for fixed normal and defect probe identities;
- tail-entry count, tail-exit count, first/last entry epoch and longest consecutive residence;
- adjacent-epoch and key-checkpoint Jaccard overlap of tail sets;
- per-seed rank correlation and tail-membership agreement for identical probe identities;
- effective unique video/group/cluster count inside each tail and maximum group concentration;
- exact percentile boundary rank, fractional endpoint weight, tie count and comparator convention;
- hard pair-violation count and identity-level partner concentration;
- compute time and memory by tail fraction, including number of ranking/selection passes;
- source-role and group-split checks for every calibration and evaluation identity.

These are diagnostics. They do not alter `yolo11l`, batch 128, image size 224, workers 4, optimizer, schedule, augmentation, AMP or any other canonical field.

## Concrete Experiment Consequence

P027 adds no training arm. Within each fixed causal replay arm, it tests whether late replay changes who occupies the dangerous tail:

```text
tail turnover
+ pair-violation concentration
+ weak-defect score movement
+ realized replay exposure
```

If the same replay IDs produce opposite outcomes across seeds, compare whether those seeds entered different model-defined tail sets before the outcome split. A stable association would support conditional value. It is not causal by itself; continuous, decayed, dose-matched and no-replay paired interventions remain necessary.

## Stage1 Decision Status

- Evidence status: `FULL_READ_COMPLETE`
- Claim eligibility: yes, for state-dependent tail membership, non-decomposable ordering structure and extreme-tail optimization cost
- Replication-depth eligibility: no; no immutable executable code snapshot was available
- Direct support for static replay ranking: no
- Direct support for dynamic replay timing: indirect only
- Direct support for a new formal arm: no
- Direct support for process fields: yes
- Permission to change canonical Stage1 hyperparameters: no
- Reviewed at: 2026-08-07
