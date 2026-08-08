# P025 - Neyman-Pearson Classification Algorithms and NP Receiver Operating Characteristics

## Identity

- Paper ID: P025
- Authors: Xin Tong, Yang Feng and Jingyi Jessica Li
- Venue and year: Science Advances 4(2), 2018, article eaao1659
- DOI: 10.1126/sciadv.aao1659
- Published full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC5804623/
- Author manuscript: `source_papers/Neyman_Pearson_Umbrella_NP_ROC_2018.pdf`, SHA256 `89C21BF2FC961433B27C261B36C0FFE613821267BDEFFEBBCD3D73F5A75C2DE7`
- Official reproduction bundle: `source_code/PMC5804623_supplementaryFiles.zip`, SHA256 `22836AC57472DC464639CEA1EDC184469072B6D4B5DBAA40E6BC1E3BC44B6FBF`
- Paper-declared software: `nproc` 2.0.9; current CRAN 2.1.5 was audited separately

## Reading Coverage

- Main manuscript: 18/18 pages read, including formulation, umbrella algorithm, Proposition 1, empirical ROC comparison, NP-ROC construction, simulations, both applications, discussion and limitations.
- Supplement: 14/14 pages read, including the proposition proof, conditional type-II bounds, Simulations S1 and S2, all tables and figures.
- Official reproduction code: all 915 lines of the 2017-11-26 R Markdown read; bundled data and precomputed-result inventory checked.
- Package versions: CRAN 2.0.9, 2.1.1, 2.1.4 and 2.1.5 source paths and relevant changes inspected.
- Executable checks: R was unavailable, so official R code was not run. Independent Python checks reproduced the exact order-statistic bound, its minimum sample size and the qualitative Simulation-1 violation pattern.
- Visual verification: all 32 pages and eight contact sheets under `audit/visual_checks/P025_NP_Umbrella_SciAdv_2018/` were inspected.

## Research Question

The paper asks how to minimize one classification error while controlling a prioritized population error with high probability. Its oracle problem is

```text
minimize_phi  R1(phi)
subject to    R0(phi) <= alpha,
```

where `R0` is the prioritized error, `alpha` is its population upper bound and `delta` is the tolerated probability that the learned classifier violates that bound.

For Stage1, ordinary labels make false negatives a type-II error. The labels or score orientation must therefore be reversed before transferring the theorem: treat defect as the prioritized class 0 and use a score increasing toward normal, equivalently apply the rule to `-p_defect`. The theorem does not control Stage1 FN under the package's default label orientation.

## Core Method

Fit any scoring model `f` without using a held-out prioritized-class calibration sample of size `n`. Sort its calibration scores:

```text
T_(1) <= ... <= T_(n).
```

For threshold rank `k`, Proposition 1 gives

```text
P[R0(phi_k) > alpha]
  <= v(k)
  = sum_{j=k}^n C(n,j) (1-alpha)^j alpha^(n-j).
```

The bound is exact for continuous score distributions. Choose

```text
k_star = min{k : v(k) <= delta}.
```

The smallest calibration size for which any order statistic can satisfy the requested guarantee is

```text
n_min = ceil(log(delta) / log(1-alpha)).
```

The theorem requires the fitted scoring function and held-out calibration observations to be independent and the prioritized-class calibration observations to be i.i.d. It needs only score ordering, not calibrated probabilities.

NP-ROC constructs a pointwise high-probability upper bound on prioritized error and conditional bounds on the other error. The vertical interval has at least `1-2*delta` coverage under the stated derivation. Averaging bands across random splits is an empirical summary, not the same one-split theorem.

## Stage1 Numerical Consequence

Stage1's recall target is approximately `alpha=0.005`. At `delta=0.05`:

```text
n_min = ceil(log(0.05) / log(0.995)) = 598.
```

The exact best-case violation probabilities are:

```text
n=95:  (1-0.005)^95  = 0.6211445383  -> impossible to claim 95% control
n=598: (1-0.005)^598 = 0.0499116912  -> just feasible
n=600: (1-0.005)^600 = 0.0494138221  -> feasible
```

Thus `FN <= 95` is an outcome count, not a calibration sample-size argument. Ninety-five held-out defects cannot certify a 0.5% population FN bound with 95% confidence. With 30,000 independent defect calibration scores, the transformed-score rank is 29,871, corresponding approximately to the 130th lowest original defect score.

The printed Discussion examples contain an internal arithmetic error. The paper reports 45 for `alpha=0.1, delta=0.05` and 29 for `alpha=0.05, delta=0.1`; the formula yields 29 and 45 respectively. The `alpha=delta=0.05` example, 59, is correct.

## Experimental Evidence

- Simulation 1 uses 1,000 datasets of size 1,000 from two one-dimensional Gaussians. Direct empirical-error and five-fold-CV thresholds violate the population bound in roughly half the datasets; the NP threshold stays within the requested violation tolerance.
- Independent Python reproduction over 1,000 new datasets found violation rates `0.554` for the naive rule, `0.515` for the code-equivalent CV rule and `0.033` for NP at `alpha=delta=0.05`.
- Simulation S1 generates 2,000 datasets, pairing 1,000 training datasets with 1,000 test datasets. At `alpha=delta=0.05`, the selected empirical-ROC classifiers have 30.8% population violation, versus 3.1% for NP.
- Simulation S2 uses 1,000 datasets from each of logistic and LDA generators, a common one-million-observation test set, six base methods and split counts `{1,5,9,11,15}`. Majority voting reduces reported dispersion and retains empirical violation below delta, but this is simulation evidence rather than Proposition-1 coverage for the ensemble.
- The Early Warning application has only 60 prioritized observations among 6,365. It demonstrates the asymmetric framing but operates at a much looser error level than Stage1.
- The neuroblastoma application has 176 high-risk and 322 non-high-risk samples. NP reduces empirical prioritized error but often pays substantial type-II cost. Main text says 100 train/test repetitions, while the supplement table and reproduction code use 1,000.
- The paper explicitly states that no distribution-free oracle optimality for the non-prioritized error is available.

## Code Reproduction Audit

1. The official R Markdown is dated 2017-11-26 and explicitly states that paper results use `nproc` 2.0.9. That version predates publication; 2.1.1 was released 11 days after publication and current 2.1.5 in 2020.
2. The reproduction bundle contains prepared EWP and SEQC data plus cached RData outputs. It is much stronger than a demo, but dependencies are only lower-bounded and no environment lock is supplied.
3. Simulation 1 and Simulation S1 declare seed variables but do not call `set.seed` before generating data when caches are absent. Clean regeneration therefore depends on ambient RNG state.
4. The neuroblastoma code uses `reps=1000`, confirming the supplement rather than the main text's 100 repetitions.
5. In 2.0.9, insufficient calibration size makes `min(which(...))` non-finite; the intended stop branch is not reached and the cutoff path is invalid.
6. Version 2.1.1 added a proper `min.alpha` check and `nsmall` state. The 2.1.4 speed rewrite, retained in 2.1.5, again takes `min(which(s<=delta))` without guarding an empty result and leaves `nsmall=FALSE`.
7. The package infers score direction by comparing class means that include the held-out class-0 scores. Strictly, the scoring rule is then not fixed independently of the calibration sample as required by Proposition 1. Stage1 must fix score orientation from semantics before calibration.
8. `split=0` reuses the prioritized samples for fitting and calibration, so that mode is outside the theorem.
9. Multi-split fits reuse overlapping observations and majority vote. The paper gives simulation support, not the one-split guarantee, for this output.
10. NP-ROC code searches probability bounds on a 0.001 grid and injects random jitter when scores have fewer than ten unique values. These are implementation approximations that need explicit provenance.
11. No package tests exist in the inspected source archives. R is absent locally, so the official code was not executed.
12. The independent order-statistic implementation reproduced `n_min=598`, exact violation `0.0499116912` and a 500,000-trial estimate `0.049212`.

## Evidence Limitations

1. This is a threshold-calibration paper, not a sample-replay or training-dynamics paper.
2. It does not identify valuable training samples, replay timing, replay dose, gradient direction or guard composition.
3. The guarantee is marginal over the random calibration sample for a fixed trained scorer. It does not survive repeated policy selection on the same calibration identities without further correction.
4. I.i.d. class-conditional sampling is questionable for correlated video frames. Grouped clips, temporal dependence and distribution shift require separate treatment.
5. The score continuity condition is needed for equality; discrete or heavily tied scores retain only the inequality.
6. The one-split guarantee does not automatically apply to majority-vote ensembles or averaged NP-ROC bands.
7. Population FN-rate control does not prove an improvement in `TN_at_FN95`, raw-frontier area or replay efficiency.
8. The paper supplies no Stage1 epoch boundary, replay percentage, optimizer choice or augmentation setting.

## Direct Support For Stage1

1. Separate model/policy discovery, threshold calibration and blind confirmation by sample identity and access role.
2. Treat `alpha` and `delta` as separate preregistered quantities: allowed FN rate versus probability of violating that rate.
3. Compute and store exact order-statistic feasibility, rank, bound and tie policy for every saved checkpoint.
4. Use a fixed semantic orientation for defect protection; do not estimate sign from the calibration sample.
5. Fail before evaluation when the independent defect calibration count is below `n_min`.
6. Report the operational raw frontier and an NP-conservative operating point separately. The latter is a safety statement, not a replacement for utility analysis.
7. Preserve group/video identities so that an i.i.d. claim is not silently made for adjacent frames.
8. Keep the blind holdout closed until replay schedule, selection, checkpoint rule, `alpha`, `delta` and calibration procedure are frozen.

## What It Does Not Support

1. Changing any canonical Stage1 training hyperparameter.
2. Treating `FN=95` as a statistically guaranteed threshold merely because the observed count is 95.
3. Reusing OOF, `val_op` or blind samples for both repeated schedule discovery and final certification without role accounting.
4. Claiming majority-vote multi-split control from Proposition 1.
5. Importing `nproc` defaults, split ratios, split count 11 or any paper classifier settings into yolo11l training.
6. Replacing raw score-frontier comparisons with a single conservative threshold.

## Stage1 Field Contract

Under the exact hash-locked 240-run hyperparameters, add evaluation fields rather than changing training:

- `evaluation_role`: discovery, policy calibration, threshold calibration or blind confirmation;
- calibration manifest hash, defect identity count, video/group count and overlap checks;
- `np_alpha`, `np_delta`, `np_n_min`, actual calibration `n`, chosen rank and exact violation bound;
- score orientation, comparator strictness, tie count and tie-breaking policy;
- checkpoint and scoring-function hashes proving the model was fixed before calibration;
- raw observed FN/TN and confidence/conservative bounds at every key checkpoint;
- number of times each calibration identity influenced schedule, checkpoint or threshold selection;
- dependence warnings for repeated frames and group-aware sensitivity results.

These fields belong to 120, 140, 150, 160, 180 and 200 checkpoint evaluation. Low-cost feasibility metadata can be emitted at every epoch if the fixed probe predictions are already available.

## Concrete Experiment Consequence

P025 does not add a training arm. It adds a threshold-calibration gate after a replay policy and checkpoint are frozen:

```text
OOF/train                 -> candidate and mechanism discovery
fixed val_op discovery    -> schedule and checkpoint rule only
independent defect sample -> one-split NP threshold calibration
blind holdout             -> final utility and safety evaluation
```

If no untouched defect set of at least 598 independent identities exists at `alpha=0.005, delta=0.05`, the experiment may still report observed `FN<=95`, exact binomial intervals and raw frontiers, but it may not claim the paper's distribution-free 95% population control. Video dependence may require substantially more conservative group-level analysis even when the raw image count exceeds 598.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, for independent order-statistic threshold calibration and explicit alpha/delta feasibility
- Replication-depth eligibility: yes, because main, supplement, official reproduction code, exact package versions and independent numerical probes were audited
- Direct support for static replay ranking: no
- Direct support for dynamic replay timing: no
- Direct support for a new training arm: no
- Direct support for a separate threshold-calibration gate: yes
- Permission to change canonical Stage1 hyperparameters: no
- Reviewed at: 2026-08-07
