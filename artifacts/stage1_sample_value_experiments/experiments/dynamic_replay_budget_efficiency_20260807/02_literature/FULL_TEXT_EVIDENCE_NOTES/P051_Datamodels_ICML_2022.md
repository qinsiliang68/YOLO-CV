# P051 - Datamodels: Understanding Predictions with Data and Data with Predictions

## Identity

- Paper ID: P051
- Authors: Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry
- Venue and year: ICML 2022; Proceedings of Machine Learning Research 162
- Official article page: https://proceedings.mlr.press/v162/ilyas22a.html
- Official paper: `source_papers/Datamodels_ICML_2022.pdf`, SHA256 `90EAB4FD0CDCAEC55A1FD08CB135E843E730AF5B35FB72015F7D44A0F55F048D`
- Official implementation: https://github.com/MadryLab/datamodels at commit `61e590a6d857b31b6b11be10800f7c9bba6b400e`
- Official data release: https://github.com/MadryLab/datamodels-data at commit `9464be5f227ee6f473e6d13c24c54113a746aa3c`

The PDF and arXiv materials also use the shorter subtitle "Predicting Predictions from Training Data." The PMLR bibliographic title above is used as the canonical identity.

## Reading And Audit Coverage

- Main paper and appendices: 63/63 pages read.
- Coverage includes the formal datamodel definition; the linear LASSO instantiation; output-function choice; all CIFAR-10 and FMoW training settings; held-out subset prediction; data support; removal and mislabeling counterfactuals; stochastic retraining analysis; architecture transfer; stress tests; train-test leakage; datamodel embeddings; PCA interventions; the connection to empirical influence; future work; and the complete proof appendix.
- Visual verification: all 63 pages inspected at original detail under `audit/visual_checks/P051_Datamodels_ICML_2022/`.
- Text extraction: the complete PDF was extracted and searched for definitions, estimands, subset distributions, model counts, stochastic repetitions, counterfactual protocols, correlations, limitations, and future work.
- Code audit: all 14 commits and all 29 tracked files in the official implementation repository were provenance-scanned; its 16 Python files passed AST parsing. All five commits and all three tracked files in the official data repository were inspected. The hundreds-of-gigabytes to multi-terabytes released arrays were not downloaded.
- Replication boundary: this is recorded as `REPLICATION_DEPTH` because the released data schema, worker, regression path, examples, history, and implementation gaps were audited. It is not an assertion that the four million model trainings or the paper's exact numerical results were rerun.

## Formal Estimand

For a fixed target example `x`, fixed full training set `S`, fixed learning algorithm `A`, and a distribution `D_S` over subsets of `S`, define the stochastic outcome:

```text
f_A(x; S') = output obtained by training A on subset S' and evaluating x.
```

A datamodel is a predictor from subset identity to this outcome:

```text
g_theta(1_S') approximately equals f_A(x; S').
```

The paper's principal instantiation is target-specific sparse linear regression:

```text
g_theta(1_S') = theta_x^T 1_S' + theta_0,x
```

fitted with LASSO. Each coefficient is indexed by one training identity, but it is not an intrinsic, context-free value of that identity. It is conditional on:

```text
target x
learning algorithm A
full data universe S
subset distribution D_S
subsampling fraction alpha
chosen model-output functional
regularization and available subset-model observations
```

This conditionality is central for Stage1. A coefficient learned for one target, learner, alpha, or output cannot be relabeled as universal sample value.

## Why Margin Is Modeled

The paper compares correctness, correct-class confidence, cross-entropy, and correct-class margin. It chooses:

```text
margin(x) = logit(correct class) - max logit(other classes)
```

because it is continuous, less saturated than correctness or confidence, and yields residuals closer to the paper's linear-regression assumptions. This is empirical and task-specific. It supports preserving raw margins or logits in Stage1, but it does not prove that a probability gap, quantile gap, or weighted combination is a sufficient business objective.

## Experimental Scale And Training Protocol

The experiments use target-specific datamodels for all 50,000 CIFAR-10 training examples, 10,000 CIFAR-10 test examples, 21,404 FMoW training examples, and 3,138 FMoW test examples.

The number of subset-trained models is:

| Dataset | alpha | Datamodel-train models |
| --- | ---: | ---: |
| CIFAR-10 | 0.10 | 1,500,000 |
| CIFAR-10 | 0.20 | 750,000 |
| CIFAR-10 | 0.50 | 300,000 |
| CIFAR-10 | 0.75 | 600,000 |
| FMoW | 0.20 | 375,000 |
| FMoW | 0.50 | 150,000 |
| FMoW | 0.75 | 300,000 |

An additional 10,000 models per setting are retained for held-out evaluation. CIFAR uses a ResNet-9 trained for 24 epochs with batch 512 and peak learning rate 0.5; FMoW uses a ResNet-18 trained for 15 epochs with batch 512 and peak learning rate 0.4. These settings make the enormous subset experiment feasible and are explicitly non-transferable to Stage1.

For the reported CIFAR setup, one alpha=0.5 ResNet-9 model takes about 17 seconds on an A100 and reaches about 90% accuracy. The appendix reports more than 5,000 such models per GPU-day and roughly 40,000 per day on eight A100s. Only subset masks and final outputs are retained; the millions of checkpoints are not.

## Predictive Fit

- The linear datamodel predicts held-out subset-trained outcomes with very high rank correlation; representative CIFAR plots report Spearman correlations around 0.991 to 0.997.
- The authors distinguish datamodel estimation error from irreducible outcome variance caused by stochastic training.
- LASSO regularization is selected for each target using held-out subset models. Even hundreds of thousands of subset models do not make unregularized high-dimensional regression automatically safe.
- Different alpha values expose relations at different scales. Larger alpha is more local to removals from a nearly complete set; smaller alpha exposes broader group relations. No alpha is universally best for every counterfactual.

This means a single Stage1 ranking learned from one replay ratio cannot be presumed stable at another ratio. Ratio-specific effect estimates and rank churn must be retained.

## Counterfactual Validation

The paper does not stop at coefficient plausibility. It retrains models after target-specific data interventions.

- It evaluates 300 CIFAR-10 and 100 FMoW target examples.
- CIFAR removal sizes are `10, 20, 40, 80, 160, 320, 640, 1280`; larger removals are added in stress tests.
- Each CIFAR counterfactual is averaged over 20 independent retrainings and each FMoW counterfactual over 10; CIFAR baselines use 10.
- Full-data controls average 10,000 CIFAR models or 500 FMoW models.
- Aggregate predicted-versus-observed counterfactual correlations are about Spearman 0.98/0.94 and Pearson 0.96/0.90 for CIFAR/FMoW.
- A one-retraining estimate has lower rank agreement than the 20-retraining estimate, approximately 0.966 versus 0.995 in the reported CIFAR analysis. Training randomness is therefore part of the measurement error, not a nuisance that can be ignored.
- Transferring the same datamodel counterfactuals from ResNet-9 to ResNet-18 degrades agreement but retains reported Spearman correlation around 0.949. This is evidence of partial transfer, not architecture invariance.
- Datamodel ranking outperforms representation distance, influence functions, and TracIn in the tested counterfactual tasks. Among those baselines, TracIn is the strongest in the reported comparison.

The paper also stress-tests its own method:

- removals extend to roughly 20% of the training set;
- rank agreement can remain high while unexplained magnitude error grows;
- interventions chosen to have near-zero predicted effect are observed to have small effects;
- negatively predicted interventions have much smaller absolute effects and lower correlation, around 0.716 in one stress plot;
- counterfactuals relative to a random alpha=0.5 control are also tested.

For Stage1, a static score becomes credible only after actual frozen-set replay interventions across paired seeds. Predicting a sign is not the same as demonstrating stable business-tail gain.

## Data Support And Interaction

The paper defines target-specific data support as the smallest removal set predicted and then verified to flip a target prediction. It searches top datamodel coefficients, interpolates the predicted crossing, and verifies by retraining.

- The initial estimated removal count flips only about 67% of targets.
- Inflating the estimate by 20% raises empirical certification to about 92%.
- Roughly half of CIFAR targets have estimated support at most 250 training images, about 0.4% of the dataset.
- Roughly 20% have support below 40 images, about 0.08%.
- Label-flipping interventions are substantially more brittle: around half of targets can be flipped by target-specific mislabeling of about 35 training examples.

These results show that a small set can strongly control one target. They do not show that adding those identities to training improves an aggregate endpoint, nor that individually large coefficients compose additively under replay. The linear model is an approximation to a set function and the paper explicitly studies finite-set counterfactuals to validate it.

## Embeddings, Leakage, And Aggregate Cancellation

The coefficient vector for each target acts as a model-behavior embedding indexed by training identities.

- Positive and negative coefficients expose model-relevant similarity and opposing evidence.
- CIFAR human review uses nine annotators per target and identifies same-scene or near-duplicate train-test leakage for about 10% of test examples under majority judgment.
- FMoW geodesic leakage is identified more effectively by datamodel relations than by standard representation distance.
- Spectral clustering and PCA find model-faithful subpopulations.
- Removing training examples from opposite ends of a datamodel principal component has opposite effects on aligned test groups while having almost no average effect over the complete test set.

That last result is directly relevant to Stage1: an aggregate endpoint can hide cancellation between difficult-normal and weak-defect subgroups. Their trajectories and intervention effects must be stored separately.

## Relation To Influence

For 50% fixed-size subset sampling and binary correctness, Lemma 1 shows that the ordinary least-squares datamodel coefficient vector and the difference-of-conditional-means empirical influence vector agree up to a known scaling in the infinite-subset limit:

```text
||(1 + 2/n) w_OLS - (1/2) w_influence||_2 -> 0.
```

The appendix also interprets alpha-subsampled influence as a first-order Taylor approximation of the multilinear extension of the training-set set function around the inclusion-probability vector `alpha * 1`.

This supplies a useful conceptual bridge but also a warning:

```text
influence is local to the chosen subset regime;
changing alpha changes the point around which the set function is approximated.
```

Empirical influence performs much worse than the explicit margin datamodel at matched sample counts. Table J.1 separates three causes: too few subset models, binary correctness rather than margin, and difference-of-means/OLS rather than sparse LASSO. The reported Spearman correlation rises from 0.028 with 25,000 correctness models to 0.320 with 100,000 margin models and LASSO. Thus an influence coefficient is not automatically a reliable value estimate simply because it has a formal interpretation.

## Limitations And Future Work

The paper identifies unresolved issues rather than claiming a final valuation method:

1. Outputs for many targets are correlated, but target-wise estimation does not exploit this structure.
2. The work provides point estimates, not calibrated confidence intervals for coefficients or counterfactual outcomes.
3. LASSO feature selection followed by naive inference creates post-selection inference problems.
4. Uniform random fixed-alpha subset sampling may be statistically inefficient; adaptive or intervention-designed sampling could reduce the number of trained models.
5. Better structured priors and nonlinear datamodels may be needed.
6. The success of a linear surrogate for a nonlinear end-to-end training process lacks a complete explanation.
7. The role of alpha is empirically clear but not theoretically characterized.
8. Human-facing interpretation and data exploration based on datamodels require further validation.

These limitations prohibit treating one fitted coefficient table as ground truth.

## Official Code And Data Audit

The official data release describes masks, margins, logits, and fitted datamodel tensors. Depending on dataset and alpha, masks plus margins range from roughly 10.6 GB to 245 GB and CIFAR logits add up to terabytes. It recommends memory mapping. This is an evidence release, not a lightweight reproduction fixture.

The later official `datamodels` repository provides a generic worker and sparse-regression pipeline. Material findings are:

1. The CIFAR example samples each identity independently with probability 0.5, whereas the paper defines uniform fixed-size alpha subsets. The example is not the exact paper subset generator.
2. The example does not bind NumPy, PyTorch, CUDA, augmentation, or loader randomness to the worker index, so a failed row cannot be reconstructed from index alone.
3. The worker writes output arrays one by one and only then marks `_completed`; a crash can leave partial row data. Completion prevents use of the row, but there is no row-level checksum or transactional cleanup.
4. Timeout exits with status zero. External schedulers can mistake a timed-out incomplete job for success unless they also inspect `_completed`.
5. The completion store supports skipping finished indices, but command provenance is only appended after success and concurrent appends are not guarded.
6. `initialize_store` opens arrays in `w+`, so rerunning initialization on an existing directory can destroy prior output.
7. Regression output refuses an existing directory and has no checkpoint/resume path for a long sparse fit.
8. The final repository's test script points at a missing `examples.imagenet.integration_test` module.
9. `requirements.txt` is unpinned and omits imported `ffcv` and `fast_l1`; `setup.py` installs only `fastargs`, names an incorrect homepage, and references a missing license filename. The released environment is not reproducibly locked.
10. The full paper experiment scripts, exact fixed-size mask construction, exact RNG lineage, and cluster scheduler manifests are not present. The repository is a framework for constructing datamodels, not a bitwise reproduction package.

These findings are not used to dismiss the paper's reported evidence. They define what can and cannot be imported into the Stage1 engineering design.

## Direct Support For Stage1

1. Replace the idea of universal `V(x)` with a conditional estimand such as `V(x | target role, theta_t, ratio, schedule, dose, seed, context)`.
2. Preserve target roles separately: difficult-normal suppression and weak-defect protection cannot be collapsed before analysis.
3. Store raw logits or margins at every formal epoch. Binary correctness and thresholded labels discard information needed for counterfactual modeling.
4. Treat replay ratio as part of the estimand. A coefficient or ranking at 0.5% need not transfer to 2.5%.
5. Estimate and report stochastic intervention error across paired seeds; one retraining is a noisy effect observation.
6. Evaluate selected sets with actual replay interventions. Correlation or coefficient magnitude alone does not establish positive downstream effect.
7. Record selected-set identity, inclusion mask, nested-prefix relation, cumulative exposure, and target-role outputs so later subset-effect models remain possible.
8. Keep the canonical 240-run learner fixed. Datamodel value changes when `A` changes, so a hidden batch, optimizer, LR, augmentation, or epoch change changes the scientific object.
9. Use group-aware subset diagnostics for video frames. Independent identity masks would overstate effective variation when adjacent frames are near duplicates.
10. Compare aggregate effects with role-specific effects to detect cancellation.

## What It Does Not Support

1. It does not justify using datamodel coefficients as the next Stage1 selection arm; millions of subset trainings are not available and our 240 historical runs were not randomized subset experiments.
2. It does not justify any replay percentage, decay epoch, guard proportion, number of seeds, or business success threshold.
3. It does not prove that high positive coefficients are high-value replay images or that individual top-k selection is optimal under interactions.
4. It does not make test-set-driven selection permissible. Development targets and blind holdout must remain separated.
5. It does not justify changing the canonical model, optimizer, batch, workers, AMP, augmentation, LR path, or 200-epoch duration.
6. It does not establish cross-seed stability. Its own counterfactual appendix shows that stochastic retraining averaging materially changes measurement quality.
7. It does not validate a single weighted scalar over normal and defect endpoints.

## Consequence For The Next Campaign

The most faithful low-compute translation is not to fit a full datamodel. It is to make the planned replay experiment itself produce a small, valid set-function dataset:

```text
for each canonical seed block and frozen selected set:
  preserve the exact inclusion/exposure schedule
  save every-epoch target-role margins
  compare same-selection timing/dose interventions
  estimate paired finite effects and their seed distribution
```

The first causal question remains whether continuous, same-peak decay, or dose-matched decay changes the effect of the same selected identities. If enough independently varied replay masks are later accumulated, a restricted datamodel can be fit to development-tail outcomes as a secondary analysis. It must not delay the registered experiment or consume the blind holdout.

## Decision

- Reading status: REPLICATION_DEPTH
- New formal arm: no
- New hyperparameter: no
- Canonical lock change: no; the paper reinforces that learner identity is part of the value definition
- Added fields: target identity and role, raw correct-class margin/logits, subset/replay mask identity, replay ratio, alpha-like context scale, selection prefix, realized exposure, seed/retraining replicate, paired finite intervention, prediction uncertainty, rank and magnitude error, role-specific effect, aggregate cancellation, code/config hash, and blind-holdout access state
- Remaining uncertainty: whether a compact Stage1 intervention design has enough independently varied masks to estimate interactions beyond the preregistered timing/dose contrasts
