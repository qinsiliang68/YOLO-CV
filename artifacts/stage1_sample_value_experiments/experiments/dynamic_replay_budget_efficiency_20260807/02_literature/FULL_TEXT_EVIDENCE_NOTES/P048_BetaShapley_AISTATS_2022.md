# P048 - Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for Machine Learning

## Identity

- Paper ID: P048
- Authors: Yongchan Kwon and James Zou
- Venue and year: AISTATS 2022, PMLR 151, oral presentation
- Official proceedings page: https://proceedings.mlr.press/v151/kwon22a.html
- Paper and supplement: `source_papers/BetaShapley_AISTATS_2022.pdf`, SHA256 `A1E69807CC9E6025EBE56F720C61A28781CC4051D83AEA68B1A0468D6273E94F`
- Official code: https://github.com/ykwon0407/beta_shapley
- Audited HEAD: `bb9b5e46363efdc7d52caf465f999abd364ca207`

## Reading And Audit Coverage

- Main paper and embedded supplement: 23/23 pages read.
- Coverage includes Definitions 1-2, Theorems 1-5, Proposition 3, Equations 1-5, Algorithm 1, all main and supplementary tables and figures, proofs, experimental settings, negative results, and limitations.
- Visual verification: all 23 pages inspected at original detail under `audit/visual_checks/P048_BetaShapley_AISTATS_2022/`.
- Code: all three Python modules, all 18 notebook cells and saved outputs, all nine commits, estimator and convergence paths, evaluation helpers, data splitting, and current file hashes inspected. The code was not treated as a runnable Stage1 dependency.

## Research Question

Data Shapley averages one point's marginal contribution uniformly over every training-set cardinality. This paper asks whether that uniform average is statistically and operationally appropriate for machine learning.

Its strongest transferable result is not that one Beta weighting is universally best. It is that the usefulness and noise of a marginal contribution depend on the surrounding set size and the downstream action. That directly challenges a single static Stage1 score used unchanged at every replay ratio.

## Formal Definitions

For candidate `z*`, utility `h`, database `D`, and a context containing `j-1` other points:

```text
Delta_j(z*; h, D)
  = average over all |S|=j-1 of [h(S U {z*}) - h(S)]
```

Data Shapley is the uniform cardinality average:

```text
psi_shap(z*) = (1/n) * sum_{j=1}^n Delta_j(z*)
```

The paper removes the efficiency axiom and retains linearity, null player, and symmetry. Theorem 2 shows that every resulting semivalue is a weighted mean of the same cardinality-specific marginal contributions. This makes the weight distribution a scientific choice rather than an axiomatically unique truth.

Beta Shapley uses a Beta-distribution family of weights. In the paper's convention:

```text
w_alpha,beta^(n)(j)
  = n * Beta(j + beta - 1, n - j + alpha) / Beta(alpha, beta)
```

After the semivalue normalization:

- `Beta(1,1)` is ordinary Data Shapley;
- `Beta(16,1)` concentrates on small contexts;
- `Beta(1,16)` concentrates on large contexts and approaches leave-one-out behavior.

This is still one scalar obtained by collapsing a full marginal-contribution curve. The curve itself is the more informative object for Stage1.

## Signal-To-Noise Result

For a fixed candidate and a random i.i.d. database, Theorem 1 studies the asymptotic variance of `Delta_j`. Under:

```text
j = o(sqrt(n))
and lim_{j->infinity} zeta_j / (j * zeta_1) is bounded,
```

the normalized variance converges so that:

```text
Var(Delta_j) is asymptotically proportional to j^2 * zeta_1 / n.
```

If the expected marginal signal also decays with context size, the signal-to-noise ratio is expected to fall as `j` grows. The paper verifies the bound condition empirically but explicitly says the needed upper bound has not been established generally. It is not a theorem that every large-context contribution is useless.

The experiments show clean/noisy separation is often largest at small cardinality and can overlap or reverse at large cardinality. However, this is label-noise discrimination under small deterministic classifiers, not a proof about Stage1 replay benefit.

## Optimal-Subsampling Claim And Boundary

Theorem 4 and formal Theorem 5 connect a Beta-Shapley-derived importance weight to asymptotically minimum-variance Horvitz-Thompson estimation for an M-estimator. The result requires Hadamard differentiability, bounded inclusion probabilities, an appropriate sampling measure, and asymptotic convergence of the normalized semivalue to the influence function.

The paper states that the convergence rate is unknown and that the optimal `(alpha,beta)` may depend on task and data distribution. This theorem does not establish a finite-sample optimal replay schedule, a top-k set, or a non-convex SGD training effect.

## Experimental Contract

- Fifteen binary classification datasets are used. Most valuation experiments use only 200 valued samples, 200 validation samples, and 1,000 held-out test samples.
- Image datasets use 32 principal components of penultimate-layer features from an ImageNet-pretrained ResNet18; utility is then computed with logistic regression or SVM, not end-to-end image training.
- Synthetic label noise flips 10% of both the valued training labels and the validation labels. The held-out test set is used for downstream evaluation.
- Main comparisons include LOO-First, `Beta(16,1)`, `Beta(4,1)`, Data Shapley, large-context Beta variants, LOO-Last, and KNN Shapley.
- Noise detection clusters the scalar values into two groups and thresholds below the lower cluster center. Results use 50 repetitions.
- Subsampling draws 50 of 200 samples with replacement using nonnegative value-derived probabilities and inverse-propensity weighting.
- Point addition begins from 10 samples; point removal deletes the lowest-valued half. The model is retrained after each addition or removal.
- Main figures report Gaussian-style confidence bands from 50 repetitions. No source/video clustered uncertainty or multiplicity-adjusted inferential claim is supplied.

## Main Results And Important Failures

1. Average synthetic-noise F1 is `0.458` for Beta(16,1), `0.451` for Beta(4,1), and `0.411` for Data Shapley. The winner varies by dataset; LOO-First or KNN wins several rows.
2. For 25% subsampling, Beta(16,1) and Beta(4,1) both average `0.757`, Data Shapley `0.749`, and random `0.722`. The numerical advantage is real in this setup but not universal across datasets.
3. Small-context Beta is most often best for noise detection, subsampling, and point addition.
4. Data Shapley, with uniform context weights, is slightly better for point removal. Large-context information matters when deleting from an already complete dataset.
5. LOO-First, which focuses almost exclusively on the smallest context, can fail because very small sets cannot train a stable classifier. More small-context emphasis is not monotonically better.
6. The paper itself concludes that the optimal weight can depend on the ML task and data distribution. It does not recommend one universal Beta pair.
7. These results support a context-size interaction, not a new Stage1 weighting coefficient. Repeated replay changes optimizer exposure rather than simply adding one identity to a retrained subset.

## Official Code Audit

- The repository has nine commits, eight tracked files, no release tag, no dependency lock, no tests, and no experiment manifest.
- The implementation uses deterministic scikit-learn models with hard-coded `random_state=666`; it does not test stochastic deep-training seed reversal.
- The paper states ten chains and a Gelman-Rubin threshold of `1.0005`. The example notebook changes the threshold to `1.05`; its saved run stops at 1,000 permutations with `R-hat=1.0065`, which would not pass the paper threshold.
- The convergence code reshapes one sequential Monte Carlo stream into ten contiguous arrays rather than running ten independently initialized chains. It checks an aggregate identity vector, not every cardinality-specific marginal used by extreme Beta weights.
- The smallest-context contribution `Delta_1 = U({z}) - U(empty)` is never recorded. The first identity initializes `old_score` and is skipped. This omission is especially important for a method designed to emphasize the smallest contexts.
- Fit failures and single-class prefixes are silently replaced by a random-guess utility, so the small-context signal partly reflects an undocumented failure policy.
- Per-cardinality sums are divided by a global mean observation count rather than a per-identity/per-cardinality completion mask. Truncation or unequal first-position counts can bias finite estimates.
- The code variable names `alpha` and `beta` are reversed when constructing the paper-facing label. The notebook passes `(1,16)` to report `Beta(16,1)`. This is internally intentional but easy to misuse.
- `beta_constant` only supports an integer second argument even though the paper defines positive real Beta parameters.
- The truncation expression is not the paper's stated marginal-increment convergence rule and can behave oppositely when the accumulated marginal sum is near zero.
- The class stores results only in memory. There is no atomic persistence, resume state, RNG snapshot, cardinality-completion manifest, or provenance lock.
- The balancing helper oversamples the full dataset before train/validation/test splitting. Exact duplicated minority rows can therefore cross split boundaries.
- The point-addition helper uses the first 5% of source IDs as its initial set, while the paper says the initial set is randomly selected.
- The public repository lacks the complete image preprocessing/experiment pipeline needed to reproduce the reported 15-dataset tables from a clean checkout.

These issues prevent using the code as a Stage1 estimator. They also make the paper's conceptual cardinality result more useful than its exact public implementation.

## Direct Support For Stage1

1. Preserve a context-scale profile instead of one combined value: replay ratio, selected-set size, local context size, and budget-specific sign/rank.
2. Analyze `0.5%`, `1.0%`, and `2.5%` as distinct interventions. Do not infer one ratio from another or choose a global weighting by visual preference.
3. Separate the downstream action. A score useful for adding examples need not be useful for deleting, weighting, or repeated replay.
4. Record whether a candidate's effect changes sign across budget, seed, checkpoint, and context. Report the full profile and uncertainty before any scalar summary.
5. Use trusted OOF/development tail probes for utility. The paper's noisy-validation setup is not suitable for the Stage1 weak-defect safety constraint.
6. Preserve the exact canonical hyperparameters. The observed scale effect must be tested through preregistered replay ratios and schedules, not by importing the paper's classifier or Beta parameters.

## What It Does Not Support

1. It does not establish that `Beta(16,1)` is a universal high-value score.
2. It does not identify the best Stage1 replay ratio or justify an arbitrary weighted formula.
3. It does not test 120,000 images, end-to-end `yolo11l`, repeated replay, weak-defect guards, raw FN frontiers, or no replay.
4. It does not explain cross-seed sign reversal under stochastic deep optimization.
5. It does not justify treating video frames as independent repetitions.
6. It does not permit changing batch, workers, optimizer, learning rate, augmentation, model, or epoch count.
7. It does not validate the public code's convergence threshold or smallest-context estimator.

## Transfer Boundary And Observable Consequence

The transferable hypothesis is:

```text
sample-set effect is a function of context scale and downstream action,
not a context-free scalar rank.
```

For Stage1, collect and report ratio-specific paired effects and budget-dependent rank/sign churn. The formal training experiment should remain a canonical-locked factorial comparison of replay ratio, timing, cumulative dose, and weak-defect protection. Beta-Shapley-style curves can be an offline diagnostic only if their context, missingness, and uncertainty are explicit.

## Decision

- Reading status: REPLICATION_DEPTH
- New formal arm: no
- New hyperparameter: no
- Canonical lock change: no
- Added fields: context cardinality, context-size regime, marginal profile by scale, scale-specific sign and uncertainty, downstream action identity, budget rank churn, small-context fit-failure count, convergence threshold, chain/RNG identity, and cardinality completion count
- Remaining uncertainty: whether low-cost context-scale diagnostics computed around canonical checkpoints predict replay-policy treatment effects better than static OOF ranks
