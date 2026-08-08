# P046 - Data Shapley: Equitable Valuation of Data for Machine Learning

## Identity

- Paper ID: P046
- Authors: Amirata Ghorbani and James Zou
- Venue and year: ICML 2019, PMLR 97
- Official proceedings page: https://proceedings.mlr.press/v97/ghorbani19c.html
- Main paper: `source_papers/DataShapley_ICML_2019.pdf`, SHA256 `48979F27F610C834FFDA28AB48194C1AF13C453D32ADD965FC817BBA1E9B4407`
- Supplement: `source_papers/DataShapley_ICML_2019_supplement.pdf`, SHA256 `55E67B04B876E5AE32F86DE0F2DDAF04B5181EA48DCE24411E15FC80DF8E0121`
- Official code: https://github.com/amiratag/DataShapley
- Audited HEAD: `303d91d988a149948fb357ac82dc72af1bc7430d`

## Reading And Audit Coverage

- Main paper: 10/10 pages read, including Proposition 2.1, Equations 1-2, Algorithms 1-2, Figures 1-5, every experimental setting, discussion, and references.
- Supplement: 3/3 pages read, including convergence, truncation, exact-versus-TMC comparisons, and G-Shapley-versus-TMC correlations.
- Visual verification: all 13 pages inspected at original detail under `audit/visual_checks/P046_DataShapley_ICML_2019/` and `audit/visual_checks/P046_DataShapley_ICML_2019_supplement/`.
- Code: all tracked Python files and all 19 notebook cells inspected; complete one-commit repository history and file hashes recorded. The legacy TensorFlow 1.12 implementation was not executed.

## Research Question

The paper defines equitable data value for a fixed triple:

```text
training data D
learning algorithm A
performance functional V
```

Its notation is explicitly:

```text
phi_i(D, A, V)
```

The paper states that there is no universal value for a datum. Value changes with the learner, performance metric, evaluation population, and the other training data. This directly supports the Stage1 conclusion that value is conditional, but it defines value over retrained subsets rather than over one observed OOF trajectory.

## Formal Definition

For datum `i`, Data Shapley averages its marginal contribution over all coalitions that do not contain it:

```text
phi_i = E_permutation[
  V(predecessors_i U {i}) - V(predecessors_i)
]
```

Equivalent subset form:

```text
phi_i = C * sum_{S subset D\{i}}
  [V(S U {i}) - V(S)] / choose(n-1, |S|)
```

The three axioms used are null-player, symmetry, and additivity of performance functionals. They uniquely determine this allocation up to scale. The theorem establishes an equitable allocation under those axioms; it does not prove that sorting individual Shapley values and taking top-k is the optimal interacting subset.

The value averages marginal effects across every coalition size. A point can have a positive effect at one replay budget and a negative effect at another while receiving one average scalar. This matters for Stage1, where `0.5%`, `1.0%`, and `2.5%` are distinct interventions.

## Approximation Methods

### TMC-Shapley

Sample random permutations, retrain on growing prefixes, and average each identity's change in `V`. Truncation stops evaluating the tail of a permutation when the current score is sufficiently close to the full-data score.

The paper says convergence usually occurs around `3n` Monte Carlo permutations and defines a relative change criterion below `0.05`. These are empirical implementation choices, not transferable Stage1 guarantees. At 120,000 images, prefix retraining across this scale is computationally infeasible for canonical `yolo11l`.

### Gradient Shapley

G-Shapley replaces full prefix retraining with one single-example SGD pass through each sampled permutation. The paper explicitly tunes a separate larger learning rate for this one-pass surrogate. It is therefore a different learning algorithm from the canonical multi-epoch Stage1 trainer and only an approximation to the requested `phi_i(D,A,V)`.

### Group Shapley

The same allocation can be applied to groups. This suggests cluster/video/source-level diagnostics, but grouping changes the players and does not recover individual value.

## Experimental Contract

- Every main experiment separates a valuation set used for `V` from a held-out set used for final reporting.
- UK Biobank: 1,000 training identities, 1,000 valuation identities, balanced binary logistic regression; TMC converged at 4,000 permutations and G-Shapley at 1,500 in the reported runs.
- Synthetic study: 20 generated datasets per setting, train sizes 100 and 1,000, linear versus cubic relationships, logistic regression versus a one-hidden-layer network.
- Label-noise studies: 3,000 spam examples with 20% flips; 1,000 flower embeddings with 10% flips; 1,000 Fashion-MNIST images with 10% flips.
- Image-quality study: 100 train and 1,000 valuation images with 10% corrupted by increasing white noise.
- Group study: 60,000 training patients grouped into 146 demographic intersections.
- Compute: most valuations below 24 hours on four 4-CPU machines; one ConvNet study used four GPUs for 120 hours.
- The supplement reports TMC versus exact Shapley Pearson correlation `98.4%-99.5%` only for synthetic logistic problems with 4-14 points.
- G-Shapley versus TMC correlation falls to `0.57` for flowers and `0.62` for Fashion-MNIST label-flip tasks, showing substantial approximation disagreement in the most relevant image/noise cells.

## Main Results And Negative Evidence

1. Removing high-Shapley points generally hurts the studied model faster than random removal; removing low-Shapley points can improve it.
2. Low values enrich synthetic label flips and increasingly noisy images, but this is not proof that every low-value hard Stage1 sample is noise.
3. Similar-looking acquisitions predicted from high-value points can help in the two disease tasks, while predicted low-value acquisitions can hurt.
4. The same synthetic identity can be useful for a nonlinear model and harmful for logistic regression. This is direct empirical evidence against intrinsic image value.
5. A single Shapley scalar depends on a chosen scalar `V`. Accuracy in the paper does not encode Stage1's asymmetric requirement to suppress hard normals without harming the weakest defects.
6. The framework allows dependent data mathematically, but the reported Monte Carlo uncertainty and experiments do not address repeated video frames or source-level dependence.
7. Truncation can preserve approximate rank in the studied synthetic cases, but 25% truncation only achieves rank correlation around 0.8 in the supplement. It is not lossless.
8. The authors explicitly caution that their three axioms may be inappropriate in some ML settings and that all high/low-value claims assume the full context is fixed.

## Official Code Audit

- The repository has one commit, no tag, no dependency lock, no test suite, and targets Python, NumPy, TensorFlow 1.12, and scikit-learn.
- `directory=None` is advertised as a default but skips data initialization and leaves required attributes unset.
- Existing `data.pkl` silently overrides newly supplied arrays without content hashes or an identity check.
- Parallel worker numbering scans filenames and chooses `max+1` without a lock; simultaneous workers can claim the same output name.
- Pickle result writes are non-atomic. Merge deletes worker files before the final merged file is durably written, so interruption can lose evidence.
- Only global NumPy and TensorFlow seeds are set from the caller; many sklearn constructors hardcode `random_state=666`, and full RNG/environment state is not persisted.
- String identity comparisons use `is` instead of equality in multiple branches.
- The automatic truncation path assigns `self.tol` but later reads `self.tolerance`; callers normally pass a ratio instead, which differs from the paper's bootstrap-tolerance description.
- TMC requires six consecutive near-full scores before truncating, unlike the one-condition pseudocode. Unvisited identities remain represented as zeros without a completion mask.
- G-Shapley tunes its learning rate on the valuation set, then records only differences between post-update scores. It omits the first identity's contribution relative to the random initialization in every permutation.
- Neural-network early stopping references undefined `val_acc`; batch scores average batches equally rather than weighting by batch size.
- `my_f1_score` references undefined `x`; binary AUC uses probability assigned to each true class rather than the positive-class probability.
- The legacy helper `shapley()` calls an undefined `one_pass` and its running-average update omits the previous-iteration multiplier.
- The example notebook is mostly unexecuted and does not provide a clean end-to-end reproduction record.

These defects do not invalidate the conditional-value concept or the paper's reported experiments. They prevent importing the code as a reliable Stage1 valuation pipeline.

## Direct Support For Stage1

1. Define any value claim with its context: data version, canonical learner hash, checkpoint/state, replay policy, selection budget, and evaluation functional.
2. Do not build one arbitrary weighted score. Estimate separate marginal effects on normal-tail suppression and weak-defect protection, then report a joint admissibility rule.
3. Use only development/calibration identities to define `V`; reserve the blind holdout for final evaluation.
4. Treat `0.5%`, `1.0%`, and `2.5%` as separate coalition/budget regimes. Record set churn and interaction rather than assuming one global rank transfers.
5. Group-level valuation by video/cluster may be a feasible diagnostic for redundancy, but it must be validated against identity-level and tail-specific outcomes.
6. If a finite intervention diagnostic is computed, preserve permutation/batch/context identity and uncertainty. One gradient step is not canonical retraining.
7. Keep the exact old hyperparameters. Changing the learner changes `A` and therefore changes the quantity being called value.

## What It Does Not Support

1. It does not support a context-free scalar `V(x)`.
2. It does not establish a universal top-k replay ranking or a stable high-value percentage.
3. It does not justify averaging normal benefit and defect harm with invented coefficients.
4. It does not show that G-Shapley is accurate enough for Stage1 images; the relevant image/noise correlations with TMC are only `0.57-0.62`.
5. It does not support using test/blind identities to select replay samples.
6. It does not permit a one-pass tuned learning rate or any change to the canonical optimizer, batch, augmentation, model, or epoch settings.
7. It does not evaluate repeated replay exposure, timing schedules, weak-defect guards, raw FN frontiers, or cross-seed sign reversal.

## Transfer Boundary And Observable Consequence

Use Shapley as a conceptual definition and interaction warning, not as the next 120,000-image production algorithm:

```text
Value must be indexed by:
  learner/config hash
  model/checkpoint state
  development utility
  selected context/set
  replay ratio
  replay schedule and cumulative dose
  seed/machine/RNG block
```

For a limited diagnostic, compare finite marginal effects of small candidate groups on two separate development utilities:

```text
U_normal = hard-normal tail suppression
U_defect = weak-defect tail protection
```

An admissible candidate must improve `U_normal` without violating the preregistered `U_defect` constraint. Do not collapse them into one coefficient-weighted scalar. The formal campaign remains a controlled replay-policy experiment under the canonical lock.

## Decision

- Reading status: REPLICATION_DEPTH
- New formal arm: no
- New hyperparameter: no
- Canonical lock change: no; Data Shapley makes the learner part of the value definition
- Added fields: value-context identity, evaluation-functional identity, budget/coalition size, interaction context, approximation method, Monte Carlo uncertainty, and development-versus-blind role
- Remaining uncertainty: whether low-cost finite group interventions can predict canonical full-run tail-safe effects without changing the learner or leaking the final holdout
