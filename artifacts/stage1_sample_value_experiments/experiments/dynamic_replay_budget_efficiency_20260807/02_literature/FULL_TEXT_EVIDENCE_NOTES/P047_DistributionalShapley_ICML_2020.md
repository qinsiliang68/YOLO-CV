# P047 - A Distributional Framework for Data Valuation

## Identity

- Paper ID: P047
- Authors: Amirata Ghorbani, Michael P. Kim, and James Zou
- Venue and year: ICML 2020, PMLR 119
- Official proceedings page: https://proceedings.mlr.press/v119/ghorbani20a.html
- Main paper: `source_papers/DistributionalShapley_ICML_2020.pdf`, SHA256 `77C35F4C1BF8743B83743E20A61582CBBE09AC7B3537AF5401AD9EF68C5631BF`
- Supplement: `source_papers/DistributionalShapley_ICML_2020_supplement.pdf`, SHA256 `2B252D8D0977A2B96AA6DA848B21A03FA17CC4E2296996C0967462891B6685E5`
- Official code: https://github.com/amiratag/DistributionalShapley
- Audited HEAD: `d6f67cdac0f8f50081c31cb8a7a7f85724eb4b9b`

## Reading And Audit Coverage

- Main paper: 10/10 pages read, including Definitions 2.1-2.6, Theorems 2.3, 2.7, 2.8, 3.1, and 3.2, Algorithms 1-2, all experiments, discussion, and references.
- Supplement: 16/16 pages read, including all omitted proofs, runtime/sample-complexity derivations, model-specific speed-up results, CIFAR-10 setup, and the four-dataset pricing case study.
- Visual verification: all 26 pages inspected at original detail under `audit/visual_checks/P047_DistributionalShapley_ICML_2020/` and `audit/visual_checks/P047_DistributionalShapley_ICML_2020_supplement/`.
- Code: repository structure, complete one-commit history, all three Python modules, all 19 notebook cells, saved notebook outputs, and relevant estimator, persistence, interpolation, metric, and model paths inspected. The TensorFlow 1.12 implementation was not executed.

## Research Question

Data Shapley values a point inside one fixed training set. This paper instead asks for the expected value of a point when the surrounding data set is drawn from an underlying distribution. The proposal removes dependence on one particular database draw, but it does not make value intrinsic to the image.

The value is explicitly indexed by:

```text
candidate z
potential U = learner plus evaluation functional
data distribution D
maximum training-set size m
```

For Stage1 this is closer to the observed conditional-value phenomenon than a static `V(x)`, while still omitting checkpoint state, replay schedule, optimizer path, and seed unless those are included inside `U`.

## Formal Definition

For a point `z`, potential `U`, distribution `D`, and size `m`, distributional Shapley is:

```text
nu(z; U, D, m)
  = E_{B ~ D^(m-1)}[phi(z; U, B U {z})]
```

Theorem 2.3 gives the equivalent expected marginal contribution:

```text
nu(z; U, D, m)
  = E_{k ~ Uniform{1,...,m}}
    E_{S ~ D^(k-1)}[U(S U {z}) - U(S)]
```

Thus the scalar averages over every context size from 0 through `m-1`. It can conceal a sign change between Stage1 replay ratios, just as ordinary Data Shapley can conceal coalition-size interactions. Changing `m`, `D`, or `U` changes the estimand.

The supplement proves an expectation form of efficiency:

```text
E_{z ~ D}[nu(z; U, D, m)]
  = (E_{B ~ D^m}[U(B)] - U(empty)) / m
```

This is an average allocation identity, not a guarantee that the highest-valued identities form the best replay subset.

## Stability Results And Their Assumptions

Deletion stability requires:

```text
abs(U(S U {z}) - U(S)) <= beta(k)
```

Lipschitz stability strengthens this by requiring the replacement effect to be bounded by `beta(k) * d(z,z')` for a chosen metric `d`.

Under Lipschitz stability, Theorem 2.7 bounds distribution shift:

```text
abs(nu(z; U, Ds, m) - nu(z; U, Dt, m))
  <= (2/m) * sum_{k=1}^{m-1} k * beta(k) * W1(Ds, Dt)
```

The theorem initially holds `U` fixed. The supplement adds a model-specific corollary for regularized RKHS empirical-risk minimization when `U` itself changes with the evaluation distribution. Its proof assumes a convex Lipschitz loss, explicit regularization, bounded expected feature norm, and the paper's constructed metric.

Theorem 2.8 states that similar points receive similar values only under the selected metric and Lipschitz-stability assumptions:

```text
abs(nu(z; U, D, m) - nu(z'; U, D, m))
  <= E_{k ~ Uniform[m]}[beta(k)] * d(z,z')
```

The RKHS example assigns infinite distance to points with different labels. Therefore this result cannot be used as evidence that arbitrary YOLO embeddings, cross-class weak defects, or adjacent video frames have similar value.

## Estimation Algorithms

Algorithm 1 samples `k` uniformly, samples an i.i.d. context `S ~ D^(k-1)`, evaluates the marginal contribution for every candidate, and averages over iterations. Theorem 3.1 gives an absolute additive-error concentration rate proportional to `log(|Z|/delta)/epsilon^2`. The paper explicitly notes that values commonly scale as `O(1/m)`, so fixed absolute error becomes progressively worse in relative terms as `m` grows.

Fast D-Shapley combines two approximations:

1. evaluate a random subset of candidates and regress/interpolate values for the rest;
2. sample context sizes non-uniformly according to stability and importance-reweight the marginal contributions.

The guarantees depend on a valid stability rate and unbiased reweighting. The method still requires repeated model fits and fresh samples from `D`; the finite-database bound relies on i.i.d. reuse of database samples. A 120,000-image canonical `yolo11l` valuation is therefore not made practical by the theorem alone.

## Experimental Contract

- UK Biobank: 10,000 patients, 9,000-point database, 500-point holdout, 120 features, balanced 5,000 positive cases, logistic-regression breast-cancer task, and `m=1,000`.
- Adult Income: about 50,000 identities, 40,000-point database, 5,000-point evaluation set, 14 features, logistic-regression income task, and `m=5,000`.
- Both main speed-up studies stop when average absolute value change over the previous 100 iterations is below 1%. The paper varies computational cost and reports point-removal curves and value-recovery `R^2`.
- CIFAR-10: values for 50,000 images using an ImageNet-pretrained Inception-v3 with every layer except the last frozen; biased sampling gives a factor-10 speed-up and interpolation a factor-50 speed-up. Only 1,000 identities are directly valuated. Removing the top 50% by estimated value reduces reported accuracy from 77% to 68%.
- Pricing study: CoverType with Random Forest, Diabetes130 with AdaBoost, Postures with multinomial logistic regression, and Sensorless with Gradient Boosting. Buyers and sellers each hold 100 or 500 points.
- Some seller-versus-buyer rank correlations are only about 0.6 even when aggregate value error is low. This is direct evidence that set-level agreement does not imply a reliable identity ranking.
- The paper reports no cross-seed success probability, no paired no-replay control, no replay schedule, no weak-defect constraint, and no raw FN safety frontier.

## Main Results And Negative Evidence

1. Averaging over database draws can reduce fixed-dataset sampling artifacts, but value remains conditional on `U`, `D`, and `m`.
2. Distributional value is an expectation over contexts, not a realized guarantee for one seed or one optimizer path.
3. The empirical speed-up curves are smooth even for 0-1 accuracy, where the formal stability guarantees do not necessarily hold. The authors present this as empirical behavior, not a theorem.
4. Point-removal and point-addition rankings generally beat random in the studied tabular and frozen-feature settings, but these interventions differ from repeated replay under non-convex end-to-end training.
5. A scalar value averages all context sizes uniformly. It does not establish one stable ranking across Stage1 ratios `0.5%`, `1.0%`, and `2.5%`.
6. Sampling frames i.i.d. from an empirical database ignores video/source dependence and can overstate the effective sample size of near-duplicate frames.
7. Distribution averaging may reduce variance while washing out a rare subgroup whose business value lies specifically in the weakest defect tail.
8. The paper's explicit limitation is reliance on a known, fixed task, algorithm, and metric. Stage1 must therefore freeze the canonical learner to preserve the meaning of any comparison.

## Official Code Audit

- The repository has one commit, no release tag, no dependency lock, no tests, and targets TensorFlow 1.12.
- `directory=None` is accepted by the constructor but skips instance initialization and then accesses missing attributes.
- A pre-existing `data.pkl` silently replaces newly supplied arrays, without hashes or identity checks.
- Result filenames are allocated by scanning existing files, which races across workers. Pickle writes are non-atomic, and merge deletes inputs after writing without a verified durable sidecar.
- `dist_iteration` returns all-zero contributions when `k=1`, although the paper requires `U({z}) - U(empty)`. This makes the estimator biased.
- Fit failures and missing-class contexts are silently represented as zero contributions rather than failed observations with a completion mask. Later statistics look for `-1`, so the missingness convention is internally inconsistent.
- The importance-sampling notebook reconstructs context size using `len(idxs_dist)=k-1`, then applies a power weight intended for `k`. The `k=1` record has length zero, creating an off-by-one and zero-power problem.
- Group/source iteration assumes integer keys `0..n-1`; arbitrary source dictionaries can fail.
- Non-null sample weights in TMC use `sample_weight_batch` before initialization.
- Global NumPy/TensorFlow seeds coexist with hard-coded estimator seeds of 666; no complete RNG, environment, or resume state is persisted.
- Multiple branches use string identity (`is`) rather than equality. F1 references undefined `x`, and binary AUC scores the probability of each observation's true class instead of positive-class probability.
- Neural early stopping references undefined `val_acc`; batch metrics average batches equally instead of weighting by batch size.
- The helper interpolation paths rely on undefined globals such as `truncation` and `init`; the public class does not provide a durable end-to-end Algorithm 2 pipeline.
- The example notebook has nonsequential execution counts. Its saved output gives only `0.437` rank correlation and `11.6%` aggregate percentage error for the demonstrated 100-point split, which is a useful reminder that aggregate agreement can coexist with weak identity ordering.

These implementation defects do not refute the paper's mathematical definition or reported experiments. They rule out importing the repository as Stage1 production code.

## Direct Support For Stage1

1. Replace any context-free value label with a conditional record indexed by canonical config hash, checkpoint/state, replay ratio, schedule, cumulative dose, seed, source distribution, context composition, and development utility.
2. Treat `0.5%`, `1.0%`, and `2.5%` as separate context-size regimes. Record nested-set churn and effects separately rather than assuming one global ranking.
3. Keep normal-tail suppression and weak-defect protection as separate utilities and constraints. Do not hide them in an arbitrary weighted scalar.
4. Estimate uncertainty with source/video-group resampling or a hierarchical distribution; naive frame-i.i.d. intervals are not credible for repeated video frames.
5. Keep discovery on OOF/development identities. The blind holdout must not define `D`, train the value regressor, tune stopping, or select an arm.
6. Preserve the old canonical hyperparameters exactly. In the paper's notation, changing the learner or metric changes `U` and therefore changes the value being estimated.

## What It Does Not Support

1. It does not support an intrinsic image score or a universal top-k replay list.
2. It does not prove cross-seed sign stability or explain one realized Stage1 seed reversal.
3. It does not justify i.i.d. frame sampling for correlated videos.
4. It does not show that a last-layer frozen-feature result transfers to end-to-end `yolo11l` training.
5. It does not test replay timing, cumulative exposure, dynamic decay, defect guards, or no replay.
6. It does not justify importing any paper optimizer, model, batch size, epoch count, sampling alpha, stopping tolerance, or interpolation setting.
7. It does not establish that its public code is an unbiased or resumable implementation of the published algorithm.

## Transfer Boundary And Observable Consequence

Distributional Shapley is useful as a definition-level warning:

```text
value is conditional on U, D, and m,
and realized replay benefit is additionally conditional on
theta_t, optimizer path, schedule, dose, context, and seed.
```

The low-cost Stage1 consequence is not a new Shapley training arm. Add distribution/context fields and test whether paired treatment effects vary with source/video composition, replay ratio, and checkpoint state. Any optional valuation diagnostic should use group-resampled development utilities and report identity rank uncertainty separately from aggregate set utility.

## Decision

- Reading status: FULL_READ_COMPLETE
- New formal arm: no
- New hyperparameter: no
- Canonical lock change: no; the learner is part of `U`
- Added fields: source-distribution identity, empirical-distribution version, source/video group, maximum context size, realized context-size draw, context composition, distribution-shift measure, group-bootstrap identity, value uncertainty, aggregate-versus-rank agreement, and approximation missingness
- Remaining uncertainty: whether a source-aware conditional group diagnostic can predict paired canonical replay effects better than static OOF ranks without requiring infeasible retraining
