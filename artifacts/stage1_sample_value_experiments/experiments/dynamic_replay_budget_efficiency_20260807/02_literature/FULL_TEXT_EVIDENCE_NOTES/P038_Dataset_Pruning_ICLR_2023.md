# P038 - Dataset Pruning By Generalization Influence

## Identity

- Paper ID: P038
- Full title: Dataset Pruning: Reducing Training Data by Examining Generalization Influence
- Authors: Shuo Yang, Zeke Xie, Hanyu Peng, Min Xu, Mingming Sun, and Ping Li
- Venue and year: ICLR 2023
- Official conference page: https://openreview.net/forum?id=4wZiAXD29TQ
- Full-text version read: arXiv `2205.09329v2`, revised 2023-02-27 and marked as the ICLR 2023 conference paper
- Local PDF: `source_papers/Dataset_Pruning_ICLR_2023.pdf`, SHA256 `BD66704C09AC2C7FD693E6FDBE41E1C05C984924C59B9E75AAC74933EAD36CE0`
- Verifiable official code: none found

## Reading Coverage

- Full paper: 13/13 pages read, including Definitions 1, Equations 1-14, Algorithm 1, Theorem 1 and proof, Figures 1-4, Table 1, all experimental settings, cross-architecture experiment, NAS proxy experiment, conclusion, and references.
- Visual verification: all 13 pages inspected at original detail under `audit/visual_checks/P038_DatasetPruning_ICLR_2023/`. The Poppler renderer emitted a Symbol-font warning, but every equation, plot, table, and reference page remained visible and legible.
- Version check: arXiv records v1 on 2022-05-19 and v2 on 2023-02-27. The v2 PDF carries the ICLR 2023 publication header and is the evidence asset.
- Peer-review limitation: the OpenReview page and both API endpoints returned access challenges or HTTP 403. Reviewer text was not available and no reviewer claim is used.
- Code limitation: the submission metadata certified an anonymous URL during review but did not expose one, the final paper names no repository, and no author-owned implementation for this paper was found. No unofficial reimplementation is treated as source evidence.

## Research Question

The paper asks which subset can be removed while keeping the retrained model close to the full-data model. It argues that individual scalar scores miss group cancellation: two samples can each have a large influence norm while their influence vectors sum to nearly zero.

Stage1 asks a different intervention: repeatedly replay a labeled set during one training trajectory and improve an FN-constrained two-tail objective. The paper is directly useful for showing that sample effects interact and that direction matters. It does not establish replay value, temporal scheduling, weak-defect protection, or raw-frontier safety.

## Core Definitions And Mathematics

The desired pruned subset is first defined through the parameter distance between two separately optimized models:

```text
epsilon-redundant(D_hat):
    ||theta_hat_(D without D_hat) - theta_hat_D||_2 <= epsilon
```

For a training sample `z`, infinitesimal upweighting gives the parameter influence:

```text
I_param(z) = -H(theta_hat)^(-1) grad_theta L(z, theta_hat)
```

Removing one point is approximated by multiplying this derivative by `-1/n`. Removing a subset is then approximated by summing individual changes:

```text
theta_hat_(D without D_hat) - theta_hat_D
    approximately sum_(z in D_hat) (1/n) H^(-1) grad L(z)
```

The fixed-cardinality optimization chooses a binary removal vector `W` that minimizes the norm of the aggregate influence. This is the central transferable idea:

```text
minimize_W ||W^T S||_2
subject to sum_i W_i = m
```

where each row of `S` is a vector, not a scalar norm. Opposing vectors can cancel. Therefore, the marginal effect of one identity depends on the other identities in the selected set.

The paper then derives a first-order expected-loss expression and states a bound of order:

```text
O(epsilon / n + m / n^2)
```

This is not a distribution-free replay guarantee. It relies on local Taylor expansion, an empirical-risk minimizer, a positive-definite Hessian, a linear sum of infinitesimal leave-out influences, and an implicit bound on the expected test gradient.

## Important Mathematical Ambiguity

The epsilon scaling is inconsistent across the algorithm and theorem:

- Section 4 and Algorithm 1 constrain `||sum (1/n) I_param(z)|| <= epsilon`.
- Theorem 1 states `||sum I_param(z)|| <= epsilon` and derives an `epsilon/n` term.
- Figure 2 is described as comparing the observed gap with `epsilon/n`, without disambiguating which epsilon was computed.

If the algorithmic epsilon is defined after the `1/n` scaling, the theorem's epsilon is `n` times larger. The notation can be repaired algebraically by renaming the two quantities, but the paper does not do so. Stage1 must never use an influence threshold unless every scaling factor, target gradient, averaging convention, and unit is explicit in the collector contract.

The proof also writes a supremum without defining its domain and absorbs the expected test-gradient norm into big-O without stating a uniform bound. The second-order remainder is represented by `m/n^2` from the perturbation vector norm, but no empirical finite-removal error bar shows how the local approximation degrades with subset size. These are transfer limits, not proof that the group-cancellation observation is false.

## Experimental Protocol

- Datasets: CIFAR-10, CIFAR-100, and TinyImageNet.
- Architectures: SqueezeNet, ResNet-18, and ResNet-50; most pruning comparisons use a randomly initialized ResNet-50.
- Training: 200 epochs, batch 128, learning rate 0.01 with cosine annealing, SGD momentum 0.9, weight decay `5e-4`, random crop, and random horizontal flip.
- Influence computation: last linear layer only, with an approximate inverse-Hessian method.
- Subset optimization: simulated annealing under an unspecified time budget and unspecified annealing schedule.
- Baselines: random, herding, forgetting, GraNd, EL2N, and individual influence norm.
- Evaluation: retrain a new randomly initialized ResNet-50 on each retained subset.
- Architecture transfer: select with SENet and retrain ResNet-18 or ResNet-50.
- NAS proxy: train 720 candidate ConvNets on a 2% proxy set; compare rank correlation and final selected-architecture performance.
- No seed count, random-seed identities, confidence intervals, standard deviations, or error bars are reported for Figures 1-4 or Table 1.
- No independent validation protocol is specified for selecting pruning thresholds, simulated-annealing settings, checkpoints, or method hyperparameters.
- These settings are literature context only. They may not alter the Stage1 canonical `yolo11l` hyperparameter lock.

## Direct Support

1. Gradient or influence norm alone discards direction. Two individually large effects can cancel, while a set of modest aligned effects can accumulate.
2. Value is at least set-conditioned. The relevant object is an aggregate update or finite intervention, not a sorted list of isolated magnitudes.
3. Cross-architecture transfer is possible in this benchmark, so a diagnostic embedding need not be perfectly architecture-specific. The evidence is descriptive because uncertainty is absent.
4. Last-layer approximations can be computationally practical, which supports bounded Stage1 key-checkpoint diagnostics.
5. A finite retrain test is still needed because the method itself evaluates final retrained subsets after constructing them from a local proxy.

## Non-Support And Negative Evidence

1. Removing data and retraining from scratch is not equivalent to replaying data repeatedly inside an existing optimizer path.
2. Parameter proximity does not imply Stage1's asymmetric tail behavior. Two models can be close in global parameter norm yet cross an extreme operating threshold for particular weak defects.
3. The objective preserves average expected loss, not `TN_at_FN95`, `FN_at_TN68253`, or the raw `FN=0..95` safety frontier.
4. Group influence is estimated by adding infinitesimal individual influences. Large finite subsets can violate the local linear approximation.
5. Positive-definite invertible Hessian assumptions are fragile for overparameterized neural networks; only the last layer is used in practice.
6. The epsilon scaling ambiguity and unstated gradient/remainder constants prevent direct numeric transfer.
7. Simulated annealing adds stochastic optimization variance, but its seed, schedule, budget, convergence, and repeated-run stability are unreported.
8. The paper has no same-selection cross-seed analysis and no uncertainty estimates.
9. Class balance, video/group dependence, label noise, and protected-tail identity are not encoded.
10. No public implementation or exact experiment manifest was found.

## Stage1 Transfer Boundary

The valid transfer is a diagnostic decomposition, not a new arm:

```text
candidate_gradient_or_influence_vector
aggregate_vector_of_selected_set
sum_of_individual_norms
norm_of_vector_sum
cancellation_ratio
pairwise_cosine_distribution
within_role_and_cross_role_alignment
weak_defect_violation_count
finite_one_step_and_short_horizon_effect
approximation_residual
state_checkpoint_hash
selection_set_hash
subset_optimizer_seed_and_budget
```

A useful cancellation statistic is descriptive:

```text
cancellation_ratio = ||sum_i v_i|| / sum_i ||v_i||
```

Values near zero mean strong cancellation; values near one mean aligned accumulation. It is not a value score because a small aggregate can mean either harmless redundancy or cancellation between one strongly helpful and one strongly harmful sample. The numerator must be computed separately for difficult-normal and weak-defect target directions.

## Current Experiment Consequence

- Add vector-sum, sum-of-norms, cancellation, pairwise-sign, and finite-approximation-residual fields at key checkpoints.
- Keep difficult-normal correction and weak-defect protection as separate signed targets; do not collapse them into global parameter distance.
- Calibrate any local influence or gradient proxy against actual one-step and short-horizon replay interventions at the same checkpoint.
- Keep the first causal timing/dose block unchanged: no replay, continuous replay, same-peak decay, and cumulative-dose-matched decay on one frozen selection.
- Do not add a Dataset-Pruning arm and do not import its epsilon, pruning ratios, simulated annealing, optimizer, or augmentation.
- Do not change any Stage1 canonical training hyperparameter.

## Bottom Line

P038 gives direct mathematical support for the claim that isolated gradient magnitude cannot define sample value because vector effects interact and cancel. Its own guarantee is too local, ambiguously scaled, average-loss-oriented, and empirically under-replicated to guarantee Stage1 tail safety. The correct use is to collect set-level vector interaction and approximation-error fields, then test them against finite replay outcomes under the fixed canonical experiment.
