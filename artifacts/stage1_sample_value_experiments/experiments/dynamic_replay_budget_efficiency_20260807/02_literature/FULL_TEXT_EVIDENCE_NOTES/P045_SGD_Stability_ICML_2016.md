# P045 - Train Faster, Generalize Better: Stability of Stochastic Gradient Descent

## Identity

- Paper ID: P045
- Authors: Moritz Hardt, Benjamin Recht, and Yoram Singer
- Venue and year: ICML 2016, PMLR 48
- Official proceedings page: https://proceedings.mlr.press/v48/hardt16.html
- Conference paper: `source_papers/SGD_Stability_ICML_2016.pdf`, SHA256 `E50B29E0F621998864CF8A403A3B5CB0D417264B9D98AA7B085B3BF051A11FB6`
- Author full version: `source_papers/SGD_Stability_ICML_2016_full_arxiv.pdf`, SHA256 `F25C44207C24F6A940EB92E06CB36BC53ED4C57A7632850BE1B78B334022749E`
- Official experiment code: none located on the PMLR page, arXiv record, or Google Research publication page.

## Reading Coverage

- Conference paper: 10/10 pages read, including Definitions 2.1-3.5, Lemmas 2.4 and 3.3-4.4, Theorems 2.2/3.7/3.8/5.2, Proposition 5.4, all three figures, settings, references, and stated limitations.
- Full version: 32/32 pages read, including the omitted strongly-convex proofs, projected updates, proximal steps, iterate averaging, all ten figures, full experimental details, future-work limitations, and appendix proofs.
- Visual verification: all 42 conference/full-version pages inspected at original detail under `audit/visual_checks/P045_SGD_Stability_ICML_2016/` and `audit/visual_checks/P045_SGD_Stability_ICML_2016_full_arxiv/`.
- Source verification: official PMLR proceedings, arXiv full version, and Google Research publication identity checked.

## Research Question

The paper asks how changing one training identity changes the output of a stochastic-gradient learning algorithm and how that algorithmic stability relates to expected generalization. Its object is a learning procedure, not an intrinsic score attached to a photograph.

This is directly relevant to Stage1 because replay deliberately changes identity frequency. The same replay set can have a different effect when its first exposure, subsequent repetitions, learning-rate path, and surrounding batches differ. The paper does not show that any particular difficult sample is useful, nor does it evaluate a constrained low-FN safety frontier.

## Formal Model

For data sets `S` and `S'` differing in at most one example, randomized algorithm `A` is uniformly stable when:

```text
sup_z E_A[f(A(S); z) - f(A(S'); z)] <= epsilon
```

Uniform stability bounds expected generalization. The proof tracks the recursive parameter separation:

```text
delta_t = ||w_t - w'_t||
```

When the differing identity is not sampled, separation evolves through the expansiveness of the common update. When it is sampled, separation can increase by a term proportional to the step size and gradient bound.

For smooth convex losses with admissible step sizes, Theorem 3.8 in the full version gives:

```text
epsilon_stab <= (2 L^2 / n) * sum_t alpha_t
```

For bounded smooth non-convex losses with monotonically decreasing `alpha_t <= c/t`, Theorem 3.12 gives an expectation bound whose leading dependence is:

```text
epsilon_stab approximately T^(1 - 1/(beta*c + 1)) / n
```

The exact constants require global boundedness, Lipschitzness, smoothness, i.i.d. samples, and the paper's random-index or random-permutation SGM model. These assumptions do not hold exactly for modern YOLO training with augmentation, momentum-like optimizer state, batch normalization, video dependence, and replay duplication.

The risk decomposition is:

```text
expected population risk
  <= expected minimum empirical risk
   + optimization error(T)
   + stability error(T)
```

More steps may reduce optimization error while increasing instability. This is a process tradeoff, not a static sample-value formula.

## Experimental Contract

- Data/model pairs: CIFAR-10 with a three-convolution cuda-convnet-like model, MNIST with a LeNet-like model, ImageNet with AlexNet, and Penn Treebank with a two-layer LSTM.
- Perturbation: remove a random training example, replace one identity to create `S'`, and train `S` and `S'` with the same random seed.
- Observations: per-layer Euclidean parameter distance every 100 SGM updates; train/test error once per epoch.
- CIFAR-10 intentionally omits regularization and augmentation and uses constant step sizes to isolate the mechanism; it is not a competitive training recipe.
- ImageNet early-versus-late substitution tests where within the epoch the differing identity first enters the path.
- The full version reports error bars but does not state a complete independent-run count or machine/software provenance contract.
- No executable official experiment repository, checkpoint contract, or resume-equivalence test was located.

Every paper setting is evidence context only. None may change the Stage1 canonical hyperparameters.

## Main Results And Negative Evidence

1. Training on data sets differing by one identity produces path-dependent parameter divergence even under a shared seed.
2. Larger step sizes generally produce larger generalization gaps in the reported controlled experiments.
3. Parameter distance grows sublinearly in the tested networks, while the non-convex theory can be exponentially pessimistic. The bound is therefore not a calibrated predictor of Stage1 harm.
4. Parameter distance and train-test generalization gap often move together, but parameter distance is explicitly only a stronger proxy, not a necessary characterization of stability.
5. Early substitution in ImageNet causes much larger later parameter divergence than late substitution. A perturbation's effect depends on how many and which updates follow it.
6. This early-substitution result does not prove that late Stage1 replay is harmless. Stage1 repeats selected identities and measures two protected tails, while this experiment changes one identity once per paired data set.
7. The full version states that its results are in expectation and that useful high-probability bounds remain open.
8. It also states that momentum's stability effect is unclear, non-convex full gradient descent is not uniformly stable under the same argument, and the work offers no recipe for simultaneously obtaining low training error and stable training.
9. Weight decay, clipping, dropout, projections, and iterate averaging improve specific theoretical bounds, but those are mathematical analyses under paper assumptions, not permission to alter the frozen Stage1 recipe.

## Direct Support For Stage1

1. Treat sample influence as a trajectory intervention. Record first replay epoch, last replay epoch, cumulative appearances, exposure after each checkpoint, and the number of optimizer updates remaining after each exposure.
2. Record realized learning rate and optimizer-step index for every exposure because the local perturbation scale is step-size dependent.
3. Compare identical selections under continuous, same-peak decay, and dose-matched decay schedules. The schedule and total dose must be separated causally.
4. Use paired seeds and, where feasible, paired base order/augmentation streams so that path divergence is attributable to replay policy rather than unrelated randomness.
5. At saved checkpoints, measure prediction-space and protected-tail divergence in addition to parameter or gradient distance. A parameter norm alone is not the business endpoint.
6. Keep an identity-level exposure ledger so that resume, sampler behavior, or finite rounding cannot silently change the treatment.
7. Preserve the exact canonical hyperparameters. Stability is itself step-size, optimizer-path, regularization, and iteration dependent, so an arm-specific training change would confound the intervention.

## What It Does Not Support

1. It does not support calling high-gradient, high-loss, or frequently replayed images valuable.
2. It does not establish that stopping replay after epoch 160 improves Stage1. Its early-substitution result actually warns that earlier perturbations can have more time to propagate.
3. It does not justify changing learning rate, weight decay, dropout, clipping, optimizer, batch, image size, augmentation, or epoch count.
4. It does not cover Adam/auto optimizer state, mini-batch interaction, batch normalization, dependent video frames, class imbalance, label noise, or quantile-constrained tail metrics.
5. Its expected uniform-stability bound is not a seed-level success probability, not a confidence interval, and not a per-sample ranking.
6. Its one-identity replacement experiment does not identify the effect of repeatedly oversampling a selected set.

## Transfer Boundary And Observable Consequence

Add process fields rather than a new formal arm:

```text
identity_first_exposure_epoch
identity_last_exposure_epoch
identity_cumulative_exposure_by_epoch
optimizer_step_at_exposure
learning_rate_at_exposure
remaining_optimizer_steps_after_exposure
paired_prediction_divergence_by_checkpoint
normal_tail_divergence
weak_defect_tail_divergence
resume_exposure_equivalence
```

The observable Stage1 question is not simply whether late replay is harmful. It is whether, under the same canonical settings and same selected identities, redistributing a fixed cumulative dose changes protected-tail trajectories and cross-seed sign reversals. If dose-matched decay beats continuous replay, timing matters beyond total exposure. If only lower-total-dose decay wins, excessive dose is the more credible mechanism. If neither wins, the late-exposure hypothesis is rejected for that selection and ratio.

## Decision

- Reading status: FULL_READ_COMPLETE
- New formal arm: no; it strengthens the existing continuous/same-peak-decay/dose-matched-decay contrast
- New hyperparameter: no
- Canonical lock change: no
- Added fields: first/last exposure, cumulative identity exposure, update and learning-rate position, remaining updates, paired path/prediction/tail divergence, and resume exposure equivalence
- Remaining uncertainty: whether Stage1 reversals are driven mainly by cumulative dose, temporal placement, nonlinear set interactions, or protected-tail gradient conflict
