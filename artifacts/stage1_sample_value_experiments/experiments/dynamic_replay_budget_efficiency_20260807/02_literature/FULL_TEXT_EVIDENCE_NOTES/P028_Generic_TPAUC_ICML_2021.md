# P028 - When All We Need is a Piece of the Pie: A Generic Framework for Optimizing Two-way Partial AUC

## Identity

- Paper ID: P028
- Authors: Zhiyong Yang, Qianqian Xu, Shilong Bao, Yuan He, Xiaochun Cao, and Qingming Huang
- Venue and year: ICML 2021, PMLR 139:11820-11829
- Published page: https://proceedings.mlr.press/v139/yang21k.html
- Main PDF: `source_papers/Generic_TPAUC_ICML_2021.pdf`, SHA256 `D56147C96D2287AF5FF09FF9E7EEDBF87BC0967B9C3FB76646B72B81A9C30DF5`
- Supplement: `source_papers/Generic_TPAUC_ICML_2021_supp.pdf`, SHA256 `0AB93585DDD9DC88440E2F8F8813C88C957967077954876FE54B4E8673A3024F`
- Paper-declared code: none found on the PMLR page, manuscript, or supplement; an exact-title repository search did not yield a verifiable author implementation.

## Reading Coverage

- Main manuscript: 10/10 pages read, including TPAUC definition, inconsistency with OPAUC, bi-level formulation, calibrated penalties and weights, Propositions 1-3, Theorem 1, all experiments, sensitivity figures, and conclusions.
- Supplement: 15/15 pages read, including all proofs, implementation details, dataset construction, and complete sensitivity plots.
- Visual verification: all 25 rendered pages under `audit/visual_checks/P028_Generic_TPAUC_ICML_2021/` inspected at original detail.
- Code: no immutable official implementation was available, so this paper is not replication-depth evidence.

## Research Question

The paper asks how to optimize the ROC region that simultaneously has high true-positive rate and low false-positive rate. In the paper's notation the target region is

```text
TPR >= 1 - alpha and FPR <= beta.
```

The finite empirical objective selects the bottom-scored `floor(n_pos * alpha)` positives and the top-scored `floor(n_neg * beta)` negatives. These identities are functions of the current model scores. Consequently, a weak-defect set and a difficult-normal set are not immutable lists:

```text
positive_tail(theta) and negative_tail(theta) change as theta changes.
```

This is directly relevant to Stage1's state-dependent value hypothesis, but the paper's `alpha` and `beta` values must not be mapped numerically to `FN<=95` without a separate finite-sample definition.

## Core Formulation

The hard TPAUC empirical loss compares only bottom positives with top negatives. Even after replacing the pairwise 0-1 loss with a differentiable surrogate, selecting those two tails still requires sorting, so the objective remains non-differentiable.

Proposition 1 rewrites this as a bi-level problem. Inner variables select hard positives using `1 - f(x_pos)` and hard negatives using `f(x_neg)` under cardinality constraints; the outer problem minimizes pairwise loss over their product weights.

The paper replaces sparse cardinality penalties with smooth convex penalties. Their dual weighting functions satisfy:

```text
v_pos = psi_gamma(1 - f(x_pos))
v_neg = psi_gamma(f(x_neg))
```

and the pair contribution is proportional to `v_pos * v_neg * pair_loss`. The required weighting function is strictly increasing and strictly concave. Concavity preserves nonzero mass on easier examples while still upweighting difficult ones.

Two instantiations are proposed:

```text
polynomial: psi(t) = t^(1 / (gamma - 1))
exponential: psi(t) = 1 - exp(-gamma * t)
```

The upper-bound result is conditional. Proposition 2 requires a data-, score-, and weight-dependent sufficient condition. Theorem 1 additionally assumes no ties, a bounded surrogate that upper-bounds 0-1 loss, iid class samples, and `(alpha,beta)` inside a sample-dependent sufficient set. It is not an unconditional guarantee for every extreme tail.

## Experimental Protocol

- Nine long-tailed binary image subsets are constructed from CIFAR-10, CIFAR-100, and Tiny-ImageNet-200.
- Positive counts range from 885 to 4,200 and negative counts from 7,898 to 68,600.
- Random train/validation/test splits use 70%/15%/15%; no group structure is involved.
- CIFAR uses ResNet-20 on 32x32 images; Tiny-ImageNet uses ResNet-18 on 224x224 images.
- SGD with Nesterov momentum is used, with batch size 128 and a fixed 1:10 positive-to-negative batch ratio.
- Methods are tuned independently for TPAUC `(0.3,0.3)`, `(0.4,0.4)`, and `(0.5,0.5)` and the validation-selected model is reported on test.
- The paper introduces a delay epoch `E_k`: ordinary AUC optimization is used first, then TPAUC weighting is activated.
- The number of random training seeds/repetitions, total training epochs, stopping rule, and uncertainty for Table 1 are not reported.

All architecture, optimizer, rate, batch composition, augmentation, delay, and tail values above are literature context only. They cannot change the Stage1 canonical configuration.

## Positive And Negative Evidence

1. The paper directly establishes that a two-sided operating region is not equivalent to an OPAUC objective with one fixed FPR interval. The induced FPR lower bound depends on the current scoring model.
2. The hard positive and hard negative identities are jointly model-dependent, and their contribution is pairwise. This supports collecting both tails and their pair relations rather than assigning an intrinsic scalar to each image.
3. The authors explicitly warn that emphasizing hard examples at the beginning risks overfitting. Their CIFAR sensitivity plots show that delayed activation can materially improve performance, especially for the exponential weighting function. This supports training-stage dependence.
4. That evidence does not determine Stage1's direction of timing intervention. The paper delays the start of hard-tail optimization, whereas the current Stage1 hypothesis reduces late replay. It supports “timing matters,” not “stop at epoch 140.”
5. The sensitivity result is method- and subset-dependent. Polynomial weighting has weaker delay trends and large dispersion across hyperparameter combinations; several final benchmark cells are not won by the proposed methods.
6. Table 1 reports single values without standard deviations or confidence intervals. The text says results are “significantly” better, but no test, repeated-run count, or uncertainty is supplied. The claim cannot support cross-seed stability.
7. Sensitivity boxplots aggregate different hyperparameter combinations, not stochastic replicates. Their spread measures tuning sensitivity, not seed variance.
8. Reproduction details are internally incomplete. Supplement G.2 lists delay values `{3,5,8,10,12,15,18,20}`, while the plotted/main discussion includes 30. It labels polynomial `gamma` values below 1 even though the stated polynomial formula requires `gamma>2`; figures appear to parameterize an inverse transform instead. Total epochs are absent.
9. The theoretical analysis assumes no ties and iid class draws. Stage1 video frames require explicit group-effective sample size, tie handling, and dependence audits.
10. The evaluated tails, 30%-50% on each side, are far broader than Stage1's extreme recall-constrained operating region. Numerical efficacy does not transfer.

## Direct Support For Stage1

1. Persist model-defined weak-defect and difficult-normal tail membership at every epoch, alongside fixed probe identities.
2. Record tail entry/exit, rank, residence duration, Jaccard turnover, and cross-seed membership agreement.
3. At key checkpoints, record pairwise weak-defect/difficult-normal violations, pair weights, unique partner counts, and concentration by image/video/cluster.
4. Keep hard-indicator and smooth-weight diagnostics separate. Their disagreement is an outlier-sensitivity signal, not a value score.
5. Record the exact activation/exposure schedule and cumulative replay dose. Training-stage effects cannot be inferred from final predictions alone.
6. Validate any tail summary against the original raw `FN=0-95` frontier. Surrogate improvement is not an operational guarantee.
7. Report seed uncertainty separately from hyperparameter sensitivity and tail-membership turnover.

## What It Does Not Support

1. Replacing Stage1 cross-entropy training with TPAUC loss, pairwise AUC training, or a new optimizer.
2. Importing the paper's `alpha`, `beta`, `gamma`, delay epoch, learning rate, batch composition, augmentation, architecture, or any other hyperparameter.
3. Claiming that a current hard positive or hard negative is permanently valuable for replay.
4. Claiming that delayed onset proves late replay should stop at epoch 140 or 160.
5. Claiming cross-seed robustness, extreme-tail calibration, group-independent evidence, or significance from Table 1.
6. Treating a smooth sample weight as a causal replay-value estimate.

## Stage1 Field Contract

Under the exact canonical hyperparameter lock, add or retain:

- all-epoch dynamic tail identity, rank, boundary distance, tie count, entry/exit, and residence fields for both classes;
- hard-tail and smooth-tail weights computed after training as diagnostics, with formulas and parameters versioned;
- weak-defect/difficult-normal pair violation, pair margin, pair-weight mass, unique partner count, and partner concentration;
- effective independent image/video/cluster counts in each tail and pair graph;
- replay activation state, realized per-epoch exposure, cumulative exposure, and schedule phase;
- fixed-probe loss and score trajectories before, during, and after replay activation or decay;
- seed-replicate uncertainty, hyperparameter sensitivity, and tail-set turnover as three distinct outputs;
- exact finite tail fractions, integer boundary ranks, endpoint/tie policy, and the original raw safety-frontier outcome.

These are diagnostics only. They do not alter `yolo11l`, 200 epochs, batch 128, image size 224, workers 4, AMP, optimizer, learning-rate schedule, augmentation, or any other canonical field.

## Concrete Experiment Consequence

P028 adds no formal arm. It strengthens the evidence card for a timing intervention and makes its interpretation more careful:

```text
same selection + same seed + same cumulative dose
different replay timing
=> causal test of timing, not sample identity
```

The existing continuous, same-peak decay, cumulative-dose-matched decay, and no-replay comparison remains appropriate. P028 predicts that tail membership and pair-weight concentration should change around replay schedule transitions. It does not predict which schedule must win.

If Stage1 data show that damage begins immediately rather than late, delayed-onset replay becomes a future preregistered candidate. It must not be added after seeing the same confirmatory outcomes.

## Stage1 Decision Status

- Evidence status: `FULL_READ_COMPLETE`
- Claim eligibility: yes, for joint model-dependent tail selection, pairwise non-decomposability, and timing sensitivity
- Replication-depth eligibility: no; no immutable official code and incomplete stochastic/training details
- Direct support for static replay ranking: no
- Direct support for dynamic replay timing: indirect and direction-agnostic
- Direct support for a new formal arm: no
- Direct support for process fields: yes
- Permission to change canonical Stage1 hyperparameters: no
- Reviewed at: 2026-08-07
