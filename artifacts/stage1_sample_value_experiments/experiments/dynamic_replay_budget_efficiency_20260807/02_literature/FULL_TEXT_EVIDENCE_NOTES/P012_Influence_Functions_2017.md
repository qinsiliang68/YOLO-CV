# P012 - Understanding Black-box Predictions via Influence Functions

## Identity

- Paper ID: P012
- Authors: Pang Wei Koh and Percy Liang
- Venue and year: ICML 2017, PMLR 70
- Official page: https://proceedings.mlr.press/v70/koh17a.html
- Official main PDF: `source_papers/Influence_Functions_2017.pdf`, SHA256 `715FB9F935B0AF32C7AE1F0D834FD5158EDB279A575A4AE91196D1830F281F65`
- Official supplement: `source_papers/Influence_Functions_2017_supp.pdf`, SHA256 `23DD20059B6A4354B660E7442B5846788920AB754076467598FD91727319F13B`
- Author full version: `source_papers/Influence_Functions_2017_arxiv_full.pdf`, SHA256 `4AC350D88D44FFF186B89E96048306C702D09EAAC812303F12220C012E00BADC`
- Distinct content: 10 main + 2 appendix pages; arXiv v3 combines them into 12 pages.
- Code: https://github.com/kohpangwei/influence-release
- Code snapshot: commit `0f656964867da6ddcca16c14b3e4f0eef38a7472`, dated 2020-12-29

## Reading Coverage

- Official main: 10/10 pages read.
- Official supplement: 2/2 pages read.
- ArXiv v3: 12/12 pages cross-checked.
- Sections checked: upweight/removal derivation; input perturbation; Hessian geometry; CG and LiSSA; convex/non-convex/nondifferentiable validation; every application; derivation and non-convergence appendix.
- Public code checked: Hessian-vector products, CG/LiSSA, target gradient aggregation, influence cache, LOO self-influence, model seed handling and spam-repeat script.
- Visual verification: author-version pages 2-9 under `audit/visual_checks/P012_Influence_Functions/`.

## Research Question

Can the local effect of infinitesimally upweighting, removing or perturbing a training example be approximated without retraining, and can that approximation diagnose which training data drive a target prediction?

## Core Definition

Let the empirical-risk minimizer be:

```text
theta_hat = argmin_theta (1/n) * sum_i L(z_i, theta)
```

Under twice differentiability, strong convexity and positive-definite empirical Hessian:

```text
H = (1/n) * sum_i Hessian_theta L(z_i, theta_hat)
```

The parameter derivative from infinitesimally upweighting sample `z` is:

```text
I_up_params(z) = -H^-1 * grad L(z, theta_hat)
```

The corresponding change in target loss is:

```text
I_up_loss(z, z_target)
  = -grad L(z_target)^T * H^-1 * grad L(z)
```

Removing one point is approximated by upweighting with `epsilon=-1/n`:

```text
delta_remove_target_loss
  approximately (1/n) * grad L(z_target)^T * H^-1 * grad L(z)
```

The sign must always be reported with its intervention. `I_up_loss` and predicted leave-one-out loss change have opposite signs.

## Why It Is More Than Gradient Alignment

Raw alignment uses `g_target^T g_i`. Influence uses `g_target^T H^-1 g_i`. The inverse Hessian transforms the sample gradient by the local curvature of the full empirical risk. In the paper's logistic-regression interpretation, this represents how strongly the rest of the training data resists movement in that direction.

Therefore two samples with the same loss or gradient norm can have different local removal effects because:

1. their directions relative to the target differ;
2. surrounding samples provide different support/redundancy;
3. curvature makes one direction easier to move than another.

This is a principled reason that sample value is contextual and set-dependent.

## Computational Methods

For a target gradient `v`, first solve `s = H^-1 v`, then score every training sample with `s^T g_i`.

### Conjugate gradient

- Avoids materializing `H` and uses Hessian-vector products.
- Requires a positive-definite or suitably damped local quadratic.
- Each full-HVP iteration can traverse all training data.

### Stochastic inverse-HVP / LiSSA

The paper uses a stochastic truncated Neumann recursion. Its convergence requires a scaled Hessian with spectrum below one. Practical settings include scale, damping, recursion depth, sampled mini-batch size and number of repeated recursions.

The MNIST experiment uses 10 repeats and 5,000 recursion steps; even one repeat identifies major points but is noisier. These constants are task-specific approximation settings, not Stage1 defaults.

## Experimental Contract And Results

### Convex leave-one-out validation

- Ten-class MNIST logistic regression, `n=55,000`, `p=7,840` and L2 regularization 0.01.
- Exact CG and stochastic inverse-HVP predictions closely match leave-one-out retraining for selected influential points.
- Figure 1 shows that removing either the training-loss factor or `H^-1` yields materially different and incorrect rankings.

### Non-convex and non-converged validation

- A small seven-convolutional-layer tanh network with only 2,616 parameters, 10% MNIST, batch 500 and Adam.
- Training remained non-converged after 500,000 iterations; damping `lambda=0.01` was added.
- For the 100 most influential points, predicted versus retrained loss changes have Pearson correlation 0.86.
- Retraining starts from the nearby state and runs 30,000 steps. This validates correlation in one small setting, not calibrated causal accuracy for modern yolo11l.

### Nondifferentiable loss

- Raw hinge derivatives fail badly.
- SmoothHinge with temperature 0.001 reaches Pearson correlation 0.95; temperature 0.1 still reaches 0.91.
- The smoothing choice changes the diagnostic quantity and requires validation.

### Data debugging

- Enron spam: 4,147 train, 1,035 test, bag-of-words logistic regression, random 10% label flips.
- Influence-based human review beats loss and random review; uncertainty is shown over 40 independently flipped subsets.
- No method accesses test data for selecting review candidates.
- The 40 repeats vary corruption/review sampling, not neural-network initialization.

### Other applications

- Domain mismatch case study finds four child-patient examples 30-40 times more influential than the next points.
- Dog-versus-fish uses frozen Inception features/top-layer training and demonstrates training-data poisoning.
- These are debugging/security demonstrations, not replay-benefit experiments.

## Assumptions And Failure Modes

### Local intervention only

Influence is a derivative at infinitesimal upweighting. The paper explicitly states that larger subpopulation changes remain an open problem because the model must not move too far. Stage1 replay percentages repeated over many epochs are finite, path-changing interventions.

### Endpoint geometry, not training path

The method evaluates curvature around a fitted endpoint or local state. It does not attribute when replay occurred, cumulative exposure or optimizer history. Two seeds can have different Hessians and influence signs even with the same selected IDs.

### Deep-network theory does not hold directly

Strong convexity, differentiability, positive-definite Hessian and exact empirical-risk minimization fail for modern deep networks. Damping forms a different local quadratic and its value can alter ranking.

### Curvature approximation is fragile

CG, LiSSA scale, damping, recursion depth, mini-batch sampling and numerical tolerances are real hyperparameters. A score without these identities and residual diagnostics is not reproducible.

### Duplicate and group effects are not additive at finite dose

First-order group upweighting can sum individual derivatives at epsilon zero. Once a duplicated video pattern is replayed repeatedly, curvature and parameters change, and individual local influences no longer identify the set effect.

### Average targets can hide tails

The code averages target gradients. Difficult-normal and weak-defect gradients can cancel before `H^-1` is applied. Stage1 must retain the two target axes and worst-case probe distribution rather than one average validation loss.

### High self-influence is not high value

Self-influence detects points that strongly support their own fitted label. That includes useful rare examples, mislabeled points, ambiguous outliers and poisonable points. It is an audit signal.

## Code Audit

- The base model constructor hard-codes NumPy and TensorFlow seeds to zero.
- `get_test_grad_loss_no_reg_val` averages gradients over target examples before the inverse-HVP.
- LiSSA defaults are scale 10, damping 0, one sample and recursion depth 10,000; the MNIST script overrides them. None transfers numerically.
- CG uses damped full-dataset HVPs, tolerance `1e-8` and maximum 100 iterations in the release.
- Cached inverse-HVP filenames include model name, method, loss type and a text description, but omit checkpoint SHA, dataset SHA, damping, scale, recursion depth and code/config identity. Stale caches can silently be reused.
- Cache writes use direct `np.savez` without atomic completion sidecars.
- The latest repository commit fixes the exact logistic LOO expression from `sigma * quad_x` to `sigma^2 * quad_x`. Older code snapshots produce a different score.
- The stack is TensorFlow 1.1, Keras 2.0.4 and Python-2-era syntax. No automated tests or immutable environment lock are present.

## What It Supports For Stage1

1. Raw gradient direction is incomplete; local data curvature and redundancy add information.
2. Separate target axes are required for difficult-normal benefit and weak-defect non-harm.
3. Influence must be conditioned on model state/seed and cannot be a permanent `V(x)`.
4. High loss/self-influence is a candidate-audit signal, not a replay policy.
5. Approximation residuals, damping, scale, recursion depth and checkpoint identity must be logged.
6. Group/video-cluster influence deserves a separate probe because correlated frames alter local resistance.
7. Influence probes belong on OOF/val_op discovery targets, never the blind test.
8. Actual replay remains a required causal experiment because finite exposure exceeds the local derivative regime.

## What It Does Not Support

1. It does not support selecting a fixed top percentage by influence and declaring those samples valuable.
2. It does not support a replay ratio, decay epoch or guard share.
3. It does not establish cross-seed sign stability.
4. It does not study additive replay under an FN constraint.
5. It does not validate full yolo11l inverse-Hessian computation.
6. It does not guarantee that damping preserves ranking or causal magnitude.
7. It does not justify combining normal and defect target losses with an arbitrary scalar weight.

## Stage1 Field Contract

For every influence probe, persist:

- `target_namespace`: difficult-normal, weak-defect, or separately defined probe subgroup;
- checkpoint/model/initial-weight/hyperparameter-lock SHA identities;
- layer scope and parameter count;
- target sample IDs and gradient aggregation rule;
- raw gradient dot/cosine/norm;
- inverse-HVP method, damping, scale, recursion depth, repeats, batch size and random seed;
- linear-system residual `||Hs-v|| / ||v||` or equivalent convergence diagnostic;
- curvature-adjusted dot, sign and normalized rank;
- cross-check against exact/CG or finite-difference removal on a tiny calibration subset;
- checkpoint and seed sign/rank stability;
- group/video-cluster aggregate and concentration;
- replay exposure before/after the checkpoint.

## Concrete Experiment Consequence

- Do not attempt all-network inverse Hessians for 120,000 samples across 200 epochs.
- At a few key states, calibrate a final-layer or low-rank curvature-aware probe against raw alignment and tiny finite-difference interventions.
- On same-selection cross-seed reversal pairs, ask whether raw alignment, actual optimizer update alignment or curvature-adjusted influence better tracks outcome direction.
- Keep these measurements observational in the first mechanism block; do not let them adapt arms mid-run.
- Use causal timing/dose/guard arms to determine whether a locally favorable direction survives finite cumulative replay.

## Reproduction Notes

- Main formulas, experiment scripts and data links are public.
- Exact modern rerun requires restoring an obsolete TF1/Keras environment and historical data/CodaLab assets.
- The code's 2020 LOO correction means repository commit identity is necessary when comparing published or third-party outputs.
- `REPLICATION_DEPTH` denotes full paper, appendix and code audit; no benchmark rerun was performed.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, with strict local/endpoint/curvature assumptions
- Direct support for curvature-aware diagnostic fields: yes
- Direct support for an influence-selected replay arm or numeric schedule: no
- Reviewed at: 2026-08-07
