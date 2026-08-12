# P014 - If Influence Functions are the Answer, Then What is the Question?

## Identity

- Paper ID: P014
- Authors: Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi and Roger B. Grosse
- Venue and year: NeurIPS 2022
- Official page: https://proceedings.neurips.cc/paper_files/paper/2022/hash/7234e0c36fdbcb23e7bd56b68838999b-Abstract-Conference.html
- Main PDF: `source_papers/Influence_Question_PBRF_2022.pdf`, SHA256 `943184BEDD774849911E39C0B02FC35AC88241ABCB3A2A82DB40371400C72E14`
- Supplement: `source_papers/Influence_Question_PBRF_2022_supp.pdf`, SHA256 `93025EC6E25E353CD9FC42B5A41C7F09FC1251F4414DBE74DF86CE26EFF02086`
- Public PyTorch library: https://github.com/alstonlo/torch-influence
- Code snapshot: `8b4f0756acf642a38a8b24a517da89343a9de7d0`, dated 2022-11-16
- Related JAX library: https://github.com/pomonam/jax-influence

## Reading Coverage

- Main paper: 15/15 pages read.
- Official supplement: 11/11 pages read.
- Derivations checked: response function, proximal response, proximal Bregman response, linearized PBRF, GNH influence, CG and LiSSA.
- Experiments checked: all model families, five-gap decomposition, width/depth/training-time/regularization/damping/group-removal ablations, two-stage LOO and mislabeled-data audit.
- PyTorch implementation checked: objective contract, exact/CG/LiSSA modules, GNH path, data-loader reconstruction, score sign, tests, dependency metadata and repository history.
- Visual verification: main pages 2-10 and supplement pages 5-10 under `audit/visual_checks/P014_Influence_Question_PBRF/`.

## Research Question

When a practical neural-network influence estimate disagrees with leave-one-out retraining, is it simply inaccurate, or is it accurately estimating a different, local and proximity-constrained counterfactual?

## Five-Part Decomposition

The paper decomposes the distance from cold-start leave-one-out retraining to practical influence into:

1. `warm_start_gap`: cold-start versus continuation from the current model state;
2. `proximity_gap`: ordinary warm-start versus an objective penalizing departure from current parameters;
3. `non_convergence_gap`: ordinary continued training versus an objective for which the current predictions are already optimal;
4. `linearization_error`: nonlinear PBRF versus its locally linearized network/loss surrogate;
5. `solver_error`: exact linearized solution versus truncated CG/LiSSA.

The first three are different estimands, not merely numerical errors. This distinction is central for Stage1: a mathematically accurate score can still answer the wrong intervention question.

## Core Definitions

The paper defines deletion through downweighting:

```text
Q_minus_z(theta, epsilon)
  = J(theta) - epsilon * L_z(theta)

epsilon = 1/N means one-point deletion up to a constant scale.
```

With damping, influence linearizes a proximal response:

```text
argmin_theta Q_minus_z(theta, epsilon)
             + (lambda / 2) * ||theta - theta_current||^2
```

The PBRF replaces ordinary training loss with a Bregman penalty around the current model predictions:

```text
PBRF(theta, epsilon)
  = mean_i D_L_i(f_theta(x_i), f_current(x_i))
    - epsilon * L_z(theta)
    + (lambda / 2) * ||theta - theta_current||^2
```

For cross-entropy or squared loss, the first term can be viewed as training against current soft predictions. It removes ordinary continued-learning drift from the local deletion question.

After linearizing the network and using the output-loss quadratic, its solution is:

```text
theta_linear_PBRF
  = theta_current
    + (G_current + lambda I)^-1 * grad L_z * epsilon
```

where `G` is the Gauss-Newton Hessian. Thus damping is not just a numerical trick; it changes the proximity-constrained question being answered.

## Experimental Contract

- Base model is trained for `K` epochs.
- Twenty random training points are deleted one at a time.
- Cold-start deletion retrains from the same initialization and repeats the same batch order for the first `K` epochs, then trains another `K/2` epochs.
- Warm/proximal/PBRF/linearized-PBRF branches start from the base state and train `K/2` epochs.
- Influence uses LiSSA with GNH; scale is selected from `{10,25,50,100,150,200,250,300,400,500}` until convergence.
- Main discrepancy is mean output L2 distance over the training set, not a constrained business-tail metric.
- Reported `mean +/-` values primarily vary over the 20 deleted examples; they are not a 20-seed initialization study.

## Main Results

### The estimand dominates

Across logistic regression, MLPs, an autoencoder, LeNet, AlexNet, VGG13, ResNet-20 and a Transformer, warm-start, proximity and non-convergence gaps usually dominate linearization and solver errors. Influence often correlates poorly with cold-start deletion but strongly with PBRF.

For MNIST models, Table 2 reports influence-versus-PBRF Spearman values from 0.52 to 0.99, while influence-versus-cold-start Spearman ranges from -0.08 to 0.12. This is evidence that influence can be internally accurate while failing to predict the actual retraining intervention.

### Factors are not monotone or universal

- More training reduces the non-convergence gap.
- More damping reduces solver and linearization error but increases the proximity gap.
- More weight decay generally reduces several gaps in the studied MLP.
- Increasing width reduces linearization error in this study.
- Unlike P013, the authors do not find a strong depth relationship. The disagreement itself shows that architecture/dataset-specific local calibration is required.
- Removing 10-90% of data sharply increases linearization and warm-start gaps. Those fractions are much larger than Stage1's per-epoch replay ratio, but repeated replay also creates a finite path-changing exposure.

### Two-stage drift control

The supplement subtracts two continuation branches: one trained longer with all data and one trained longer after deletion. This two-stage estimate correlates better with influence than ordinary warm-start deletion, although still worse than PBRF. This directly motivates a same-checkpoint no-intervention drift branch for Stage1 micro-intervention calibration.

### Noise detection is not replay value

On 10% MNIST with 10% random label corruption, PBRF self-influence finds over 80% of corrupted points after inspecting 20% of data. This supports auditing high self-influence for noise. It does not show that replaying those points improves a model; those examples are deliberately bad training data.

## Reproduction And Code Audit

- The NeurIPS checklist explicitly says the main experiment code was not attached. The later PyTorch/JAX repositories are generic influence libraries, not a reproducible release of the five-gap experiment suite.
- The PyTorch repository has seven commits and no changes after November 2022. Dependencies are broad lower bounds, not a lock file.
- Exact, CG and LiSSA methods are tested on small linear/logistic problems; no PBRF implementation or paper benchmark reproduction is present.
- `CGInfluenceModule` discards SciPy CG's convergence `info`, so a failed solve can silently yield a score. It exposes no residual artifact.
- LiSSA samples from the global PyTorch RNG and does not accept or persist a dedicated seed, sampled-batch identity or convergence criterion.
- `model.eval()` makes the diagnostic fixed-view, but the reconstructed DataLoader drops options such as generator, pin-memory, persistent workers and prefetch identity.
- Model parameters are temporarily removed and reinserted; exceptions before reinsertion can leave the model in a mutated state because cleanup is not in `finally`.
- The library docstring says scores estimate removal plus retraining, while the paper's central result says practical scores often approximate PBRF instead. Users must name the estimand themselves.
- `AutogradInfluenceModule` optionally accepts zero Hessian eigenvalues as non-negative even though direct inversion requires nonsingularity.

## Direct Support For Stage1

1. Define the intervention before defining value: cold-start replay, warm-start local upweighting and PBRF are different quantities.
2. The formal campaign's paired arms must share initialization, base data order, canonical hyperparameters and total training horizon.
3. A micro-intervention probe should include a same-checkpoint continuation control so ordinary training drift is subtracted.
4. Damping, GNH scope and solver settings are part of the scientific estimand and must be preregistered, not tuned until an attractive ranking appears.
5. The gradient/influence probe should report whether it predicts local PBRF-like change or actual end-to-end replay outcome; these labels must not be conflated.
6. Separate difficult-normal and weak-defect target axes. PBRF's average preservation term can still hide the weakest defect tail.
7. Full-epoch state and exposure records are justified because non-convergence and checkpoint location change the question.
8. The actual replay timing/dose/guard experiment remains necessary because Stage1 changes the path from epoch 1, not only an endpoint weight.

## What It Does Not Support

1. It does not provide a Stage1 replay percentage, decay epoch, guard fraction or success threshold.
2. It does not show that PBRF/influence predicts additive replay under an `FN <= 95` constraint.
3. It does not study a frozen selection across many initialization seeds.
4. It does not establish that high self-influence is beneficial training data.
5. It does not justify adapting the final experiment based on blind test outcomes.
6. It does not make one average output-distance target adequate for tail-risk control.
7. It does not provide runnable code for its central five-gap decomposition.

## Stage1 Field Contract

Persist, at every calibrated mechanism probe:

- `estimand_namespace`: cold-start finite replay, warm-start finite upweight, PBRF, linearized PBRF, raw local gradient or actual optimizer step;
- base checkpoint/model/optimizer/hyperparameter-lock/initial-weight/data-order identities;
- continuation-control identity and output/target-loss drift;
- deleted/replayed sample or set manifest and finite epsilon/exposure;
- parameter-space and function-space proximity penalties;
- target axes and exact normal-tail/weak-defect sample manifests;
- Hessian versus GNH, damping, scale, solver, iteration/depth/repeats and solver seed;
- solver status and explicit normalized residual;
- linearization, solver and total finite-difference errors where measurable;
- rank/sign agreement over the full calibrated pool and selected tail;
- checkpoint and seed stability;
- runtime, peak CPU/GPU memory and failure status.

## Concrete Experiment Consequence

- Use a tiny set of same-checkpoint branches to compare continued training, finite local replay and predicted raw/GNH/PBRF directions.
- Keep this calibration observational and small; do not create a ten-machine PBRF-ranked Treatment arm yet.
- In the main campaign, compare continuous, decayed and dose-matched replay under exactly the canonical 240-run hyperparameters. Their paired no-replay/random arms estimate the real end-to-end causal quantity.
- Allow influence-derived selection only after it predicts unseen seed-state intervention signs better than simpler raw and optimizer-aware alignment.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, for estimand decomposition and drift-control requirements
- Direct support for static influence ranking as replay value: no
- Direct support for numeric replay scheduling: no
- Reviewed at: 2026-08-07
