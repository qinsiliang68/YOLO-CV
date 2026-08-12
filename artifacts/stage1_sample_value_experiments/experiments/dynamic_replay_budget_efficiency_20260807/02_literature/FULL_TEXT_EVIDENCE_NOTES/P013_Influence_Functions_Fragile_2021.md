# P013 - Influence Functions in Deep Learning Are Fragile

## Identity

- Paper ID: P013
- Authors: Samyadeep Basu, Phillip Pope and Soheil Feizi
- Venue and year: ICLR 2021
- Official forum: https://openreview.net/forum?id=xHKVVHGDOEk
- Author full text: https://arxiv.org/abs/2006.14651
- Local PDF: `source_papers/Influence_Functions_Fragile_2021.pdf`
- PDF SHA256: `C03CED98DEA27D0EE6882C3594B77F3617F8DD4F1C0B49A14E12C16271A7557F`
- Author TeX source: `source_papers/Influence_Functions_Fragile_2021_source.tar`
- Source SHA256: `FC7295E581522BE5700EB51E07E593AF6339E9BB3F99E97549792AA08CB5F2E3`
- Public experiment code: none located; the author source archive contains the paper and figures, not runnable experiment code.

## Reading Coverage

- Full paper and appendix: 22/22 pages read.
- Sections checked: definition and assumptions; exact-Hessian Iris study; shallow-CNN MNIST study; MNIST/CIFAR-10/100 deep-network studies; ImageNet study; ground-truth retraining discussion; inverse-HVP appendix; initialization/optimizer/sample-selection studies; group influence; multiple-target study.
- Author TeX checked for formulas, hyperparameters, URLs, implementation references and manuscript inconsistencies.
- Visual verification: pages 4-9, 15-16 and 19-21 under `audit/visual_checks/P013_Influence_Functions_Fragile/`.

## Research Question

How faithfully does the classical first-order influence approximation predict leave-one-out or group-reweighting effects once the model is a non-convex, over-parameterized deep network?

## Estimand And Formula

For fitted parameters:

```text
theta_star = argmin_theta (1/n) * sum_i loss(z_i, theta)
```

Infinitesimal upweighting of training point `z` gives:

```text
I_up_params(z) = -H^-1 * g_z
I_up_loss(z, z_target) = -g_target^T * H^-1 * g_z
```

Removal corresponds to `epsilon = -1/n`, so the predicted target-loss change is:

```text
delta_remove_loss approximately -I_up_loss / n
                         = +(1/n) * g_target^T H^-1 g_z
```

The manuscript states after Equation 5 that `I(z, z_t)/n` approximates removal. Given its own definition of `I` as the upweight derivative, this drops the minus sign. Stage1 must encode the intervention name and sign explicitly and test both against a finite-difference calibration.

## What Fragility Means Here

The paper distinguishes several failure sources:

1. the local Taylor approximation can fail in a non-convex landscape;
2. deeper and wider networks can have larger curvature and larger parameter-change approximation gaps;
3. exact inverse-Hessian computation is infeasible at scale, while stochastic inverse-HVP adds another error source;
4. damping and scale alter the surrogate Hessian;
5. the chosen target point and the subset on which correlation is measured materially alter apparent quality;
6. leave-one-out retraining itself can be noisy when ordinary continued training changes the target loss more than deleting one sample.

This is not evidence that all influence diagnostics are useless. It is evidence that a reported influence number without a calibrated estimand, checkpoint, target, approximation identity and finite intervention check is not trustworthy.

## Experimental Contract And Results

### Exact-Hessian Iris study

- Full-batch gradient descent for 60,000 iterations; leave-one-out reference retraining for 7,500 steps from the fitted state.
- Evaluation uses the maximum-loss test point and the top 16.6% of training points.
- A depth-1, width-5 ReLU network reaches Spearman 0.97 with weight decay, versus 0.508 without it; the singular no-decay Hessian is damped by 0.001.
- Correlation falls when depth exceeds roughly five and when width increases from 8 to 50.
- The paper observes correlation between curvature growth and Taylor error, but does not prove that the top Hessian eigenvalue is a usable error bound.

### Small-MNIST shallow CNN

- Ten percent of MNIST, 2,600 parameters, 500,000 training iterations and 30,000 nearby-retraining iterations.
- Weight decay is 0.001 when enabled.
- For each selected target, the authors retrain the top 100 estimated training influences and another 100 around the 30th percentile.
- Top-tail influence ranks can correlate well, but the 30th-percentile subset degrades sharply.
- Across target points, top-tail Spearman ranges from 0.92 to 0.38 even with weight decay.
- Correlation begins to fall again above weight decay 0.01, so regularization is not a monotone cure.

### Deeper MNIST, CIFAR-10 and CIFAR-100

- Two target points per MNIST/CIFAR-10 architecture: maximum loss and median loss.
- Only the top 40 estimated training influences are leave-one-out retrained for 6% of the original training steps.
- Table 1 varies strongly by architecture, dataset, target point and weight decay. On MNIST, ResNet-50 Spearman is 0.22 or 0.19 with decay and 0.13 without. On CIFAR-10, some no-decay VGG values exceed decay values.
- CIFAR-100 ResNet-18 with weight decay `5e-4` remains poor across several high-loss and median-loss targets and multiple initializations.
- Correlation on an influence-selected top tail is conditional evidence; it does not establish calibration, sign accuracy or ranking quality over the full candidate pool.

### ImageNet

- ResNet-50 with about 25.5M parameters and a fastai/DAWNBench-style schedule; reported top-5 validation accuracy 92.302%.
- Two randomly selected target points, top 50 estimated training points, then two extra retraining epochs, about 5% of original training time.
- Pearson and Spearman are below 0.15 for both targets.
- Continuing training with all data changes one target loss by -0.679 and another by +0.066, which can dominate the leave-one-out signal.
- Gradient norms of 20.18 and 15.89 are called relatively small given the parameter count, yet target losses still move materially. Raw norm alone is not a convergence certificate.
- Manuscript identity typo: the text and Figure 11 use target index 13,923, while Figure 13's caption says 13,293. Sample IDs must be manifest-validated, not copied manually.

### Group influence

- Larger group sizes break the infinitesimal perturbation assumption even more directly.
- On ResNet-18, group-influence correlations range only 0.01-0.21 on MNIST and 0.01-0.18 on CIFAR-100.
- This is the closest published failure mode to Stage1 repeated replay: a replay set is not the sum of stable per-image local derivatives once model state and curvature move.

### Multiple targets, initialization and optimizer

- Eight-target aggregation gives Pearson/Spearman 0.91/0.78 for small MNIST but only 0.15/0.11 for ResNet-18 on CIFAR-100.
- Small CNN and LeNet initialization experiments report low variance but inconsistent correlations and only test the top 40 points for the maximum-loss target.
- LeNet across Adam, gradient descent, Nesterov and RMSProp reports Pearson `0.72 +/- 0.04` and Spearman `0.56 +/- 0.11`; optimizer choice changes rank fidelity.
- These small-model results cannot answer same-selection yolo11l cross-seed reversal.

## Reproduction Audit

- The author archive provides TeX and figures but no runnable implementation, dependency lock, exact seeds, complete optimizer settings, hardware identity or exact stochastic inverse-HVP settings for every experiment.
- Table 2 reports influence computation times from roughly 136 to 4,620, but does not label the unit or identify enough hardware/software context for transfer.
- Deep inverse-HVP uses stochastic recursion with recursion depth, scale and damping, but the paper does not provide a complete reproducible parameter table for all deep experiments.
- Ground truth is mostly nearby retraining from the fitted model for a small fraction of original steps, not full retraining of each intervention from the original initialization.
- Reported correlations are often computed only on points first selected by estimated influence, creating a conditional evaluation that can hide errors outside the selected tail.
- No confidence intervals or seed-level tables accompany most central deep-network correlations.

## Direct Support For Stage1

1. Do not promote inverse-Hessian influence to a large replay-selection arm without local calibration on yolo11l.
2. Preserve target identity: difficult-normal and weak-defect probe sets need separate scores and signs.
3. Record checkpoint, model state, seed, optimizer, regularization, target composition and approximation parameters with every influence result.
4. Compare raw gradient alignment, optimizer-aware actual-update alignment and curvature-adjusted alignment at the same state.
5. Use finite-difference micro-interventions and exact final-layer solves where feasible to estimate approximation error.
6. Measure rank and sign stability across target subsets, checkpoints and seeds, not only top-tail Spearman.
7. Treat group/set replay as a finite causal intervention whose effect cannot be recovered by summing static sample scores.
8. Add ordinary continued-training controls when measuring tiny removal/upweighting effects, because baseline drift can dominate the intervention.

## What It Does Not Support

1. It does not prove all influence functions fail on every deep model.
2. It does not validate a replay ratio, decay epoch, guard share or Stage1 arm.
3. It does not show that weight decay makes influence reliable; several results remain poor or non-monotone.
4. It does not evaluate additive replay, an `FN <= 95` target or the Stage1 raw safety frontier.
5. It does not support replacing causal retraining with one endpoint Hessian calculation.
6. It does not identify high-value photos or distinguish helpful hard-clean points from harmful noise.
7. It does not establish cross-seed sign stability for a fixed selected set.

## Stage1 Field Contract

For each calibrated influence probe, persist:

- intervention namespace: upweight, remove, replay occurrence, replay set or group;
- explicit sign convention and expected target-loss direction;
- checkpoint, seed, model, optimizer-state, hyperparameter-lock, initial-weight and data-manifest SHA256;
- target namespace and exact target sample-ID manifest;
- candidate sample or set identity, video/group identity and prior cumulative replay exposure;
- raw gradient norm, dot product and cosine for each target axis;
- actual optimizer update dot product for each target axis;
- Hessian scope, estimator, damping, scale, recursion depth, repeat count, random seed and residual;
- exact/final-layer/low-rank approximation namespace;
- top-tail and full-pool rank diagnostics, sign agreement and finite-difference error;
- ordinary continued-training drift control;
- checkpoint- and seed-level rank/sign stability;
- computation time, peak memory and failure state.

## Concrete Experiment Consequence

- Keep influence collection observational in the first seven-day mechanism block.
- At key checkpoints, use the frozen candidate set and fixed normal/defect tail probes to compare three direction estimators plus a tiny finite-difference replay or upweight intervention.
- Do not spend ten-machine capacity on full-network inverse Hessians for all 120,000 samples.
- Use the real replay timing, dose and weak-defect-guard arms as the causal test of finite set value.
- A curvature-aware score can enter later selection only if it improves sign prediction over raw/optimizer alignment on unseen seed-state pairs and passes provenance/residual gates.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, for fragility and calibration requirements
- Direct support for a curvature-aware large replay arm: no
- Direct support for numeric replay settings: no
- Reviewed at: 2026-08-07
