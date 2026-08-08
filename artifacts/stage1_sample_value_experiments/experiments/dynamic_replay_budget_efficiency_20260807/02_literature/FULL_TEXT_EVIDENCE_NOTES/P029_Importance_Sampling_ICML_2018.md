# P029 - Not All Samples Are Created Equal: Deep Learning with Importance Sampling

## Identity

- Paper ID: P029
- Authors: Angelos Katharopoulos and Francois Fleuret
- Venue and year: ICML 2018, PMLR 80:2525-2534
- Published page: https://proceedings.mlr.press/v80/katharopoulos18a.html
- Main PDF: `source_papers/Importance_Sampling_2018.pdf`, SHA256 `1D6726829DD83572323499D6DBE238B9251E529362BDE12937D0969104487FD8`
- Supplement: `source_papers/Importance_Sampling_2018_supp.pdf`, SHA256 `7E76283DAD6C5460D54A5AEA1A08C89CDD8818C72180641D094D2847965BDDA6`
- Official code: https://github.com/idiap/importance-sampling
- Audited snapshot: tag `v0.7`, commit `69e29b091420c19419293123f9b67232f878fda3`, dated 2018-06-11.

## Reading Coverage

- Main manuscript: 10/10 pages read, including Equations 1-26, Algorithm 1, all three task experiments, figures, limitations, and conclusions.
- Supplement: 4/4 pages read, including the variance derivation, distance-to-optimum assumption, SVRG comparison, presampling-size ablation, and loss-sampling analysis.
- Visual verification: all 14 rendered pages under `audit/visual_checks/P029_Importance_Sampling_ICML_2018/` inspected at original detail.
- Code audit: official tag `v0.7` inspected at the training loop, sampler, condition, score layer, reweighting, logging, seed, checkpoint, tests, and dependency levels. The legacy stack was not executed because the unpinned Keras/TensorFlow interface is obsolete and execution would not change the Stage1 design decision.

## Research Question

The paper asks whether non-uniform sampling can reduce stochastic-gradient variance enough to improve convergence within a fixed wall-clock budget. It does not ask whether a sample improves a held-out business objective or the final `FN=0-95` safety frontier.

This distinction is central:

```text
gradient magnitude       -> instantaneous parameter leverage
gradient covariance      -> optimizer noise / efficiency
target gradient alignment -> local direction toward a chosen target
finite replay outcome    -> observed Stage1 utility under a full training path
```

The first two quantities are not substitutes for the last two.

## Core Formulation

For sample probability `p_i` and inverse-probability weight

```text
w_i = 1 / (N * p_i),
```

the expected stochastic gradient remains the full empirical gradient. In the paper's derivation, the sampling distribution that minimizes the trace of the stochastic-gradient covariance is proportional to the current per-sample gradient norm:

```text
p_i proportional to ||grad_theta L_i(theta_t)||.
```

Because exact full-gradient norms are expensive, Equations 16-20 derive a sample-dependent score from the loss gradient with respect to the final preactivation. The omitted multiplicative terms are treated as approximately sample-independent under slope-bounded activations and roughly uniform hidden activations. This is an approximation under stated network assumptions, not an identity for every deep model.

The score is explicitly state-dependent. The paper says it cannot be computed once and reused because sample importance changes with the model.

Algorithm 1 first samples a uniform candidate batch of size `B`, computes current scores, and resamples `b` examples with replacement. A statistic `tau` estimates the batch-size increase that would give an equivalent variance reduction. Importance sampling is activated only when an exponential moving average of `tau` exceeds a threshold. Before activation, current batch scores are still collected after each uniform update.

## Experimental Protocol

- The gradient-approximation study uses WRN-28-2 on CIFAR-100, a uniform candidate batch of 1,024, a selected batch of 128, checkpoints every 3,000 updates, and ten resampling repetitions per measurement.
- CIFAR-10/100 use WRN-28-2, SGD with momentum, batch 128, 50,000 updates, and a wall-clock learning-rate schedule. Reported curves average three independent runs.
- MIT67 fine-tuning replaces the last layer of GoogLeNet and trains end to end; results average three independent runs.
- Pixel-permuted MNIST uses an LSTM with Adam and gradient clipping. The main text does not state a repeated-run count for that figure.
- Primary comparisons equalize wall-clock time, not optimizer updates or total sample presentations.
- No confidence intervals or formal statistical tests are reported.

All architectures, optimizers, batch sizes, learning rates, candidate-batch sizes, thresholds, warmups, and time budgets are literature context only. They cannot alter the Stage1 canonical hyperparameter lock.

## Positive And Negative Evidence

1. The paper directly supports fresh, state-dependent measurement. A one-time gradient or loss ranking is theoretically inconsistent with its own objective.
2. Under inverse-probability correction, large current gradient norm identifies a sample that can contribute strongly to stochastic-gradient variance. It identifies leverage, not beneficial direction for a validation or safety objective.
3. The last-preactivation gradient provides a cheap approximation channel worth collecting at key Stage1 checkpoints, but its tightness is demonstrated on one WRN/CIFAR-100 setup.
4. Loss is a poor proxy for full gradient norm outside the very-small-gradient regime. Figure 1 shows high-loss sampling increasing gradient variance, especially early in training.
5. Loss-based sampling helps on easier CIFAR-10 only with fresh scores and warmup, fails to speed CIFAR-100, and hurts pixel-permuted MNIST. This is direct negative evidence against `loss high => useful`.
6. The proposed method optimizes convergence efficiency. Lower training loss or earlier test error at the same wall-clock time does not establish better final quality at the same update path, nor positive Stage1 tail utility.
7. Three independent runs and unreported uncertainty are insufficient evidence for cross-seed stability.
8. The theoretical speed expression measures Euclidean distance to one optimum and uses smoothness-related assumptions. Deep non-convex training can have multiple relevant solutions, so the convergence interpretation has a limited transfer boundary.
9. Presampling from a local random candidate batch is materially different from globally taking a fixed top percentage and replaying it for 200 epochs.
10. Importance correction is essential to the unbiased-gradient argument. A replay system that duplicates selected samples without correction intentionally changes the training objective; the paper's guarantee does not apply.

## Official Code Audit

The official `v0.7` code agrees with the main algorithm on several important points:

- `ModelSampler` draws a fresh candidate batch, scores it with the current model, and resamples with replacement.
- `BiasedReweightingPolicy(k=1)` implements inverse-probability-style correction despite the confusing class name.
- `ImportanceTraining` computes a default variance-reduction threshold from `B` and `b` and wraps the sampler in `ConditionalStartSampler`.
- Before activation, `ConditionalStartSampler` trains a uniform batch and updates its condition from the actual score tensor returned by that same training call. After activation, it updates from the current presampled candidate scores.
- The training script seeds NumPy and TensorFlow once, evaluates `dataset.test_data` during training, and can log sampled identities and predicted scores.

The snapshot also has important reproducibility limits:

- checkpoints save model weights, and optionally a secondary model, but not NumPy RNG state, TensorFlow RNG state, condition EMA (`_vr`/`_previous_vr`), sampler caches, iteration counters, or logger state;
- loading `--initial_weights` therefore does not reproduce an interrupted sampling or activation path;
- checkpoint writes are not atomic and have no completeness sidecar or content-bound config/data identity;
- dependencies are only `keras>=2`, `blinker`, and `numpy`; TensorFlow/Keras versions are not pinned;
- no direct test was found for variance-condition transitions, interruption/resume, atomic artifacts, or exact sampling-path replay;
- the generic script evaluates `test_data` repeatedly, so role separation depends on how the caller constructs the dataset.

These are reasons to require full sampler/collector state and immutable manifests in Stage1, not reasons to copy this legacy runner.

## Direct Support For Stage1

1. Keep per-sample gradient magnitude separate from gradient direction. A large norm is a high-leverage candidate, not a high-value conclusion.
2. At key checkpoints, collect last-head gradient norm plus dot product and cosine against separate difficult-normal and weak-defect target gradients.
3. Collect gradient dispersion/covariance summaries and a `tau`-like heterogeneity diagnostic separately from target alignment. Optimization efficiency and business value answer different questions.
4. Record score checkpoint, score age, parameter-update age, stale/fresh status, replay sampling probability, inverse-probability or replay weight, realized presentation count, and cumulative exposure.
5. Compare good and bad seeds for the same selected IDs. The falsifiable question is whether magnitude remains similar while target alignment, covariance, or downstream finite effect changes.
6. Persist condition and RNG state if any future adaptive collector or replay policy uses an EMA or stochastic sampling decision.
7. Use same-state finite interventions on a small audited subset to calibrate whether local gradient alignment predicts actual short-horizon tail change.

## What It Does Not Support

1. Defining high-value samples as the largest gradient norms.
2. Eliminating every low-gradient sample; low individual leverage can coexist with representation, coverage, calibration, or aggregate value.
3. Replacing Stage1 replay with importance sampling or inverse-probability weighting in the current formal campaign.
4. Adding a gradient-magnitude Treatment arm before a separate preregistration and falsifier.
5. Importing any paper hyperparameter, optimizer, batch size, warmup, threshold, architecture, candidate-pool size, or learning-rate schedule.
6. Claiming cross-seed stability, weak-defect safety, raw-frontier improvement, or causal sample value from variance reduction.

## Stage1 Field Contract

Under the exact canonical hyperparameter lock, add or retain at key checkpoints:

- `head_grad_norm`, `head_grad_norm_sq`, and percentile/rank within a frozen candidate pool;
- `dot_to_normal_tail_target`, `cos_to_normal_tail_target`;
- `dot_to_weak_defect_target`, `cos_to_weak_defect_target`;
- target-gradient norms so cosine and dot products remain interpretable;
- candidate-batch gradient-norm sum, squared sum, coefficient of variation, effective sample size, and `tau`-like variance-reduction statistic;
- current score checkpoint, score age in optimizer steps and epochs, and stale/fresh flag;
- intended sampling probability/weight, realized sample presentations, epoch-weighted exposure, and cumulative exposure;
- gradient-to-actual-optimizer-update alignment, because momentum, weight decay, AMP scaling, and optimizer state can change the realized update;
- paired same-ID, same-checkpoint, cross-seed summaries and finite-intervention outcome.

All-epoch low-cost training dynamics and the six heavy checkpoints remain unchanged. These fields do not alter `yolo11l`, 200 epochs, batch 128, image size 224, workers 4, AMP, optimizer, learning-rate schedule, augmentation, or any other canonical field.

## Concrete Experiment Consequence

P029 adds no formal arm. It sharpens the mechanism analysis inside the already planned timing block:

```text
large norm + positive target alignment + favorable finite effect
    => locally useful high-leverage evidence

large norm + negative target alignment
    => high-leverage harmful candidate

large norm + unstable alignment across checkpoints/seeds
    => conditional or risky candidate

small norm
    => low immediate leverage, not proof of zero aggregate value
```

The continuous, same-peak decay, cumulative-dose-matched decay, and no-replay comparison remains the correct causal intervention. Gradient channels explain why the same selection may reverse; they do not replace paired outcome validation.

## Stage1 Decision Status

- Evidence status: `FULL_READ_COMPLETE`
- Claim eligibility: yes, for state-dependent gradient-norm sampling, unbiased importance correction, variance reduction, and the failure of loss as a general gradient proxy
- Replication-depth eligibility: no; the official implementation was audited but not executed, the legacy environment is incompletely pinned, and reproducing its task does not change the Stage1 design
- Direct support for static replay ranking: no
- Direct support for gradient magnitude as value: no
- Direct support for process fields: yes
- Direct support for a new formal arm: no
- Permission to change canonical Stage1 hyperparameters: no
- Reviewed at: 2026-08-07
