# P031 - Automated Curriculum Learning for Neural Networks

## Identity

- Paper ID: P031
- Authors: Alex Graves, Marc G. Bellemare, Jacob Menick, Remi Munos, and Koray Kavukcuoglu
- Venue and year: ICML 2017, PMLR 70:1311-1320
- Published page: https://proceedings.mlr.press/v70/graves17a.html
- Main PDF: `source_papers/Automated_Curriculum_2017.pdf`, SHA256 `AE2C83B04CEB61E45224E8C3FC26EC7EF1F7CF22398CAA3FFF3256B7D15EA7C1`
- Supplement: none listed by PMLR
- Official code: none listed by PMLR and none identified from the paper

## Reading Coverage

- Main manuscript: 10/10 pages read, including Equations 1-8, Algorithm 1, every progress signal, all three task suites, all figures, negative results, conclusion, and references.
- Visual verification: all 10 pages inspected at original detail under `audit/visual_checks/P031_Automated_Curriculum_ICML_2017/`.
- Supplement and public review: no supplement or public reviewer reports are linked by the official PMLR record.
- Code audit: not available. Numeric reproduction and exact interruption behavior cannot be verified from an official implementation.

## Research Question

The paper asks how to adapt a stochastic curriculum over tasks so that a network spends training time where it is currently making progress. The action is a task distribution, not an individual image identity, and the objective is learning speed on synthetic sequence problems rather than final raw `FN=0-95` safety-frontier utility.

Its transferable conceptual distinction is:

```text
instantaneous self gain       -> change on the just-trained sample
same-task fresh-sample gain   -> local generalization to an independent sample
target gain                   -> change on the desired target distribution
gradient squared norm         -> first-order self-gain approximation
Stage1 value                  -> paired tail-safe outcome under a full replay path
```

These quantities are not interchangeable.

## Core Formulation

A curriculum is an ensemble of tasks and a syllabus is a time-varying distribution over them. The authors use the nonstationary Exp3.S bandit because the best task can change as the model learns. The policy retains an exploration floor and updates importance-corrected task rewards.

The raw reward is a progress signal divided by estimated processing time. Because reward scales are unknown and nonstationary, the method stores a reservoir sample of reward history, clips at approximate 20th and 80th percentiles, and maps the result to `[-1, 1]`.

The loss-driven progress signals are:

```text
PG  = L(x, theta) - L(x, theta_after_x)
GPG = ||grad L(x, theta)||^2
SPG = L(x_fresh, theta) - L(x_fresh, theta_after_x), x_fresh from same task
TPG = target loss before - target loss after
MPG = uniformly sampled task loss before - after
```

The complexity-driven signals use changes or directional derivatives of variational KL complexity or L2 norm.

Under the paper's local first-order and SGD assumptions, true expected progress is proportional to the squared norm of the mean gradient, whereas expected prediction gain is proportional to the mean squared gradient norm. Therefore:

```text
expected PG = true expected progress + gradient variance.
```

For equal true progress, a PG curriculum prefers the higher-variance task. GPG additionally depends on the local Taylor approximation. SPG is unbiased under the derivation because it evaluates the update on an independent same-task draw, but it has higher estimator variance.

This is direct mathematical evidence that a large gradient or large same-sample loss drop can reward variance rather than useful generalization.

## Experimental Protocol

- All experiments use stacked unidirectional LSTMs and cross-entropy optimized with RMSProp and momentum.
- Every experiment is repeated ten times with different network initializations.
- Time is total processed input steps rather than epochs or optimizer updates.
- Baselines are uniform task sampling and, where meaningful, direct target-task training.
- Independent samples not used for training or reward calculation measure reported loss and error, except in the modified bAbI experiment described below.
- N-gram language modeling uses 11 generated task distributions and two-layer 512-cell LSTMs.
- Repeat Copy uses 169 tasks from two difficulty axes and a one-layer 512-cell LSTM.
- bAbI uses 20 tasks but replaces the original small benchmark with one million generated stories per task; training and evaluation performance are reported as indistinguishable, so training performance is shown.

All architectures, optimizers, learning rates, bandit parameters, clipping quantiles, exploration rates, batch sizes, generated-data sizes, and stopping horizons are literature context only. None may modify the Stage1 canonical hyperparameter lock.

## Positive And Negative Evidence

1. The optimal curriculum policy is explicitly nonstationary. A task can be useful during one phase and unhelpful later, supporting conditional value rather than a static ranking.
2. The policy can move across multiple difficulty axes and need not visit every task. This supports measuring transfer and redundancy between selected groups rather than equating presentation count with information.
3. Same-sample prediction gain and gradient squared norm mix learning progress with gradient variance. Large magnitude is not a reliable positive-value criterion.
4. Independent same-task and target probes better approximate generalization effects, but they increase variance and can be uninformative when the target is too difficult early in training.
5. Uniform sampling is described as a surprisingly strong benchmark. Adaptive sampling is only useful when the progress signal is suitable.
6. On Repeat Copy, GPG, GL2G, and L2G are much worse than uniform. Direct target training fails entirely, while GVCG is about twice as fast as uniform under variational training.
7. On bAbI, PG improves over uniform, SPG improves less, other signals are roughly equal or worse, GVCG starts faster and becomes slightly worse later, VCG performs poorly for unknown reasons, and variational inference generally hampers progress.
8. The N-gram curriculum is superfluous because direct training on the target task is already most efficient. A successful-looking policy visualization does not prove the curriculum intervention was necessary.
9. Ten initialization replicates and standard-deviation bands are better than one trajectory, but no confidence intervals for paired differences, formal statistical tests, worst-seed analysis, or selection-versus-seed variance decomposition are reported.
10. The tasks are synthetic sequences grouped into known task arms. The paper does not solve identity-level image selection, replay duplication, label noise, weak-defect protection, or a constrained operating frontier.
11. Reward clipping, exploration, and reservoir state are part of the adaptive path. A checkpoint that stores only model weights would not reproduce the same syllabus.
12. Dividing reward by processing time changes the objective from pure statistical utility to progress per compute. Stage1 should report compute efficiency separately rather than silently folding it into scientific sample value.

## Direct Support For Stage1

1. Treat replay value as state- and schedule-dependent and measure it over the full 200-epoch path.
2. Separate pre/post loss change on the replay sample from change on an independent same-stratum probe and from change on difficult-normal and weak-defect targets.
3. Estimate within-stratum gradient variance separately from squared mean-gradient norm. Their difference diagnoses magnitude-based selection bias.
4. Record policy or replay score freshness, model-state hash, score age, intended probability, exploration floor, realized presentation count, and cumulative exposure.
5. Record progress-signal histories, clipping or normalization state, policy entropy, task/group allocation, and collector RNG state if any adaptive policy is later tested.
6. Measure transfer: after training one replay group, evaluate score changes in unpresented related groups and in both operational tails.
7. Report progress per optimizer step and per wall-clock/GPU-second as secondary operational metrics, while keeping raw safety utility separate.
8. Use same-selection, same-seed timing contrasts to test whether progress turns negative or weak-defect harm appears after prolonged exposure.

## What It Does Not Support

1. Defining `||gradient||^2` as sample value.
2. Calling a large same-sample loss decrease generalized benefit.
3. Replacing the preregistered Stage1 timing experiment with an adaptive bandit arm.
4. Importing Exp3.S parameters, clipping quantiles, exploration rates, model architecture, optimizer, batch size, learning rate, or training horizon.
5. Inferring a universal replay stop epoch from synthetic sequence curves.
6. Claiming an image-level value function, cross-seed stability, weak-defect safety, or raw-frontier improvement.

## Stage1 Field Contract

Under the exact canonical hyperparameter lock, add or retain:

- all-epoch base and replay loss before update;
- low-cost post-update loss on the replay batch where instrumentation is behaviorally equivalent;
- at key checkpoints, pre/post loss on an identity-disjoint same-stratum probe, difficult-normal probe, and weak-defect probe;
- per-stratum `mean_grad_norm_sq`, `norm_mean_grad_sq`, and their difference as a gradient-dispersion estimate;
- separate normal-tail and weak-defect gradient dot products and cosines;
- progress-signal mean, variance, sign, quantiles, stale age, and cross-checkpoint sign consistency;
- replay group/task identity, group probability, allocation entropy, exploration status, presentation count, cumulative exposure, and transfer to unpresented related groups;
- training compute time, DataLoader wait, evaluation time, and progress per GPU-second as operational fields;
- adaptive-state, RNG, reservoir, quantile, policy, and checkpoint hashes if an adaptive controller is ever introduced.

The existing all-epoch lightweight collector and six heavy checkpoints remain the appropriate storage pattern. None of these fields changes `yolo11l`, 200 epochs, batch 128, image size 224, workers 4, AMP, optimizer, learning-rate schedule, augmentation, or any other canonical setting.

## Concrete Experiment Consequence

P031 adds no formal arm. It sharpens the timing mechanism and its falsifier:

```text
large self gain + large gradient variance + weak independent-probe gain
    => high-variance leverage, not demonstrated value

positive independent normal-tail gain + non-negative weak-defect gain
    => local gap-positive evidence

gain positive early and negative late for the same replay group
    => stage-conditional value and support for timing intervention

uniform equal to or better than adaptive proxy
    => proxy rejected, not evidence that useful samples do not exist
```

The formal causal block remains no replay, continuous replay, same-peak decay, and cumulative-dose-matched decay on the same frozen selection and seed. An adaptive controller would require a later evidence card and separate preregistration.

## Stage1 Decision Status

- Evidence status: `FULL_READ_COMPLETE`
- Claim eligibility: yes, for nonstationary curricula, prediction-gain bias, the gradient-variance decomposition, and mixed/negative adaptive-sampling results
- Replication-depth eligibility: no; no official code or supplement was identified and no benchmark was rerun
- Direct support for static replay ranking: no
- Direct support for gradient magnitude as value: no
- Direct support for independent progress probes and stateful schedule fields: yes
- Direct support for a new formal arm: no
- Permission to change canonical Stage1 hyperparameters: no
- Reviewed at: 2026-08-07
