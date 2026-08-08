# P020 - Early-Learning Regularization Prevents Memorization of Noisy Labels

## Identity

- Paper ID: P020
- Authors: Sheng Liu, Jonathan Niles-Weed, Narges Razavian and Carlos Fernandez-Granda
- Venue and year: NeurIPS 2020
- Official page: https://proceedings.neurips.cc/paper/2020/hash/ea89621bee7c88b2c5be6681c8ef4906-Abstract.html
- Main paper: `source_papers/Early_Learning_Regularization_NeurIPS_2020.pdf`, SHA256 `D29F8D401B62E9E7CDABAAF5B45E08AB2CBB8CB0122A7D610F03573062C4F331`
- Supplement: `source_papers/Early_Learning_Regularization_NeurIPS_2020_Supplemental.pdf`, SHA256 `A09CBB6905E6A59B64F3F7826EB611D1E438E75445A32A8D119C426927471EA3`
- Official code: https://github.com/shengliu66/ELR
- Audited code commit: `934af53434a336b6db80d05d7649d23216e8ca6d`

## Reading Coverage

- Main paper: 12/12 pages read, including all methods, experiments, ablations, discussion and references.
- Supplement: 14/14 pages read, including the full proof, algorithms, experiment settings, hyperparameter sensitivity and timing analysis.
- Peer-review evidence checked: official NeurIPS reviews, meta-review and author response.
- Code checked: ELR and ELR+ loss implementations, training entry points, data construction, update order, random seeds, checkpoint/resume logic, configurations and dependencies.
- Visual verification: all 26 pages and seven contact sheets under `audit/visual_checks/P020_Early_Learning_Regularization_NeurIPS_2020/`.

## Research Question

Why do models trained with incorrect labels first predict the latent clean classes and later memorize the observed wrong labels, and can a temporal target regularizer preserve the useful early direction?

This is relevant to Stage1 because a repeatedly exposed subset can become disproportionately influential after the base examples have small residual gradients. It is not a direct match: Stage1 replay labels are not known to be wrong, and its operational target is a constrained normal-tail versus weak-defect tradeoff rather than average multiclass accuracy under label corruption.

## Formal Setup

The theory studies binary softmax regression. Clean examples are drawn from two Gaussian clusters with means `+v` and `-v`; an observed label is replaced by an independent random class with probability `Delta`. Consequently the actual wrong-label probability is `Delta / 2`, not `Delta`.

The central asymptotic regime is highly specific:

```text
p, n -> infinity
p / n in (1 - Delta/2, 1)
sigma sufficiently small
fixed gradient-descent step eta < 1
random initialization on the radius-2 sphere
```

Let `T` be the first iteration at which the classifier direction has correlation at least `0.1` with `v`. The supplement establishes, under its assumptions:

```text
T = order(1 / eta)
cos(-grad L(theta_t), v) >= 1/6 for t < T
accuracy on wrong-labeled examples rises from <= 0.5001 to > 0.9999 by T
mean squared clean-label gradient coefficients fall by more than 0.05
mean squared wrong-label coefficients rise by more than 0.05
```

The eventual memorization argument uses linear separability when `p / n > 1 - Delta/2`. It proves existence of a stylized high-dimensional regime, not a universal phase transition for deep networks. The meta-review explicitly questions the unstated scaling of `sigma` and whether the clusters become nearly identical; the final paper still states only that `sigma` is sufficiently small.

## Gradient Mechanism

For a deep classifier with logits `N_x(theta)` and probabilities `p`, the per-example cross-entropy contribution is

```text
grad_theta L_i = Jacobian_theta N_x(theta) * (p_i - y_i)
```

As clean examples become confidently correct, their `p-y` factors shrink. Conflicting examples can then dominate the aggregate gradient even if they are fewer. This is a relative-contribution claim: the dangerous group need not have an ever-growing absolute norm if the rest of the gradient vanishes faster.

ELR stores one temporal target per training identity:

```text
t_i(k) = beta * t_i(k-1) + (1-beta) * p_i(k)
```

and optimizes

```text
L_ELR = L_CE + (lambda/n) * sum_i log(1 - <p_i, t_i>)
```

The logarithmic term makes high agreement lower the objective. Its derivative adds a classwise vector `g_i`; when the early target's true-class component dominates, `g_i` preserves useful clean-example pressure and counteracts the reversed wrong-label component.

The negative ablations are as important as the positive result:

- replacing ELR with a KL penalty only delays memorization and can overfit the initial targets;
- temporal targets without the directional ELR term eventually follow the noisy labels;
- `beta=0`, which uses current outputs without averaging, falls to about 38% in the reported CIFAR-10 sensitivity experiment;
- too small a regularization coefficient fails to neutralize conflicting gradients, while too large a coefficient suppresses cross-entropy learning.

Therefore a large, stable or temporally averaged signal is not sufficient. Its update direction and balance against the base objective matter.

## Experimental Contract

- CIFAR-10 and CIFAR-100 use simulated symmetric and class-dependent noise; Clothing1M and mini-WebVision use real-world noisy annotations.
- Basic ELR uses ResNet-34, SGD momentum `0.9`, weight decay `0.001`, batch size `128`, 120 epochs on CIFAR-10 and 150 on CIFAR-100. The paper also reports a cosine-restart variant.
- ELR+ uses two PreActResNet-18 networks on CIFAR, EMA weights, cross-network targets and mixup. CIFAR-10 uses 200 epochs; CIFAR-100 uses 250.
- CIFAR uses a 45k/5k train/validation split. Hyperparameters are grid-searched on validation data.
- Table 1 and Table 5 report means and standard deviations over five noise realizations. Hyperparameter sensitivity uses four runs.
- Several state-of-the-art comparison values are copied from prior papers. Table 2 explicitly notes that competing methods often report the best validation accuracy during training while the authors use a held-out split, so columns are not fully comparable.
- The real-world gains are mixed: ELR+ is close to DivideMix on Clothing1M and WebVision, but DivideMix is materially better on ILSVRC12 top-1.

The paper does not report paired same-data, same-initialization schedule contrasts, all per-seed outcomes, sign-reversal probabilities, confidence intervals for the main comparisons, or multiplicity correction.

## Code Audit

The public implementation matches the main ELR loss and target update, but it is not a reproducible orchestration template for Stage1.

1. `random`, PyTorch and CUDA seeds are set, but NumPy is not seeded. CIFAR train/validation splits, synthetic label noise and mixup lambdas use NumPy randomness.
2. ELR+ constructs its two data loaders independently. Each construction re-randomizes the split and noise realization, so the two networks may not observe the same sample partition or corrupted labels.
3. The code evaluates the official CIFAR test loader after every epoch. Validation is the configured monitor, but repeated test access weakens operational blinding.
4. The basic ELR target tensor and ELR+ prediction histories are ordinary attributes, not registered persistent buffers. Checkpoints omit criterion state, EMA target histories, scheduler state and RNG states.
5. Basic ELR saves only a best checkpoint even though `save_period=1`; its resume path expects `checkpoint['config']`, which the save path does not write.
6. ELR+ writes epoch checkpoints directly, but its resume path also expects an absent `config` field and references nonexistent generic `self.model` and `self.optimizer` attributes. Resume is not viable as released.
7. Checkpoint writes are non-atomic and have no source, data, code or completion sidecars.
8. The dependency lock targets PyTorch 1.2.0, torchvision 0.4.0a0 and NumPy 1.16.4, but it does not lock CUDA, datasets or platform identity.
9. The repository has no scientific unit or integration test suite; `test.py` is an evaluation entry point rather than a test harness.
10. `update_ema_variables` references undefined loop variables if `ema_alpha=0`; released configurations avoid that branch, but it is still an implementation defect.

These defects do not by themselves refute the published results. They reduce the strength of seed-stability and exact-reproduction claims and reinforce the need for Stage1's canonical lock, immutable manifests, atomic writes and complete resume state.

## Direct Support For Stage1

1. Aggregate gradient dominance is dynamic and relative: once common/base examples are learned, a smaller conflicting group can control the update.
2. Per-example loss or gradient magnitude alone cannot determine value; gradient sign relative to separate operational targets matters.
3. All-epoch low-cost trajectories are justified because the transition is a process, not a single checkpoint property.
4. Separate group contributions should be measured: base clean support, replay normal correction and weak-defect harm can move differently.
5. Temporal smoothing, direction and exposure must be stored independently; a smoothed confidence score is not a substitute for directional evidence.
6. Same-state finite interventions should test whether an observed dominance transition actually predicts replay harm.
7. A replay decay hypothesis is scientifically plausible if late replay gradients become relatively dominant and anti-aligned with the weak-defect target, but this requires direct Stage1 measurement.

## What It Does Not Support

1. Treating selected Stage1 normals as mislabeled or noisy without audit evidence.
2. A universal early/late boundary, including epochs 140 or 160.
3. Any Stage1 replay percentage, cumulative dose or guard ratio.
4. Stopping replay solely because training loss is small or a confidence curve has plateaued.
5. Applying ELR as a new formal Treatment while changing the canonical 240-run objective or optimizer.
6. Calling high loss, high gradient norm or target disagreement a positive sample-value score.
7. Assuming a result is stable across initialization seeds from five noise realizations.
8. Using the paper's noisy-label average accuracy as evidence for the Stage1 raw `FN <= 95` safety frontier.

## Stage1 Field Contract

Persist at every epoch when inexpensive:

- base, replay and total loss separately, with classwise normal/defect components;
- mean and tail `|p-y|`, margin and correctness for base normal, replay normal, ordinary defect and weak-defect probes;
- target/probability EMA state used only as an observational feature, including momentum and update count;
- counts of learned, forgotten and never-learned identities by source group;
- realized replay occurrences and cumulative exposure per identity;
- group gradient norm, replay-to-base gradient-norm ratio and contribution share;
- separate dot products and cosines against difficult-normal and weak-defect target gradients;
- aggregate base-plus-replay gradient, optimizer-transformed update and finite probe-loss change;
- epoch, learning rate, optimizer step, schedule phase, seed, base-order hash, augmentation state and checkpoint hash.

At the preregistered gradient checkpoints, compute at minimum:

```text
dominance_normal = ||g_replay_normal|| / (||g_base|| + epsilon)
harm_weak_defect = dot(g_replay_normal, g_weak_defect)
benefit_hard_normal = dot(g_replay_normal, g_hard_normal)
actual_update_harm = -dot(g_weak_defect, delta_theta)
```

Sign conventions must be tested with a one-step finite difference. Group means, quantiles, concentration and cancellation must be retained; one scalar composite is not sufficient.

## Concrete Experiment Consequence

P020 strengthens, but does not numerically define, the first timing experiment. For the same frozen selected IDs and seed, compare no replay, continuous replay, same-peak decay and equal-cumulative-dose relocation under the exact canonical 240-run hyperparameter lock. Measure whether late exposure produces the predicted sequence:

```text
base gradient contribution shrinks
replay/base contribution ratio rises
replay remains useful for difficult normals
replay becomes anti-aligned with weak defects
actual updates lower normal scores while lowering weak-defect scores
```

If decay improves outcomes without this mechanism, the memorization analogy is not supported even if the arm succeeds. If the mechanism appears but outcomes do not improve, it is descriptive rather than decision-sufficient. No adaptive stopping rule should be introduced in the first confirmatory block.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, for relative gradient dominance, temporal target/update separation, directional regularization and required process fields
- Direct support for static replay ranking: no
- Direct support for numeric Stage1 schedule boundaries or percentages: no
- Reviewed at: 2026-08-07
