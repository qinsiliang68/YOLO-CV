# P033 - AutoSampling: Search for Effective Data Sampling Schedules

## Identity

- Paper ID: P033
- Authors: Ming Sun, Haoxuan Dou, Baopu Li, Lei Cui, Junjie Yan, and Wanli Ouyang
- Venue and year: ICML 2021, PMLR 139:9923-9933
- Published page: https://proceedings.mlr.press/v139/sun21a.html
- Main PDF: `source_papers/AutoSampling_2021.pdf`, SHA256 `B9218612E2072FD3F047408F0440CD807425427D78DEB12942EB84BFB0F6A745`
- Supplement: none listed by PMLR
- Official code: none linked by PMLR or the paper and none identified by exact-title search
- Related public-review record: https://openreview.net/forum?id=AJTAcS7SZzf for an earlier ICLR 2021 submission

## Reading Coverage

- Main manuscript: 11/11 pages read, including Equations 1-7, Algorithms 1-2, all implementation details, Tables 1-6, Figures 1-4, every ablation, the static/dynamic comparison, image/frequency analysis, discussion, conclusion, and references.
- Visual verification: all 11 pages inspected at original detail under `audit/visual_checks/P033_AutoSampling_ICML_2021/`.
- Supplement: no supplement is linked by the official PMLR record.
- Code audit: unavailable. No official implementation or paper snapshot was identified.
- Public review: the earlier OpenReview forum identity was checked, but the review bodies were not used as evidence because they were not retrievable through the public page in this session.

## Research Question

The paper treats the complete ordered sequence of sampled training identities as a high-dimensional hyperparameter and asks whether a population search can learn a useful schedule without a handcrafted loss, confidence, imbalance, or noise proxy.

Its central object is explicitly dynamic:

```text
h = (x_1, ..., x_N) in D^N
```

where order and repeated appearances are part of the schedule. This is closer to conditional replay value than a static top-k list. However, the optimization procedure also performs repeated validation-based model selection and weight cloning, so the reported result is not the isolated effect of sample scheduling.

## Core Method

AutoSampling alternates two stages.

### Multi-exploitation

A population of child models starts from the same cloned state. Each worker consumes a different sampled sub-schedule for a short interval. The worker with the best held-out validation score is selected, its sub-schedule is appended to the winning schedule, and its model weights are copied to every worker.

```text
(h_t_star, theta_t_star) = argmax_i eval(theta_i, h_t_i)
theta_i <- theta_t_star for every worker i
```

The reward therefore depends jointly on:

```text
sample sequence
stochastic update path
current model state
short evaluation window
winner selection among N_p workers
repeated validation queries
```

It is not an identity-level data value.

### Distribution exploration

The frequency of identities in the winning schedule estimates a sampling distribution:

```text
P(x) = count(x in H_star) / sum_z count(z in H_star)
```

The raw distribution becomes extremely skewed and gives most identities zero probability. The method therefore applies logarithmic smoothing and mixes in `N_u` uniform schedules before drawing the next population of sub-schedules.

The final artifact is an ordered dynamic schedule plus a history of estimated distributions, not one static ranking.

## Experimental Protocol

- CIFAR-10 and CIFAR-100 use 50,000 training images, 240 epochs, batch 128 per V100 worker, base learning rate 0.1, and step decay every 60 epochs.
- The first distribution exploration occurs after 20 epochs; later exploration occurs every `N_u + 1` epochs with `N_u = 3`.
- The principal CIFAR configuration uses 20 workers and 20-batch exploitation intervals. It costs 4,800 aggregate worker-epochs and about 14 hours.
- ImageNet uses 100 epochs, eight workers, eight V100s per worker, total batch 512 per worker, base learning rate 0.2, cosine decay, and FP16. It costs 800 aggregate worker-epochs and about four days.
- The method compares one-worker uniform training, random exploration with population winner selection, and mixture exploration with the learned nonuniform distribution.
- The paper reports `mean +/- value` for most CIFAR cells but never defines the repetition count or whether the value is standard deviation, standard error, or another quantity. ImageNet cells have no uncertainty.
- All numeric settings are literature context only and cannot modify the Stage1 canonical hyperparameter lock.

## Positive And Negative Evidence

1. Dynamic schedules outperform schedules sampled from the final aggregate distribution in Table 4. Stage-dependent exposure matters in this setting.
2. Sampling distributions at epochs 80, 160, and 240 differ visibly. A single frequency or rank computed at one checkpoint cannot represent the whole path.
3. Frequency and loss are not visibly correlated for a random set of 500 images at three epochs. Loss-only selection is not an adequate reconstruction of the learned schedule, although the paper reports no correlation coefficient or uncertainty.
4. Static distributions transfer modestly across ResNet-18, ResNet-50, and DenseNet-121 in Table 5. This supports a possible model-shared component but does not establish a model-independent scalar value.
5. The learned distribution can become so concentrated that most samples receive zero frequency and training destabilizes. Uniform exploration and smoothing are necessary risk controls.
6. High-frequency examples look more obscure in a small qualitative figure, but low-frequency examples include both clear easy cases and malformed/low-quality cases. Visual hardness, noise risk, and usefulness remain mixed.
7. Shorter 20-batch reward windows beat 80-batch windows in the reported experiments. This shows reward horizon matters, not that a short reward is unbiased or transferable to Stage1.
8. Increasing from 20 to 80 workers yields only marginal gains, so more search compute is not linearly useful.
9. Mixture exploration improves over random exploration for ResNet-18/50 on CIFAR-100 and ImageNet, but not for DenseNet-121 or CIFAR-10. The learned nonuniform distribution is not consistently better than population search with uniform samples.
10. Random exploration already produces large gains over one-worker uniform training. Because it still samples uniformly but selects the best short-run worker and clones its weights, much of the gain can arise from stochastic-path selection rather than discovery of valuable identities.
11. Dynamic versus static also changes whether online validation winner selection and weight cloning occur. It is not a clean causal estimate of temporal sampling alone.
12. Repeatedly maximizing held-out validation performance among many workers creates winner's-curse and validation-adaptation risk. The paper does not report an untouched selection-calibration split or query-count correction.
13. The exact training/validation/test split, seed list, repetition count, uncertainty definition, paired design, and worst-seed behavior are not documented sufficiently for reliability claims.
14. Comparisons to prior methods in Table 6 use different architectures and roughly aligned rather than identical protocols, so percentage improvements are not a direct controlled ranking.
15. No official code, environment lock, schedule artifact, per-worker trajectory, RNG state, or executable reproduction path is provided.

## Direct Support For Stage1

1. Treat replay value as a path object: identity, order, model state, epoch, repeat count, and neighboring samples all matter.
2. Save per-epoch intended and realized identity frequencies, cumulative exposures, unique coverage, duplicate concentration, and distribution-drift metrics.
3. Compare schedule distributions across epochs using rank correlation, top-k churn, Jensen-Shannon divergence, class/video/cluster concentration, and tail-role transitions.
4. Keep static selection and timing schedule as separate immutable manifest dimensions.
5. Record short-window finite-intervention reward separately from endpoint reward and test whether reward signs persist across horizons and seeds.
6. If any adaptive search is later introduced, persist parent checkpoint, branch ID, worker ID, candidate sub-schedule, evaluation score, selection event, clone event, validation query count, policy state, and all RNG states.
7. Use an identity-disjoint selection probe and a still-blind final holdout. Repeated validation winner selection cannot share the final confirmation endpoint.
8. Preserve a uniform exploration floor or explicit coverage constraint if a learned policy can drive identities to zero exposure.
9. Decompose any adaptive gain into population winner selection, static nonuniform distribution, and temporal distribution changes with factorial controls.
10. For the current campaign, avoid population branching entirely so continuous and decay arms differ only in preregistered replay timing/dose.

## What It Does Not Support

1. Calling the AutoSampling frequency an intrinsic fixed value of an image.
2. Claiming that dynamic replay is proven for Stage1's `FN <= 95` objective.
3. Adding a population-based adaptive arm to the first formal campaign.
4. Treating Random Exploration as equivalent to Stage1 global-random replay; it includes validation winner selection and weight cloning.
5. Inferring that short-term reward is causal sample value without correcting winner selection, model-state inheritance, and validation reuse.
6. Importing its epochs, learning rates, batch sizes, worker counts, interval lengths, smoothing, FP16 policy, architectures, or schedules.
7. Changing any Stage1 canonical hyperparameter.

## Stage1 Field Contract

Under the exact canonical hyperparameter lock, add or retain:

- all-epoch replay identity, order, intended probability, realized probability, presentation count, and cumulative exposure;
- per-epoch distribution rank correlation, top-k churn, Jensen-Shannon divergence, entropy, effective sample size, Gini/concentration, and zero-exposure count;
- short-, medium-, and endpoint finite-intervention reward on separate normal-tail and weak-defect probes;
- reward horizon, probe identity, query count, score age, state/checkpoint hash, and sign consistency across horizons;
- base-stream and replay-stream RNG/sampler state plus exact resume lineage;
- candidate, parent, branch, worker, selection, and clone metadata if branching is ever used;
- unique validation role identities and cumulative adaptive query count;
- canonical lock hash on every schedule, checkpoint, resume record, and report.

These fields do not change `yolo11l`, 200 epochs, batch 128, image size 224, workers 4, AMP, optimizer, learning-rate schedule, augmentation, or any other canonical setting.

## Concrete Experiment Consequence

P033 adds no formal arm. It sharpens the current first-cycle comparison:

```text
same selection + same seed + same base stream + no branching
continuous vs same-peak decay vs dose-matched decay
```

The collector should measure whether the selected set's realized exposure distribution and independent tail reward change across the full path. If early local reward does not predict paired endpoint safety utility, it is rejected as a value proxy. If schedule distributions drift but the fixed timing intervention has no benefit, dynamic descriptive variation is not evidence that an adaptive policy will help.

## Stage1 Decision Status

- Evidence status: `FULL_READ_COMPLETE`
- Claim eligibility: yes, for dynamic schedule variation, static/dynamic differences, loss-frequency mismatch, concentration failure, and population-search confounding
- Replication-depth eligibility: no; no official code, supplement, complete seed protocol, or executable reproduction artifact was identified
- Direct support for static replay ranking: no
- Direct support for conditional value and all-epoch exposure fields: yes
- Direct support for a new formal arm: no
- Permission to change canonical Stage1 hyperparameters: no
- Reviewed at: 2026-08-08
