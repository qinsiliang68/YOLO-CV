# P007 - An Empirical Study of Example Forgetting during Deep Neural Network Learning

## Identity

- Paper ID: P007
- Authors: Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, Geoffrey J. Gordon
- Venue and year: ICLR 2019
- OpenReview paper: https://openreview.net/pdf?id=BJlxm30cKm
- ArXiv full text: https://arxiv.org/pdf/1812.05159
- Local PDF: `source_papers/Example_Forgetting_2019.pdf`
- PDF SHA256: `0CEEF9C602D22B977BAF61E1E62897EE0D3D1A043E93D34129B3A448A9120C89`
- Page count: 19, including appendices
- Code: https://github.com/mtoneva/example_forgetting
- Code snapshot inspected: commit `d51c1aaa51c5a5cd3b8a584f8d5dee2c8957653e`, dated 2020-10-02
- Relevant code blobs: `run_cifar.py` = `f00187eb9d339bc7c1f90475cc355cb23d9e29a2`; `order_examples_by_forgetting.py` = `276962ebb862d5a38db4cf19f8b14afbacb23a4a`

## Reading Coverage

- Full PDF: 19/19 pages read.
- Sections checked: definition and procedural estimator; seed stability; first learning; margin relation; synthetic label and pixel noise; continual-learning intervention; removal experiments; architecture/time transfer; chance-gradient control; 100-seed confidence analysis; CIFAR-100 duplicate-label audit.
- Experimental details checked: all architectures, optimizers, learning rates, schedules, augmentation descriptions, seed counts and reported uncertainty.
- Public code checked: per-presentation collection, sample-identity mapping, sorting over runs, never-learned handling, removal workflow, seed controls and dependency versions.
- Visual verification: pages 3, 4, 5, 7, 8, 14, 16 and 18 under `audit/visual_checks/P007_Example_Forgetting/`.

## Research Question

Do individual examples repeatedly transition from correct to incorrect during ordinary single-task SGD training, are those transitions stable, and do they identify removable easy examples or important boundary-supporting examples?

## Definitions

For sample `i` at presentation `t`, define:

```text
acc_i^t = 1[argmax_y p(y | x_i; theta_t) == y_i]
```

A forgetting event occurs when:

```text
acc_i^t = 1 and acc_i^(t+1) = 0
```

A learning event is the reverse transition. An example is unforgettable only if it is learned at least once and never subsequently forgotten. Never-learned samples are assigned maximal/infinite forgetting for sorting, not placed in the unforgettable group.

The classification margin is:

```text
margin_i = logit(correct class) - max logit(other classes)
```

Crucially, the practical estimator does not evaluate every sample after every optimizer step. It observes each sample only when that sample is presented in a mini-batch and compares consecutive presentations. It is therefore a lower bound on all possible parameter-step-induced forgetting events.

## Experimental Contract

- Datasets: MNIST, fixed-permutation MNIST, CIFAR-10 and appendix CIFAR-100.
- CIFAR-10 uses ResNet-18 with data augmentation/cutout, SGD, Nesterov momentum 0.9, initial learning rate 0.1 divided by 5 at epochs 60, 120 and 160, and 200 epochs.
- MNIST uses a small convolutional network, SGD learning rate 0.01 and momentum 0.5.
- Architecture transfer uses a smaller CNN and a WideResNet-28-10; the WideResNet uses Adam at 0.001 in the stated setup.
- Core forgetting histograms use five seeds; pairwise stability uses ten seeds.
- Confidence analysis uses 100 seeds, grouped into 20 averages of five seeds, and reports empirical 2.5/97.5 percentile bands.
- Removal and continual-partition figures report means and standard errors across five seeds.
- Synthetic noise changes 20% of labels or pixels; an appendix also noised every image at several pixel-noise scales.

## Main Results

- Across five seeds, examples common to the unforgettable group make up 91.7% of MNIST, 75.3% of permuted MNIST and 31.3% of CIFAR-10.
- Across ten seeds, per-example forgetting counts have average Pearson correlation 89.2%. Two random groups of five runs produce cumulative counts correlated at 97.6%.
- First-learning presentation and forgetting count have only moderate Spearman correlation, 0.56; the signals are related but not equivalent.
- Forgetting count and mean misclassification margin have Spearman correlation -0.74.
- Randomly relabeled or heavily pixel-corrupted examples shift strongly toward high forgetting; no synthetic mislabeled example is unforgettable in the reported run.
- Removing least-forgotten examples retains performance much longer than random removal. On CIFAR-10, about 30% can be removed without material loss in the reported setup.
- Removing blocks with progressively higher forgetting generally harms generalization more, but the far-right extreme reverses upward: some most-forgotten samples are harmful outliers or mislabeled examples.
- Ranking stabilizes progressively; the paper reports strong retrieval by roughly 75 epochs, but the exact time is task/schedule-specific.
- A smaller architecture's ordering can remove 30% of CIFAR-10 while a WideResNet retains near-optimal performance in the reported setup.

## Ablations And Failure Cases

### Extreme forgetting mixes value and corruption

The most-forgotten tail contains both useful boundary-supporting examples and harmful/noisy samples. The paper explicitly identifies the upward turn in the right end of the removal curve. Thus forgetting count is not a monotone value score.

### One-run ordering is less reliable

The strongest stability comes from aggregating several seeds. This directly rejects treating one trajectory as a definitive ranking.

### Dataset complexity changes the distribution

The proportion of unforgettable examples varies enormously across MNIST, CIFAR-10 and CIFAR-100. Numeric thresholds and removal percentages do not transfer.

### Presentation sampling is a lower bound

Only transitions between consecutive presentations are counted. A sample can be forgotten and relearned between its own presentations without being observed.

### Augmentation is part of the measurement

In the public CIFAR script, the recorded pre-update prediction is produced after the training transform. A transition can therefore reflect a different stochastic augmentation as well as parameter interference.

### It is pruning, not additive replay

The causal test removes easy examples and retrains from scratch. It does not duplicate hard samples, vary exposure schedules, protect a rare tail or optimize an FN-constrained frontier.

### Cross-seed stability is strongest for the easy end

The 100-seed confidence discussion specifically emphasizes tight intervals for least-forgotten examples. It does not prove precise stable ordering within the extreme hard/noisy tail.

## Code Audit And Reproduction Gaps

The released code records pre-update per-presentation loss, correctness and margin for each original sample ID and persists the full dictionary after every epoch. The sorting script sums forgetting counts across all matched run files and assigns `npresentations` to a never-learned sample in each run.

The repository is unusually useful for reconstructing the main pipeline, but it has several limitations:

- `requirements.txt` pins old `torch==0.4.1.post2` and `torchvision==0.1.8`; the code uses deprecated APIs.
- The margin code is evaluated on stochastic training transforms, not a fixed canonical view.
- `compute_forgetting_statistics` initializes `margins_per_presentation` as a dictionary but overwrites it with the current sample's NumPy array instead of assigning by sample ID. The returned margin object therefore only represents the last processed sample. The forgetting-count path itself is separate and unaffected.
- Results depend on file-name matching and mutable pickles; there is no content-hashed manifest or sample-list sidecar.
- Exact CUDA/cuDNN versions and all machine-level determinism settings are absent.

## What It Supports For Stage1

1. Per-sample training dynamics carry information not present in one terminal confidence score.
2. Correct-to-incorrect transitions, first-learned time and margin trajectory should be separate fields.
3. Multi-seed aggregation is essential before declaring a trajectory rank stable.
4. Stable easy examples can be identified more reliably than the exact ordering of the hardest tail.
5. High forgetting is a candidate-review signal that must be separated into useful hard, corrupted, ambiguous and redundant strata.
6. Replay duplicates require occurrence-aware presentation logs because selected samples may appear more than once per epoch.
7. Fixed-view checkpoint evaluation should be separated from stochastic augmented presentation dynamics.

## What It Does Not Support

1. It does not support `more forgetting = more replay value`.
2. It does not support a fixed forgetting threshold, percentile, replay ratio or stop epoch for Stage1.
3. It does not show that forgetting improves `TN_at_FN95` or protects weak defects.
4. It does not study same-selection additive replay across seeds.
5. It does not show that an OOF held-out epoch trajectory is the same phenomenon as in-training presentation forgetting.
6. It does not distinguish all clean hard samples from label noise in the extreme tail.
7. It does not justify opening blind test data or using test trajectories for selection.

## Transfer Boundary

Stage1 currently has 10-fold OOF checkpoint trajectories. For each sample, its OOF model did not train on that sample. An OOF correct-to-incorrect transition is therefore a held-out generalization reversal, not the paper's training-presentation forgetting event.

The two measurements must be named and modeled separately:

```text
train_presentation_forgetting:
  transition on consecutive augmented mini-batch presentations

oof_epoch_correctness_reversal:
  transition across fixed-view held-out checkpoint predictions
```

Their agreement, disagreement and seed stability are empirical questions. Neither may be silently substituted for the other.

## Concrete Field Requirements

For every base or replay presentation where collection cost is acceptable:

- `sample_id`, `epoch`, `global_step`, `presentation_index` and `occurrence_in_epoch`;
- `is_replay`, replay source/rule and cumulative exposure before this presentation;
- pre-update loss, correctness, correct-class logit, strongest-other logit and margin;
- learning/forgetting transition from the previous presentation;
- data transform identity or at minimum deterministic augmentation seed/occurrence key;
- time since previous presentation and optimizer-step distance;
- never-learned flag, first-learned presentation and cumulative forgetting count.

For fixed-view OOF or validation checkpoint trajectories:

- epoch-level correctness reversal count;
- probability and margin first differences;
- sign-change count, maximum adverse drop and recovery time;
- trajectory summaries per seed and cross-seed rank/sign agreement.

The two namespaces must remain separate in schemas and reports.

## Concrete Experiment Consequence

- Add presentation-level lightweight fields to the final training worker without changing canonical optimization behavior.
- Keep all-epoch fixed-view telemetry, but label it OOF/validation reversal rather than classic forgetting.
- Use forgetting to stratify candidate pools, not to define the Treatment arm by itself.
- In any forgetting-selected pilot, exclude or separately audit never-learned/extreme-noise samples and match random controls on class, fold, video and trajectory severity.
- Compare single-seed and multi-seed stability before promoting any rank.
- Test whether replay timing changes forgetting/reversal dynamics before using those dynamics as an adaptive controller.

## Reproduction Notes And Missing Information

- The paper and code together provide definitions, algorithms, training settings, seed ranges and end-to-end commands for the central results.
- The public code is sufficient to reproduce the conceptual pipeline after adapting deprecated dependencies and correcting/auditing the margin-return bug.
- Exact source-data hashes, environment hashes, seed-specific sorted-ID manifests and hardware determinism settings are not supplied.
- `REPLICATION_DEPTH` records a complete method/artifact audit; no CIFAR benchmark rerun was performed here.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, with explicit presentation/OOF distinction and hard-tail contamination caveats
- Direct support for multi-seed training-dynamics fields: yes
- Direct support for forgetting-only replay or numeric Stage1 parameters: no
- Reviewed at: 2026-08-07
