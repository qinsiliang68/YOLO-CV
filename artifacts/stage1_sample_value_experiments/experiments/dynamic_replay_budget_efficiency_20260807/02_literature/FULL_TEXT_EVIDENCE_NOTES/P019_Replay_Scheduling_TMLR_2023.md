# P019 - Learn the Time to Learn: Replay Scheduling in Continual Learning

## Identity

- Paper ID: P019
- Authors: Marcus Klasson, Hedvig Kjellstrom and Cheng Zhang
- Venue and year: Transactions on Machine Learning Research, September 2023
- Official page: https://openreview.net/forum?id=Q4aAITDgdP
- Full paper and appendix: `source_papers/Replay_Scheduling_TMLR_2023.pdf`, SHA256 `C10C3AC54DD62ED5D38ED16E0AD446194A6F2779011B2489FA0EBF1007829D4E`
- Official code: https://github.com/marcusklasson/replay_scheduling
- Audited code commit: `926b825cdf063f85e19ee385c9d182d5c8061e8e`

## Reading Coverage

- Main paper, references and appendix: 56/56 pages read.
- Method checked: discrete replay schedules, MCTS, UCT, validation reward, DQN and A2C policy learning.
- Experiments checked: all six single-environment datasets, varying memory sizes, tiny memory, recent replay methods, schedule transfer to seeds, new task orders and new datasets.
- Statistics checked: five-seed means, Welch tests, per-environment tables, rank aggregation and the absence of a visible multiplicity correction.
- Code checked: action space, MCTS node/search logic, trainer, replay batch construction, checkpoint reuse, seeds, train/validation split, result persistence, configuration and environment lock.
- Visual verification: all 56 pages and 14 contact sheets under `audit/visual_checks/P019_Replay_Scheduling_TMLR/`.

## Research Question

When replay capacity is limited in continual learning, can choosing *when and which previous tasks to replay* outperform uniform or random replay, and can a learned scheduling policy transfer to unseen task orders or datasets?

This is related to Stage1 replay timing, but the intervention is different. The paper schedules proportions of old *tasks* while learning sequential new tasks. Stage1 repeatedly adds a fixed selected subset to one stationary binary-classification training run.

## Formal Setup

A replay schedule is

```text
S = (p_1, ..., p_{T-1})
```

where `p_t` is a vector of proportions over the tasks seen before task `t+1`. A fixed memory of size `M` is filled according to these proportions. The paper discretizes the action space by placing `t-1` interchangeable dividers into `t-1` bins, then converts the resulting counts into task proportions.

For MCTS, each complete schedule is evaluated by retraining a continual-learning model. The reward is final average validation accuracy:

```text
R(S) = (1/T) * sum_i A_validation[T, i]
```

The UCT rule is unusual but explicit in both paper and code:

```text
UCT(v_child) = max(q(v_child))
               + C * sqrt(2 * log(n(parent)) / n(v_child))
```

Thus `q(v)` is a list of rollout rewards and exploitation uses its maximum, not its mean. This optimizes the best validation result observed under a schedule prefix and can be more sensitive to validation noise than ordinary mean-return MCTS.

The RL framework uses validation accuracies of learned tasks as state, a discrete task-proportion action, and average validation accuracy as reward. DQN and A2C are trained on precomputed continual-learning transition tables.

## Exact Replay Semantics

- Historical examples are stored in a fixed memory and task proportions choose which stored examples enter replay.
- A current-task mini-batch of size `B` is concatenated with a replay mini-batch of size `M`, or `B` replay examples when `M > B`.
- Current and replay observations receive equal per-example loss weight in the concatenated cross-entropy.
- The same selected replay pool is sampled throughout all epochs of a task.
- Replay selection and replay timing are not separated from task identity; the schedule changes composition at task boundaries, not at every epoch.
- The code deterministically seeds the current-task DataLoader by `seed + task_id` and uses a separate NumPy RandomState with the same expression for replay sampling.

## Experimental Contract

- Datasets: Split MNIST, Split FashionMNIST, Split notMNIST, Permuted MNIST, Split CIFAR-100 and Split miniImageNet; policy transfer additionally uses Split CIFAR-10.
- Continual-learning settings: task-, domain- and class-incremental variants.
- Models: two-layer MLPs, a small ConvNet and reduced ResNet-18, not a modern large image classifier.
- Optimizer: Adam with learning rate `0.001`, betas `0.9/0.999`, and no reset between tasks in central experiments.
- Batch sizes: 128 for MNIST-family tasks, 256 for CIFAR-100/CIFAR-10, and 32 for miniImageNet.
- Epochs per task vary by dataset; miniImageNet uses one epoch per task after a five-epoch first-task warmup.
- MCTS uses 100 rollouts and `C=0.1`.
- The central train/validation split reserves 15% of each task's training data using the run seed.
- Central tables report five seeds. Heuristic thresholds are grid-searched per dataset and setting.
- The RL environment search space has 1,050 complete schedules for five tasks. One Split MNIST environment takes about 9.5 hours and Split CIFAR-10 about 16.1 hours to precompute on a 2080Ti.

## Empirical Findings

1. The motivating experiment replays the same ten Task-1 samples once at different later tasks. Across five seeds, final ACC changes from 89.66% to 94.49%. This directly establishes that replay timing can matter even when memory identity and size are held fixed.
2. MCTS often beats equal-task scheduling and random scheduling, especially on long task horizons, but the advantage is not universal. Multiple memory-size and dataset cells are statistically indistinguishable, and some MCTS cells are worse than random or heuristic schedules.
3. A schedule found with one seed can sometimes transfer to five new seeds, but transferred schedules differ materially by source seed. There is no single schedule that dominates across all datasets.
4. Learned DQN/A2C policies are highly environment dependent. On new FashionMNIST environments they are generally worse than the strongest fixed heuristic and often near or below random/equal scheduling.
5. Tiny-memory experiments show that timing/composition can partly compensate for fewer replay examples, but not on every dataset. This is evidence for replay efficiency as an empirical possibility, not a universal law.
6. Forward-transfer estimates have high seed variance, which the authors explicitly acknowledge.

## Statistical Limitations

1. Central comparisons use five seeds, which is too small to characterize the kind of sign reversal already observed in Stage1.
2. The paper performs many pairwise Welch tests across datasets, memories and baselines without a visible family-wise or false-discovery correction.
3. Deterministic baselines are copied across RL policy seeds in rank calculations, producing zero variance and infinite Welch statistics in several appendix cells. Those p-values do not represent independent replications of the fixed baselines.
4. MCTS chooses the maximum validation reward after 100 full retrains. The selected schedule therefore includes an optimizer's-curse or multiple-search component that is not reflected by a fresh nested validation analysis.
5. Test accuracy is computed during every MCTS rollout in the public code. It is not used as the programmatic reward, but this weakens operational blinding and makes manual leakage possible.
6. The test-environment transfer study is closer to external validation, yet it still uses only five policy seeds or five transferred training seeds per reported condition.

## Code Audit

The public repository is substantial and pins an old CUDA/PyTorch environment. It includes configurations, transition tables and many result artifacts. However:

- `ReplaySchedulingNode.best_child` uses `max(c.q)`, matching the paper but not conventional mean-return UCT.
- `run_search` overwrites the outer `rs` history variable with the current rollout schedule and then appends a copy to that same list. `res['rs']` is therefore not a trustworthy audit history, although `best_rs` is separately retained.
- checkpoint cache names encode only task/action prefixes inside an output directory. They do not hash the full config, data split, code revision or initial model identity.
- `torch.save`, pickle and text writes are direct, with no temporary file, sidecar validation or completion marker.
- there is no automated unit/integration test suite for the scientific contracts.
- the MCTS README commands name `configs/rs_mcts`, while the checkout contains `configs/mcts`.
- raw downloaded datasets and preprocessed notMNIST/miniImageNet assets are not content-hashed.

These gaps do not invalidate the paper's reported result, but they make the repository unsuitable as the orchestration substrate for the final Stage1 campaign.

## Direct Support For Stage1

1. Replay timing is a causal variable distinct from memory identity and nominal memory size.
2. Schedule value is conditional on dataset, model state, task/history context and seed; it is not a permanent scalar attached to a replay set.
3. A fixed selected set should be tested under multiple preregistered time schedules before creating another static ranking.
4. Replay efficiency must be reported as benefit per realized occurrence or optimizer example, not only final quality.
5. Schedule discovery and schedule confirmation need separate seeds; adaptive validation search must not be reported as confirmatory evidence.
6. Full per-seed results, worst-seed behavior and sign consistency are necessary because mean gains conceal schedule failures.
7. Replay composition, memory identity, batch co-occurrence and timing must be persisted independently.

## What It Does Not Support

1. A universal cutoff at epoch 140, decay through epoch 160, or any other Stage1 numeric boundary.
2. Any Stage1 replay percentage. Its memory sizes and task proportions have different denominators and semantics.
3. Direct transfer from continual task replay to stationary duplicate-sample replay.
4. Using MCTS, DQN or A2C to adapt the first confirmatory Stage1 block after looking at the same validation outcomes.
5. Replacing no-replay, equal-dose and random/matched controls with an intelligent scheduler.
6. A static per-image value score, gradient-magnitude rule or confidence-only selection.
7. Averaging difficult-normal benefit and weak-defect harm into one unconstrained validation reward.
8. Changing the canonical Stage1 optimizer, learning rate, augmentation or batch size by schedule arm.

## Stage1 Field Contract

Persist at every epoch when inexpensive:

- nominal and realized replay ratio;
- integer replay slots and cumulative replay occurrences;
- selected identity, class, selection-set hash and schedule hash;
- base examples, replay examples, total optimizer examples and optimizer steps;
- first and last replay occurrence for each selected identity;
- batch-level replay count, replay/base mix and repeated-identity concentration;
- current learning rate, optimizer state summary and actual update norm;
- difficult-normal and weak-defect probe trajectories separately;
- schedule phase and distance since phase transition;
- run seed, base-order hash, augmentation seed/state and initial-weight hash;
- validation/search role and whether a result is discovery, transfer or locked confirmation;
- machine, duration, resource phases, failure/retry state and artifact completion status.

At preregistered checkpoints, collect separate normal-tail and weak-defect gradient alignment plus actual-update alignment. Do not collapse them into one average reward.

## Concrete Experiment Consequence

The first Stage1 cycle should use one frozen selection and one paired seed block to compare at least:

```text
NR: no replay
C: continuous replay
D_peak: same peak ratio followed by preregistered decay/stop
D_dose: time-relocated schedule with cumulative integer replay slots matched to C
```

This isolates whether any gain comes from removing late exposure or merely lowering total dose. The schedule boundaries must come from a separate preregistered rationale and existing Stage1 trajectory analysis, not from optimizing all candidate boundaries on the same six seeds.

Weak-defect guard is a separate causal factor and should enter only after, or factorially with, the minimal timing/dose block if capacity allows. A learned/adaptive scheduler is a later discovery tool only after fixed schedules show that the timing mechanism exists.

All arms must load a machine-generated lock from the prior 240-run artifacts. Except for seed and the preregistered replay intervention, model, batch, image size, optimizer resolution, learning-rate path, augmentation, base data/order and initial-weight identities must match.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, for replay timing as a causal variable, schedule context dependence, discovery/confirmation separation and detailed exposure accounting
- Direct support for static replay ranking: no
- Direct support for numeric Stage1 schedule boundaries or percentages: no
- Reviewed at: 2026-08-07
