# P021 - Identifying Mislabeled Data using the Area Under the Margin Ranking

## Identity

- Paper ID: P021
- Authors: Geoff Pleiss, Tianyi Zhang, Ethan Elenberg and Kilian Q. Weinberger
- Venue and year: NeurIPS 2020
- Official page: https://proceedings.neurips.cc/paper_files/paper/2020/hash/c6102b3727b2a7d8b1bb6981147081ef-Abstract.html
- Main paper: `source_papers/AUM_NeurIPS_2020.pdf`, SHA256 `EB66138E223D17AE8D39C5545C02672CA3055C9B42853F3E035FE560FDCDB503`
- Supplement: `source_papers/AUM_NeurIPS_2020_Supplemental.pdf`, SHA256 `593BDB7CBC025462048E1DEE6030C13FE71E6B61D62FDFB16536744E6500F25B`
- Official code: https://github.com/asappresearch/aum
- Audited code commit: `892e76eda4b6b85e21bda441d86134e0430ded7a`

## Reading Coverage

- Main paper: 13/13 pages read, including the method, experiments, failure cases, discussion and references.
- Supplement: 4/4 pages read, including experiment settings, cross-architecture consistency, threshold ablations and qualitative examples.
- Peer-review evidence checked: official NeurIPS reviews, meta-review and author response.
- Code checked: calculator semantics, sample identity wrapper, threshold construction, paper-replication runner, shell commands, tests, persistence and dependencies.
- Code execution: the two released unit tests pass in a contemporary isolated `uv` environment; a separate one-item-batch probe reproduces an untested `TypeError`.
- Visual verification: all 17 pages and five contact sheets under `audit/visual_checks/P021_AUM_NeurIPS_2020/`.

## Research Question

Can a temporally averaged assigned-label margin distinguish persistently mislabeled examples from clean but difficult examples more reliably than a single loss or margin snapshot?

This is relevant to Stage1 because all 200 OOF epochs are available and a trajectory can separate persistent disagreement from one-off confidence. It is not a direct match: AUM diagnoses and removes suspected label errors, whereas Stage1 keeps the 120,000 base examples and adds repeated replay to optimize an asymmetric `FN <= 95` operating region.

## Definition And Threshold Procedure

For logits `z^(t)(x)` and assigned label `y`, the epoch margin is

```text
M^(t)(x, y) = z_y^(t)(x) - max_{i != y} z_i^(t)(x)
```

and the Area Under the Margin statistic is the arithmetic mean

```text
AUM(x, y) = (1 / T) * sum_{t=1}^T M^(t)(x, y).
```

Low AUM means the assigned class repeatedly loses to another class. The threshold procedure is not an intrinsic zero cutoff:

1. randomly choose `N / (C + 1)` training examples;
2. relabel them into a new artificial class `C + 1` with no coherent positive pattern;
3. train only until the first learning-rate drop;
4. set `alpha` to the 99th percentile of the threshold-sample AUM distribution;
5. flag original examples with `AUM <= alpha`;
6. repeat with a disjoint threshold set so every original identity receives a usable score.

The artificial class approximates examples that can only be fitted by memorization. Its percentile is a heuristic calibration device, not a universal decision boundary.

## Experimental Contract

- CIFAR-10, CIFAR-100 and Tiny ImageNet use ResNet-32 or similarly scaled models; WebVision50, Clothing1M and ImageNet use ResNet-50.
- Standard small-dataset training uses 300 epochs, batch size 256, SGD with Nesterov momentum `0.9`, learning rate `0.1`, weight decay `1e-4`, and learning-rate drops at epochs 150 and 225.
- AUM measurement deliberately stops at the first learning-rate drop and lowers batch size from 256 to 64. The authors state that the extra SGD variance reduces memorization and makes AUM more salient.
- After removal, batch size is reduced in proportion to the retained set so the retraining run keeps approximately the same number of optimizer iterations.
- Small-dataset results use four random seeds. Large-dataset results without confidence intervals are single trials.
- The running average is much more architecture-stable than a single margin: the supplement reports more than 98% cross-architecture Spearman correlation for AUM in one CIFAR-10 40% uniform-noise study, versus roughly 75% for a single margin or training loss and roughly 40% for validation loss.
- On synthetic uniform noise, AUM separates mislabeled examples well and generally improves cleaned retraining.
- On real data, removing 17.8% of WebVision50 lowers error from 21.4% to 19.8%; removing 16.7% of Clothing1M lowers error from 35.8% to 33.5%; removing 13% of CIFAR-100 lowers error from 33.0% to 31.8%.
- The result is not universal. On ImageNet, removing 2.7% changes error from 24.2% to 24.4%, a small degradation.
- Pairwise asymmetric noise is a stated failure mode. At 40% asymmetric CIFAR-10 noise, AUM reports 41.3% error, substantially worse than several noise-learning baselines and only slightly better than standard training at 43.7%.
- Real-world AUM distributions are not cleanly bimodal, so thresholding cannot be interpreted as discovering two naturally separated populations.

The official reviews were mixed. Reviewers questioned novelty, class imbalance after removal, the validity of the threshold class, unfair removal baselines, systematic noise and weak Clothing1M comparisons. The meta-review accepted the work mainly as a simple, potentially useful strategy. The author response reports augmentation robustness and full-Clothing1M improvement, while acknowledging that AUM is not state of the art there and that the threshold-set rule may need adjustment under extreme imbalance.

## Code Audit

The released package implements the published statistic, but its semantics and replication runner expose boundaries that matter directly to Stage1.

1. `AUMCalculator.update` averages once per sample presentation, not once per epoch. Repeated sampling therefore changes the statistic's weighting. In Stage1, replay occurrence count must be stored separately and an epoch-normalized margin trajectory must coexist with any presentation-weighted summary.
2. The paper runner skips epoch 1 and records pre-update logits after each later mini-batch. It casts logits to FP16 and back to FP32 before accumulation, introducing deliberate precision loss that is undocumented in the paper.
3. The calculator tracks a count and sum but does not validate tensor lengths, sample-ID uniqueness, expected epoch coverage or monotone update identity. Duplicate IDs are accumulated; the returned dictionary silently retains only the final record for an ID repeated within one call.
4. Batch size one is broken because unrestricted `squeeze()` converts the margin vector to a scalar. The released tests cover only two-item batches. The defect was reproduced locally with `TypeError: 'float' object is not iterable`.
5. The two released calculator tests pass, but they test only basic arithmetic and final CSV equality. They do not cover threshold construction, paper replication, duplicates, missing samples, resume, interruption, atomic writes or scientific provenance.
6. Calculator state is in Python dictionaries and is never checkpointed. A failed training run cannot resume its AUM trajectory faithfully even if model weights are available.
7. `finalize`, epoch logs, prediction CSVs and model files are written directly rather than atomically. They have no completion sidecars, hashes, row-count validation or source manifests.
8. Sample IDs are positions in a wrapped subset rather than content-bound identities. The saved metadata provides indices and labels but no dataset manifest or image hash, so identity can drift when a split or ordering changes.
9. The threshold sets are slices of one seeded random permutation and are disjoint under the intended two-run protocol, but the code does not validate the set index, candidate coverage or extreme class imbalance.
10. With `num_valid=0`, the runner evaluates the test set after every epoch and saves the best checkpoint using test error. The large-dataset scripts set `num_valid=0`, so the released replication path is not a blind test protocol.
11. The small-dataset shell script checks for five arguments while reading a sixth output-directory argument. The usage contract and implementation disagree.
12. The package requires only lower bounds for PyTorch and pandas, and the replication environment does not lock torchvision, NumPy, CUDA, dataset bytes or platform identity.

These findings do not invalidate the AUM definition. They mean the released pipeline cannot be copied as the Stage1 collector or treated as evidence of exact cross-seed reproducibility.

## Direct Support For Stage1

1. A sample trajectory can contain materially more stable information than a single confidence, loss or margin snapshot.
2. Assigned-label margin is a useful all-epoch persistence and noise-risk coordinate because it records both the target logit and strongest competitor.
3. Temporal aggregation must preserve its measurement protocol. Presentation-weighted and epoch-weighted summaries answer different questions under replay.
4. Cross-model or cross-seed rank agreement is measurable and should be reported rather than assumed.
5. The difficult-versus-suspect distinction cannot be made from one extreme score. Label audit, class balance, systematic subgroups and downstream retraining are necessary.
6. Negative results under asymmetric noise and ImageNet justify reporting failure regions and avoiding one global cutoff.
7. Full epoch trajectories are worth collecting when cheap, but they should remain separate fields rather than an arbitrarily weighted value formula.

## What It Does Not Support

1. Calling low-AUM Stage1 examples mislabeled without independent annotation evidence.
2. Calling high AUM or low AUM a positive replay-value score.
3. Removing base examples or adding an artificial output class in the formal Stage1 campaign.
4. Importing the paper's batch size 64, 150-epoch measurement run, learning-rate schedule, 99th percentile, removal rate or adjusted retraining batch size.
5. Changing the exact canonical 240-run hyperparameters to make a diagnostic more separable.
6. Any Stage1 replay percentage, decay boundary, guard ratio or adaptive stopping rule.
7. Cross-seed stability from four small-dataset runs or a single large-dataset trial.
8. Improvement on a constrained raw `FN=0-95` safety frontier from average classification error after data deletion.

## Transfer Boundary

AUM is a model-state and protocol-dependent diagnostic. Its value changes with architecture, augmentation, batch noise, stopping point, label-noise structure and class balance. Stage1 must compute an observational AUM-like trajectory inside the unchanged canonical run, rather than start a separate lower-batch diagnostic run and then claim comparability.

For binary Stage1 classification, retain at each epoch:

```text
assigned_margin = z_assigned - z_other
signed_probability_margin = p_assigned - p_other
loss
correctness
presentation_count_this_epoch
cumulative_presentation_count
base_or_replay_role
```

Compute both:

```text
epoch_weighted_AUM = mean(one aggregate margin per identity per epoch)
presentation_weighted_AUM = mean(margin over every realized presentation)
```

Their difference is itself an exposure diagnostic. Never compare Treatment and controls unless the same identity definition, epoch coverage, augmentation observation point and schedule accounting are verified.

## Concrete Experiment Consequence

P021 strengthens the all-epoch collector but does not add a formal Treatment arm. Under the exact canonical 240-run lock, use AUM-like fields to ask whether the same selected normal IDs in good and bad seeds exhibit different transitions:

```text
persistent assigned-label conflict
increasing replay-versus-base exposure weighting
late margin reversal
class- or subgroup-concentrated disagreement
simultaneous weak-defect margin decline
```

The causal value still comes from paired no-replay, continuous, same-peak decay and cumulative-dose-matched replay. AUM-like trajectories explain when persistent disagreement appears; they do not prove that replaying or removing the corresponding image improves the Stage1 target.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, for all-epoch assigned-margin trajectories, protocol-aware temporal aggregation, cross-seed rank stability and noise-risk diagnostics
- Direct support for static replay ranking: no
- Direct support for numeric Stage1 schedule boundaries or percentages: no
- Canonical hyperparameter implication: strict non-drift; paper-specific batch and schedule changes are observational context only
- Reviewed at: 2026-08-07
