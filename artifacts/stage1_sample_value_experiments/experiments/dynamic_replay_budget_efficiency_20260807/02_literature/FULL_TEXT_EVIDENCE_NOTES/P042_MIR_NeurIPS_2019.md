# P042 - Online Continual Learning with Maximal Interfered Retrieval

## Identity

- Paper ID: P042
- Authors: Rahaf Aljundi, Eugene Belilovsky, Tinne Tuytelaars, Laurent Charlin, Massimo Caccia, Min Lin, and Lucas Page-Caccia
- Venue and year: NeurIPS 2019
- Official proceedings page: https://proceedings.neurips.cc/paper_files/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html
- Main paper: `source_papers/MIR_NeurIPS_2019.pdf`, SHA256 `400761B89077A1C1AF05F202F7437B426660A33E803FA388DAD7505C79E6516F`
- Official supplement: `source_papers/MIR_NeurIPS_2019_supplemental.zip`, SHA256 `0652BDA8C101DD29AEAD2E6D8020DABFBEC28A43E4805A2B4ADEEA4500206188`
- Official code: https://github.com/optimass/Maximally_Interfered_Retrieval
- Audited public HEAD: `35eda78bcdd35025b16b0b1039a926d34aad4851`
- Embedded submission checkout HEAD: `80f1e819b88bbd3a81f26bfac4a848fcbd1c5550`, with a materially dirty and partial worktree in the official supplement

## Reading And Audit Coverage

- Main paper: 12/12 pages read, including Equations 1-3, Algorithms 1-2, Figures 1-4 and 7, Tables 1-7, all experience/generative/hybrid experiments, ablations, conclusion, and references.
- Supplement: 3/3 pages read, including Tables 4-5, exact ER settings, searched generative settings, Algorithm 3, and hybrid ablations.
- Peer review: all three official reviews, the meta-review, and one-page author response checked.
- Visual verification: all 15 paper and supplement pages inspected at original detail under `audit/visual_checks/P042_MIR_NeurIPS_2019/` and `audit/visual_checks/P042_MIR_NeurIPS_2019_supplemental/`.
- Code: all 16 Python files in public HEAD and all seven Python files present in the submission archive parse under Python; the full 32-commit public graph, submitted Git state, experiment scripts, retrieval path, buffer path, data splits, seed path, evaluation path, and uncertainty calculation were audited.
- Finite implementation probe: 50,000 independent `K=10, N=100, batch=10` reservoir simulations compared the published code semantics with exact sequential reservoir sampling.

## Research Question

MIR asks which stored samples should be replayed when a new non-IID batch arrives. It does not rank samples by current loss. It predicts which old samples would be harmed by the impending update and replays the most harmed candidates.

This is directly relevant to the user's request to observe training dynamics rather than a static confidence snapshot. It is not yet a Stage1 value function: MIR estimates damage from one incoming update to one stored sample, while Stage1 needs the benefit and harm of replay on two protected tail populations over a 200-epoch optimizer path.

## Method And Mathematics

For incoming batch `B_t`, MIR forms a virtual SGD update:

```text
g_t       = grad_theta L(B_t; theta)
theta_v   = theta - alpha * g_t
s_MI1(x)  = loss(x; theta_v) - loss(x; theta)
s_MI2(x)  = loss(x; theta_v) - min(loss(x; theta), best_historical_loss(x))
```

It uniformly pre-samples `C` memories from the buffer and returns the top `B` scores. For experience replay, this random candidate stage is the only diversity mechanism; it neither guarantees coverage nor finds the global top `B` whenever `C` is smaller than the buffer.

A first-order expansion exposes the directional meaning:

```text
s_MI1(x) = -alpha * dot(g_x, g_t) + O(alpha^2)
```

A positive score therefore means the current batch gradient conflicts with the stored sample gradient. This is not the same as a replay candidate being aligned with the Stage1 target gradient. MIR answers "which protected sample is about to be hurt?", whereas gradient-aligned value asks "which candidate update lowers the protected target objective?"

The distinction matters because a candidate can protect one normal sample while harming weak defects. Stage1 must compute separate normal-tail and weak-defect interference or alignment fields and retain the joint constraint.

## Experimental Contract

- Setting: online continual learning with a shared classifier and a non-IID sequence, generally one pass through each stream.
- Datasets: Split MNIST, Permuted MNIST, Split CIFAR-10, and Split MiniImageNet; generative and hybrid studies use smaller subsets of these settings.
- Baselines: fine-tuning, IID online/offline upper bounds, GEM, iCaRL, random experience replay, and random generative replay.
- ER memory: reservoir sampling; incoming and replay batch sizes are both ten in the central experiments.
- MIR search: candidate size `C=50`, selected from validation; replay budget is ten in central ER experiments.
- Learning rates: paper and supplement report `0.05` for MNIST and validation-selected `0.1` for CIFAR-10.
- Repeats: 20 for the main MNIST table, 15 for CIFAR-10 and MiniImageNet, and five for hybrid experiments. The supplement's Permuted-MNIST table and submitted script instead say ten, an unresolved provenance inconsistency.
- Reported uncertainty: public code computes `2 * population_std / sqrt(n)`, an approximate two-standard-error interval, not an exact t confidence interval.
- Compute: author response reports the unoptimized CIFAR-10 ER-MIR implementation is approximately three times slower than random ER.

All numeric settings above are evidence context only and are forbidden from entering the Stage1 canonical configuration.

## Main Results And Negative Cells

On Split MNIST at 50 memories per class, ER-MIR reports `87.6 +/- 0.7` accuracy and `7.0 +/- 0.9` forgetting versus ER's `82.1 +/- 1.5` and `15.0 +/- 2.1`. On Permuted MNIST, ER-MIR raises accuracy from `78.9 +/- 0.6` to `80.1 +/- 0.4`, but forgetting is `3.9 +/- 0.3` versus ER's `3.8 +/- 0.6` and GEM's `3.1 +/- 0.5`. Thus MIR does not improve every endpoint in every central cell despite the surrounding prose.

On CIFAR-10, the gap grows with memory size. At 100 memories per class, accuracy is `47.6 +/- 1.1` versus `41.3 +/- 1.9`, and forgetting is `17.4 +/- 2.1` versus `23.3 +/- 2.9`. At 20 memories per class, forgetting differs only `50.2` versus `50.5`, well inside the displayed uncertainty. Candidate-pool quality and budget therefore condition the effect.

Five real updates improve both ER and ER-MIR only modestly in Table 3, but this ablation changes cumulative exposure and optimizer steps. It cannot distinguish selection quality from extra dose. The author response reports a non-monotone generative result at 100 iterations, where GEN-MIR becomes slightly worse than GEN; additional updates can increase forgetting or overfit.

The generative method needs entropy and diversity terms, and the paper identifies blurry/mixed generated classes as a failure mode. The hybrid method also initially learns a shortcut separating real current images from reconstructed old images; autoencoding current inputs is required to reduce that distribution shift. These failures reinforce representation and provenance dependence.

## Official Code Audit

### Virtual update and candidate search

- The public ER path computes gradients on the incoming batch, deep-copies the full model, applies plain `theta - lr * gradient`, and scores pre-sampled buffer examples before the actual joint update.
- This matches the paper's plain-SGD approximation. It is not an optimizer-general virtual update: momentum, adaptive moments, weight decay, gradient scaling, clipping, parameter groups, and scheduler state are absent.
- Stage1's virtual-update probe must either reproduce the canonical resolved optimizer state or explicitly label a last-layer/simple-SGD score as an approximation and measure its disagreement with the realized update.
- Candidate identities and candidate RNG draws are not logged. A selected top score is conditional on the random `C`-subset and cannot be interpreted as a global sample property.

### Reservoir implementation defects

- The submitted code increments `n_seen_so_far` only after at least one replacement. A batch with no accepted item returns early, so later replacement probabilities are too large. Public commits in May 2020 explicitly fix this after publication.
- The current code still draws every item in an incoming batch using the same pre-batch denominator and permits duplicate destination indices. This is not exact sequential reservoir sampling.
- In 50,000 finite simulations with uniform target probability `0.10`, exact reservoir sampling produced probabilities `0.0964-0.1038`; current-code semantics produced `0.0517-0.1583`; submitted-code semantics produced `0.0498-0.1504`.
- When excluding current-task entries, `Buffer.sample` returns indices relative to the filtered tensor, but the old-logit path later uses them as indices into the unfiltered buffer. This can update or read the wrong historical identity for `s_MI2` after current-task samples enter memory.

These defects do not prove the reported means are false, but they prevent treating the implementation as a clean realization of uniform reservoir sampling or identity-correct historical-loss tracking.

### Reproduction and statistical provenance

- The official repository has no release tag. Public master starts in October 2019, while the supplement embeds a May 2019 development branch whose extracted worktree has tracked modifications, many tracked deletions, and untracked experiment files.
- The 2020 public script changes Split-MNIST learning rate from the paper's `0.05` to `0.1` in a commit titled "Fix lr to match experiments". The paper, supplement, submitted script, and current reproduction script therefore do not define one unambiguous numeric run contract.
- The submitted Permuted-MNIST script uses ten runs, the main table caption says 20, and the later public script uses 20.
- Validation and test loaders are evaluated after every task. Hyperparameters are described as validation-selected, but test trajectories remain visible during development.
- Data split randomness is fixed once before the run loop; run seeds change model/training randomness but share one validation split.
- There is no checkpoint/resume contract, sampler state, immutable manifest, data/code hash in outputs, atomic completion marker, automated scientific test suite, or paired-run identity table.

## Direct Support For Stage1

1. A dynamic loss change after a virtual update is more informative than current loss magnitude alone.
2. First-order interference is a signed gradient relation, not a gradient norm. Large magnitude without the correct sign is insufficient.
3. Dynamic value is conditional on checkpoint, incoming batch, optimizer update, candidate pool, random pre-sample, replay budget, and surrounding identities.
4. Protected normal and weak-defect objectives must be scored separately; one aggregate loss can hide an unsafe tradeoff.
5. Candidate-set identity, RNG draw, score age, virtual-update identity, and actual-update discrepancy must be persisted.
6. Extra virtual or real steps alter compute, optimizer exposure, and overfitting risk, so dose-matched controls remain mandatory.

## What It Does Not Support

1. It does not establish a static high-value image ranking.
2. It does not show that the most interfered sample improves Stage1 `FN <= 95` safety or score-gap behavior.
3. It does not justify importing `C=50`, replay batch ten, memory sizes, learning rates, SGD, architecture, task splits, or any paper hyperparameter.
4. It does not test same-selection cross-seed sign reversals, replay decay, cumulative-dose matching, weak-defect guards, or a no-replay safety frontier.
5. The virtual plain-SGD step is not automatically valid for the canonical Stage1 optimizer.
6. The public reservoir and historical-logit bugs mean the code cannot serve as Stage1 infrastructure.

## Transfer Boundary And Observable Consequence

MIR adds a diagnostic, not a new formal training arm:

```text
At checkpoints 120/140/150/160:
1. take a preregistered base batch or realized aggregate update;
2. estimate its actual optimizer-aware parameter delta;
3. measure per-probe loss change for difficult normals and weak defects;
4. compare finite loss change with last-layer gradient dot-product prediction;
5. save candidate pool, score rank, selected identity, and approximation residual.
```

If harmful seeds show larger late weak-defect interference under the same selection and schedule, the training-dynamics mechanism gains support. If interference scores do not separate beneficial and harmful paired seeds, MIR-style local damage is insufficient and should not become a scheduler. The causal training block remains no-replay, continuous, same-peak decay, and dose-matched decay under the exact canonical 240-run lock.

## Decision

- Reading status: REPLICATION_DEPTH
- New formal arm: no
- New hyperparameter: no
- Canonical lock change: no
- Added fields: finite virtual loss change, first-order interference dot product, optimizer-aware virtual delta, finite-minus-linear residual, candidate-set identity, candidate sampling probability, selected rank, replay concentration, and actual-update disagreement
- Remaining uncertainty: whether late weak-defect interference predicts Stage1 cross-seed reversal beyond existing trajectory and gradient-alignment fields
