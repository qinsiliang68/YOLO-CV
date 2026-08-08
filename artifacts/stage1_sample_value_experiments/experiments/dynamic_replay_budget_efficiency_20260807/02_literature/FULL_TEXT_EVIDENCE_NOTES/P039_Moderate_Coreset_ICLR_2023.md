# P039 - Moderate Coreset

## Identity

- Paper ID: P039
- Full title: Moderate Coreset: A Universal Method of Data Selection for Real-world Data-efficient Deep Learning
- Authors: Xiaobo Xia, Jiale Liu, Jun Yu, Xu Shen, Bo Han, and Tongliang Liu
- Venue and year: ICLR 2023
- Official conference page: https://openreview.net/forum?id=7D5EECbOaf9
- Full-text version read: archived snapshot of the official OpenReview conference PDF, captured 2023-03-14
- Local PDF: `source_papers/Moderate_Coreset_ICLR_2023.pdf`, SHA256 `311FF558997CFB4FCE94C7A5688C10B46502599452D61BFF86743DA6E43FFE82`
- Official code: https://github.com/tmllab/Moderate-DS
- Audited code commit: `7f64d319624aab981dab1604a1aa3d528eb82f76`

## Reading Coverage

- Full paper: 20/20 pages read, including Equations 1-3, Definition 1, Figures 1-7, Tables 1-13, Algorithm 1, all ideal-data and robustness experiments, architecture transfer, higher-noise appendices, simultaneous-corruption results, and the MINE appendix.
- Visual verification: all 20 rendered pages inspected at original detail under `audit/visual_checks/P039_ModerateCoreset_ICLR_2023/`; equations, plots, tables, and references are legible.
- Source limitation: direct OpenReview PDF and API calls returned HTTP 403. The evidence PDF is the 2023-03-14 Wayback capture of the official URL, not an unofficial manuscript.
- Peer-review limitation: reviewer text could not be retrieved reliably, so no reviewer statement is used.
- Code coverage: all seven public commits were inspected; all 20 Python files parse with `ast.parse`; selection, training, dataset, perturbation, model, and output paths were audited.

## Research Question

The paper asks for a static coreset that is reasonably robust when the deployment scenario, corruption regime, architecture, or desired coreset size differs from the scenario assumed by a specialized score. It argues that neither the easiest nor hardest score tail is universally useful and proposes selecting a central score interval.

Stage1 asks whether repeated exposure to a fixed labeled set improves an FN-constrained operational tail across seeds. Moderate Coreset is relevant evidence against extreme-score ranking as a universal value rule. It does not estimate replay treatment effects, replay timing, weak-defect harm, or conditional value across optimization states.

## Method And Mathematics

For a trained classifier `f = g(h(x))`, the paper extracts penultimate representations `z_i = h(x_i)`. Equation 1 defines each class center as the arithmetic mean:

```text
z_bar_j = sum_i 1[y_i = j] z_i / sum_i 1[y_i = j]
```

Each sample receives a Euclidean distance score:

```text
d_i = ||z_i - z_bar_(y_i)||_2
```

The distances are sorted globally. For requested coreset size `m` from `n` samples, the paper sets `a = (n - m) / 2` and retains the contiguous rank interval around the global distance median:

```text
S_star = sorted_samples[a : n-a]
```

This is a rank-band heuristic. The paper does not prove that the median band minimizes generalization loss, tail risk, or any replay objective. Its justification is that the score median is a proxy for the full score distribution, plus an information-bottleneck interpretation evaluated with MINE. Table 1 repeats the MINE estimate 20 times, but this is not a theorem linking median proximity to operational value.

The paper explicitly states the premise most transferable to Stage1: the preferred score interval changes with scenario and coreset size. That premise argues against declaring either the highest-loss tail or any one fixed quantile band intrinsically valuable.

## Experimental Protocol

- Datasets: CIFAR-100, Tiny-ImageNet, and ImageNet-1k.
- Main architecture: ResNet-50; transfer experiments include SENet, EfficientNet-B0, VGG-16, ShuffleNet V2, and ViT-small.
- Baselines: Random, Herding, Forgetting, GraNd, EL2N, an influence-based optimizer, and self-supervised selection.
- Selection ratios: generally 20%, 30%, 40%, 60%, 80%, and 100%; ImageNet uses 60%-100%.
- Non-ImageNet experiments use five random seeds and report mean plus standard deviation. ImageNet is run once per condition.
- Robustness scenarios: five image corruptions, symmetric label noise, PGD and gradient-sign attacks, and a mixed corruption/noise/attack condition.
- The endpoint is ordinary test accuracy or top-5 accuracy, not a fixed-FN frontier, paired seed effect, or lower-tail constraint.

The reported pattern is robust-average-rank performance, not universal cell-wise dominance:

- On ideal CIFAR-100, Forgetting exceeds Moderate at 60% and 80% selection in Table 5.
- On ideal Tiny-ImageNet, Forgetting exceeds Moderate at 80% in Table 6.
- On ImageNet at 80%, self-supervised selection slightly exceeds Moderate in Table 7.
- With 35% label noise, Herding or Forgetting exceeds Moderate on both CIFAR-100 budgets, while Moderate wins the Tiny-ImageNet cells in Table 10.
- With 30% corrupted images, Herding slightly exceeds Moderate on the two CIFAR-100 cells, while Moderate wins Tiny-ImageNet in Table 12.
- In the CIFAR-100 PGD 30% cell, Random exceeds Moderate in Table 4.

The paper reports no paired tests, confidence intervals, multiplicity correction, seed identities, or worst-seed analysis. ImageNet has no repeat uncertainty.

## Official Code Audit

The current public HEAD is still `7f64d319624aab981dab1604a1aa3d528eb82f76`, with seven commits and no release tag. The initial public commit is `999385e1c38d4476ab3b5c9c8e4cf4268435d93a`. Important findings follow.

### Paper-code center mismatch

The paper's Equation 1 and prose define an arithmetic class mean. `selection.py` instead computes a coordinate-wise class median with `np.median`, and this implementation has been unchanged since the first public commit. A finite six-point probe shows that the mean-center and code-median procedures can retain different identities. This is a semantic method mismatch, not a cosmetic naming issue.

### Feature extraction state is uncontrolled

`selection.py` loads the extractor but never calls `model.eval()` and does not use `torch.no_grad()`. ResNet BatchNorm therefore remains in training mode while the dataset is traversed in fixed batches of 64. The resulting representation depends on batch composition and ordering, and extraction unnecessarily builds autograd graphs. No representation hash or batch-state audit is saved.

### The retained set is globally, not class-wise, constrained

The implementation sorts all class-conditional distances in one global array and keeps one global middle rank band. It enforces no class quota. A balanced two-class synthetic probe retained one sample from one class and three from the other at a 50% budget. Default NumPy quicksort also leaves exact-boundary tie order unspecified. `--rate` has no range validation, and rounding can make the realized small-dataset rate differ from the requested rate.

### Evaluation and execution defects

- `CIFAR100N` forces `train=True` even when the caller requests the test split; `CIFAR100C`, `tinyA`, and `tinyN` similarly route evaluation to perturbed training assets.
- `CIFAR100A` can combine 50,000 attacked training images with only 10,000 test labels during evaluation, causing an index failure.
- Base training reaches `args.save_dir` at final checkpoint save, but that argument is never defined. A nominal full-data run can save `best.pth` and then crash at the end.
- `--save` uses `type=bool`, so ordinary command-line strings do not reliably disable saving.
- Index output is only `index/<dataset>.bin`; different rates overwrite one another and carry no source or checkpoint manifest.
- Run outputs are keyed only by dataset and architecture; seed/config identity is not isolated. Coreset training disables checkpoint saving entirely.
- The final commit deleted `robust/corrupt.py`, while `robust/corrupt-tiny.py` still imports it before redefining the same functions, so the documented current script fails at import.
- Perturbation scripts do not seed NumPy, several paths are hard-coded, Tiny corruption mutates images in place, and writes are not atomic.
- There is no dependency lock, environment file, test suite, release tag, exact seed manifest, resume state, or license file.

The repository therefore verifies the intended rough selection idea but is not a complete executable reproduction package for the reported robustness tables.

## Direct Support For Stage1

1. A largest-score or smallest-score tail is not a task-independent definition of value.
2. The useful score region is conditioned on budget, representation, corruption regime, architecture, and downstream objective.
3. Moderate, non-extreme candidates can be useful when extreme difficulty mixes informative and corrupted identities.
4. Static methods should be compared as nested set families across ratios, not judged from one top-k point.
5. Representation identity, class composition, tie behavior, and realized set membership are necessary provenance fields.

## What The Paper Does Not Support

1. It does not show that median-ranked samples have positive replay value.
2. It does not study repeated exposure, timing, cumulative dose, optimizer state, or stopping replay.
3. It does not protect a weak-defect tail or evaluate `FN <= 95` safety.
4. It does not measure candidate gradients, target alignment, finite-step influence, or sample interactions under replay.
5. It does not test the same selection across training seeds or explain seed reversal.
6. Its published hyperparameters and selection ratios cannot replace any Stage1 canonical setting.

## Transfer Boundary And Observable Consequences

Moderate proximity can enter Stage1 only as a diagnostic coordinate or bounded candidate-pool baseline. It must not become a new formal arm merely because it performs well on average rank in unrelated coreset tasks.

Fields worth retaining are:

```text
representation_checkpoint_sha256
representation_source_and_age
distance_to_class_mean
distance_to_class_median
global_distance_rank_percentile
distance_to_global_score_median
requested_and_realized_selection_ratio
per_class_retention_rate
per_video_retention_rate
boundary_tie_count_and_policy
set_overlap_and_churn_across_budgets_or_checkpoints
tail_role_composition
```

The falsifiable Stage1 prediction is deliberately weak: if extreme static tails are contaminated or over-specialized, a central score band may have lower downside than the extreme band at some budgets, but it need not beat random replay and need not be stable across seeds. This can be checked retrospectively without adding a training arm. The primary causal schedule experiment remains unchanged.

## Decision

- Reading status: REPLICATION_DEPTH
- New formal training arm: no
- New Stage1 hyperparameter: no
- Canonical lock change: no
- Added evidence: static score utility is interval-, budget-, representation-, and task-conditioned; code provenance is essential because paper and implementation differ materially.
- Remaining uncertainty: whether any moderate-band diagnostic predicts conditional replay benefit after controlling seed, replay timing, cumulative exposure, and weak-defect harm.
