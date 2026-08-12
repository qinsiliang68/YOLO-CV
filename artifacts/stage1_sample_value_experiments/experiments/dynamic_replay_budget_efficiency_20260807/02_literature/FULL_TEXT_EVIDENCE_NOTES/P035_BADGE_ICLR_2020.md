# P035 - Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds

## Identity

- Paper ID: P035
- Authors: Jordan T. Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, and Alekh Agarwal
- Venue and year: ICLR 2020
- Official conference page: https://iclr.cc/virtual_2020/poster_ryghZJBKPS.html
- Full text read: arXiv v2 dated 2020-02-24, marked as an ICLR 2020 conference paper
- Local PDF: `source_papers/BADGE_ICLR_2020.pdf`, SHA256 `D3901AA393ACB598BEED4A5432F52F080237E50AD443659A18E65B1B1F4CE176`
- Official code: https://github.com/JordanAsh/badge
- Paper-period public commit audited: `6beb219736349fb19286faad6689cfba76b0dce3`, dated 2020-03-23
- Current HEAD audited: `a2d18acd372cf0f61d9e75bfb0c879c107fbf9f6`, dated 2024-06-04

## Reading Coverage

- Manuscript and appendices: 26/26 pages read, including Algorithms 1-2, Equation 1, Proposition 1, Figures 1-31, all learning curves, pairwise comparisons, normalized-error CDFs, and the binary-logistic argument.
- Visual verification: all 26 pages inspected at original detail under `audit/visual_checks/P035_BADGE_ICLR_2020/`; no missing, blank, or malformed page was found.
- Code audit: complete 33-commit history, paper-period first commit, current HEAD, README, runner, strategy base class, BADGE sampler, result aggregator, and syntax compilation inspected.
- Source limitation: the official ICLR page identifies the paper and links to OpenReview, but OpenReview was inaccessible behind a challenge during retrieval. The read artifact is author arXiv v2, whose first page says it is published at ICLR 2020.
- Execution limitation: current source syntax compilation passed. A benchmark training run was not attempted because neither the historical paper environment nor the external five-repeat job matrix is preserved.

## Research Question

BADGE asks which unlabeled examples should be acquired as a batch so that the batch is both uncertain under the current classifier and diverse in a last-layer gradient embedding. It is an active-label-acquisition method. Stage1 instead has labels for all candidates and asks when and how strongly selected examples should be replayed during one 200-epoch trajectory.

The paper is therefore useful for defining checkpoint-conditioned candidate geometry and redundancy. It does not directly estimate the signed effect of replay on difficult-normal and weak-defect tail objectives.

## Core Mathematics

For a softmax classifier with penultimate representation `z(x; V)`, class probability `p_i`, and hallucinated label:

```text
y_hat = argmax_i p_i
```

the last-layer gradient embedding block for class `i` is:

```text
(g_x)_i = (p_i - I[y_hat = i]) * z(x; V)
```

Proposition 1 gives, for any possible true label `y`:

```text
||g_x|| <= ||g_x^y||
```

because the predicted class minimizes the last-layer cross-entropy gradient norm over labels. The hallucinated norm is therefore a lower bound on the norm that would be induced after observing the label. This is an uncertainty/leverage statement, not a sign-of-utility statement.

BADGE applies a `k-means++`-style seeding procedure in gradient space. After choosing centers, a remaining point is sampled with probability proportional to squared distance from its nearest center. The paper's Algorithm 2 samples the first center uniformly; both the paper-period and current public implementations instead choose the maximum-norm embedding first. The executable algorithm is therefore not literally identical to the stated generic `k-means++` algorithm.

## Experimental Protocol

- Initial labeled set size is 100; acquisition batch size is 100, 1,000, or 10,000.
- The benchmark spans 11 dataset-architecture combinations, seven algorithms, and three batch sizes, for 231 stated experiment settings.
- Models are retrained from scratch after every acquisition round and trained until training accuracy exceeds 99%.
- The paper reports Adam, learning rate 0.001 for image data and 0.0001 for non-image data, no learning-rate schedule, and no image augmentation.
- Every reported experiment is repeated five times, with standard errors shown on learning curves.
- Pairwise comparisons use paired five-repeat differences and a two-sided t critical value of 2.776. Label budgets are selected while learning is progressing, using the random baseline's approach to 99% of its final accuracy.
- The many dataset, architecture, batch-size, algorithm, and label-budget comparisons are aggregated into penalty matrices; no family-wise or false-discovery correction is reported.
- These settings are literature context only. None may alter the Stage1 canonical hyperparameter lock.

## Positive And Negative Evidence

1. Pure uncertainty sampling can select many nearly identical high-magnitude examples. BADGE's gradient-space dispersion explicitly addresses this batch redundancy.
2. Diversity is conditional on representation quality. The paper reports cases where Coreset is worse than random when penultimate representations are not meaningful; diversity is not universally beneficial.
3. The learning curves support a stage-dependent description: diversity can matter more early and uncertainty more later. This is not a replay stop rule and does not identify epoch 140, 150, or 160 for Stage1.
4. BADGE is generally competitive across the paper's settings, but the discussion explicitly says the fundamental reason why it works remains open.
5. The binary-logistic argument applies in a restricted low-margin region and reasons about expected stochastic-gradient variance. It is not a proof for nonlinear image replay or an FN-constrained tail frontier.
6. The hallucinated embedding contains uncertainty and representation geometry but not the true labeled update direction. It cannot distinguish a large beneficial gradient from a large weak-defect-harming gradient.
7. The paper evaluates annotation efficiency and accuracy, not repeated exposure, cumulative dose, tail safety, `TN_at_FN95`, or the raw `FN=0-95` frontier.
8. Every acquisition round retrains from scratch. Stage1's optimizer state and path-dependent replay exposure are absent by design.
9. Five repeats are better than one, but seed identities, RNG states, exact initial sets, and run manifests are not reported in the paper or runner.
10. The paper-period code has no seed argument, checkpoint, resume state, atomic artifact contract, or environment lock. Repetitions were externally orchestrated and that job definition is absent.
11. The public trainer assigns `n_epoch` but does not use it as a cap; it loops until 99% training accuracy, so exact optimization effort depends on the stochastic path.
12. The public paper-period sampler passes hidden pool labels into the embedding API and derives class count from their unique values. It does not use the identities as gradient labels, but this is still an avoidable unlabeled-pool dependency.
13. The paper-period code explicitly builds the full class-by-feature embedding. Current HEAD uses a 2023 factorized distance speedup and is not an identified 2020 publication snapshot.
14. The current repository documents Python 3.8 and PyTorch 1.11, but has no lockfile. The paper-period dependency versions are not preserved, no release tag exists, and the original small test file was deleted three days after the first commit.
15. The result parser can aggregate separate repeat directories, but it hardcodes the parsed replicate field to zero and does not persist seed or initial-pool identity.

## Direct Support For Stage1

1. At selected checkpoints, represent each candidate by a last-layer gradient embedding and record geometry separately from target-tail alignment.
2. For labeled Stage1 candidates, compute both the hallucinated-label embedding and the true-label embedding. Their norm ratio and angular difference quantify how much the uncertainty surrogate misstates the actual labeled update.
3. Record candidate embedding norm, nearest-neighbor distance, cluster identity, distance to selected centers, within-selection pairwise-distance summaries, effective rank, and per-video concentration.
4. Compute these fields at more than one checkpoint because representation geometry and uncertainty change with model state.
5. Use diversity only after a mechanism-based candidate pool is defined. Diversity should prevent duplicate video/appearance modes from consuming replay slots; it should not manufacture value for candidates whose true gradients harm weak defects.
6. Compare diversity-constrained and unconstrained subsets through a bounded diagnostic or later transfer block, not by silently changing the frozen first-cycle selection.
7. Keep true candidate-to-difficult-normal and candidate-to-weak-defect dot products/cosines from P034. BADGE's magnitude lower bound cannot replace those signed quantities.
8. Persist all seeding, selected-center order, embedding/checkpoint hashes, and tie behavior. The paper/code first-center discrepancy shows that an underspecified seeding detail can change the selected set.

## What It Does Not Support

1. Calling a high hallucinated-gradient norm a high-value replay sample.
2. Using gradient-space diversity without checking weak-defect harm.
3. Treating the predicted-label gradient as the true labeled gradient.
4. Treating one checkpoint's embedding geometry as a permanent property of an image.
5. Adding BADGE as a formal first-cycle training arm.
6. Replacing the no-replay, continuous, same-peak-decay, and dose-matched timing controls.
7. Importing BADGE's optimizer, learning rates, stopping rule, architecture, batch sizes, augmentation setting, or acquisition budgets.
8. Changing any Stage1 canonical hyperparameter.

## Stage1 Field Contract

For the bounded gradient/diversity pilot, under the exact canonical lock, record at the preregistered key checkpoints:

- checkpoint-conditioned penultimate embedding and hash;
- hallucinated-label and true-label last-layer gradient embeddings;
- each embedding norm, squared norm, and their ratio;
- cosine and angular difference between hallucinated and true embeddings;
- candidate-to-normal-tail and candidate-to-weak-defect signed dot/cosine fields;
- nearest-neighbor and nearest-selected-center distances;
- cluster/medoid identity, effective rank, coverage radius, and within-batch distance distribution;
- video/source concentration, duplicate fraction, and unique-context coverage;
- first-center rule, complete center order, RNG seed/state, and tie-resolution record;
- compute time, memory, candidate-pool size, and embedding dimensionality;
- finite-intervention and endpoint outcomes on identity-disjoint OOF/val_op probes;
- canonical hyperparameter lock SHA256 on every artifact.

## Concrete Experiment Consequence

P035 adds no formal training arm. It adds a low-cost candidate-geometry diagnostic to the frozen causal schedule block:

```text
same selection + same seed + exact canonical hyperparameters
no replay vs continuous vs same-peak decay vs dose-matched decay
```

If timing/dose causality is established, a later preregistered transfer block may compare the same candidate rule with and without diversity constraints. That block must hold replay ratio, timing, total exposure, seed, and all canonical parameters fixed and must retain both random controls.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, for hallucinated last-layer gradient construction, uncertainty/diversity separation, and representation-dependent failure modes
- Replication-depth eligibility: yes, with explicit version, environment, seed, and paper/code discrepancy caveats
- Direct support for static replay ranking: no
- Direct support for checkpoint-conditioned diversity diagnostics: yes
- Direct support for a new formal arm: no
- Permission to change canonical Stage1 hyperparameters: no
- Reviewed at: 2026-08-08
