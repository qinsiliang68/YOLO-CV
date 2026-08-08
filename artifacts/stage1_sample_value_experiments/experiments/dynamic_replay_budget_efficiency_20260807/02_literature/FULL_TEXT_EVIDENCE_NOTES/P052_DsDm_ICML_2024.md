# P052 - DSDM: Model-Aware Dataset Selection with Datamodels

## Identity

- Paper ID: P052
- Authors: Logan Engstrom, Axel Feldmann, and Aleksander Madry
- Venue and year: ICML 2024; Proceedings of Machine Learning Research 235
- Official article page: https://proceedings.mlr.press/v235/engstrom24a.html
- Official paper: `source_papers/DsDm_ICML_2024.pdf`, SHA256 `5025DBC19CC5298AE09E4A9841216AECFC7F3941AB6C2871A8DF3EC74EB43827`
- Official project page: https://gradientscience.org/dsdm/
- Official repository: https://github.com/MadryLab/dsdm at commit `50e970a7c9af7d835ff645a8e3ae732244bfa65d`

## Reading And Audit Coverage

- Main paper and appendices: 36/36 PDF pages read.
- Coverage includes the task-optimal subset estimand; linear datamodel derivation; target-set construction; candidate C4 construction; all targeted and broad language-model experiments; random, classifier, DSIR, and SemDeDup controls; target-task ablations; inverse-selection counterfactual; model-training settings; leakage audit; TRAK derivation and LM adaptation; compute accounting; limitations; and all reported figures and tables.
- Visual verification: all 36 pages inspected under `audit/visual_checks/P052_DsDm_ICML_2024/`.
- Text extraction: the complete PDF was extracted and searched for equations, target/holdout counts, candidate size, model and token budgets, reference-model count, projection dimension, random repetitions, hyperparameter selection, counterfactuals, and limitations.
- Code audit: both commits and all 30 tracked files in the official repository were inspected. All three Python files passed `ast.parse`. Git LFS payloads were intentionally not downloaded: 17 pointers declare 14,054,956,805 bytes, while the candidate dataset is separately described as approximately 400 GB.
- Official issue audit: all three public issues and their comments were inspected on 2026-08-08. Issues 1 and 3 remain open, and issue 1 still lacks the promised implementation details for gradient projection. The repository itself states that data selection code is "Coming soon."
- Replication boundary: this is recorded as `REPLICATION_DEPTH` because the paper, appendices, repository history, selection loader, LFS identities, dependencies, and public issue history were audited. The paper's reference-model training, projected-gradient collection, 14 GB score release, 400 GB candidate corpus, and downstream LM experiments were not rerun.

## Formal Estimand

For candidate universe `S`, subset size `k`, target distribution `D_targ`, loss `ell`, and stochastic learning algorithm `A`, the paper defines the task-optimal subset as:

```text
S* = argmin_{S subset of S, |S| = k} L_Dtarg(S)

L_D(S) = E_{x ~ D, A-randomness}[ell(x; A(S))]
```

The expectation explicitly includes learner randomness. Therefore the selected set is conditional on the learner, target, subset size, and training procedure. It is not an intrinsic ranking of documents.

For one target example `x`, a datamodel approximates expected trained loss from the subset inclusion vector:

```text
L_x(S) = E[ell(x; A(S))]

tau_theta_x(1_S) approximately equals L_x(S)
```

The paper instantiates a linear datamodel:

```text
tau_theta_x(1_S) = theta_x^T 1_S
```

and averages target-specific coefficient vectors. Under this linear fixed-effect approximation, the estimated task-optimal size-`k` set is:

```text
S_hat_DSDM = bottom-k((1 / n) * sum_x theta_x)
```

because smaller estimated coefficients reduce target loss. This derivation appears on PDF pages 2-3. It supplies an explicit target-conditioned set objective, but it assumes additive fixed effects and does not model candidate interactions.

## How The Datamodel Is Estimated

The paper does not train millions of subset models as P051 did. It uses a TRAK approximation adapted to language modeling.

- Four independently trained 125M-parameter reference language models are used.
- Each reference model is trained for 80,000 batches on approximately 82 million samples, about 38% of the 216,948,746-example candidate set.
- Per-candidate gradients of a language-model output functional are projected to dimension 16,384.
- Model-specific projected-gradient terms are averaged to reduce training-randomness sensitivity.
- The output functional is a mean multiclass margin across tokens rather than ordinary sequence cross-entropy.
- The paper reports a total cost near `1.2e9` forward-backward-pass units under its accounting. Projected-gradient collection is about 73% of this cost; each of the four reference trainings is about 7%.
- The authors state that projection itself is less than 1% of projected-gradient computation, but the public repository does not include that implementation.
- The authors did not optimize this pipeline and did not report an ablation over the number of reference models, number of reference batches, or projection dimension.

Thus four models and dimension 16,384 are observed implementation choices, not transferable minimums or guarantees for Stage1.

## Candidate Data And Target Splits

The candidate set is `C4 en.noblocklist`, concatenated and sliced into 1024-token examples. The exact paper count is 216,948,746 examples.

The target tasks are separated from downstream holdout evaluation:

| Target | Target samples | Holdout samples | Notes |
| --- | ---: | ---: | --- |
| SQuAD | 23,107 | 10,557 | Uses 25% of SQuAD train as the target distribution |
| Jeopardy | 876 | 876 | Word Origins category excluded |
| LAMBADA | 2,577 | 2,570 | Six leaked holdout examples removed |
| CS-Algorithms | 660 | 660 | Random half split of the benchmark test set |

The authors search the full C4 candidate set for lowercased whitespace-normalized context and continuation strings. They report six LAMBADA overlaps and none for the other three tasks under that rule. The target/holdout separation is directly relevant to Stage1: development-tail identities may define a target gradient, but blind holdout identities may not be used for ranking or schedule selection.

## Targeted Selection Experiments

For each target task, the paper trains a 125M GPT-2-style model on six billion tokens. Selection size changes while token budget is held fixed by changing the number of passes over selected examples.

- DSDM is compared with random selection and two target-similarity methods, CLASSIFIER and DSIR.
- DSDM reduces target loss more consistently than these baselines across the four tasks.
- Similarity-based methods fail to consistently beat random on SQuAD and CS-Algorithms.
- Smaller, higher-ranked DSDM subsets can outperform larger selections because the fixed token budget repeats the smaller set more times.
- The paper's Figure 9 reports the range of ten random models, each trained for one epoch. Most non-random points are not accompanied by an equivalent distribution of independent retrainings.
- The inverse experiment trains on DSDM's least preferred examples and performs much worse than random, providing a finite directional counterfactual rather than only score plausibility.
- Qualitatively selected examples are often not semantically intuitive. Some highly ranked examples look noisy or unrelated, while similarity can select text that looks relevant yet is not useful after training.

The fixed-token protocol confounds selection size with per-identity repetition count. For Stage1, ratio, cumulative exposure, number of unique identities, per-identity concentration, optimizer steps, and training stage must therefore be measured separately.

## Broad Language-Model Experiments

The broad experiment applies each targeted method using its recommended target distribution and evaluates 15 benchmarks.

- DSDM uses a mixture of SQuAD, Jeopardy, and LAMBADA target coefficients.
- The selected data are used to train 760M, 1.3B, and related GPT-style models; a 1.8B random-data model represents roughly twice the 1.3B compute budget.
- DSDM improves average benchmark accuracy relative to random and can match or exceed the 2x-compute random comparator in reading-comprehension and world-knowledge categories.
- Target choice matters. SQuAD, Jeopardy, and LAMBADA separately improve different categories, and a single target can reduce performance on unrelated categories.
- The three-target mixture performs best overall in the reported experiment.
- The authors do not tune target-task choice or number of epochs for this broad comparison.
- Model-aware rankings computed from 125M proxy models are useful for larger models in this setting, but this is empirical transfer under one LM family, not general architecture invariance.

This supplies strong evidence against collapsing heterogeneous protected targets before analysis. A ranking that helps the difficult-normal target can harm the weak-defect target even when its aggregate mean looks good.

## Training Configuration And Non-Transferability

The paper trains GPT-style language models with:

```text
Adam beta1=0.9, beta2=0.95, epsilon=1e-8
sequence length=1024
batch size=1024
cosine schedule
200 warmup batches
minimum learning-rate multiplier alpha=0.1
gradient clipping threshold=1
BF16 on A100 and FP8 on H100
```

Table 2 contains model-size-specific learning rates, token counts, weight decay, and architecture settings. The authors state that the settings were selected using held-out C4 perplexity for the 125M model and that Section 4 required a larger weight decay so larger models would not diverge.

None of these constants may enter Stage1. They belong to a different task, architecture, precision regime, tokenization, target, and compute scale. Stage1 formal runs remain locked to the complete canonical 240-run configuration, including `yolo11l`, 200 epochs, batch 128, image size 224, workers 4, AMP, optimizer, learning-rate path, augmentation, and all other source-derived fields.

## Statistical And Mechanistic Limits

1. The linear datamodel assigns each candidate a fixed additive effect and therefore omits subset interactions, redundancy, and context-dependent sign changes.
2. TRAK introduces local linearization and random-projection approximation.
3. Four reference models are averaged, but there is no ablation or uncertainty interval establishing that four is sufficient.
4. The targeted plots expose a ten-run random range, while most selected arms do not report matched retraining distributions. Cross-seed treatment stability is therefore not established.
5. The broad benchmark comparisons do not report a complete paired-seed distribution or confidence intervals.
6. Fixed token count means selection fraction changes repetition and epochs, so selection quality and exposure concentration are coupled.
7. Target choice can harm unrelated tasks, showing that an averaged target objective can hide protected-target damage.
8. The method is computationally enormous and its core implementation is not public in the audited repository.
9. The candidate corpus is language data with independent token chunks, not grouped, near-duplicate video frames.
10. The result does not identify a Stage1 replay ratio, decay epoch, guard fraction, seed count, or success threshold.

## Official Repository Audit

The official repository is an index loader for released score tensors, not an end-to-end implementation of DSDM.

1. `README.md` marks actual data selection implementation as "Coming soon." It provides no reference-model training, TRAK computation, projected-gradient collection, target preparation, downstream training, evaluation, or experiment orchestration.
2. Public issues corroborate the gap. The author wrote in July 2024 that implementation code would be released soon, but the open gradient-projection issue still had no implementation response as of 2026-08-08.
3. `process_path()` resolves data under `dsdm/data`, while the tracked LFS pointers live under repository-root `data`. A lightweight path contract check gives `expected_data_path_exists=false` and `actual_data_path_exists=true`.
4. The README imports `selections` and `utils` but calls bare `get_indices(...)`, which is undefined in that namespace. The documented call should be qualified.
5. `requirements.txt` lists only unpinned `torch`, `datasets`, and `numpy`; `dsdm/utils.py` imports `transformers`, which is omitted.
6. `TARGETS` accepts `gpt3_mix`, but `dsdm_select()` has no branch that assigns `dm_params` for it. The internal `_test()` silently skips this invalid public combination.
7. Random, DSIR, and SemDeDup paths create `np.random.default_rng()` without a supplied or recorded seed.
8. The paper derives bottom-`k` smallest loss coefficients, while the loader sorts ascending and returns the largest loaded tensor values. This may be intentional if the released tensors encode sign-reversed TRAK utility, but no metadata documents the convention. Because LFS payloads were not downloaded, this remains an unresolved paper/code sign contract rather than a proven numerical bug.
9. There is no selection-size validation, candidate-manifest hash, data-tensor schema, atomic artifact generation, test suite, run provenance, or resume protocol.
10. The 17 LFS pointers declare 14,054,956,805 bytes. The code snapshot contains only pointers; no claim was tested against the numeric tensors.

These defects do not invalidate the paper's reported experiments. They make the public repository unsuitable as a Stage1 dependency or reproducibility template.

## Direct Support For Stage1

1. Define candidate utility against explicit target roles and a fixed learner, not as a context-free image score.
2. Keep difficult-normal and weak-defect target losses, gradients, margins, and treatment effects separate before imposing any guard constraint.
3. Preserve exact target identities and prohibit blind-holdout use during selection or schedule tuning.
4. Treat replay ratio, training stage, cumulative exposure, unique identity count, and repetition concentration as parts of the treatment.
5. Compare target-aware selection against strong random and matched-random controls under identical compute, optimizer steps, seed, initialization, and canonical hyperparameters.
6. Validate score direction with actual frozen-set replay interventions; inverse or deliberately adverse controls are more informative than score histograms alone.
7. Average gradient diagnostics across model states or seeds only after retaining per-reference values, variance, sign agreement, and role-specific conflict.
8. Use low-cost final-layer or projected-gradient diagnostics at selected checkpoints rather than attempting full DSDM computation over all 120,000 images and every epoch.
9. Record target-mix cancellation. A combined scalar may improve while weak-defect behavior degrades.
10. Preserve candidate rank, selected prefix, budget, context, and estimator/model checkpoint because ranking meaning changes with all of them.

## What It Does Not Support

1. It does not justify adding DSDM as a formal Stage1 arm. The required randomized reference training and released core implementation are absent from our evidence and compute plan.
2. It does not justify treating TRAK, gradient norm, or a linear coefficient as ground-truth sample value.
3. It does not show that four reference models or projection dimension 16,384 is sufficient for our classifier.
4. It does not justify averaging difficult-normal and weak-defect gradients into one target vector before checking constraints.
5. It does not justify any Stage1 replay percentage, decay epoch, cumulative dose, guard share, or number of seeds.
6. It does not support changing batch, workers, model, optimizer, learning rate, precision, augmentation, or any other canonical training hyperparameter.
7. It does not establish cross-seed stability of selected-set treatment effects.
8. It does not eliminate redundancy or video-group dependence.
9. It does not permit test-driven selection.
10. It does not prove that a static top-`k` policy is better than a dynamic replay policy.

## Consequence For The Next Campaign

DSDM strengthens the diagnostic layer, not the formal arm matrix. At key checkpoints, Stage1 can estimate two separate target gradients:

```text
g_normal_tail(theta_t)
g_weak_defect(theta_t)
```

and store, for a stratified candidate subset:

```text
candidate last-layer gradient norm
dot and cosine with g_normal_tail
dot and cosine with g_weak_defect
per-reference-model or per-seed mean, standard deviation, and sign agreement
projected-versus-unprojected agreement on a small calibration subset
role conflict and finite virtual-update residual
```

These fields test whether the same frozen selection has different update geometry in good and bad seeds. They remain diagnostics. Formal causal evidence still comes from canonical-locked, paired same-selection interventions over replay timing, dose, and weak-defect guard.

The first seven-day block should therefore not wait for a full model-aware ranking. It should release a minimal complete seed block that compares the same frozen selection under continuous, same-peak decay, dose-matched decay, and no replay, while collecting every-epoch role-specific trajectories and key-checkpoint gradient diagnostics. This produces an interpretable result even if the gradient proxy fails.

## Decision

- Reading status: REPLICATION_DEPTH
- New formal arm: no
- New diagnostic: yes; separate target-conditioned last-layer/projected-gradient geometry at key checkpoints
- New hyperparameter: no
- Canonical lock change: no; the learning algorithm is part of the estimand, so changing canonical settings would change the research object
- Added fields: target-set identity and hash, role-specific target loss, per-target gradient, reference model/seed identity, projected-gradient dimension and seed, per-reference score, score mean/std/sign agreement, candidate rank and prefix, selection size and ratio, unique identities, per-identity repeats, total token/sample exposure, optimizer steps, role conflict, target-mixture cancellation, inverse-control identity, actual finite treatment effect, code/data tensor convention, and blind-holdout state
- Remaining uncertainty: whether low-cost last-layer Stage1 gradient geometry predicts paired replay effects across unseen seeds after timing, dose, and weak-defect protection are controlled
