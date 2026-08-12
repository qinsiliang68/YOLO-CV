# P049 - Data Banzhaf: A Robust Data Valuation Framework for Machine Learning

## Identity

- Paper ID: P049
- Authors: Jiachen T. Wang and Ruoxi Jia
- Venue and year: AISTATS 2023, PMLR 206, oral presentation
- Official proceedings page: https://proceedings.mlr.press/v206/wang23e.html
- Paper and supplement: `source_papers/DataBanzhaf_AISTATS_2023.pdf`, SHA256 `990344B923DE65E451AA7E8C5626B2EB6715D123A6509B1666BA115BE04482C9`
- Official code: https://github.com/Jiachen-T-Wang/data-banzhaf
- Audited HEAD: `c9438325f1d9b18b71ea96b240741d9a52b9bed4`

## Reading And Audit Coverage

- Main paper and embedded supplement: 34/34 pages read.
- Coverage includes Definitions 4.1, 4.2, and 4.5; Theorems 2.1, 4.4, 4.6, 4.8, 4.9, 4.10, C.2, and C.3; Equations 1-19; all eight figures and three tables; all proofs; all experiment settings; limitations; and extended discussion of the noise model.
- Visual verification: all 34 pages inspected at original detail under `audit/visual_checks/P049_DataBanzhaf_AISTATS_2023/`.
- Code: complete ten-commit history, all six experiment/estimator/data Python modules, all nineteen model-source modules at syntax and provenance level, all tracked result metadata, data files, dependency lock attempt, filenames, persistence paths, and randomization paths inspected. The code was not imported into Stage1.

## Research Question

The paper starts from a problem directly visible in Stage1: a stochastic learning algorithm makes the subset utility `U(S)` noisy, and the resulting Shapley or leave-one-out rankings can change substantially across training runs. It asks which semivalue preserves pairwise value rankings under perturbations of the entire subset-utility function.

This is closely related to, but not the same as, our cross-seed replay reversal:

```text
paper: repeated training perturbs estimated U(S), so an estimated static ranking changes
Stage1: the exact same replay set can have opposite downstream treatment effects across seeds
```

The first phenomenon can contribute to the second, but a stable ranking estimator does not imply a stable replay treatment effect.

## Formal Object

For `n` training identities and utility function `U`, a semivalue has the form:

```text
phi_semi(i; U, w)
  = (1/n) * sum_k w(k)
      * sum_{S subset N\{i}, |S|=k-1} [U(S union {i}) - U(S)]
```

with the semivalue normalization:

```text
sum_k C(n-1, k-1) * w(k) = n.
```

The Data Banzhaf value sets a constant cardinality weight:

```text
w(k) = n / 2^(n-1)
```

and is equivalently the expected marginal contribution when every context subset excluding `i` is sampled uniformly from the power set:

```text
phi_banz(i) = E_{S ~ Uniform(2^(N\{i}))}[U(S union {i}) - U(S)].
```

Like Shapley, this remains conditional on the learner, utility, validation identities, training context, and data distribution. It relaxes the Shapley efficiency axiom, so it is not an intrinsic property of an image.

## Safety Margin Result

The paper calls a pair `(i,j)` `tau`-distinguishable when its size-specific average marginal-contribution differences are at least `tau`. Its safety margin is the smallest L2 perturbation of the full `2^n`-entry utility vector that reverses at least one distinguishable pair's ranking.

Under this exact definition:

- LOO has safety margin `tau`;
- Shapley has a larger but sub-exponential expression;
- Banzhaf achieves the largest safety margin among semivalues, `tau * 2^(n/2 - 1)`;
- Banzhaf also minimizes the L2 operator Lipschitz constant among semivalues.

The result concerns exact semivalue rank order under a worst-case norm-bounded perturbation of `U`. It does not establish that Banzhaf maximizes downstream accuracy, raw-frontier area, or replay benefit. The appendix also notes that Banzhaf is not literally the only weight sequence attaining the maximum; alternating weights can do so, while the constant Banzhaf weights are the only natural uniform interpretation offered.

## Estimation Result

Simple Monte Carlo estimates each identity from separate sampled contexts. Maximum Sample Reuse (MSR) samples utility-bearing subsets once and reuses every `U(S)` for all identities:

```text
phi_MSR(i)
  = mean[U(S) | i in S] - mean[U(S) | i not in S].
```

The stated utility-call bounds are:

```text
simple MC:
  L2   O(n^2 / eps^2 * log(n/delta))
  Linf O(n   / eps^2 * log(n/delta))

MSR:
  L2   O(n / eps^2 * log(n/delta))
  Linf O(1 / eps^2 * log(n/delta))
```

The lower bound proved for constant failure probability is only `Omega(1/eps)` in Linf, so MSR remains one factor `1/eps` from that lower bound.

The noisy-utility theorem adds an irreducible term involving `gamma = ||U-U_hat||_2`. Because this norm spans all `2^n` subsets, the bound can become loose or vacuous when per-subset noise accumulates at realistic scale. It is not a finite-sample guarantee for 120,000 identities.

## Experimental Contract

- Figure 1 uses 2,000 CIFAR10 identities with 10% synthetic label flips, 50,000 potentially repeated utility subsets, and five separately trained models for every subset. Reported average pairwise Spearman indices are approximately `0.010` for LOO, `0.038` for Shapley, and `0.856` for Banzhaf.
- Figure 2 reports cross-run top/bottom-k overlap from the same setup. This measures rank repeatability, not whether the selected examples improve a separately trained target model.
- The sample-efficiency experiment with exact values uses only ten synthetic points and deterministic full-batch logistic regression.
- The larger rank-stability experiment uses CPU with 200 valued points and CIFAR10 with 500 valued points. It samples 2,000 utility subsets and treats the value obtained by averaging `k=50` stochastic trainings per subset as a reference, not exact ground truth.
- Application tables cover thirteen datasets with 200 to 2,000 valued points and 100,000 sampled utilities. The paper describes CIFAR10 as pretrained ResNet18 penultimate features and uses small neural classifiers for the other image tasks.
- Weighted training min-max normalizes each value to `[0,1]` and uses it as a sampling weight. Label detection declares the bottom 10% values mislabeled. These are different downstream actions from fixed-set repeated replay.
- Data Banzhaf wins many weighted-accuracy rows, but not all. In noisy-label F1, Shapley or Beta variants win multiple datasets. The tables therefore do not establish universal downstream dominance despite stronger rank robustness.
- There is no source/video grouped uncertainty, no raw `FN=0..95` frontier, no weak-defect safety constraint, no no-replay arm, and no 200-epoch replay timing intervention.

## Official Code Audit

- The repository has ten commits, 98 tracked files, no release tag, no test suite, no license file, no experiment manifest, and no durable run schema. Sixty tracked files are compiled `pyc` artifacts for several Python versions.
- The two included result pickles contain 10,000 subset utilities by five stochastic repeats for a 200-point OpenML Pol example. They do not contain the full paper tables, figures, image features, or run-level provenance.
- `sample_for_value.py` seeds Python, NumPy, CPU Torch, and one CUDA generator once. It does not enable deterministic algorithms, capture RNG states, record per-repeat seed identities, or save environment fingerprints.
- A long utility-sampling run writes one pickle only at the end with non-atomic `pickle.dump`. There is no checkpoint, resume cursor, heartbeat, failed-subset ledger, or partial-result validation.
- For big datasets, the producer filename includes `_Seed{random_state}` while `load_value_args` searches for a filename without the seed suffix. The public producer and consumer therefore do not connect without undocumented renaming or external files.
- `applications.py` assumes exactly five utility repeats regardless of the CLI `n_repeat`, mutates `args.n_sample` for big datasets, hard-codes `n_val=2000` for evaluation loading, and prints final summaries without persisting an auditable result artifact.
- For non-OpenML data, `get_processed_data` builds the validation set with `n_data` rather than the requested `n_val`. Argument names and actual role sizes can disagree.
- The paper describes pretrained ResNet18 CIFAR features for its application table, while the released `CIFAR10` path loads raw pixels and trains a CNN. The complete paper image pipeline is not present.
- Neural utilities pass already-softmaxed probabilities into `CrossEntropyLoss`, which expects logits. This changes the intended objective.
- Utility is the maximum validation accuracy seen over 15 or 30 epochs, so the same validation set is repeatedly used for model selection and utility estimation.
- The weighted path creates `WeightedRandomSampler(weights, num_samples=batch_size)`. It therefore executes one weighted batch per epoch, while the unweighted path traverses the entire dataset. Weighted-vs-uniform comparisons confound sampling weights with optimizer-step count and cumulative exposure.
- `normalize` divides by `max-min` without a constant-value guard. Classical model failures are caught by a bare `except` and replaced with random-guess utility. Empty-data return types are inconsistent.
- `net_best = net` stores a reference rather than a copied best checkpoint. The reported maximum accuracy need not correspond to the returned final parameters.
- The dependency file contains machine-specific Conda build paths and an editable Git dependency. It is not a portable lock.

These defects block direct import of the estimator or claimed evaluation path. They do not negate the paper's mathematical observation that utility-estimation noise can destabilize rankings.

## Direct Support For Stage1

1. Treat static sample ranking as an estimate with uncertainty. Record rank repeatability, sign repeatability, and selected-set overlap across independent OOF/model seeds.
2. Separate two variance layers: uncertainty in the score/rank used to select a set, and variance of the downstream replay treatment effect after the set is frozen.
3. Preserve the exact same replay identities across paired seeds when estimating conditional treatment effects. A rank that is stable before training can still produce seed-dependent downstream outcomes.
4. Report pairwise and set-level stability rather than only scalar score variance: Spearman/Kendall correlation, top-p overlap, Jaccard, rank confidence interval, and sign-flip rate.
5. Keep development utility and blind evaluation separate. The paper's repeated use of held-out accuracy inside valuation cannot be copied into Stage1.
6. Maintain the canonical 240-run training lock. The paper offers no transferable batch size, optimizer, architecture, epoch count, or replay ratio.

## What It Does Not Support

1. It does not show that Data Banzhaf values are replay treatment effects.
2. It does not show that a stable top-ranked set improves the Stage1 safe frontier across unseen seeds.
3. It does not justify computing exact or sampled Banzhaf values over 120,000 images within the remaining campaign window.
4. It does not justify one universal scalar, one top-k budget, or one fixed replay schedule.
5. It does not separate adjacent video frames as dependent groups.
6. It does not permit changing `yolo11l`, 200 epochs, batch 128, image size 224, workers 4, optimizer, learning-rate schedule, augmentation, AMP, or determinism settings.

## Transfer Boundary And Observable Consequence

The transferable decomposition is:

```text
observed downstream variability
  = selection-score estimation variability
  + conditional replay-policy variability
  + their interaction.
```

Stage1 already froze identical selections across seeds and observed opposite outcomes, so ranking noise alone cannot explain the evidence. The next campaign should retain the paper's stability lesson as diagnostics while testing replay timing, cumulative dose, and weak-defect protection causally. Do not add a Banzhaf training arm.

## Decision

- Reading status: REPLICATION_DEPTH
- New formal arm: no
- New hyperparameter: no
- Canonical lock change: no
- Added fields: score-estimator seed, utility-repeat identity, score/rank mean and dispersion, rank confidence interval, pairwise Spearman/Kendall, top-p overlap, selected-set Jaccard, sign-flip rate, score-estimation variance, frozen-selection downstream variance, and variance-layer identity
- Remaining uncertainty: whether cheap repeated OOF/checkpoint estimates can separate ranking instability from true conditional replay-policy instability without approximating an exponential subset utility
