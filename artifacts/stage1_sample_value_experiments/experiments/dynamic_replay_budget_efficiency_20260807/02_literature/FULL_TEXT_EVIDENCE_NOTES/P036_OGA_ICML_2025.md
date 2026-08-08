# P036 - Outlier Gradient Analysis

## Identity

- Paper ID: P036
- Full title: Outlier Gradient Analysis: Efficiently Identifying Detrimental Training Samples for Deep Learning Models
- Authors: Anshuman Chhabra, Bo Li, Jian Chen, Prasant Mohapatra, and Hongfu Liu
- Venue and year: ICML 2025, Proceedings of Machine Learning Research volume 267
- Official paper page: https://proceedings.mlr.press/v267/chhabra25a.html
- Local PDF: `source_papers/Outlier_Gradient_Analysis_ICML_2025.pdf`, SHA256 `FB335EF102851257982B587F5C9B9E36AD84E9E4D7FEC0874F946F987D053E8D`
- Official code: https://github.com/anshuman23/outlier-gradient-analysis
- Audited code commit: `a62c3807bed227efffa57ac7f0b1473efbb742f3`, dated 2025-05-03

## Reading Coverage

- Main paper and appendices: 20/20 pages read, including Equations 1-2, Observation 3.1, Hypothesis 3.2, Algorithm 1, Figures 1-4, Tables 1-13, dataset/model details, trimming-budget ablation, runtime analysis, architecture transfer, ImageNet subset, baseline comparison, distribution-shift extension, limitations, and reproducibility statement.
- Visual verification: all 20 pages inspected at original detail under `audit/visual_checks/P036_OGA_ICML_2025/`; no blank, missing, clipped, or malformed page was found.
- Code audit: complete 13-commit public history, README, CIFAR-10N/CIFAR-100N runners, custom ResNet, RoBERTa selection runner, Llama influence runner, and LoRA model implementation inspected.
- Static execution check: all 50 Python files parsed successfully with the current Python AST parser without importing dependencies or creating bytecode.
- Training execution limitation: no paper benchmark was rerun because the repository has no exact environment lock, contains large model/data artifacts, and does not preserve a deterministic five-repeat experiment manifest.

## Research Question

The paper asks whether detrimental training samples can be identified cheaply by treating their per-sample gradients as outliers. Stage1 instead asks whether replaying a labeled sample set at a particular model state, dose, and training phase improves a difficult-normal objective without harming the weak-defect tail.

The paper is directly relevant to the warning that a large or unusual gradient is not automatically useful. It does not establish a target-tail-aware value function and does not study replay timing, cumulative exposure, or same-selection cross-seed reversals.

## Core Mathematics

For training sample `z_j`, the paper writes an influence score of the form:

```text
I(z_j) = - sum_{z in T/V} grad L(z)^T H^{-1} grad L(z_j)
```

and discretizes it as:

```text
I(z_j) < 0  -> detrimental
I(z_j) >= 0 -> beneficial
```

The sign depends on three objects:

```text
evaluation-target gradient
inverse-Hessian transformation
candidate gradient
```

The proposed bridge then discards the first two fixed factors and hypothesizes that detrimental candidates are a minority of outliers in raw sample-gradient space. Observation 3.1 states that most samples are beneficial and detrimental samples are a smaller subset. Hypothesis 3.2 states that an outlier algorithm exists that can detect the detrimental subset.

This is a hypothesis, not an algebraic equivalence. In general, raw-gradient outlier status cannot preserve the sign of a target-specific influence score after the target direction and Hessian transformation are removed. Two candidates can be equally outlying but point in opposite directions relative to the weak-defect objective; a non-outlying candidate can still conflict with that objective.

Algorithm 1 therefore implements a proxy pipeline:

```text
train model
compute one gradient vector per training sample
run iForest, L1-norm, or L2-norm outlier detection
trim k predicted outliers
retrain on the retained set
```

Its output is an outlier/risk decision under a chosen geometry and budget, not a signed Stage1 replay-value estimate.

## Experimental Protocol

- Synthetic datasets use logistic regression or a two-hidden-layer MLP and manually flip 10 or 20 labels.
- CIFAR-10N and CIFAR-100N use 50,000 noisy training labels and 10,000 clean test images.
- The paper describes an ImageNet-pretrained ResNet-34 fine-tuned for 100 epochs with SGD, learning rate 0.1, momentum 0.9, weight decay 0.0005, and batch 128.
- Vision results use a 5% trimming budget and five runs. Appendix C.2 evaluates budgets from 2.5% through 12.5% on test accuracy and then chooses 5% as an overall setting.
- CIFAR-100 gradients are represented as a `50000 x 51200` last-layer matrix and passed through sparse random projection.
- RoBERTa-large is LoRA-fine-tuned on four binary GLUE tasks with 20% synthetic label flips; 30% of examples are trimmed, training lasts 10 epochs, and results use three runs.
- Llama2-13B LoRA experiments use three synthetic category-structured influence benchmarks.
- Runtime tables time influence/outlier computations after gradients are available; they do not represent full end-to-end training cost.
- These settings are literature context only. None may alter the Stage1 canonical `yolo11l` hyperparameter lock.

## Positive And Negative Evidence

1. On synthetic data, iForest gradient outliers identify most mislabeled points and trimming can improve the MLP endpoint from 90% to 96%.
2. The same Table 1 gives the critical counterexample: L1/L2 gradient outliers identify 98% of mislabeled points, yet post-trimming accuracy is 87%, below the 90% untrimmed model. Accurate noise/outlier detection is therefore not sufficient for positive model value.
3. Different outlier methods win under different CIFAR noise regimes. L2 wins Aggregate and Noisy100, L1 wins Random, and iForest wins Worst. There is no universal gradient-outlier geometry.
4. The trimming-budget optimum varies by dataset, noise regime, and detector. Appendix Table 5 selects or discusses budgets using test accuracy, so the reported 5% setting is test-informed rather than a blind transferable constant.
5. Main vision means use five runs and report standard deviations in the appendix, but they do not preserve paired seed identities or test same selected identities across seeds.
6. The method removes samples and retrains. Stage1 adds repeated presentations inside one optimizer path. Removal value and replay value need not have the same sign.
7. The paper's distribution-shift extension supplies validation/test samples as inliers to OneClassSVM. If the final test set is used this way, it is adaptation leakage for Stage1; only a role-frozen calibration stream could be eligible.
8. The LLM benchmark is explicitly a category-similarity task. It does not establish individual training-sample influence within a category.
9. The limitations section acknowledges that the trimming budget `k` is a nontrivial hyperparameter with little field-wide consensus.
10. The paper notes that high-quality data with no gradient outliers may make the method less useful. Outlier existence is task- and state-dependent.
11. The influence-to-outlier bridge assumes that detrimental points are a minority and sufficiently separated in raw gradient space. Neither condition is verified for Stage1 difficult-normal and weak-defect tails.
12. Last-layer gradients reduce cost but can miss backbone interactions, BatchNorm state, momentum, weight decay, and finite multi-step replay effects.

## Official Code Audit

1. The repository is a clean untagged HEAD with 13 commits, but includes datasets, cached Hugging Face artifacts, model adapters, Python bytecode, event logs, and result files. It is not a minimal source-only release.
2. The README lists dependency names without versions; no lockfile, container, environment export, or exact hardware-independent install contract is provided.
3. CIFAR runners seed PyTorch and CUDA only. Python `random.sample`, NumPy, cuDNN behavior, `IsolationForest`, and sparse random projection remain unbound or partially unbound.
4. The base CIFAR DataLoader uses `shuffle=False`, while a custom removal sampler later draws batches with unseeded Python randomness.
5. The CIFAR-10 runner allocates a default-float64 `50000 x 5120` gradient array, about 2.05 GB. The CIFAR-100 runner allocates `50000 x 51200`, about 20.48 GB, before projection.
6. The CIFAR runners evaluate test accuracy every epoch and report `np.max` across 100 test epochs for both the original and trimmed model. This selects a best test epoch and overstates blind endpoint evidence.
7. The CIFAR-100 projection and iForest omit `random_state`; the CIFAR-10 iForest also omits it. Repeated results are not reconstructible from the exposed seed argument.
8. The source comment says contamination 0.05 “works best so far”, reinforcing that the budget was empirically selected rather than preregistered.
9. The paper describes an ImageNet-pretrained ResNet-34 block. The repository uses a custom CIFAR ResNet initialized from scratch and never loads an ImageNet checkpoint. This is a material paper-code protocol mismatch.
10. The RoBERTa script uses unseeded Python sampling, unseeded iForest with contamination 0.3, and no complete global RNG capture. Its three repetitions are not encoded as immutable runs.
11. The Llama script fits one sparse random projection on concatenated training and validation gradients, then splits the result. It trains ten class-specific iForests on known contiguous category blocks.
12. For each validation prompt, the Llama script broadcasts one class-level score to all 90 training prompts in that category. It cannot rank individual identities within a class and structurally favors same-category retrieval.
13. No unit tests, scientific contracts, queue manifest, checkpoint/resume path, atomic output, sidecar validation, or exact five-run orchestration were found.
14. All 50 Python files are syntactically parseable, but syntax validity does not establish numerical reproduction.

## Direct Support For Stage1

1. Add a checkpoint-conditioned `gradient_outlier_score` for candidates as a risk or geometry field, never as the sole value rank.
2. Preserve the exact gradient representation, module, dtype, normalization, projection matrix hash, outlier algorithm, contamination/budget, detector seed, and checkpoint hash.
3. Record L1 norm, L2 norm, robust distance, iForest score, nearest-neighbor distance, local density, and cluster identity separately because different detectors encode different notions of outlyingness.
4. Measure outlier-score stability and rank agreement across seed and checkpoint. A sample cannot be called intrinsically anomalous from one model state.
5. Cross outlier fields with P034's separate signed alignment to difficult-normal and weak-defect probe gradients. Outlier magnitude without role-specific direction remains ambiguous.
6. Compare aggregate probe alignment with identity-level alignment dispersion; an aggregate can hide a small weak-defect subgroup that is harmed.
7. Calibrate every proxy against a same-state finite replay intervention and later paired endpoint outcomes. Proxy detection accuracy alone is not model value.
8. Keep OOF/train discovery, `val_op` calibration, and blind holdout roles physically separated. Never fit an outlier detector using blind holdout gradients.
9. Log compute time and peak memory because raw full-head gradients can exceed practical RAM before dimensionality reduction.

## What It Does Not Support

1. Defining large gradient norm, gradient outlier status, or suspected label noise as high replay value.
2. Assuming that trimming a detrimental sample and replaying a beneficial sample are symmetric interventions.
3. Choosing a 5% or 30% Stage1 replay ratio from this paper.
4. Adding OGA as a formal first-cycle training arm.
5. Replacing signed normal-tail and weak-defect alignment with an unsupervised outlier score.
6. Using test identities, test gradients, or test endpoint maxima during discovery.
7. Importing the paper's ResNet, SGD, learning rate, batch size, epoch count, trimming budget, projection, or LoRA settings.
8. Changing any Stage1 canonical hyperparameter.

## Stage1 Field Contract

At preregistered key checkpoints for a bounded candidate/probe subset, save:

- candidate identity, role, checkpoint, model-state hash, and canonical-lock SHA256;
- exact last-layer parameter names, shape, dtype, and gradient-embedding schema;
- raw and normalized L1/L2 norms plus robust norm percentiles;
- iForest score, detector prediction, contamination, detector seed, and implementation version;
- sparse-projection seed, matrix hash, input/output dimensions, and distortion diagnostics;
- local density, nearest-neighbor distance, cluster identity, and video/source concentration;
- difficult-normal and weak-defect target dot products, cosines, and violation flags;
- aggregate-versus-identity alignment dispersion and worst protected-identity conflict;
- checkpoint/seed rank stability, score age, intended and realized replay exposure;
- finite one-step and bounded multi-step intervention effects on identity-disjoint probes;
- endpoint difficult-normal benefit, weak-defect harm, and raw `FN=0-95` frontier change;
- wall time, peak host RAM, peak CUDA memory, and failure/completion state.

## Concrete Experiment Consequence

P036 adds no formal training arm and no hyperparameter change. It adds one bounded diagnostic family to the existing canonical-locked causal schedule block:

```text
same frozen selection + same seed + same canonical training configuration
no replay vs continuous vs same-peak decay vs cumulative-dose-matched decay
```

At the preregistered gradient checkpoints, test whether a candidate's outlier score adds information after signed difficult-normal and weak-defect alignment is known. The falsifier is explicit: if outlier status does not predict finite replay harm or improve held-out endpoint prediction conditional on direction and exposure, it remains only a data-quality review flag.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, for the published outlier-gradient construction, empirical failure of outlier detection as a sufficient value proxy, and detector/budget dependence
- Replication-depth eligibility: yes, with explicit environment, seeding, paper-code, leakage, and test-selection caveats
- Direct support for static replay ranking: no
- Direct support for a new formal arm: no
- Direct support for a risk/geometry field: yes
- Permission to change canonical Stage1 hyperparameters: no
- Reviewed at: 2026-08-08
