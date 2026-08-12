# P023 - Understanding and Improving Early Stopping for Learning with Noisy Labels

## Identity

- Paper ID: P023
- Authors: Yingbin Bai, Erkun Yang, Bo Han, Yanhua Yang, Jiatong Li, Yinian Mao, Gang Niu and Tongliang Liu
- Venue and year: NeurIPS 2021
- Official page: https://proceedings.neurips.cc/paper_files/paper/2021/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html
- Main paper: `source_papers/Progressive_Early_Stopping_NeurIPS_2021.pdf`, SHA256 `E25C97DF3B77DEA3501E8CAFE6498FEA2EF4B0933732D65BB4C5F5C1A7868ED4`
- Supplement: `source_papers/Progressive_Early_Stopping_NeurIPS_2021_Supplemental.pdf`, SHA256 `47B12063FB3405C834E79E45F837F07F64FD15BC91BF69CD4FA5978BEBAA9596`
- Official code: https://github.com/tmllab/PES at commit `ec5290d9fcc9efa8f302dbe8a78c448805d9e6e7`

## Reading Coverage

- Main paper: 12/12 pages read, including motivation, layer probe, equations, algorithm, all tables and figures, sensitivity, runtime and stated limitation.
- Supplement: 4/4 pages read, including noise construction, preprocessing, complete hyperparameters and additional results.
- Peer review: the official proceedings page was checked. Public reviews were not counted because the available endpoint was blocked by an automated browser challenge.
- Code: all eight Python files, README and requirements inspected; all eight Python files compiled; there is no repository test suite.
- Visual verification: all 16 PDF pages and four contact sheets under `audit/visual_checks/P023_PES_NeurIPS_2021/` were inspected.

## Research Question

Does one stopping time for a whole deep network miss the fact that later layers can absorb noisy-label effects earlier than former layers? Can later network parts be reinitialized and trained for progressively shorter durations while preserving earlier representations?

For Stage1, the transferable question is narrower: can the sign and location of replay effects change with training stage and model module? The paper does not study duplicate replay, operational score tails, paired same-selection seed reversals or a constrained `FN=0-95` frontier.

## Core Method

The network is decomposed into ordered parts:

```text
z_l = f_l(z_(l-1); theta_l), l = 1,...,L.
```

PES first trains the whole network for `T1`, keeps the resulting former part, reinitializes later parts and trains those parts while preceding parts are frozen. Durations obey:

```text
T1 >= T2 >= ... >= TL.
```

The layer-sensitivity diagnostic is not label-free. For a candidate noisy-training epoch, it freezes a selected prefix, reinitializes the remaining layers and trains the remainder on clean data. Final clean classification accuracy measures representation damage. This is a useful audit probe but cannot be embedded in Stage1 formal training without an explicitly separate clean diagnostic set.

The final noisy-label pipeline adds a confident-example criterion. A sample is confident when the average of two augmented-view predictions agrees with its supplied label. The algorithm then applies class weighting or MixMatch. Consequently, the headline final method is a bundle, not an isolated early-stopping intervention.

## Experimental Evidence

- The layer probe on CIFAR-10 reports five-run mean curves under symmetric, pairflip and instance-dependent noise. Later layers peak earlier and fall more sharply in the studied settings.
- The direct early-stopping-versus-PES comparison reports five-run means and standard deviations. PES improves mean accuracy and label precision in the listed CIFAR-10 settings, while the size varies by noise type.
- Main no-semi experiments use ResNet-18 on CIFAR-10 and ResNet-34 on CIFAR-100. Reported settings are `T1=25/30`, `T2=7`, `T3=5`, 200 base epochs, SGD `lr=0.1`, momentum `0.9`, weight decay `1e-4`, drops at epochs 100 and 150, and Adam `lr=1e-4` for the PES stages.
- Semi-supervised experiments use PreActResNet-18, 300 base epochs, `T1=20/35`, `T2=5`, cosine annealing from `0.02` to `0.0002`, weight decay `5e-4` and Adam `1e-4` for PES. Main tables report three runs.
- Clothing1M uses pretrained ResNet-50, paper-reported `T1=20`, `T2=7`, 50 paper-defined epochs, SGD `lr=0.005`, momentum `0.9`, weight decay `0.001`, LR drops at 20 and 30, and Adam `5e-6` for PES.
- Sensitivity varies `T2` and `T3` from 0 to 10. The reported optimum is 7 and 5 in that study, but the authors explicitly identify the added stopping-time hyperparameters as their main limitation.
- Final tables combine PES with confident sample selection, class weighting or MixMatch. They do not provide a complete factorial ablation that identifies the contribution of layer scheduling independently across all datasets.

## Code Reproduction Audit

1. The public repository has no tests. A syntax pass compiled all eight Python files, but this does not validate numerical reproduction.
2. The environment lock contains only Torch 1.7.1 and torchvision 0.8.2. It omits Python, NumPy, SciPy, PIL, CUDA/cuDNN, hardware, dataset identity and raw-result provenance.
3. CIFAR scripts seed Python, NumPy and Torch but enable `cudnn.benchmark=True` while deterministic mode is commented out.
4. `PES_cs.py` evaluates the noisy validation split using `transform_train`, including random crop and flip. Best-validation checkpoint selection is therefore affected by evaluation-time augmentation RNG.
5. `PES_semi.py` and `PES_noisylabels.py` evaluate test accuracy every epoch and report the maximum. Those results are test-selected, not a blind final evaluation.
6. `renew_layers` creates new module parameters after the original SGD optimizer is built. A live identity probe found 17 new layer4/classifier parameters and zero tracked by the original optimizer. Temporary Adam trains them during PES, but later calls to the original optimizer cannot update them after the code sets `requires_grad=True`. The implementation therefore behaves like implicit post-PES freezing of replacement parameters, despite the algorithm's later unfreeze step.
7. Clothing1M replaces `model.fc` after its original SGD optimizer is built and has the same stale-optimizer problem.
8. Current Clothing1M defaults do not reproduce the paper timing. Each inner chunk covers about 100,000 examples, matching the supplement's epoch definition. `T1=3` outer groups of ten chunks starts refinement around paper epoch 30 and scheduler milestones are 30 and 40, while the paper reports refinement at 20 and LR drops at 20 and 30. The earlier 2021 code used `warmup=2`; later edits changed this default.
9. Shared metric code weights each batch by `images[0].size(0)`, which is the channel dimension rather than batch size. Full equal-sized batches are effectively batch-averaged, but a final partial evaluation batch is misweighted.
10. `PES_cs.noisy_refine` calls `update_trainloader` before renewal and discards the result, adding augmented inference and RNG consumption without affecting the returned loader.
11. CIFAR paths have no faithful checkpoint/resume, atomic writes, completion sidecars, sample identity manifest or configuration hash.

## Evidence Limitations

1. The task is noisy-label learning, not replay of correctly labeled but operationally difficult samples.
2. The clean-data layer diagnostic requires information unavailable to ordinary noisy-label training and is not itself the deployed method.
3. Different architectures, datasets and training recipes use separately tuned `T1/T2/T3`; there is no universal stopping time.
4. Central no-semi and semi tables use five and three runs respectively; Clothing1M uncertainty is not reported in the same paired form.
5. The paper optimizes average test accuracy, not separate difficult-normal benefit, weak-defect non-inferiority or a raw safety frontier.
6. Progressive reinitialization changes architecture state, optimizer state and effective training history. It is not equivalent to stopping only the replay stream while canonical base training continues.
7. Current official-code defaults and optimizer behavior do not exactly instantiate the written algorithm, weakening literal replication claims.
8. No evidence identifies Stage1 epochs 140 or 160, a replay percentage, cumulative dose or guard ratio.

## Direct Support For Stage1

1. Process effects can differ by training stage and by model module, so endpoint-only metrics are insufficient.
2. Later/head representations may show harmful fitting before earlier/backbone representations, motivating module-level diagnostics at key checkpoints.
3. One global early-stop statistic can hide antagonistic behavior across components, just as one average score can hide difficult-normal gain and weak-defect harm.
4. Schedule parameters are method-defining hyperparameters and require preregistration, sensitivity analysis and exact provenance.
5. A mechanism hypothesis should be tested by an intervention under one frozen base algorithm, not inferred from a descriptive peak.

## What It Does Not Support

1. Importing PES's optimizer, layer reset, batch size, augmentation, architecture or stopping values into Stage1.
2. Calling epoch 140, 150 or 160 a literature-established memorization point.
3. Treating persistent high loss or late-layer gradient magnitude as positive replay value.
4. Omitting no-replay, dose-matched timing or weak-defect controls.
5. Selecting a final Stage1 checkpoint on the blind test set.
6. Assuming an official repository necessarily reproduces the paper without an implementation audit.

## Stage1 Field Contract

Preserve the exact canonical 240-run training configuration and collect low-cost fields for all 200 epochs:

- head and backbone parameter-update norm, raw gradient norm and optimizer-state norm;
- replay versus base gradient contribution by module;
- difficult-normal and weak-defect target alignment by module;
- head/backbone representation or logit drift on fixed probes;
- train/validation tail losses, margins, correctness transitions and replay exposure;
- actual optimizer parameter-group identity hash so replaced, frozen or untracked parameters cannot occur silently;
- deterministic/runtime settings and all RNG-state identities.

At 120, 140, 150, 160, 180 and 200, save the existing heavy artifacts and compute a fixed, non-training module probe. No module may be reset, frozen or optimized with a paper-specific optimizer in the formal arms.

## Concrete Experiment Consequence

P023 adds a module axis to the already justified timing experiment. Under the exact canonical lock and the same selected IDs, compare no replay, continuous replay, same-peak decay and cumulative-dose-matched relocation. Ask whether a late harmful arm shows this preregistered sequence:

```text
replay contribution becomes relatively larger in the head
difficult-normal target still improves
weak-defect alignment or margin turns adverse
the adverse signal precedes or accompanies safety-frontier deterioration
```

This is explanatory evidence, not an adaptive stopping rule. A schedule can be declared beneficial only from paired finite outcomes on unseen seeds. The PES values `T1/T2/T3` remain literature-only and are forbidden from changing the Stage1 canonical hyperparameters.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, for layer-dependent timing, schedule sensitivity and the need for module-level process probes
- Replication-depth eligibility: yes, because main, supplement, official code, version history, executable syntax and parameter-identity behavior were audited
- Direct support for static replay ranking: no
- Direct support for numeric Stage1 schedule boundaries or percentages: no
- Permission to change canonical Stage1 hyperparameters: no
- Reviewed at: 2026-08-07
