# P044 - Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels

## Identity

- Paper ID: P044
- Authors: Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor W. Tsang, and Masashi Sugiyama
- Venue and year: NeurIPS 2018
- Official proceedings page: https://proceedings.neurips.cc/paper_files/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html
- Main paper: `source_papers/CoTeaching_NeurIPS_2018.pdf`, SHA256 `89E578C4CD1B47ABE7FDE3A3406A373A4E6C3A8A0B473B0E8AA7BA75D0D948F8`
- Official supplement: `source_papers/CoTeaching_NeurIPS_2018_supplemental.zip`, SHA256 `6B445DFCB58449DBB6DA1DC5B6E8994404498F5F2D1F8BA9ECA7AEFB750ED257`
- Official code: https://github.com/bhanML/Co-teaching
- Audited HEAD: `7c7fbe23e15e517db76a0882b6d108e4508e09d6`

## Reading And Audit Coverage

- Main paper: 11/11 pages read, including Algorithm 1, Tables 1-8, Figures 1-6, all stated settings, ablations, conclusion, and references.
- Supplement: 2/2 pages read, including the exact pair/symmetric transition matrices and full-axis Figures 7-9.
- Peer review: all three official reviews checked. They explicitly challenge the known-noise-rate assumption, mini-batch noise-mixing assumption, lack of theoretical analysis, and absence of larger-network evidence.
- Visual verification: all 13 paper and supplement pages inspected at original detail under `audit/visual_checks/P044_CoTeaching_NeurIPS_2018/` and `audit/visual_checks/P044_CoTeaching_NeurIPS_2018_supplemental/`.
- Code: all seven Python files, the full 35-commit history, training loop, loss, schedule, data/noise generation, seed path, evaluation, output, and resume behavior inspected. The legacy Python 2/PyTorch 0.3 stack was not executed.

## Research Question

Co-teaching asks how to avoid memorizing wrong labels when a high-capacity network eventually fits them. Two independently initialized networks select their own small-loss examples in each mini-batch, then exchange those selected examples so that each network updates on its peer's choices.

This is relevant to separating useful hard samples from noise, but it is not a Stage1 sample-value function. The paper assumes synthetic label flips with known hidden clean identities and usually a known noise rate. Stage1's difficult normal and weak defect tails are not known mislabeled examples, and their value depends on the protected operating point and replay trajectory.

## Method And Dynamics

For epoch `T`, Co-teaching retains a fraction:

```text
R(T) = 1 - min((T / T_k) * tau, tau)
```

Each network ranks the current mini-batch by its own cross-entropy loss, retains the smallest-loss `R(T)` fraction, and sends those indices to its peer. The paper's mechanism is explicitly temporal:

```text
early training: keep nearly all examples
later training: increase the drop rate before noisy labels are memorized
```

The claim is therefore not that small loss identifies intrinsically valuable data. It is that, under a memorization regime, loss rank can temporarily enrich for synthetically clean labels. Cross-updating is intended to reduce self-confirming selection error.

## Experimental Contract

- Datasets: MNIST, CIFAR-10, and CIFAR-100 with synthetic pair-flip or symmetric label noise.
- Main noise settings: pair 45%, symmetric 50%, and symmetric 20%.
- Model: the paper's nine-layer CNN with dropout and batch normalization.
- Optimizer context: Adam, initial learning rate 0.001, batch 128, 200 reported epochs, and two random initializations.
- Default schedule: `T_k=10`, `tau=noise_rate`; schedule-shape and `tau` are separately ablated.
- Metrics: test accuracy and selected-label precision, where label precision requires the synthetic clean/noisy ground truth.
- Repetitions: five; curves shade one standard deviation and tables average the last ten reported epochs.
- Baselines: Standard, Bootstrap, S-model, F-correction, Decoupling, and self-paced MentorNet.

All of these numeric settings are evidence context only and are prohibited from entering the Stage1 canonical lock.

## Main Results And Negative Cells

1. Co-teaching strongly improves the reported high-noise cells. For example, CIFAR-10 pair-45 is `72.62 +/- 0.15` versus MentorNet `58.14 +/- 0.38`; CIFAR-100 pair-45 is `34.81 +/- 0.07` versus MentorNet `31.60 +/- 0.51`.
2. It is not universally best. On low-noise symmetric-20, F-correction beats Co-teaching on MNIST, CIFAR-10, and CIFAR-100 in Tables 4-6.
3. The selected-label precision decreases over training even for Co-teaching in several hard cells. The score is dynamic and becomes less reliable as memorization proceeds.
4. Table 8 shows that setting `tau` equal to the known noise rate is not generally optimal. Pair-45 and symmetric-50 peak at `1.25 * epsilon`; symmetric-20 peaks at `1.5 * epsilon`. Too much dropping then collapses performance because too little training data remains.
5. Table 7 shows schedule-shape and ramp duration winners vary by noise regime. The paper says the defaults are stable but not best.
6. Five runs, synthetic corruption, and last-ten-epoch averaging do not establish seed-stable behavior on naturally ambiguous data.
7. The conclusion leaves theory and generalization analysis as future work.

## Official Code Audit

- The repository pins a Python 2.7.12, CUDA 8.0, PyTorch 0.3.0.post4 environment and has no release tag.
- Only PyTorch CPU/CUDA seeds are set. Python, global NumPy, worker, cuDNN, and complete environment state are not persisted.
- Noise generation ignores the function's supplied `random_state` and hardcodes `random_state=0`, so all training seeds receive the same synthetic corruption identities without an explicit paired-design manifest.
- The paper describes only the initial Adam settings. Code additionally linearly decays learning rate after epoch 80 or 100 and changes Adam beta1 from 0.9 to 0.1; this materially affects the training-dynamics interpretation.
- The training loop first evaluates epoch 0 and then trains `range(1, n_epoch)`, yielding 199 training epochs for `n_epoch=200`.
- Both train and test loaders use `drop_last=True`; the 10,000-example test sets are evaluated on 9,984 examples, not the full stated test set.
- Under legacy defaults, the peer-selected update cross entropy is already mean-reduced, then divided again by `num_remember`. The effective gradient scale therefore changes with the retained count in addition to changing sample composition.
- `num_iter_per_epoch` uses `i > limit`, which admits 401 iterations when the loader is long enough. Exposure is dataset-dependent.
- Test accuracy is evaluated and logged every epoch. There is no separate validation-only development contract.
- Output filenames omit training seed and code/data identity; reruns move the old text file to a timestamped backup. There are no checkpoints, resume/RNG restoration, atomic completion markers, manifests, or automated tests.

These defects do not negate the high-level phenomenon, but they prevent treating the repository as an exact, modern reproduction contract.

## Direct Support For Stage1

1. Hardness is state-dependent. Loss rank must be stored by epoch/checkpoint, not reduced to one static score.
2. Extreme hard examples can include noisy or contradictory supervision; gradient magnitude or loss alone is an influence screen, not positive value.
3. Selection quality can decay as memorization proceeds, supporting all-epoch loss/margin/selection-state fields and late-exposure tests.
4. Selection intensity has a non-monotone effect: retaining too much risks noise, while retaining too little removes useful coverage. Realized exposure and effective identity count are required.
5. Self-selection can accumulate bias. Independent model/state views may be useful diagnostics, but any disagreement or cross-selection score must be tested rather than assumed clean.
6. When labels can be audited, selection precision against adjudicated clean/noisy status is a useful diagnostic separate from downstream value.

## What It Does Not Support

1. It does not establish that high-loss Stage1 images are mislabeled or harmful.
2. It does not show that low-loss images are the valuable replay set for an `FN <= 95` objective.
3. It does not justify a Co-teaching arm, two-network training, a drop schedule, `tau`, `T_k`, Adam, or any paper hyperparameter in the final Stage1 campaign.
4. It does not evaluate replay duplication, continuous-versus-decayed replay, cumulative-dose matching, weak-defect guards, raw safety frontiers, or same-selection cross-seed reversal.
5. Synthetic independent label flips do not model video-frame dependence, visual ambiguity, domain shift, or asymmetric business harm.
6. Label precision is unavailable on unaudited real samples and cannot be silently approximated by model confidence.

## Transfer Boundary And Observable Consequence

Co-teaching contributes dynamic risk fields, not a formal training arm:

```text
For each replay identity and every epoch:
- loss and margin under the current model;
- rank within class, role, and mini-batch/context;
- learned/forgotten transitions;
- agreement across paired seeds or checkpoint models;
- realized replay exposure and effective identity count;
- optional human adjudication status and visibility quality.
```

The testable Stage1 consequence is whether harmful seeds show increasing late replay of persistently high-loss, high-disagreement, or weak-defect-conflicting identities while beneficial seeds do not. If these fields fail to separate signs under the same selection, schedule, and canonical lock, noise/memorization is not the primary explanation.

## Decision

- Reading status: REPLICATION_DEPTH
- New formal arm: no
- New hyperparameter: no
- Canonical lock change: no
- Added fields: per-epoch loss rank, role-conditioned rank, selected-state duration, cross-model/seed disagreement, retained/exposed identity count, optional adjudication, and late memorization indicators
- Remaining uncertainty: whether Stage1's hard normal tail contains true annotation noise, legitimate rare normals, weak defects mislabeled as normal, or clean but representation-conflicting samples
