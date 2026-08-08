# P040 - InfoBatch

## Identity

- Paper ID: P040
- Full title: InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning
- Authors: Ziheng Qin, Kai Wang, Zangwei Zheng, Jianyang Gu, Xiangyu Peng, Zhaopan Xu, Daquan Zhou, Lei Shang, Baigui Sun, Xuansong Xie, and Yang You
- Venue and year: ICLR 2024 oral
- Official conference page: https://openreview.net/forum?id=C61sk5LsK6
- Official proceedings page: https://proceedings.iclr.cc/paper_files/paper/2024/hash/ad1d7a4df30a9c0c46b387815a774a84-Abstract-Conference.html
- Local PDF: `source_papers/InfoBatch_ICLR_2024.pdf`, SHA256 `1376FDBE53778E012D03BBAD863E3445A67C7CE314E5F1EA3071FF6E527ABA69`
- Official supplementary archive: `source_papers/InfoBatch_ICLR_2024_supplementary.zip`, SHA256 `4C41CF0F729FF34D5241EA409DB35ADB8A56BED9AF7A14E0C00728FB78B65403`
- Official code: https://github.com/NUS-HPC-AI-Lab/InfoBatch
- Audited code commit: `8bbb012c256c9eb4de3bb209f8a5e73899de8d02`

## Reading Coverage

- Full paper: 22/22 pages read, including Equations 1-25, Figures 1-9, Tables 1-12, the complete method, all reported ablations, experimental details, proofs, variance analysis, rescaling limit, and limitations.
- Visual verification: all 22 rendered pages inspected at original detail under `audit/visual_checks/P040_InfoBatch_ICLR_2024/`; equations, plots, tables, appendix material, and references are legible.
- Supplement coverage: all six files from the official ICLR supplementary archive were extracted and inspected. Its five Python files are semantically identical to the repository's `research/` versions after normalizing line endings; only the short usage README differs.
- Code coverage: all 87 public commits were enumerated, the current remote HEAD was verified, all 17 Python files across the public checkout and submitted supplement parse with `ast.parse`, and the sampler/rescaling semantics were checked with a finite numerical probe.
- Peer-review limitation: the OpenReview forum and both API generations returned a browser challenge or HTTP 403. No reviewer statement is used as evidence.

## Research Question

InfoBatch asks whether dynamic removal of currently low-loss examples can reduce training cost without degrading ordinary benchmark accuracy. It is directly relevant to Stage1 because it treats sample exposure as time-dependent and restores the undisturbed data distribution late in training for stability.

The intervention is nevertheless different. InfoBatch removes part of the base dataset and later adds the full base dataset back. Stage1 always retains the canonical base dataset, adds extra replay exposure, and may remove only that extra exposure late. Both manipulate late-stage distributional pressure, but they are not the same treatment and cannot share an assumed effect size or schedule.

## Method And Mathematics

For sample `z`, InfoBatch stores the most recently observed loss as score `H_t(z)`. Before epoch `t`, it uses the global mean score as threshold. Equation 3 assigns pruning probability

```text
P_t(z) = r,  if H_t(z) < mean(H_t)
P_t(z) = 0,  otherwise.
```

Pruned samples retain stale scores; retained samples receive their current loss as the next score. A retained low-score sample is multiplied by

```text
gamma_t(z) = 1 / (1 - P_t(z)).
```

Equations 5-7 show an importance-sampling statement: under the random inclusion law and rescaling, the expected objective or gradient direction is proportional to the full-data quantity. It is not equality of the realized mini-batch trajectory, optimizer state, prediction tail, or final model. The paper explicitly states that compensating for fewer optimizer steps is approximate and requires a pruning ratio that is not too high and a learning rate that is not too large.

Equation 8 turns pruning off after `delta * C` epochs and trains on the full dataset to epoch `C`. Appendix B.3 gives the important finite-path reason: late in training there may be too little time to revisit omitted identities, so expectation over the sampling process cannot guarantee that a realized run is unbiased. The appendix also says this realized bias is not always harmful; annealing is introduced for stability, not as a universal monotone improvement theorem.

The variance argument in Equations 12-18 assumes low-loss examples generally have smaller gradients. The rescaling limit in Equations 19-22 depends on curvature, gradient covariance, batch size, and learning rate. Thus the mathematical claim is conditional even before considering Stage1's asymmetric normal-tail and weak-defect objective.

## Experimental Protocol And Results

- Datasets and tasks include CIFAR-10/100, ImageNet-1K, Tiny-ImageNet, ADE20K segmentation, FFHQ latent diffusion, MAE pretraining, and LLaMA instruction tuning.
- Main comparisons use ordinary accuracy, mIoU, FID, or aggregate language-task scores. None uses a fixed-FN safety frontier, paired seed benefit, weak-defect constraint, or replay-budget return.
- Default paper settings are reported as `r=0.5` and `delta=0.875`, with other task-specific values. These are evidence context only and are forbidden from entering the Stage1 canonical lock.
- Table 4 separates soft pruning, rescaling, and annealing. Rescaling alone raises the mean but has larger variance; annealing alone changes less; the combination matches the full-data mean. Figure 8 shows several run points, but the paper does not disclose their seed identities or count in the text.
- Figure 3 shows degradation at `r >= 0.6`. Larger `delta`, meaning fewer final full-data epochs, lowers mean performance and increases variance. This supports testing late exposure separately from total dose, not selecting a numeric Stage1 cutoff from this paper.
- Table 5 shows that pruning high-loss samples or pruning randomly with rescaling and annealing can also be near the full-data result at lower realized pruning ratios. The default low-loss rule is slightly better in that benchmark, but identity ranking is not the only operative mechanism.
- Table 1 reports mean plus spread for CIFAR conditions, yet the number of repeats, seed identities, paired structure, confidence intervals, and multiplicity handling are not specified. ImageNet results generally have no repeat uncertainty.
- Appendix A changes the CIFAR-100 ResNet-18 maximum learning rate from `0.03` for the baseline to `0.05` for InfoBatch because of reduced steps. Other tasks also adapt optimizer or training length. Therefore some reported lossless comparisons are not strict fixed-hyperparameter interventions.

## Official Code Audit

The repository has 87 commits, no release tag, and no exact publication environment lock. The current core implementation differs from the January 2024 accepted-period implementation only by a later `__getattr__` delegation, but no tag identifies a definitive paper snapshot. The submitted supplement provides the strongest publication-time code evidence.

### Sampling and rescaling contract

The current implementation follows the broad paper algorithm: it retains all high-score samples, samples a fixed fraction of low-score samples, and weights retained low-score losses. Several finite implementation details weaken a literal unbiasedness claim:

- It selects `int(keep_ratio * m)` of `m` low-score samples but weights them by `1 / keep_ratio`. When `keep_ratio * m` is not an integer, the exact inclusion probability is `floor(keep_ratio*m)/m`, not `keep_ratio`.
- A seven-sample probe with three low-score identities and `keep_ratio=0.5` retained one low-score identity per draw. The expected weighted inclusion was `2/3`, not `1`; 5,000 trials produced per-identity values around `0.65-0.68`. The discrepancy is negligible for a large undivided pool but can matter for small protected strata and proves the implementation is not exactly unbiased for every finite set.
- `prune_ratio` is silently clamped so at least 10% is retained. Invalid requested values therefore change semantics without a failure or manifest entry.
- All initial scores are equal, so the first effective epoch uses the full dataset. The sampler also advances once during construction. Requested, realized, and cumulative pruning exposure are not written as an epoch-level provenance table.

### Randomness, resume, and distributed behavior

- The sampler calls `np.random.seed(iterations)` and mutates NumPy's process-global RNG. Sampling is tied to epoch count rather than the experiment seed, and it can perturb other NumPy randomness in the process.
- Neither dataset nor sampler implements `state_dict` or `load_state_dict`. The example parses `--resume` but never loads or writes a checkpoint. A resumed process cannot recover sample scores, weights, sampler iteration, RNG state, or the exact exposure path.
- Importing the package globally monkey-patches the private PyTorch `_BaseDataLoaderIter.__next__` method. This is version-fragile and affects every DataLoader in the process.
- DDP gathers index/loss tensors under equal-shape assumptions and wraps a changing-length sampler through `DistributedSampler`; no distributed equivalence or resume test is included.

### Training and evaluation provenance

- The example sets no Python, NumPy, Torch, CUDA, or deterministic seed. It has no run manifest, data hash, environment lock, checkpoint identity, atomic output, or test suite.
- The OneCycle scheduler is reconstructed every epoch using the realized pruned loader length. Consequently the learning-rate path and optimizer-step count depend on the sampling path. This is part of their acceleration compensation, but it means score rule, exposure, optimizer steps, and LR trajectory are not isolated.
- The test split is evaluated every epoch. The current example tracks the best test accuracy during training, while the paper does not document whether tables use final or best test epoch. Evaluation-selection provenance is therefore incomplete.
- `setup.py` requires only `torch>=1.11` although the package imports NumPy and relies on private PyTorch internals. No tested version matrix or dependency lock is provided.

## Direct Support For Stage1

1. Sample inclusion is a dynamic process variable; a static identity score is insufficient to describe training exposure.
2. Late distributional perturbation can increase realized bias and variance even when an update is unbiased in expectation.
3. Schedule and cumulative dose must be separated. Same-peak decay and dose-matched decay are therefore a justified causal pair.
4. Configured replay ratio is not enough. Stage1 must record realized identities, weighted exposure, optimizer steps, LR, score age, resampling decisions, and cumulative class/tail exposure every epoch.
5. The final part of training can be used as a residual-correction phase, but its direction and duration must be established under Stage1's canonical configuration.

## What The Paper Does Not Support

1. It does not show that stopping extra replay after epoch 140 or 160 improves Stage1; its intervention restores omitted base data instead.
2. It does not identify high-value normal or defect identities, protect a weak-defect tail, or optimize `FN <= 95`.
3. Expected gradient preservation does not imply paired seed stability, realized-path equivalence, or positive tail-specific treatment effect.
4. It does not provide a no-replay versus continuous versus dose-matched replay experiment.
5. It does not justify importing `r`, `delta`, learning rates, optimizer choices, batch sizes, augmentations, or any other paper hyperparameter.
6. It does not resolve the same-selection cross-seed reversals observed in the 240 canonical Stage1 runs.

## Transfer Boundary And Observable Consequences

InfoBatch supports the existing mechanism experiment but adds no new formal arm. The relevant Stage1 evidence card should state:

```text
mechanism:
late repeated distributional pressure can leave a realized optimizer path biased;
removing that pressure while continuing canonical base training may improve stability.

minimum falsifiable contrast:
same selection + same seed + same canonical lock,
continuous replay versus same-peak decay versus cumulative-dose-matched decay.

required process fields:
configured and realized replay ratio,
identity-level exposure and resampling probability,
weighted examples and optimizer steps,
learning-rate trajectory,
normal-tail and weak-defect trajectories,
checkpoint-conditioned gradient alignment,
resume lineage and RNG state.
```

If both decay schedules beat continuous replay, late timing is implicated. If only same-peak decay wins, reduced total dose is the stronger explanation. If dose-matched decay wins while process fields remain canonical, timing matters beyond total exposure. If neither wins, the late-exposure mechanism does not transfer from pruning to Stage1 replay.

## Decision

- Reading status: REPLICATION_DEPTH
- New formal training arm: no; strengthens the already proposed continuous/same-peak-decay/dose-matched-decay contrast
- New Stage1 hyperparameter: no
- Canonical lock change: no
- Added evidence: expectation-level gradient correction, realized optimizer-path correction, cumulative dose, and late-stage residual correction are distinct quantities and must be observed separately
- Remaining uncertainty: whether removing extra normal replay late protects Stage1 weak defects under identical initialization, base-data order, optimizer path contract, and canonical hyperparameters
