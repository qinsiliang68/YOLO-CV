# P015 - Training Data Attribution via Approximate Unrolling

## Identity

- Paper ID: P015
- Authors: Juhan Bae, Wu Lin, Jonathan Lorraine and Roger Grosse
- Venue and year: NeurIPS 2024
- Official page: https://proceedings.neurips.cc/paper_files/paper/2024/hash/7af60ccb99c7a434a0d9d9c1fb00ca94-Abstract-Conference.html
- Main PDF: `source_papers/SOURCE_Approximate_Unrolling_2024.pdf`, SHA256 `C0B6A9E52F16D2AA87782A479716B8D247E7FB52414ACFDE1D061F00F5491271`
- Paper-linked library: https://github.com/pomonam/kronfluence
- Paper-linked snapshot: `19088a899dd333aae086fe82213a972db5e6c493`
- Later SOURCE reference implementation: https://github.com/pomonam/simple-influence
- Later implementation snapshot: `2e4e8f04752f89fabd55db2f6a6ba278e3dee4d3`

## Reading Coverage

- Official paper and appendices: 40/40 pages read.
- Derivations checked: exact unrolled derivative, expected trajectory effect, segment factorization, stationarity and independence approximations, spectral filter, influence-function limit, preconditioned optimizer extension and EK-FAC construction.
- Experiments checked: LDS, subset-removal counterfactuals, full/non-converged/noisy/multi-stage tasks, segment count, parameter averaging, single-versus-multiple model attribution, sampling-ratio sensitivity and linear-model controls.
- Limitations and checklist checked: computational scaling, checkpoint/hyperparameter requirements, segment stationarity, ignored optimization autocorrelation, publication-time code availability and statistical uncertainty.
- Code checked: current `kronfluence` repository, publication-time commit identity, later `simple-influence` SOURCE implementation, tests and dependency metadata.
- Visual verification: all 40 pages and four ten-page contact sheets under `audit/visual_checks/P015_SOURCE_Approximate_Unrolling/`.

## Research Question

Can training-data attribution retain information about a finite, non-converged or multi-stage optimization trajectory without storing and differentiating through every SGD step?

## Target Quantity

For one realized mini-batch trajectory, upweighting sample `m` changes an SGD step through:

```text
theta_(k+1)(epsilon)
  = theta_k(epsilon)
    - eta_k / B * sum_i (1 + delta_ki * epsilon) grad L(z_ki, theta_k(epsilon))

d theta_T / d epsilon
  = -sum_k eta_k / B * delta_k * J_(k+1:T) * g_k
```

The paper then takes an expectation over batch-selection trajectories while fixing the initialization in the derivation. Removal is approximated by the negative derivative at `epsilon=0`. This is already more path-aware than endpoint influence, but it remains a first-order counterfactual around a specified training procedure.

## SOURCE Approximation

The trajectory is partitioned into `L` chronological segments. Within segment `l`, the Hessian, sample gradient and learning rate are approximated as stationary, while Jacobians from different segments are treated as statistically independent:

```text
S_l ~= exp(-eta_l * K_l * H_l)

r_l ~= (1/N) * (I - exp(-eta_l * K_l * H_l)) * H_l^-1 * g_l

E[d theta_T / d epsilon]
  ~= -sum_l (product of later S factors) * r_l
```

The segment spectral response is:

```text
F_r(sigma) = (1 - exp(-eta_l * K_l * sigma)) / sigma
```

It approaches `eta_l*K_l` in very low-curvature directions and `1/sigma` in high-curvature directions. A one-segment approximation resembles damped influence with `lambda ~= 1/(eta*K)`, but the paper only claims a qualitative approximation, not algebraic identity.

For momentum or adaptive optimizers, the appendix introduces a preconditioner `P_k`. With diagonal `P_l`, the local inverse resembles `(H_l + lambda P_l^-1)^-1`. The practical EK-FAC implementation uses a diagonal Hessian approximation for matrix exponentials under AdamW and EK-FAC for the inverse term. Optimizer state is therefore part of the estimand.

## Experimental Contract

- LDS uses `M=100` uniformly sampled data subsets, usually at `alpha=0.5`, with `R` retrainings per subset: 100 for UCI, 10 for MNIST, 20 for CIFAR-10, 5 for GLUE/WikiText and 20 for RotatedMNIST/PACS.
- The reported LDS is a Spearman correlation between retrained subset outputs and the sum of individual attribution scores, averaged over up to 2,000 queries with 95% bootstrap intervals over sampled subsets.
- Subset-removal evaluation first selects 100 test examples correctly classified by all five full-data seeds, removes top-ranked proponents, then retrains under three seeds. The maximum nominal cost is 1,800 retrainings per task-method pair, with early stopping after a query first becomes misclassified.
- Ordinary tasks save six checkpoints and use three early/middle/late segments. Noisy tasks train only three epochs with 30% corrupted targets or labels. Multi-stage tasks use two segments aligned with two datasets.
- Hyperparameters are selected by validation performance averaged over five seeds. Ground-truth construction uses models that train in less than 20 minutes on A6000/A100-class hardware; one task-and-alpha LDS ground truth can still cost up to 210 GPU-hours.
- SOURCE is approximately `C` times the final-checkpoint influence cost when using `C` checkpoints; the parameter-averaged variant is approximately `L` times that cost.

## Main Results

### Trajectory information matters

SOURCE generally outperforms representation similarity, TracIn, TRAK and final-checkpoint influence on LDS, especially for three-epoch non-converged/noisy models and two-stage training. Multiple segments usually outperform one segment. The gain narrows near convergence, consistent with endpoint influence becoming a better approximation there.

### Multiple models improve attribution

Averaging scores from ten independently trained models materially improves LDS for several methods. This supports measuring seed-conditioned attribution and its between-seed dispersion. It does not support averaging away Stage1's seed reversal: both the conditional values and the population summary must be retained.

### Single-point ground truth collapses

At `alpha=(N-1)/N`, all neural-network attribution methods show a large LDS drop. The appendix attributes this to training stochasticity overwhelming the one-point removal signal. This directly rejects treating one noisy leave-one-out estimate as a reliable per-image ground truth.

### Additivity is an evaluation assumption

LDS predicts a subset by summing individual scores. It is useful as a standardized attribution diagnostic but does not test interaction-aware replay, repeated exposure or threshold-tail constraints. SOURCE's success under LDS cannot establish an additive Stage1 value function.

### Subset removal is still not replay

SOURCE more often identifies removal sets that flip a query classification in the studied tasks. The intervention removes data and retrains; Stage1 keeps the entire base pool and adds repeated exposure. Direction, magnitude, optimizer path and set interactions can differ.

## Failure Modes And Limitations

1. Segment stationarity may fail when gradients or curvature change rapidly. The paper proposes more segments as a possible remedy but does not solve automatic segmentation.
2. The factorization neglects autocorrelation between optimization iterates. Continuous replay deliberately creates correlated repeated exposures, so this assumption is especially questionable for Stage1.
3. The expected-trajectory derivation fixes initialization, while empirical evaluation averages over random seeds. Stage1 must report within-initialization intervention effects and across-initialization heterogeneity separately.
4. EK-FAC uses a positive-semidefinite GNH approximation and excludes some parameter types; it is not the exact full Hessian or exact full-network path.
5. The main experiments use small, fast-training benchmarks and no imbalanced `FN <= 95` business constraint.
6. The 95% LDS intervals resample subsets/queries; they are not confidence intervals for deployment success probability across training seeds.
7. The top-removal test uses only three retraining seeds and conditions queries on correct classification across five initial models.
8. The paper gives no Stage1 replay ratio, decay epoch, guard share, selection rule or success threshold.

## Reproduction And Code Audit

- The NeurIPS checklist says code was not released with the paper, although Appendix D says SOURCE code will be provided through `kronfluence`.
- The current `kronfluence` repository is an EK-FAC attribution backend. A source search finds the paper citation but no SOURCE segment/unrolling implementation. It does not reproduce the central LDS or subset-removal experiment suite.
- `simple-influence` added SOURCE in May 2026. It is useful for formula inspection but post-dates the paper and is not evidence that the 2024 results are directly reproducible.
- Its `SourceComputer` supports only `Linear` and `Conv2d` EK-FAC blocks, assumes SGD-style updates, requires manual momentum scaling and explicitly omits the paper's Adam/preconditioner path.
- The implementation omits the global `1/N` factor as a ranking convention. That is harmless for a fixed dataset rank but prevents comparing absolute scores across pool sizes or exposure definitions without restoring scale.
- It allocates the full `(num_query, num_train)` score table on one device. Full Stage1 pairwise attribution is infeasible; a fixed small tail-probe set and streamed training candidates are required.
- True-Fisher pseudo-label sampling has no dedicated persisted generator in `SourceComputer`; reproducibility requires external RNG capture.
- Model mutation and checkpoint loading are not enclosed in a failure-safe restoration context. There are no data/checkpoint hashes, atomic caches, resume sidecars or row-identity checks.
- The tests do not compare SOURCE to exact unrolling, validate non-identical checkpoint averaging, test adaptive optimizers or reproduce the paper. A local `uv run pytest tests/test_source.py -q` attempt stopped before collection because the active environment had no `pytest`; no passing test claim is made.

## Direct Support For Stage1

1. Treat sample value as state-, segment-, optimizer- and exposure-conditioned rather than static.
2. Preserve all-epoch low-cost trajectories and exact replay occurrence/cumulative-dose fields; sparse endpoint summaries cannot diagnose when a sample changes sign.
3. At key checkpoints, compute separate difficult-normal and weak-defect target gradients and retain per-seed attribution before any aggregation.
4. Record the actual optimizer class/state, effective learning rate, momentum/preconditioner and iteration count. `optimizer=auto` is not a sufficient mechanism label after training starts.
5. Add segment diagnostics: gradient/covariance drift, within-segment variance, lag autocorrelation and attribution sign stability. Equal early/middle/late segments are only a baseline.
6. Calibrate any approximate-unrolling score on tiny same-seed, same-checkpoint finite replay branches with a no-intervention continuation control.
7. Use a small fixed tail-probe set and streamed candidate gradients. Do not construct a 120k-by-120k score matrix.
8. Keep continuous, decayed, dose-matched and no-replay arms as the causal test. SOURCE may explain their behavior but cannot replace them.
9. Use unseen seeds to assess whether a mechanism predicts intervention sign. Report both mean effect and worst-seed/downside risk.

## What It Does Not Support

1. A static SOURCE score as a final replay ranking.
2. Summing individual scores to claim a replay set has the same value.
3. Treating gradient/Hessian stationarity as true across epochs 1-200 without diagnostics.
4. Applying the later SGD-only code to a canonical run whose resolved optimizer is AdamW or otherwise preconditioned.
5. Averaging scores across seeds and discarding the seed-specific reversal signal.
6. Choosing replay percentages, segment boundaries or weak-defect guard shares from this paper.
7. Opening a blind holdout to tune attribution settings.

## Stage1 Field Contract

Persist for each trajectory-aware probe:

- paper estimator name/version and explicit intervention sign;
- run, seed, initialization weight, canonical hyperparameter lock, data manifest and checkpoint hashes;
- checkpoint epoch, segment boundaries, iterations and sample occurrence/exposure counts;
- resolved optimizer class, learning-rate schedule, momentum/adaptive preconditioner identity and optimizer-state hash;
- raw sample gradient, target gradients for normal tail and weak-defect tail, gradient norm/dot/cosine and actual-update alignment;
- segment-average gradient/curvature approximation, within-segment dispersion, lag autocorrelation and stationarity diagnostics;
- approximation scope, supported/excluded modules, GNH/Fisher choice, damping/spectral filter and RNG identity;
- per-seed score, cross-seed mean/dispersion/sign agreement and rank stability;
- finite continuation-control delta, finite replay delta, sign/calibration error and target-axis harm;
- query and candidate sample manifests, row counts, streaming partition identity, runtime, memory and failure status.

## Concrete Experiment Consequence

- Do not add a ten-machine SOURCE-ranked arm now.
- Add a small observational trajectory-probe job at key checkpoints, initially using last-layer raw and actual-update alignment plus stationarity/autocorrelation fields.
- For a small stratified candidate pool, compare those probes with same-checkpoint finite replay minus same-checkpoint continuation drift.
- Promote approximate unrolling only if it predicts unseen seed-state intervention signs beyond raw alignment and simple trajectory baselines.
- Keep the formal campaign focused on replay timing, dose matching, weak-defect guard and no-replay under the exact canonical 240-run hyperparameter lock.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, for trajectory-conditioned attribution, segment diagnostics, single-point noise and optimizer-state requirements
- Direct support for static SOURCE replay ranking: no
- Direct support for numeric replay scheduling: no
- Reviewed at: 2026-08-07
