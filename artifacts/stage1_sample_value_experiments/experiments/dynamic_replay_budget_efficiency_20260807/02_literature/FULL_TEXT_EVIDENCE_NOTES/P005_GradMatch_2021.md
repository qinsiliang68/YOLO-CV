# P005 - GRAD-MATCH: Gradient Matching based Data Subset Selection for Efficient Deep Model Training

## Identity

- Paper ID: P005
- Authors: Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, Abir De, Rishabh Iyer
- Venue and year: ICML 2021, PMLR 139, pages 5464-5474
- Official landing page: https://proceedings.mlr.press/v139/killamsetty21a.html
- Main PDF: https://proceedings.mlr.press/v139/killamsetty21a/killamsetty21a.pdf
- Supplement: https://proceedings.mlr.press/v139/killamsetty21a/killamsetty21a-supp.pdf
- Local main PDF: `source_papers/GradMatch_2021.pdf`
- Main SHA256: `945F5AEBCDD0A134726E495AFE99141738A6241E4EFD7A444912FBAA66F56ABB`
- Local supplement: `source_papers/GradMatch_2021_supp.pdf`
- Supplement SHA256: `114BCE487443A2D002783EADA43AAF392501C266964FAB3BDE16B609A6F7190E`
- Page count: main 11; supplement 22
- Code: https://github.com/decile-team/cords
- Code snapshot inspected: commit `8d10c7f5d96e071f98c20e4e9ff4c41c2c4ea2af`; GradMatch blob `4fad8691193df15347fa9c5b80369c5bc31a6635`; OMP solver blob `0d61fa76e9472f2d0973c882520c70e586b4b322`

## Reading Coverage

- Main PDF: 11/11 pages read, including convergence motivation, objective, OMP algorithm, approximations, experiments, ablations and conclusion.
- Supplement: 22/22 pages read, including notation, full-GD and SGD proofs, per-step descent condition, weak-submodularity and set-cover proofs, CRAIG connection, complete settings, tables, variance, significance, gradient-error and redundancy diagnostics.
- Method: weighted set-gradient residual, regularized sparse approximation, OMP, adaptive reselection every `R` epochs, validation-target and training-target variants.
- Approximations: last-layer, per-class, per-class-per-gradient and per-batch gradients; warm start.
- Experiments: MNIST, CIFAR-10/100, SVHN and ImageNet; class imbalance; speed, energy, accuracy and gradient error.
- Visual verification: main pages 3, 5, 8 and 9; supplement pages 9, 12, 16, 20 and 21 under `audit/visual_checks/P005_GradMatch/`.

## Research Question

Can a small, dynamically refreshed, weighted training subset reproduce the full training or validation gradient closely enough to retain model quality while reducing training time and energy?

## Method And Equations

At model state `theta_t`, the paper defines the set-gradient residual:

```text
Err(w_t, X_t, L, L_T, theta_t)
  = || sum_{i in X_t} w_i,t grad L_T_i(theta_t)
       - grad L(theta_t) ||_2
```

`L` is either the full training loss or a clean validation loss. The regularized selection objective is:

```text
min_{X: |X| <= k, w}
    || G_X w - g_target ||_2 + lambda ||w||_2
```

The subset is recomputed every `R` epochs and reused between selection points. Orthogonal matching pursuit greedily chooses the gradient atom most correlated with the current residual, refits all selected weights, recomputes the residual and repeats until the budget or tolerance is reached. This is fundamentally a set objective: the marginal value of one sample depends on the gradients already selected.

For mini-batch SGD, GradMatch-PB treats complete mini-batch gradients as atoms. The paper also uses last-layer, per-class and per-class-per-gradient approximations to reduce memory and selection cost.

The released implementation inspected at the recorded commit calls OMP with `positive=True`, removes negative fitted coefficients and fills any unspent budget with random points of unit weight. The main paper's mathematical optimization does not clearly state the non-negativity constraint, so the implementation detail must be treated as part of the executable contract rather than inferred from the displayed objective.

## Theory And Proof Audit

Theorem 1 adds the average gradient-residual norm to standard convex convergence bounds. It applies to full GD and, in expectation, SGD under bounded-domain, Lipschitz/smooth/strongly-convex assumptions and normalized weights. The neural-network experiments are non-convex, so these are motivation and bounds under assumptions, not a direct guarantee for YOLO classification.

The supplement's Theorem 4 provides the more useful local condition. For aggregate selected gradient `g_X` and target gradient `g_L`, smoothness gives:

```text
L(theta - alpha g_X) - L(theta)
  <= -alpha * g_L^T g_X + (L_smooth * alpha^2 / 2) ||g_X||^2
```

Thus positive cosine is necessary but not sufficient for a finite step. The step must also be small relative to target-gradient magnitude, aggregate update norm and curvature. As training progresses, a shrinking target gradient or drifting cosine can make a formerly useful fixed replay update overshoot.

There is a material proof inconsistency in Supplement equations 64-69. The displayed SGD derivation replaces a signed residual inner product by `-D ||residual||`, making a larger mismatch appear to improve the upper bound. The main-paper theorem and the full-GD derivation use the expected positive residual penalty. The sign in the SGD appendix should therefore be treated as a typographical/algebraic error unless independently corrected; no Stage1 claim should rely on that derivation alone.

## Experimental Contract

- MNIST uses LeNet for 200 epochs; the other small-image datasets use ResNet-18 for 300 epochs; ImageNet uses ResNet-18 for 350 epochs.
- Optimizer is SGD with initial learning rate 0.01, momentum 0.9, weight decay `5e-4` and per-epoch cosine annealing.
- When no validation split exists, the paper reports taking 10% and 20% from training for validation and test respectively.
- Small-image subset budgets are 5%, 10%, 20% and 30%; MNIST uses 1%, 3%, 5% and 10%; ImageNet uses 5%, 10% and 30%.
- Default reselection interval is `R=20`; warm variants use `kappa=1/2`; OMP regularization is `lambda=0.5`; the supplement reports tolerance `1e-10`.
- Class imbalance removes about 90% of examples from 30% of classes and uses clean validation-gradient matching.
- Most reported means use five runs; supplement Table 7 gives standard deviations and Table 8 reports one-tailed Wilcoxon signed-rank comparisons.
- Runs use one V100 except the reported ImageNet setup, which uses RTX 2080. Time and energy include selection cost.

## Main Results

- GradMatch-PB-Warm gives the strongest reported accuracy-efficiency tradeoff across most datasets and budgets.
- At 30% subsets it approaches full-data accuracy with roughly threefold speedups; extended training narrows the remaining accuracy gap while retaining a smaller speed advantage.
- Warm start matters most at small budgets; too little warm start begins selection from a poor representation, while too much approaches early stopping.
- Per-batch variants are faster and often lower variance than per-example variants.
- Validation-gradient matching improves class-imbalance results and can outperform biased full-data training in the reported synthetic setting.
- Direct residual minimization yields lower measured gradient error than CRAIG's upper-bound objective in the MNIST diagnostic.

## Ablations And Failure Cases

### Very small budgets remain unstable

Supplement Table 7 states that standard deviations are larger for small subsets. GLISTER can have higher variance than random at small budgets. This is directly relevant to Stage1's low replay ratios: small-budget claims need more seeds, not fewer.

### Warm start and timing are not optional details

The method performs worse without a useful representation. `kappa=1/2` is best only in the tested setup; it is evidence that timing matters, not a transferable Stage1 stop fraction.

### Reselection interval trades accuracy for cost

`R=5,10,20` changes both selection freshness and overhead. A stale subset can be cheaper but less state-aligned. The paper does not test a permanently frozen replay set across 200 epochs.

### Regularization prevents concentration

`lambda=0` performs poorly because OMP can assign excessively large weights to individual atoms when the subset is reused. Large lambda also hurts by over-constraining the fit. This supports exposure/concentration monitoring, not copying `lambda=0.5`.

### Approximation changes the scientific object

Last-layer, per-class and per-batch variants are not interchangeable. Per-batch selection can outperform per-example selection partly because batch atoms already include interaction and smoothing. Final-layer alignment requires a smaller full-gradient audit before it can stand in for full-network behavior.

### Complete subset training differs from replay

GradMatch removes unselected base samples for many epochs. Stage1 retains all 120,000 base samples and adds repeated exposure. The same selected gradients therefore act as a correction to an existing base gradient, not as its replacement.

### Statistical comparison is not paired to one fixed set

Five-run means and a broad Wilcoxon comparison do not establish that one identical selected set is stable across initialization seeds. The reported low standard deviation cannot resolve Stage1's observed same-selection sign reversals.

### Theory is convex and the supplement has a sign issue

The principal guarantees do not directly cover the non-convex network, adaptive optimizer, augmentation, raw quantile safety metric or additive duplicate replay. The SGD proof discrepancy further limits theorem-level transfer.

## What It Supports For Stage1

1. Sample value is a set-level residual-reduction problem, not an independent top-k ranking.
2. Measure selected-set aggregate gradient, cancellation and residual reduction in addition to per-sample alignment.
3. Recompute or at least observe alignment across training stages; a frozen set can become stale.
4. Include timing and cumulative exposure as causal variables.
5. Use non-negative weights or replay counts because physical replay cannot apply a negative sample frequency.
6. Add concentration regularization, video/cluster coverage and maximum exposure limits.
7. Compare last-layer diagnostics against a smaller full-network audit set.
8. Use more unseen seeds at low replay ratios because low-budget variance is high.

## What It Does Not Support

1. It does not justify using 1%, 5%, 10%, 20% or 30% subset budgets as Stage1 replay percentages.
2. It does not justify `R=20`, 50% warm start, `lambda=0.5` or any fixed decay epoch for Stage1.
3. It does not study additive replay, weak-defect guards, FN-constrained safety frontiers or raw score tails.
4. It does not prove that validation-gradient matching is safe when the validation objective underweights rare weak defects.
5. It does not establish static sample value or cross-seed stability for one fixed set.
6. It does not validate OOF/test-derived selection without a clean, isolated, non-test target set.
7. It does not prove that last-layer gradients preserve ranking or sign for the full network in this task.

## Transfer Boundary

For additive replay, the transferable objective is not to replace the full base gradient. Let:

```text
g_base,t   = gradient induced by the unchanged 120k base training distribution
g_replay,t = weighted gradient of the candidate replay set
g_target,t = non-test operational probe gradient
```

The actual update direction is approximately:

```text
g_combined,t = a_t * g_base,t + b_t * g_replay,t
```

Replay should reduce the target residual left by the base update:

```text
residual_before = ||g_target,t - a_t g_base,t||
residual_after  = ||g_target,t - g_combined,t||
correction_gain = residual_before - residual_after
```

This is a Stage1-specific residual-gradient-matching hypothesis. It is inferred from the paper's set objective and must be tested; the paper does not evaluate it. It also makes clear why an individually aligned sample can be useless after the base gradient has already supplied the same direction, or harmful after cumulative replay overshoots.

Normal-tail and weak-defect probe gradients must remain separate. A replay set that improves normal-tail residual while worsening the weak-defect residual is not gap-positive.

## Concrete Field Requirements

At each key checkpoint and for every formal arm:

- `g_base_norm`, `g_replay_norm`, `g_combined_norm` and `g_target_norm`;
- dot products and cosines among base, replay, normal-tail target and weak-defect target gradients;
- residual norm before and after adding replay, by target stratum;
- `correction_gain` and normalized correction gain;
- aggregate cancellation ratio `1 - ||sum_i w_i g_i|| / sum_i ||w_i g_i||`;
- per-cluster gradient contribution and maximum concentration;
- effective non-negative replay weights/counts and cumulative exposure;
- sign/rank stability across checkpoints and seeds;
- a finite-step probe-loss change at a small preregistered virtual step, where affordable, to detect second-order overshoot;
- final-layer versus full-network agreement on a stratified audit subset.

## Concrete Experiment Consequence

- Keep the formal base optimizer and hyperparameters unchanged; use gradient matching first as a checkpoint diagnostic.
- Add a set-level `Residual-GradMatch` candidate only after the diagnostic code proves deterministic identities, non-negative weights, zero test leakage and acceptable cost.
- If promoted, compare it with per-sample positive alignment, magnitude-only, matched random and no replay under identical replay ratio, schedule and seed.
- Do not collapse normal-tail correction and weak-defect correction into one weighted score before an explicit constrained rule is preregistered.
- The timing experiment should report whether correction gain and descent margin decay or reverse before final performance diverges.

## Reproduction Notes And Missing Information

- Main and supplement provide the objective, algorithms, theory, architecture, optimizer, schedules, budgets, reselection interval, warm-start fraction, OMP regularization/tolerance and extensive result tables.
- The inspected released implementation enforces positive OMP coefficients and documents per-class, per-batch and per-class-per-gradient variants, but the inspected commit postdates publication and may not be the exact experiment commit.
- Exact random seeds, augmentation details and every data-order choice are not fully specified in the PDFs.
- The supplement's SGD residual-sign inconsistency requires correction before reproducing that proof.
- `REPLICATION_DEPTH` records a bounded reconstruction and audit; no benchmark rerun was performed here.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, with explicit theory and task-transfer caveats
- Direct support for set-level gradient residuals and adaptive timing: yes
- Direct support for additive replay efficacy or numeric Stage1 parameters: no
- Reviewed at: 2026-08-07
