# P006 - GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning

## Identity

- Paper ID: P006
- Authors: Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, Rishabh Iyer
- Venue and year: AAAI 2021, volume 35 issue 9, pages 8110-8118
- DOI: https://doi.org/10.1609/aaai.v35i9.16988
- Official landing page: https://ojs.aaai.org/index.php/AAAI/article/view/16988
- Official PDF: https://cdn.aaai.org/ojs/16988/16988-13-20482-1-2-20210518.pdf
- ArXiv full version: https://arxiv.org/pdf/2012.10630
- Local official PDF: `source_papers/GLISTER_2021_AAAI.pdf`
- Official PDF SHA256: `88ECDA6540A7E528B2CB7C7763492CFB5763A0D421580ADB50E130FB9551BAF6`
- Local full PDF: `source_papers/GLISTER_2021_arxiv_full.pdf`
- Full PDF SHA256: `A88ECD0FAEAE0756724B7AC07EAC815BD713F6189FA8CFB04714031788BC688C`
- Page count: official 9; full version 30
- Code: https://github.com/dssresearch/GLISTER
- Code snapshot inspected: commit `7b70e18d67bdffd0ea7cf43a7f13b8fb9bf9e1aa`, dated 2020-10-28
- Relevant code blobs: `models/set_function_all.py` = `f5bc1889c5774b8bbada0ff80b10c6b8a8de5ab3`; `dss_deep.py` = `2482330394b14b25d737b9cd728c4f2c5a27837e`

## Reading Coverage

- Official paper: 9/9 pages read.
- ArXiv full version: 30/30 pages read, including all proofs, algorithms, experimental settings, additional experiments, ablations and synthetic diagnostics.
- Proofs checked: NP-hardness; exact or approximate submodularity for logistic, squared, hinge, perceptron and cross-entropy losses; one-step descent; convex convergence; Naive Bayes, nearest-neighbor and linear-regression special cases.
- Experiments checked: efficient subset training, imbalance, synthetic label noise, active learning, refresh-interval and Taylor-recomputation ablations, synthetic boundary and covariate-shift examples.
- Public code checked at the recorded commit: one-step Taylor greedy implementation, deep training loop, seed handling, data split, subset refresh and random regularization.
- Visual verification: full-version pages 3, 4, 5, 6, 7, 25, 28 and 29 under `audit/visual_checks/P006_GLISTER/`.

## Research Question

Can a training subset be selected against held-out validation likelihood, updated as the model changes, so that training is more efficient and robust to imbalance or label noise?

## Method And Equations

The ideal objective is a mixed discrete-continuous bilevel problem:

```text
max_{S subset U, |S| <= k}
    LL_V(argmax_theta LL_T(theta, S), V)
```

The exact inner optimization is replaced by one gradient step at the current state:

```text
G_theta_t(S)
  = LL_V(theta_t + eta * sum_{j in S} grad LL_T(theta_t, j), V)
```

The first-order Taylor marginal for adding candidate `e` is proportional to:

```text
eta * grad LL_T(theta_t, e)^T grad LL_V(theta_t^S, V)
```

The validation gradient is recomputed as the greedy set grows. The selected set is refreshed every `L` training epochs; the `r` approximation refreshes the Taylor state only `r` times within one greedy construction and takes a block of roughly `k/r` candidates between refreshes. This makes value state-dependent and set-dependent rather than an independent permanent score.

The regularized objective is:

```text
argmax_{|S| <= k} G_theta_t(S) + lambda * R(S)
```

where the paper studies supervised facility-location and random regularization to reduce overfitting to a small validation set.

## Theory And Proof Audit

The exact selection problem is NP-hard. Under the paper's one-step form, several simple losses yield cardinality-constrained submodular objectives. Cross-entropy is only approximately submodular, with a bound depending on a bounded feature norm. These results concern the discrete objective at a fixed model state; they do not prove stable cross-seed neural-network replay value.

For one update using selected-set training gradient `g_S` and validation gradient `g_V`, smoothness gives the same finite-step structure relevant to Stage1:

```text
L_V(theta - eta g_S) - L_V(theta)
  <= -eta * g_V^T g_S
     + (L_smooth * eta^2 / 2) * ||g_S||^2
```

Theorem 2 therefore requires both non-negative alignment and a sufficiently small learning rate. Positive cosine alone is not enough when the selected aggregate gradient is too large. The theorem assumes bounded gradients and a Lipschitz-smooth validation loss.

The convergence theorem additionally uses convexity, bounded parameters, non-vanishing selected gradients and training/validation gradient similarity over all encountered subsets. Its residual term grows with `1 - cos(theta_l)`. None of those assumptions directly covers non-convex YOLO training, augmentation, Adam-like optimizer selection, duplicate replay or a quantile-constrained raw safety frontier.

The cross-entropy submodularity proof also relies on bounded feature norms and transformations made under a fixed cardinality. It is not a proof that greedy target alignment remains monotone after repeated additive exposure.

## Experimental Contract

- Models: two-layer fully connected network with 100 hidden units for small tabular data; LeNet for MNIST; ResNet-18 for CIFAR-10.
- Training: 200 epochs for shallow models, 100 for MNIST and 150 for CIFAR-10; simple SGD, with learning rate 0.05 stated for shallow experiments.
- When an explicit validation/test split is absent, the paper uses 10% for validation and 20% for test.
- Subset budgets are 10%, 30% and 50% in the reported efficient-training experiments.
- Default subset refresh is every `L=20` epochs.
- Shallow-model Taylor recomputation uses approximately `r=0.03k`; the text says deep models use a larger value near `k`.
- Random regularization uses `lambda=0.9`; facility-location regularization uses `lambda=100`, with scales chosen so components have roughly similar contribution.
- Synthetic imbalance removes 90% of instances from 30% of classes. Main noisy experiments state a 30% flip rate; additional appendix figures use 80% noise.
- The PDFs do not report seed lists, error bars, standard deviations, confidence intervals or paired statistical tests for the displayed results.

## Main Results

- In the paper's tested subset-replacement setting, GLISTER variants generally outperform random, CRAIG and KNN-submodular baselines at the same subset budget.
- Facility-location or random regularization can help at larger budgets, which the authors attribute to reducing validation-set overfitting.
- A clean or balanced validation objective helps on the synthetic noisy/imbalanced tasks and can beat training on all contaminated data.
- Lower `L` means fresher selection but higher cost; larger `L` can reduce accuracy.
- Very low `r` is explicitly reported as unstable, while larger `r` costs more. This is direct evidence that stale/coarse objective updates can alter stability.
- Synthetic examples show validation-targeted selection concentrated nearer a decision boundary, while CRAIG and KNN-submodular selection is more representative.

## Code Audit And Reproduction Gaps

The released code confirms the one-step mechanism: it computes per-example gradients, takes a virtual parameter step using the aggregate selected gradient, recomputes the validation gradient, and ranks remaining candidates by the resulting Taylor gain.

The code snapshot also has important limits:

- `dss_deep.py` fixes `torch.manual_seed(42)` and `np.random.seed(42)` within methods.
- It defines `num_runs = 1` and `warm_method = 0` locally, even though `runs_deep.py` appends run-count and warm-method command-line arguments. Those arguments are not consumed by the shown deep script.
- The train/validation split is produced with `random_split`, but the implementation does not provide an experiment manifest containing the realized sample identities.
- The deep script uses a hard-coded CUDA device and many loaders use `num_workers=1`.
- The README only states Python 3.7, PyTorch 1.4.0 and scikit-learn; it provides no lockfile or exact environment.

Consequently, the public artifact is sufficient to inspect the algorithmic mechanism, but it is not evidence for same-selection cross-seed stability and is not a turnkey reproduction package for the paper's full result matrix.

## Ablations And Failure Cases

### Low Taylor refresh is unstable

The appendix explicitly reports high instability at low `r`. This means an approximate gradient score can fail because its target state is stale, not merely because the candidate sample is intrinsically bad.

### Refresh interval trades quality for compute

Selection every 20, 35 or 50 epochs changes the result. The best reported `L=20` is task-specific and does not justify Stage1's epoch 140/160 schedule.

### Validation objective can overfit

The paper introduces facility-location and random regularization because a small validation set can be overfit. This directly warns against using one average validation tail as an unquestioned target.

### No seed-stability evidence

The paper does not report paired runs for one identical selected set across multiple initializations. The public deep script's fixed seed further prevents the artifact from answering Stage1's observed sign reversals.

### Synthetic corruption is not operational weak-defect risk

Random label flips and artificial class removal do not reproduce SewerML-like correlated video frames, weak visual defects, asymmetric FN constraints or duplicate replay exposure.

### Subset replacement differs from additive replay

GLISTER trains on the selected subset between refreshes. Stage1 always retains the 120k base pool and adds replay slots, so the intervention and optimizer-step distribution are different.

## What It Supports For Stage1

1. Value should be conditioned on current model state and refreshed or observed over time.
2. Candidate value is a set marginal, because the validation gradient changes after previously selected gradients are added.
3. Gradient direction against a non-test target is more informative than gradient magnitude alone.
4. Refresh/coarsening frequency is a genuine stability variable and should be measured, not assumed.
5. A small validation target can overfit, so diversity and separate tail objectives are needed.
6. Timing and cumulative exposure should be manipulated while canonical training hyperparameters remain locked.
7. The local descent margin should include both the directional term and the quadratic update-norm penalty.
8. Cross-seed and same-selection replication must be added because the paper does not provide it.

## What It Does Not Support

1. It does not justify any Stage1 replay percentage, guard percentage, stop epoch or seven-day run count.
2. It does not justify copying `L=20`, `r=0.03k`, `lambda=0.9` or `lambda=100`.
3. It does not test additive replay, weak-defect guards, raw `FN <= 95` frontiers or unit-exposure efficiency.
4. It does not show that one static selected set is stable across seeds.
5. It does not establish that last-layer alignment is equivalent to full-network alignment.
6. It does not resolve conflict between lowering difficult-normal scores and preserving weak-defect scores.
7. It does not make blind holdout data eligible for target construction or selection.

## Transfer Boundary

For Stage1, the transferable quantity is not a permanent GLISTER rank. It is the checkpoint-specific marginal change in a non-test operational probe objective after adding a candidate or set to the unchanged base update.

The target must remain vector-valued:

```text
benefit_normal_t = expected decrease in difficult-normal probe loss
harm_defect_t    = expected increase in weak-defect probe loss
```

A candidate can enter the admissible set only if weak-defect harm satisfies a preregistered non-inferiority constraint. Diversity or set-residual coverage is then optimized among admissible candidates. These are Stage1 hypotheses inferred from GLISTER's validation-targeted greedy mechanism; the paper does not validate them for this task.

## Concrete Field Requirements

At each key checkpoint and for each formal arm, record:

- per-stratum target-gradient norm and replay-gradient norm;
- target/replay dot product, cosine and finite-step descent margin;
- marginal gain after the already selected replay set, not only independent per-sample alignment;
- set-level cancellation, gradient concentration and cluster/video coverage;
- target identity, sample manifest, fold and SHA256;
- selection refresh age and time since the last gradient measurement;
- cumulative replay exposure and the base/replay update ratio;
- sign and rank stability across checkpoints and seeds;
- separate difficult-normal benefit and weak-defect harm;
- final-layer versus full-network agreement on a smaller stratified audit subset.

## Concrete Experiment Consequence

- Preserve all canonical 240-run hyperparameters and manipulate only preregistered replay policy variables.
- Use gradient alignment first as an observational checkpoint probe, not as an immediate permanent ranking.
- Include a same-selection continuous-versus-decay causal comparison to test whether late stale exposure creates harm.
- Add a dose-matched schedule so timing is separated from cumulative replay amount.
- Keep no-replay and matched random controls; neither GLISTER nor its code removes the need for them.
- Do not merge difficult-normal and weak-defect gradients into an arbitrary weighted scalar. Apply a guard/non-inferiority rule first.

## Reproduction Notes And Missing Information

- The full paper supplies objectives, pseudocode, complexity, proof assumptions, datasets, broad architectures, major budgets, refresh intervals and several ablations.
- The public code snapshot supplies executable mechanism details but contains fixed-seed and unused-run-argument issues.
- Exact seeds, all augmentation/preprocessing choices, complete environment lock, all result-producing commands and statistical uncertainty are missing.
- No benchmark rerun was performed here; `REPLICATION_DEPTH` means the method and artifact contract were reconstructed and audited, not that the paper's numerical results were reproduced.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, with explicit task, objective, finite-step and reproducibility caveats
- Direct support for dynamic validation-gradient selection and refresh sensitivity: yes
- Direct support for additive replay, numeric Stage1 schedules or cross-seed stability: no
- Reviewed at: 2026-08-07
