# P026 - When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee

## Identity

- Paper ID: P026
- Authors: Dixian Zhu, Gang Li, Bokun Wang, Xiaodong Wu and Tianbao Yang
- Venue and year: ICML 2022, PMLR 162:27548-27573
- Published page: https://proceedings.mlr.press/v162/zhu22g.html
- Published PDF: `source_papers/When_AUC_Meets_DRO_ICML_2022.pdf`, SHA256 `CE0C5E4494774DE414D5844CC69F05CBF351961A11DD83460B357744A93D8B40`
- Author preprint: https://arxiv.org/abs/2203.00176
- Official implementation family: https://github.com/Optimization-AI/LibAUC
- Audited code snapshot: LibAUC `v1.4.0`, commit `d542e131ac82d494ffec8642c0d047c65d30cbbd`, dated 2024-08-09

## Reading Coverage

- Main manuscript: 11/11 pages read, including definitions, OPAUC and TPAUC estimators, CVaR/KL-DRO formulations, Algorithms 1-3, convergence statements, all datasets, training settings, comparisons, tables and conclusions.
- Supplement: 15/15 pages read, including dataset statistics, all convergence plots, all gamma ablations, complete tables, Lemmas 4-13, proofs of Theorems 1-6 and the exact-CVaR TPAUC Algorithm 4.
- Visual verification: all 26 pages and seven contact sheets under `audit/visual_checks/P026_When_AUC_Meets_DRO_ICML_2022/` inspected.
- Code inspection: official LibAUC v1.4.0 implementations of `pAUC_DRO_Loss`, `tpAUC_KL_Loss`, `SOPAs` and `SOTAs`, plus the bundled SOPAs CIFAR-10 script, were inspected. This is a post-publication library tag, not an identified ICML 2022 reproduction snapshot.
- Execution: the official experiment was not rerun. The tagged example has constructor and forward-call mismatches against the tagged loss API, and the paper does not publish an exact run count or locked environment.

## Research Question

The paper asks how to optimize only the operationally important portion of the ROC curve with scalable stochastic deep-learning algorithms. With positives `S+`, negatives `S-` and score `h`, one-way pAUC restricts attention to high-scoring negatives. Two-way pAUC also restricts attention to low-scoring positives.

For Stage1, let defect be positive and normal be negative. The useful conceptual object is therefore not an isolated image score. It is the set of pairwise ordering violations between difficult normals and weak defects:

```text
violation(defect_i, normal_j) = I[p_defect(defect_i) <= p_defect(normal_j)].
```

A normal may have high value or high harm depending on which defect identities it interacts with and at which model state. This directly supports conditional set value, not a static `V(x)`.

## Core Definitions

For FPR restricted to `(0, beta)`, the empirical OPAUC estimator compares every positive against the top `n_minus * beta` negatives. With a decreasing pairwise surrogate `L(w; x_i, x_j)`, the exact objective is

```text
mean over positives i of
mean over top-beta negatives j of L(w; x_i, x_j).
```

For TPAUC with the paper's parameters `(alpha, beta)`, the estimator compares the bottom-ranked `n_plus * alpha` positives against the top-ranked `n_minus * beta` negatives. These paper parameters do not equal Stage1's `FN<=95` definition and must not be copied numerically.

The CVaR DRO identity converts a top-fraction mean into

```text
min_s  s + (1 / beta) * E_negative[(L - s)_+].
```

This is exact for the paper's OPAUC surrogate when the stated monotonicity and integer-rank conditions hold. The auxiliary threshold is positive-specific: each positive example has its own tail-loss threshold.

The KL-DRO alternative is

```text
lambda * log E_negative[exp(L / lambda)].
```

It is smooth but approximate. As `lambda -> 0`, it approaches the single worst negative pair; as `lambda -> infinity`, it approaches ordinary full AUC. Thus it supplies a continuum between extreme-pair sensitivity and average-pair behavior, not a calibrated mapping from `lambda` to a requested FPR.

The TPAUC soft objective applies a second DRO layer across positives, producing a three-level compositional objective. This is a mathematical expression of context: the same normal's contribution is reweighted by both its pairwise loss against each defect and the current aggregate difficulty state of that defect.

## Optimization And Dynamic State

- SOPA uses hard CVaR pair indicators and maintains one threshold `s_i` per positive identity.
- SOPA-s uses soft exponential pair weights and an exponential moving estimate `u_i` per positive identity.
- SOTA-s adds another moving aggregate across positives and gradient momentum.
- Exact CVaR TPAUC is formulated as a weakly-convex-concave problem and solved with a stagewise stochastic primal-dual algorithm.

The state variables matter. They change after every presentation and depend on which positive and negative examples co-occur. A static CSV score cannot reconstruct this path unless it also records identity, batch context, exposure order and moving-state history.

The paper proves convergence to nearly stationary solutions under boundedness, smoothness and Lipschitz assumptions. Exact CVaR OPAUC is weakly convex and non-smooth; the KL-DRO version is smooth under its assumptions. These are optimization guarantees, not guarantees of global optimality, cross-seed stability, threshold safety or Stage1 replay benefit.

## Experimental Protocol

- Datasets: CIFAR-10, CIFAR-100, Melanoma, ogbg-moltox21, ogbg-molmuv and ogbg-molpcba.
- Models: ResNet18 for images; five-layer GIN with 64 hidden units and dropout 0.5 for molecular data.
- Image imbalance: the first half of CIFAR classes are negative, the last half positive, then 80% of positive training examples are removed.
- Training: CE pretraining with Adam, classifier reinitialization, then all-layer fine-tuning for 60 epochs; batch size 64; weight decay `2e-4`; learning rate drops tenfold every 20 epochs.
- Hyperparameter selection: learning rate and method-specific parameters are tuned on training performance for convergence plots and validation performance for test comparisons.
- OPAUC regions: FPR upper bounds 0.3 and 0.5. TPAUC regions: `(TPR>=0.6,FPR<=0.4)` and `(TPR>=0.5,FPR<=0.5)`.
- Repetitions: the paper says multiple train/validation splits and random seeds and reports means and standard deviations, but does not state the exact number of runs.

Every numeric setting above is literature context only. None is permission to change the Stage1 `yolo11l` canonical hyperparameters.

## Positive And Negative Evidence

1. The proposed methods are often strongest, especially on the most imbalanced Melanoma and molmuv tasks, and the convergence curves generally favor the compositional algorithms over naive mini-batch selection.
2. The results are not universal. On image OPAUC, exact SOPA often beats smooth SOPA-s. On moltox21 OPAUC at FPR 0.3, the mini-batch baseline exceeds both SOPA variants. Several molecular TPAUC entries have large standard deviations.
3. The smooth KL estimator is compared with exact CVaR only on one molecular task using 100 randomly generated model parameter vectors. The experiment shows that some tuned `lambda` can approximate a chosen FPR, not that one `lambda` transfers across datasets, epochs or Stage1's much more extreme operating region.
4. The gamma ablation shows that the fixed main-paper value 0.9 is often not best. Performance surfaces are dataset- and metric-dependent, and rare molecular tasks remain highly variable.
5. The paper's FPR ranges 0.3/0.5 and TPR ranges 0.5/0.6 are far from Stage1's safety requirement. Tail estimators become statistically and computationally more fragile at much smaller tail fractions.
6. No experiment isolates replay timing, cumulative duplicate exposure, a weak-defect guard, no-replay, identical-selection seed reversals or raw `FN=0-95` frontier behavior.

## Code Audit

1. LibAUC v1.4.0 requires a stable sample index and allocates `u_pos` state by `data_len`. This confirms that per-example dynamic state and identity are part of the method.
2. `u_pos` and the TPAUC aggregate `w` are plain tensors, not registered module buffers or parameters. A normal `state_dict()` checkpoint does not preserve them. Resume without explicit loss-state serialization changes the algorithm path.
3. The tagged SOPAs example calls `pAUC_DRO_Loss(pos_len=...)`, while the tagged constructor requires `data_len`. It also calls `loss_fn(..., index_p=index)`, while `forward` requires `index`. The example cannot run against that tagged API without edits.
4. `pAUC_DRO_Loss` asserts a positive is present but does not symmetrically assert a negative. Correct batch composition is delegated to `DualSampler`.
5. No repository test suite covering the four audited pAUC loss/optimizer classes was found in the snapshot.
6. The tag is dated 2024-08-09, about two years after ICML 2022. It documents the implementation family but cannot establish the exact paper code, dependency versions or random-seed protocol.

## Direct Support For Stage1

1. Represent tail behavior as a bipartite defect-normal relation, not only as two marginal quantiles or one scalar per image.
2. At every key checkpoint, freeze a defect-tail probe and normal-tail probe and compute pairwise margin distributions, violation counts and concentration by identity, video and cluster.
3. For each replay normal, record how many weak-defect probes it outranks, its pairwise surrogate mass and whether these quantities rise or fall after replay exposure.
4. For each weak defect, record how many replay normals outrank it and how concentrated those violations are in a few normal patterns.
5. At selected checkpoints, compute separate gradients for normal-tail correction and weak-defect protection, plus per-candidate alignment with both. Pairwise relevance and gradient direction remain separate channels.
6. Preserve stable global sample IDs, batch composition, presentation count, schedule position, augmentation identity and all dynamic collector state across resume.
7. Report hard top-tail and smooth log-sum-exp summaries together. Their disagreement is a useful extreme-outlier sensitivity diagnostic; neither becomes an unvalidated value formula.
8. Keep all formal training arms under the exact canonical 240-run hyperparameter lock. These diagnostics can be computed from raw predictions and key-checkpoint gradient probes without replacing the loss.

## What It Does Not Support

1. Replacing canonical cross-entropy training with SOPA, SOPA-s, SOTA-s or any pAUC loss in the first confirmatory campaign.
2. Importing batch 64, 60 epochs, Adam, paper learning rates, weight decay, gamma, lambda, alpha or beta.
3. Calling a high pairwise-loss normal intrinsically valuable. It may be a repeated artifact, mislabeled, out-of-distribution or harmful to weak defects.
4. Treating OPAUC/TPAUC improvement as proof of lower `FN_at_TN68253`, higher `TN_at_FN95` or raw-frontier dominance.
5. Treating convergence to a stationary point as a cross-seed or safety guarantee.
6. Choosing a Stage1 tail fraction from the paper's moderate operating ranges.

## Stage1 Field Contract

Under the hash-locked canonical configuration, add the following diagnostic fields at 120, 140, 150, 160, 180 and 200; low-cost summaries may be emitted every epoch:

- immutable defect-tail and normal-tail probe manifest hashes and exact identity counts;
- `pair_margin = p_defect(defect) - p_defect(normal)` quantiles and signed mean;
- hard violation count/rate, number of violating partners per identity and maximum identity concentration;
- hard top-tail CVaR-style pair loss at preregistered Stage1 fractions;
- smooth log-sum-exp pair loss at a small diagnostic grid of temperatures, clearly marked as sensitivity analysis;
- replay-normal to weak-defect pair matrix in sparse/top-k form for the fixed probe subset;
- per-identity presentation count, last-seen epoch, batch co-occurrence digest and augmentation-view digest;
- normal-correction gradient norm/alignment, defect-protection gradient norm/alignment and their sign conflict;
- raw-gradient versus optimizer-update alignment and cross-checkpoint direction consistency;
- collector moving-state hash, schema version, completion mask and resume lineage.

The full pair matrix over all 120,000 images is unnecessary and wasteful. Use fixed, preregistered tail probes plus stratified random clean controls, store aggregate all-epoch summaries, and retain sparse identity-level detail at key checkpoints.

## Concrete Experiment Consequence

P026 adds no formal training arm and changes no hyperparameter. It adds a mechanism analysis nested inside the planned causal timing experiment:

```text
same selected normal IDs
same canonical seed and hyperparameters
continuous vs same-peak decay vs dose-matched relocation vs no replay
    -> compare pairwise weak-defect violations over time
    -> compare normal-correction and defect-protection gradient directions
    -> test whether late exposure changes interaction sign
```

If decay improves the raw frontier while reducing late pair violations against weak defects at matched cumulative dose, the timing mechanism gains support. If pairwise diagnostics change without paired outcome improvement, they remain descriptive. If outcome improves without these diagnostics changing, this mechanism is falsified and another path variable must explain the effect.

## Stage1 Decision Status

- Evidence status: `FULL_READ_COMPLETE`
- Claim eligibility: yes, for pairwise tail-risk structure, CVaR/KL-DRO distinctions and dynamic per-identity state
- Replication-depth eligibility: no; the exact paper environment and run count are unavailable, and the audited official tag is post-publication and API-inconsistent
- Direct support for static replay ranking: no
- Direct support for dynamic replay timing: indirect only; the paper optimizes tail pairs but does not manipulate replay timing
- Direct support for a new formal arm: no
- Direct support for additional mechanism fields: yes
- Permission to change canonical Stage1 hyperparameters: no
- Reviewed at: 2026-08-07
