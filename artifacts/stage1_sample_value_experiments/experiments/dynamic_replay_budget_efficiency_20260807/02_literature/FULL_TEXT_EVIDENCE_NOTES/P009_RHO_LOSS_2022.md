# P009 - Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt

## Identity

- Paper ID: P009
- Short name: RHO-LOSS
- Authors: Soren Mindermann, Jan M. Brauner, Muhammed T. Razzak, Mrinank Sharma, Andreas Kirsch, Winnie Xu, Benedikt Holtgen, Aidan N. Gomez, Adrien Morisot, Sebastian Farquhar, Yarin Gal
- Venue and year: ICML 2022, PMLR 162, pages 15630-15649
- Primary landing page: https://proceedings.mlr.press/v162/mindermann22a.html
- Primary PDF: https://proceedings.mlr.press/v162/mindermann22a/mindermann22a.pdf
- Local PDF: `source_papers/RHO_LOSS_2022.pdf`
- PDF SHA256: `C8FA73E9256CB70ADE58D23B498B4164D601514850ADD790E1F63A1B3E46CBBE`
- Page count: 20
- Code: https://github.com/OATML/RHO-Loss

## Reading Coverage

- PDF pages read: 20/20, including references, ethics statement and Appendices A-G.
- Abstract and introduction: motivation, hard/easy failure modes, claimed speedups.
- Background: online batch selection and prebatch/top-k contract.
- Method: Equations 1-3, Bayesian derivation, three approximations, Algorithm 1, concurrent selection approximation.
- Experiments: seven datasets, all baselines, architectures, metrics and Tables 1-4.
- Robustness: architecture/hyperparameter transfer, no-extra-holdout construction, three noise patterns.
- Ablations: approximation rank correlation, IL-model size and update, selected percentage 5/10/15/20%, active-learning baselines.
- Limitations and ethics: selection bias, minority/rare-group ambiguity, BatchNorm interaction, speed measured in epochs rather than end-to-end time.
- Visual verification: PDF pages 3, 8, 18 and 19 rendered and inspected under `audit/visual_checks/P009_RHO_LOSS/`.

## Research Question

Given a large candidate prebatch, can training prioritize points that are simultaneously not yet learned, learnable, and relevant to unseen data, instead of treating high loss or high gradient norm as value?

## Method And Equations

The ideal objective selects a point that would minimize cross-entropy on a holdout set after training on that point (Eq. 1). Directly retraining on each candidate is infeasible, so the paper derives a tractable approximation.

The final score (Eq. 3) is:

```text
RHO_t(x, y) = L[y | x; D_t] - L[y | x; D_ho]
```

where:

- `L[y | x; D_t]` is the current target model's loss on the candidate;
- `L[y | x; D_ho]` is an irreducible-loss estimate from a separate model trained on holdout data.

The first term is dynamic. The second is computed once and frozen in the practical approximation. The paper interprets the difference as reducible holdout loss:

- already-learned points have low current loss and low score;
- noisy/ambiguous labels have high current loss but also high irreducible loss, lowering the score;
- input-space outliers or points less represented in the holdout distribution tend to have high irreducible loss and are deprioritized;
- learnable points still poorly fit by the current model have high current loss and low irreducible loss, giving high score.

Algorithm 1 uniformly draws a large prebatch `B_t`, evaluates both losses, selects the top `n_b` by RHO score, and takes one gradient step on the selected mini-batch. The default experimental ratio is `n_b/n_B = 0.1`.

## Assumptions And Approximations

The derivation makes three practical approximations:

1. replace Bayesian conditioning with neural-network training by SGD;
2. approximate a model trained on `D_ho union D_t` by a fixed model trained only on `D_ho`;
3. allow a much smaller and less accurate irreducible-loss model.

Table 1 reports Spearman rank correlation against the most faithful expensive approximation as the approximations accumulate: 0.75, 0.76, 0.63 and 0.51. These correlations are meaningful but far from identity; practical RHO-LOSS is not an exact estimator of leave-one-step-out holdout gain.

For selecting multiple samples, the paper takes the individual top scores and assumes one point has little effect on another point's score. This is an explicit weak-interaction approximation. It is particularly important for Stage1 because our same-selection cross-seed reversals suggest interactions cannot simply be ignored.

## Experimental Contract

Seven datasets are used: QMNIST, CIFAR-10, CIFAR-100, CINIC-10, Clothing-1M, CoLA and SST-2. Small clean datasets reserve half of training data for the irreducible-loss model; Clothing-1M reuses 10% of its large training pool. A two-model cross-fit construction is also tested so no extra holdout data is required.

Main models include an MLP on QMNIST, small-image ResNet-18 on CIFAR/CINIC, ImageNet-pretrained ResNet-50 on Clothing-1M, and pretrained ALBERT v2 on NLP. The Clothing-1M irreducible-loss model is a randomly initialized ResNet-18. A small CNN with 21x fewer parameters and 29x fewer forward-pass FLOPs is also tested as the IL model.

Vision defaults are AdamW with learning rate 0.001, betas 0.9/0.999 and weight decay 0.01. Selected batch size is 32, or 64 on CINIC-10; candidate prebatch is 320, or 640. NLP also selects 32 from 320. Experiments use 2-10 seeds depending on the comparison. CIFAR/CINIC augmentation is random crop and horizontal flip; frozen irreducible losses are computed on unaugmented images.

Baselines include uniform shuffling, training loss, gradient norm, importance-sampled gradient norm, Selection-via-Proxy, negative irreducible loss, and active-learning acquisition functions in Appendix G.

The primary metric is epochs or gradient steps required to reach target test accuracy, plus final test accuracy. It is not end-to-end wall time: candidate forward-pass and communication overhead are discussed theoretically but not implemented as the primary systems benchmark.

## Main Results

Table 2 reports RHO-LOSS best in required epochs and final accuracy across the listed tasks. Examples:

- Clothing-1M reaches 69% in 2 epochs versus 30 for uniform, ending at 72% versus 70%.
- Half-CIFAR-10 reaches 80% in 39 epochs versus 79; with 10% label noise, 75% is reached in 27 versus 62.
- Half-CIFAR-100 reaches 40% in 48 versus 65; with label noise, in 49 versus 79.
- CoLA reaches 75% in 3 epochs versus 34.

Figures 3 and 6 support the intended mechanism: high-loss and high-gradient-norm selection overselect corrupted labels, whereas RHO-LOSS selects fewer corrupted, less relevant and already-correct points. Results are usually averaged over 2-4 seeds in the main table, with 3 seeds for several robustness plots and 4 or more for NLP curves.

The fixed-IL approximation can outperform the theoretically more direct updated-IL version late in training. With 20% CIFAR-10 label noise, updating the IL model on adaptively selected data causes its accuracy to deteriorate and corrupted selection to increase; the fixed approximation achieves 88.6% versus 86.1% in the described comparison.

## Ablations And Failure Cases

- A small, inaccurate IL model can still rank useful samples and transfer across several target architectures and hyperparameter settings.
- RHO-LOSS does not accelerate CIFAR-10 training for VGG11, a setting where uniform training itself performs poorly.
- The selected fraction has dataset-dependent effects. Appendix F tests 5%, 10%, 15% and 20%; no single fraction is best everywhere.
- Active-learning acquisition functions help on MNIST but not CIFAR-10 when naively reused for online batch selection.
- Loss and gradient norm degrade under uniform label noise, structured class-confusion noise and ambiguous MNIST.
- BatchNorm statistics differ between the large candidate batch used for selection and the selected training batch, creating a nontrivial implementation interaction.
- The authors explicitly acknowledge selection bias. They also warn that rare/minority groups may either be prioritized because the majority is learned early or deprioritized because rare groups contribute less to average holdout loss.

## What It Supports For Stage1

1. Large current loss or gradient norm is an influence/risk signal, not sufficient evidence of positive value.
2. A useful candidate should be both currently underlearned and plausibly learnable from the surrounding distribution.
3. Static confidence can be improved by subtracting a cross-fitted reference estimate of irreducible difficulty/noise.
4. Value changes with the current target model because the current-loss term evolves during training.
5. Repeatedly updating a reference model on adaptively selected data can create feedback bias and late deterioration; frozen probes/reference terms are scientifically useful.
6. Dynamic exposure should naturally fall when selected points become learned, rather than replaying a fixed top-k at constant strength forever.
7. Noise, redundancy and relevance need separate fields; one hardness field cannot identify all three.

## What It Does Not Support

1. The paper removes 90% of each candidate prebatch from ordinary training; it does not add duplicate replay on top of a complete base epoch.
2. Its 5/10/15/20% ablation is the selected fraction inside a prebatch, not a replay percentage relative to the full Stage1 training pool. It cannot justify Stage1's candidate percentages.
3. It optimizes average holdout loss/accuracy, not a constrained `FN <= 95` safety frontier.
4. It does not test normal high-tail versus weak-defect low-tail protection.
5. It does not test an identical frozen selection over many target initialization seeds.
6. It does not estimate collection-level interaction beyond an assumption that per-point effects are nearly independent within one step.
7. Speedup in training epochs does not prove GPU wall-clock speedup after selection overhead.
8. The reported 2-10 seed experiments do not establish a paired operational success probability or worst-seed guarantee.

## Transfer Boundary

The closest safe transfer is a candidate feature, not a final value formula:

```text
learnable_residual_i(t) = current_loss_i(t) - cross_fitted_reference_loss_i
```

For Stage1 this must be class- and tail-aware. A reference model trained to minimize average loss may assign high irreducible loss to rare but safety-critical weak defects and wrongly deprioritize them. Therefore weak defects require a protected stratum or target-weighted reference objective.

The reference data must come from training/OOF partitions. Test data cannot train the IL model, choose its checkpoint, tune thresholds or rank replay samples.

## Concrete Field Requirements

For each replay candidate or fixed monitor at every epoch:

- current cross-entropy loss and raw logit;
- frozen OOF/reference loss and its provenance;
- reducible-loss difference and rank within the eligible class/stratum;
- whether it entered the candidate prebatch and whether it was replayed;
- cumulative exposure and time since previous replay;
- correctness, confidence and forgetting state;
- class, weak-defect/normal-tail membership and representation density/cluster;
- target-model seed, machine, epoch, schedule and immutable selection ID.

At key checkpoints, compare RHO-like scores with last-layer gradient alignment to the actual Stage1 tail objective. This tests whether learnability and target direction add distinct information.

## Concrete Experiment Consequence

- Add `current_loss - frozen_cross_fitted_reference_loss` as a diagnostic field and candidate-pool feature, not as a fixed weighted final score.
- Keep the first timing/dose experiment's selection frozen so schedule causality is not confounded by dynamic reranking.
- In a later transfer block, compare static ranking with dynamically refreshed current-loss-minus-reference ranking under the same exposure budget.
- Preserve a weak-defect guard or tail-stratified quota; do not let average reference relevance eliminate rare safety-critical defects.
- Include set-overlap/diversity and interaction diagnostics because top individual RHO scores do not establish batch value.
- Do not import 10% as a replay budget. Percentage levels must be calibrated to the Stage1 replay semantics and prior 240-run evidence.

## Reproduction Notes And Missing Information

- Algorithm, code URL, architectures, optimizer values, batch/prebatch sizes, augmentation, seeds and IL checkpoint rule are reported.
- The code repository is available and should be inspected before any direct implementation transfer.
- The paper states 2-10 seeds depending on experiment, while each table/figure specifies narrower counts; there is no single fixed seed count for the whole study.
- End-to-end distributed selection wall time, communication cost and a paired statistical confidence interval are not provided.
- Exact seed values and every dataset preprocessing detail are not all contained in the paper; code inspection would be required for bit-level reproduction.
- `REPLICATION_DEPTH` means sufficient detail for a faithful reproduction plan, not an independent rerun of the published benchmarks.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, within the transfer boundaries above
- Direct final-arm authority: no
- Final experiment decision: pending synthesis with replay-scheduling, gradient-direction, noise-dynamics and interaction papers
- Reviewed at: 2026-08-07
