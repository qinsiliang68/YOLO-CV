# P043 - Accounting for Variance in Machine Learning Benchmarks

## Identity

- Paper ID: P043
- Authors: Xavier Bouthillier, Pierre Delaunay, Mirko Bronzi, Assya Trofimov, Brennan Nichyporuk, Justin Szeto, Nazanin Mohammadi Sepahvand, Edward Raff, Kanika Madan, Vikram Voleti, Samira Ebrahimi Kahou, Vincent Michalski, Dmitriy Serdyuk, Tal Arbel, Chris Pal, Gael Varoquaux, and Pascal Vincent
- Venue and year: MLSys 2021, Proceedings of Machine Learning and Systems 3
- Official proceedings page: https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html
- Main paper: `source_papers/Accounting_Variance_MLSys_2021.pdf`, SHA256 `7BEFBC13CC306DAA9FA7DBFF71B2C3DA1ACA245A22C7EBC1C7DEF28593AE857F`
- Official experiment code: none located. The author's publication page displays a `Code` button with an empty link, and the official proceedings page exposes only paper and BibTeX links.

## Reading Coverage

- Main paper: 23/23 pages read, including Equations 1-12, Algorithms 1-2, Figures 1-6 and C.1/F.2/G.3/H.4/H.5/I.6, case-study settings, all statistical appendices, limitations, reproducibility notes, and references.
- Visual verification: all 23 pages inspected at original detail under `audit/visual_checks/P043_AccountingVariance_MLSys_2021/`.
- Source verification: official MLSys proceedings, arXiv identity, and the first author's publication page checked. No official supplement, peer-review bundle, or experiment-code repository was exposed by those sources.
- Code boundary: `bouthilx/ml-method-video` is only code for generating the conference videos; it is not the paper's experimental pipeline and was not treated as reproduction evidence.

## Research Question

The paper asks what distribution a machine-learning benchmark is actually measuring when data sampling, initialization, data order, augmentation, dropout, numerical effects, and hyperparameter optimization all vary. It distinguishes an ideal estimator that reruns hyperparameter optimization for every independent realization from the common cheaper estimator that freezes one tuning result and repeats only part of the training randomness.

This is directly relevant to the Stage1 cross-seed reversals. The 240-run result is not a deterministic property of a selection: it is one draw from a pipeline with seed, order, augmentation, initialization, machine, and replay-path variation. The paper does not identify valuable samples or replay schedules; it governs how those hypotheses must be compared.

## Formal Model

The learning procedure is represented as:

```text
Opt(S_train, lambda) ~= argmin_h empirical_loss(h, S_train) + regularization(h, lambda)
lambda_star(S_trainval) = HOpt(S_trainval)
h_star(S_trainval) = Opt(S_trainval, HOpt(S_trainval))
```

The expected pipeline risk integrates over sampled datasets and all random choices. With one finite dataset, the practical target becomes an expectation over train/validation/test resamples and learning-pipeline randomness.

For `k` independent full-pipeline realizations, the ideal estimator has:

```text
E[mu_hat_k] = mu
Var(mu_hat_k) = sigma^2 / k
MSE(mu_hat_k) = sigma^2 / k
compute = O(k * T)
```

where `T` is the hyperparameter-search budget. The common fixed-HOpt estimator costs `O(k + T)` but conditions on one arbitrary tuning realization. Its variance contains a residual correlation term:

```text
Var(mu_tilde_k | xi)
  = Var(R | xi) / k
  + ((k - 1) / k) * rho * Var(R | xi)
```

and its MSE additionally contains squared bias. If `rho` remains positive, increasing only the number of repeated seeds cannot drive the estimator variance to zero. Randomizing more non-HOpt sources can reduce correlation and move the practical estimator toward the ideal one, but cannot remove the bias from freezing HOpt.

For paired runs, the paper estimates the probability of one procedure outperforming another:

```text
P(A > B) = (1 / k) * sum_i I(metric_A_i > metric_B_i)
```

It recommends percentile-bootstrap uncertainty around this probability and separates statistical evidence from a practically meaningful probability threshold.

## Experimental Contract

- Five task/model cases: CIFAR-10/VGG11, PascalVOC/FCN-ResNet18, SST-2/BERT, RTE/BERT, and peptide-MHC prediction with a shallow MLP.
- Individual variance sources were generally evaluated with 200 repeated trainings while other sources were held fixed.
- Hyperparameter-search variance used random search, noisy grid search, and Bayesian optimization; the study ran 20 independent procedures with budgets up to 200 trials and reports 320 optimization executions across settings.
- The ideal-versus-fixed-HOpt estimator study evaluates `k=1..100`; fixed-HOpt variance was estimated across 20 arbitrary conditioning seeds.
- Total compute is reported as approximately eight GPU years for the broader study and 6.4 GPU years for the estimator comparison.
- Reproducibility checks included five repeats per seed, forced interruption after each epoch, resumption, and iteration through seeds. The authors checkpointed RNG state and required deterministic cuDNN with benchmarking disabled for convolutional models.
- Different GPU models, CUDA-driver changes, and PyTorch versions were observed to change results. Those effects were not quantified as a formal variance component; the authors instead fixed architecture, CUDA 10.2, and PyTorch 1.2.0.

Every numeric training setting in the paper is evidence context only. None may alter the Stage1 canonical hyperparameter lock.

## Main Results And Negative Evidence

1. Data bootstrap was usually the largest measured source of variance. Initialization was generally below half of bootstrap variance and comparable to SGD data-order variance.
2. Variance components were explicitly not independent, so their bars cannot be summed to recover total variance.
3. Hyperparameter choice produced variance of similar order to initialization after tuning. Freezing one HOpt result therefore narrows the scientific claim to that tuned pipeline.
4. Randomizing only initialization improved the practical estimator little. Randomizing data splits helped all cases; randomizing all measured non-HOpt sources was the best practical estimator after the ideal estimator.
5. `IdealEst(k=100)` required about 1,070 hours versus 21 hours for `FixedHOptEst(k=100)`, yielding the headline 51-fold compute difference. This is a comparison of estimator procedures, not a promised 51-fold saving for Stage1.
6. In the authors' simulations, single-point comparison had roughly 10% false positives and 75% false negatives. Average comparison at `k=50` was conservative with below 5% false positives but roughly 90% false negatives. Probability-of-outperformance testing had roughly 5% false positives and 30% false negatives in the reported regimes.
7. The paper's `gamma=0.75` threshold was selected empirically across five case studies. Its power calculation gives 29 runs for alpha and beta both 0.05 under that setup. The appendix warns percentile bootstrap is not uniformly reliable and frames the threshold as a community choice.
8. One segmentation pipeline remained nondeterministic even with all seeds fixed. Hardware, drivers, framework versions, global RNG coupling, and resume state all caused practical reproducibility problems.
9. The framework assumes controllable training procedures and primarily i.i.d. sampling. It does not cover opaque fixed models, many-way comparisons without correction, dependent video identities, or a constrained low-FN safety frontier.

## Direct Support For Stage1

1. Compare replay policies as distributions over paired seed blocks, not as isolated best runs or only mean metrics.
2. Reuse the same initialization, base-data order, augmentation stream, and evaluation identities within each treatment/control block whenever semantically valid; this removes nuisance variation from the paired difference.
3. Still repeat blocks over unseen seeds. Pairing reduces variance but one fixed seed does not estimate robustness.
4. Balance and record machine, GPU model, driver, CUDA, PyTorch, Ultralytics, data identity, and code identity. Machine cannot be silently confounded with an arm.
5. Report paired deltas, empirical success probability, uncertainty, worst-seed behavior, and full endpoint distributions. A mean improvement alone cannot establish a stable method.
6. Preserve RNG and sampler state across resume and test interrupted-versus-uninterrupted equivalence. A successful process restart is not enough if it changes the optimizer path.
7. Keep selection-sampling variation, training-seed variation, machine variation, and evaluation-sample uncertainty as distinct fields rather than collapsing them into one standard deviation.
8. The fixed canonical Stage1 hyperparameters define a conditional treatment-effect estimand. Their source and hash must be explicit so the claim is not misrepresented as averaging over hyperparameter optimization.

## What It Does Not Support

1. It does not support changing any Stage1 optimizer, learning-rate, batch, worker, augmentation, model, image-size, epoch, or precision setting.
2. It does not show that replay decay, a weak-defect guard, gradient alignment, or any sample ranking improves Stage1.
3. It does not justify copying `gamma=0.75`, 29 runs, 50 splits, bootstrap resampling, or the paper's task-specific training settings into the preregistration.
4. It does not prove that randomizing data splits is appropriate for the first Stage1 confirmation block. Stage1 has a fixed OOF/val_op protocol and dependent video frames; changing splits would change the estimand and could break comparability.
5. It does not provide an official executable pipeline with which the reported 6.4-8 GPU-year study can be reproduced.
6. Its probability-of-outperformance endpoint is scalar. Stage1 requires a joint safety outcome over `delta_TN`, `delta_FN`, and the raw `FN=0..95` frontier, with R1 and R2 reported separately.

## Transfer Boundary And Observable Consequence

Stage1 should use a paired blocked design under an immutable canonical lock:

```text
Within one unseen seed block:
- identical canonical hyperparameters and initial-weight identity;
- identical base samples, base order, augmentation/RNG contract, and epochs;
- balanced machine assignment and randomized arm order;
- only the preregistered replay intervention changes.

Across blocks:
- use new seeds and balance machines;
- retain every valid and failed attempt with provenance;
- estimate paired deltas, joint success probability, downside, and uncertainty;
- do not promote a rule from the best run.
```

The current confirmation claim must be phrased as conditional on the frozen canonical pipeline and fixed Stage1 data protocol. A later study could randomize data splits or tuning procedures to estimate a broader pipeline effect, but doing so now would spend scarce GPU time on a different question.

If pairing materially shrinks within-block variance while continuous-versus-decay signs still reverse across unseen seeds, the remaining heterogeneity is evidence for state-conditional replay effects rather than mere machine or initialization imbalance. If arm differences disappear after proper pairing and machine balance, earlier apparent selection effects were dominated by nuisance variation.

## Decision

- Reading status: FULL_READ_COMPLETE
- New formal arm: no
- New hyperparameter: no
- Canonical lock change: no
- Design changes supported: strict paired seed blocks, machine blocking, randomized arm order, exact environment/RNG/resume provenance, unseen-seed replication, paired success probability and downside reporting
- Remaining uncertainty: how much of the observed Stage1 cross-seed reversal is attributable to seed-state interaction versus data-order, augmentation, machine, and replay-sampler differences
