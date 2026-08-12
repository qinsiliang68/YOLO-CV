# P050 - Underspecification Presents Challenges for Credibility in Modern Machine Learning

## Identity

- Paper ID: P050
- Authors: Alexander D'Amour, Katherine Heller, Dan Moldovan, et al.
- Venue and year: Journal of Machine Learning Research, 23(226), 2022
- Official article page: https://www.jmlr.org/papers/v23/20-1335.html
- Official paper: `source_papers/Underspecification_JMLR_2022.pdf`, SHA256 `0BFCEEDEC5F0A29CA74466785D2F3785E52509E6CD683FFE33318820E9B17EAF`
- Official code: none linked from the JMLR article page; no paper-specific official implementation was identified.

## Reading And Audit Coverage

- Main paper and appendices: 61/61 pages read.
- Coverage includes the formal pipeline and validation-equivalence definitions; structural versus underspecified failure; stress-test taxonomy; the epidemiological, genomics, random-feature, computer-vision, medical-imaging, NLP, and EHR case studies; all 22 figures and eight tables; the marginalization-versus-selection appendix; all reported statistical tests; the random-feature derivation through Equation 54; discussion; qualifications; and limitations.
- Visual verification: all 61 pages inspected at original detail under `audit/visual_checks/P050_Underspecification_JMLR_2022/`.
- Text extraction: the complete PDF was extracted and searched for protocol, seed, sample-size, correlation, confidence-interval, stress-test, and limitation statements.
- Replication boundary: the article supplies extensive method detail, but there is no paper-specific official code artifact and several clinical datasets are restricted. This entry is therefore `FULL_READ_COMPLETE`, not an assertion of exact computational replication.

## Core Concept

The paper studies the entire training-and-validation pipeline, not only a model class or a loss landscape. Holding the nominal data source, architecture, and validation rule fixed can still leave choices such as initialization, data order, optimization details, and random seed unresolved. The paper defines the set of practically best validation-equivalent predictors as:

```text
F* = predictors obtainable from the pipeline
     with equivalently strong in-distribution validation performance.
```

A pipeline is underspecified when `F*` contains predictors that pass the same standard validation criterion but behave differently on application-relevant evaluations. This differs from structural mismatch:

```text
structural mismatch:
  the nominal learning problem rewards a behavior that conflicts with deployment

underspecification:
  the nominal criterion admits multiple near-equivalent solutions,
  only some of which exhibit the desired deployment behavior
```

For Stage1, the important translation is not that seed makes every result arbitrary. It is that a frozen static sample set can lead to multiple validation-near-equivalent optimizer paths whose weak-defect and difficult-normal behavior differs. The estimand is therefore conditional on pipeline state and seed, and application-specific tail tests are needed to distinguish these paths.

## Empirical Detection Protocol

The paper's cross-domain protocol has four parts:

1. Perturb a small nominally irrelevant pipeline degree of freedom, usually random seed, and train an ensemble.
2. Confirm near-equivalent iid validation performance so ensemble members plausibly belong to `F*`.
3. Evaluate application-specific stress tests and compare their variation with iid variation or external baselines.
4. Test whether stress-test behavior is predictable from iid performance and whether differences are systematic rather than sample noise.

The authors explicitly call this a conservative detector, not an exhaustive map of `F*`. Their three evidential checks are variation magnitude, unpredictability from iid performance, and systematic between-model behavior.

This maps cleanly to Stage1:

```text
iid-like/common endpoint:
  ordinary held-out classification summaries

application-specific stress tests:
  complete raw FN=0..95 frontier
  weak-defect tail preservation
  difficult-normal tail suppression
  FN at the fixed TN requirement
  per-video/source and defect-subtype slices
```

It does not justify replacing these endpoints with a single weighted score.

## Experimental Evidence

### Computer Vision

- The authors train 50 ResNet-50 models on ImageNet with pipelines differing only in seed.
- They fine-tune 30 BiT ResNet-101x3 models from the same JFT-300M checkpoint: ten each with zero, uniform, and Gaussian head initialization distributions.
- ImageNet top-1 performance is tightly concentrated: `75.9% +/- 0.11` for ResNet-50 and `86.2% +/- 0.09` for BiT.
- Some level-5 ImageNet-C corruption accuracies vary by an order of magnitude more than iid accuracy. All reported 95% Pearson-correlation intervals between iid and corruption performance contain zero.
- ObjectNet variability is two times the standard-test variability for ResNet-50 and five times for BiT. Spearman correlations with standard accuracy are `0.22 (-0.06, 0.47)` and `0.47 (0.13, 0.71)` respectively.
- One-sided permutation statistics for excess ObjectNet variability are `p=0.002` for ResNet-50 and reported as `p=0.000` for BiT, while corresponding standard ImageNet statistics are `p=0.203` and `p=0.474`.
- Pairwise ensemble disagreement is much larger on ObjectNet than on ImageNet or the class-matched ImageNet subset.

The appendix shows that averaging validation-equivalent models is not a universal repair. On the pixelation stress test, averaging all 50 ResNet-50 members never exceeds the best single stress-test member. This supports reporting seed distributions and worst-case behavior instead of allowing a mean to hide failing seeds.

### Medical Imaging

- Ten retinal models differ only in fine-tuning initialization. The held-out camera has `n=287`, versus `n=3,712` in the standard test set. A jackknife-based two-sample z-test comparing AUC standard deviations reports `z=2.47`, one-sided `p=0.007`.
- Two models with nearly identical calibration on camera types seen in training have qualitatively different calibration on the unseen camera type.
- Ten dermatology predictors differ only in fine-tuning-layer initialization. Skin-type slices II and IV contain `n=437` (10.7%) and `n=798` (19.6%). Permutation results are exploratory: types III and V are consistent with sample noise (`p=0.54`, `n=2,619`; `p=0.42`, `n=109`), type II is also largely consistent (`p=0.29`, `n=437`), while type IV is more systematic (`p=0.03`, `n=798`).

The distinction between systematic seed variation and small-slice sampling noise is directly relevant to weak-defect tails. Stage1 must preserve slice identity and uncertainty rather than interpreting every tail swing as optimizer-path evidence.

### Natural Language Processing

- Five independently pretrained BERT large-cased checkpoints use the same Wikipedia and BookCorpus data. Each is fine-tuned 20 times per downstream task, yielding 100 predictors.
- STS-B iid correlations range from 0.87 to 0.90; pronoun-resolution accuracy ranges from 0.960 to 0.965.
- Stress behavior varies much more. Correlation with occupational gender statistics ranges from 0.3 to 0.7 for STS and 0.26 to 0.51 for pronoun resolution.
- Stress behavior is weakly predicted by standard performance: STS Spearman `rho=0.21`, 95% CI approximately `(0.00, 0.39)`; pronoun resolution `rho=0.08`, 95% CI `(-0.13, 0.29)`.
- The paper uses the ratio of between-pretraining to within-pretraining variance as a descriptive F-statistic. It warns that the assumptions for a valid inferential F-test likely do not hold.
- For NLI, all 100 models lie in narrow MNLI ranges: 83.4%-84.4% matched and 83.8%-84.7% mismatched. HANS and other perturbation-test performance varies substantially, depends on both pretraining and fine-tuning seed, is weakly predicted from standard MNLI performance, and is often weakly correlated across distinct stress tests.

This is strong evidence against using one ordinary validation endpoint as a surrogate for all business-tail endpoints. It is not evidence that any particular Stage1 replay rule will solve the problem.

### Electronic Health Records

- The restricted dataset contains de-identified EHRs from 703,782 patients.
- The ensemble has five seeds for each of SRU, LSTM, and UGRNN cell types, for 15 predictors total.
- Standard normalized PRAUC is tightly constrained between 34.59 and 36.61.
- Time-shift and lab-content interventions produce wider performance dispersion. Two LSTMs differing only in seed flip mostly different patient-timepoint decisions.
- A preliminary timestamp-feature ablation obtains normalized PRAUC `0.368`, compared with `0.346` to `0.366` for the original ensemble, while an auxiliary head can still predict time of day at 85% accuracy from remaining features.

The last result is only preliminary, but it illustrates why a direct intervention is more informative than a post-hoc correlation: removing one cue can preserve ordinary performance while changing the mechanism, yet correlated proxies may remain.

### Genomics And Theory

- The genomics study constructs 1,000 feature sets, each containing one representative from each of 129 correlated variant clusters.
- It trains on 82,309 British identities, evaluates on 9,662 British identities, and uses 14,898 non-British identities only for evaluation.
- The same feature choices have largely uncorrelated British and non-British performance (`r=0.131`; the main text reports Spearman `rho=0.135`, 95% CI `0.070-0.20`).
- The random-feature analysis and simulations show that independently randomized predictors can have asymptotically equal iid risk yet different risk under a predictor-targeted mean shift. The displayed simulations average 50 realizations.

These examples formalize the paper's broader point, but their shifts and linear/random-feature assumptions are not Stage1 replay estimators.

## Statistical And Causal Boundaries

1. Seed variation is an observable signature of underconstraint, not a causal explanation of which training examples are valuable.
2. Near-equal ordinary validation scores do not imply equal tail behavior, but a tail difference can still be sampling error when the tail slice is small or dependent.
3. Many reported p-values are explicitly descriptive or exploratory. They should not be copied as universal thresholds.
4. The paper changes one seed while holding a nominal pipeline fixed. It notes that varying optimizer, batch size, learning rate, parameterization, and infrastructure would expose a larger equivalence class. Stage1 deliberately does not vary those factors in the formal campaign, because doing so would confound the replay intervention.
5. Stress tests are application-specific. ImageNet-C, ObjectNet, medical camera shifts, gender templates, and EHR interventions are examples of the method, not reusable Stage1 benchmarks.
6. The study detects underspecification but does not fully characterize `F*`, estimate the probability of a successful intervention, or prove that later replay is harmful.

## Direct Support For Stage1

1. Keep the old 240-run scientific hyperparameters fixed and hash-bound. This narrows the tested equivalence class so the causal factor is replay policy rather than hidden pipeline drift.
2. Treat training seed as a blocked experimental factor. All arms inside a seed block must share initialization, base-data order policy, augmentation configuration, and environment contract.
3. Preserve the exact Treatment selection across schedule arms when testing timing and dose. Otherwise selection and optimizer path remain entangled.
4. Save application-specific tail probes at every epoch needed by the mechanism analysis. One aggregate endpoint is insufficient because different stress endpoints can be weakly correlated.
5. Report the distribution over seeds: paired differences, success count, sign reversals, double-degradation rate, confidence interval, median, worst seed, and machine block. Never report only a grand mean.
6. Separate systematic path variation from finite-slice uncertainty with identity-level paired predictions, video/source grouped resampling, and repeated-seed effects.
7. Use progressive seven-day gates. Inspect early complete seed blocks before launching later stages, while preserving preregistered decision rules and keeping the blind holdout closed.

## What It Does Not Support

1. It does not identify high-value images or define a scalar sample score.
2. It does not justify `GapCritical`, confidence, gradient norm, or any static ranking as the final selection rule.
3. It does not justify replay ratios, the 140-160 decay window, guard fractions, or 14 seeds. Those require Stage1 evidence and power/design reasoning.
4. It does not prove that continuous replay is harmful, that decay is beneficial, or that weak-defect guard examples will prevent cross-seed reversal.
5. It does not permit changing architecture, epochs, batch, image size, workers, optimizer, learning-rate path, augmentation, AMP, or deterministic settings.
6. It does not justify selecting a final checkpoint after observing blind-holdout stress performance.
7. It does not turn seed dependence into an excuse for unstructured search. The correct response is blocked intervention, explicit stress endpoints, and uncertainty reporting.

## Canonical Hyperparameter Lock Consequence

The paper explicitly lists batch size, learning rate, optimizer choice, initialization, and infrastructure as degrees of freedom that can change the returned predictor. That strengthens the need for a machine-verifiable lock in Stage1:

```text
formal run is admissible only if
  canonical args.yaml hash matches
  resolved_training_args.json matches field-for-field
  training_execution_audit.json confirms the same execution contract
  the only allowed scientific differences are preregistered replay policy fields
```

OOM or throughput pressure cannot be resolved by silently changing batch size, workers, precision, or augmentation. The full seed block must move to a compatible machine or be rerun under the same lock.

## Transfer Boundary And Observable Consequence

The transferable claim is:

```text
same nominal pipeline + near-equal ordinary validation
does not guarantee the same application-specific behavior;
therefore evaluate the behavior directly across controlled seeds.
```

For the next Stage1 campaign, this means the primary scientific object is not a static `V(x)`. It is the conditional replay-policy effect:

```text
Delta(seed, checkpoint, endpoint)
  = endpoint(canonical base + frozen selection + replay policy)
    - endpoint(canonical matched control)
```

The experiment must estimate the distribution of `Delta` over unseen seeds and across separate tail endpoints. Replay timing, cumulative dose, and weak-defect protection remain hypotheses to test, not conclusions borrowed from this paper.

## Decision

- Reading status: FULL_READ_COMPLETE
- New formal arm: no
- New hyperparameter: no
- Canonical lock change: no; this paper strengthens the lock requirement
- Added fields: canonical-config hashes, exact seed lineage, seed-block identity, per-endpoint checkpoint trajectories, application-tail slice identity, grouped uncertainty, between-seed and within-slice variance, endpoint correlation matrix, sign reversal, double degradation, worst-seed outcome, and blind-holdout access state
- Remaining uncertainty: whether Stage1 seed reversals are primarily caused by replay timing, cumulative replay dose, weak-defect interference, representation drift, or interactions among them
