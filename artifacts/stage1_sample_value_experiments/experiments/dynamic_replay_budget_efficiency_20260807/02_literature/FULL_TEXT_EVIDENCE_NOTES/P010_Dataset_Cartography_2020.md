# P010 - Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics

## Identity

- Paper ID: P010
- Authors: Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, Yejin Choi
- Venue and year: EMNLP 2020
- Official landing page: https://aclanthology.org/2020.emnlp-main.746/
- Official full text: https://aclanthology.org/2020.emnlp-main.746.pdf
- Local PDF: `source_papers/Dataset_Cartography_2020.pdf`
- PDF SHA256: `E3D807F0E007FF03A22F6B80176F4BBC9F734E7804061C61792C6FF1B97C550F`
- Page count: 19, including appendices
- Code: https://github.com/allenai/cartography
- Code snapshot inspected: commit `3df3438fcc7324e706dee2e787426389bbd8fb1a`, dated 2022-07-08
- Relevant code blobs: `train_dy_filtering.py` = `02df08fdc4d8d7655962c4b9e5930314c5d76653`; `selection_utils.py` = `cc96d27fe0dbb795633b1c740215cf093452f4a2`; `run_glue.py` = `67442f1c8178f3474999e1c30a3a82bb61599dc7`

## Reading Coverage

- Full PDF: 19/19 pages read.
- Sections checked: definitions; data-map regions; subset experiments; easy/ambiguous mixture; synthetic-noise experiment; human label audit; uncertainty interpretation; all appendices.
- Experimental details checked: datasets, model family, optimizer/search range, epoch counts, batch sizes, seed counts, ID/OOD endpoints and reporting conventions.
- Public code checked: train-mode logit capture point, epoch-file schema, aggregation formulas, sorting directions, burn-out handling and filtered-data writer.
- Visual verification: pages 1, 3, 5, 6, 7, 8, 13 and 17 under `audit/visual_checks/P010_Dataset_Cartography/`.

## Research Question

Can per-example behavior over training separate easy-to-learn, ambiguous and hard-to-learn regions, and do those regions expose useful data-selection and label-quality information that a final confidence snapshot misses?

## Definitions

For sample `i`, gold label `y_i`, parameters at training observation `e`, and `E` observations, confidence is the mean gold-label probability:

```text
confidence_i = (1/E) * sum_e p_theta_e(y_i | x_i)
```

Variability is the population standard deviation of that trajectory:

```text
variability_i = sqrt((1/E) * sum_e (p_theta_e(y_i | x_i) - confidence_i)^2)
```

Correctness is the fraction of observations at which the predicted class equals the gold label. The resulting regions are descriptive coordinates for a particular learner and training procedure:

```text
easy-to-learn: high confidence, low variability
hard-to-learn: low confidence, low variability
ambiguous:     high variability
```

These are not intrinsic labels attached permanently to a sample. The appendix and architecture comparison show that coordinates and memberships can change with architecture and training state.

## Measurement Semantics From The Released Code

The paper describes epoch-wise training dynamics, but the released implementation makes the exact observation point important:

1. During each training mini-batch, the model is in `train()` mode.
2. The code stores that batch's logits before `loss.backward()` and before `optimizer.step()`.
3. Every sample is therefore observed at the state reached when its shuffled mini-batch appears, not at one common end-of-epoch checkpoint.
4. Dropout remains active, so the coordinate also includes stochastic forward-pass variation.

The original observable is best named `train_presentation_preupdate_probability`. It is not identical to `fixed_view_checkpoint_probability` or `oof_heldout_checkpoint_probability`. Stage1 must collect and report these namespaces separately.

## Experimental Contract

- Tasks: WinoGrande, SNLI, MultiNLI and QNLI, with separate ID and OOD evaluation sets.
- Learner: primarily RoBERTa-large; appendix comparison includes BERT-large.
- Optimization: AdamW; learning rate selected from a log-uniform range `5e-6` to `2e-5`.
- Training length: five epochs for SNLI/MultiNLI and six for WinoGrande/QNLI in the reported setup.
- Batch size: 96 except WinoGrande at 64.
- Hardware: one RTX 8000 is stated.
- Central subset results use three random seeds.
- WinoGrande and QNLI tables report average and standard deviation, while the SNLI/MultiNLI table reports the best of three seeds. This reporting asymmetry limits direct uncertainty comparisons.
- Full-data training is required to obtain the map before subset retraining, so the method is a diagnostic/selection pipeline rather than immediate compute saving.

## Main Results

- On WinoGrande, the 33% most ambiguous subset achieves 78.7 +/- 0.4 ID accuracy and 87.6 +/- 0.6 WSC OOD accuracy, versus 79.7 +/- 0.2 and 86.0 +/- 0.1 for all data, and 73.3 +/- 1.3 and 85.6 +/- 0.4 for a random 33% subset.
- The paper reports similar OOD benefits from ambiguous or hard-to-learn subsets on other tasks, but the magnitude and best region vary by dataset.
- The forgetting baseline underperforms random on WinoGrande, showing that one dynamic hardness statistic is not universally interchangeable with another.
- Pure ambiguous subsets below roughly 25% fail to optimize in the reported WinoGrande setup. Replacing one tenth of the 17% ambiguous subset with easy-to-learn examples restores optimization and improves ID performance; too much easy replacement reduces performance again.
- This mixture is direct set-interaction evidence: an easy sample can be valuable because it supplies optimization support to a difficult set, not because its individual loss or gradient is large.
- Artificially flipping 1% of labels moves corrupted samples toward lower confidence and sometimes higher variability.
- A linear detector trained on balanced synthetic flips reaches 100% F1 on a similarly constructed balanced test, but real-data human audit is imperfect: 67% of WinoGrande and 76% of SNLI examples predicted noisy were judged mislabeled or ambiguous, versus 13% and 4% among predicted-clean samples.
- A variability-only noise detector reaches 70% F1 in the synthetic setting, so variability alone is insufficient.
- Confidence tracks human agreement strongly. Once confidence is known, variability adds little information about human agreement in the paper's analysis.
- Training-dynamics and dropout-based measures correlate only moderately, approximately 0.45 for confidence and 0.39 for variability; they are related views, not equivalent measurements.
- The appendix reports pairwise Pearson correlation at least 0.75 for confidence and variability across five WinoGrande seeds. This supports coarse coordinate stability, not exact hard-tail rank stability or causal replay value.

## Ablations And Failure Cases

### Ambiguous-only selection can fail optimization

The strongest data are not sufficient as a homogeneous set at small budgets. Easy examples can be necessary as optimization scaffolding. This rejects `more ambiguous = always more valuable`.

### Hard-to-learn mixes task difficulty and label problems

Low confidence can identify real ambiguity or mislabeling, but the human audit leaves false positives and task dependence. The hard tail must be stratified, not replayed wholesale.

### Coordinates depend on the learner

BERT-large and RoBERTa-large yield similar global map shapes but move individual samples between regions. A map is conditional on model family, initialization, schedule and observation protocol.

### Seed evidence is aggregate

The reported correlation is for coordinates across seeds. The paper does not freeze exactly the same selected IDs and estimate the sign of additive-replay effects across seeds.

### Burn-out is task-specific

Early trajectories increasingly correlate with converged maps, but the appendix warns about initial optimization instability. It does not identify epoch 140, 150 or 160 as a universal control point.

### Selection and replay are different interventions

The experiments train from scratch on replacement subsets. Stage1 keeps all 120,000 base samples and adds repeated exposure. Subset accuracy does not identify a safe replay dose or timing policy.

## Code Audit And Reproduction Gaps

- `compute_train_dy_metrics` uses population standard deviation (`np.std`, `ddof=0`) over gold-label probabilities.
- The optional `include_ci` branch uses `sqrt(var + var^2/(E-1))`; this is an implementation-specific adjusted spread, not a general confidence interval around the sample mean.
- `threshold_closeness` is `mean_confidence * (1 - mean_confidence)`, which peaks at 0.5. It is not closeness to Stage1's recall-constrained operating threshold.
- Never-correct samples receive forgetfulness `1000`, collapsing all such samples into an artificial extreme sentinel.
- The default writer selects `head(num_samples + 1)` but reports `num_samples`, creating an off-by-one output. The `both_ends` path constructs the intended exact count.
- `read_training_dynamics` counts every file in the dynamics directory rather than only `dynamics_epoch_*.jsonl`; unrelated files can corrupt the inferred epoch count.
- The burn-out filename suffix is computed after the loaded epoch count has been restricted or capped, making the suffix condition effectively false and allowing alternate burn-outs to overwrite `td_metrics.jsonl`.
- Dynamics JSONL writes are not atomic and have no row-count, identity or content-hash validation.
- Released dependencies pin an obsolete stack, including PyTorch 1.4-era components and an old Transformers commit.
- Exact source-data hashes, environment hashes and seed-specific selection manifests are absent.

## What It Supports For Stage1

1. Use full trajectories rather than only final confidence.
2. Keep mean level, temporal variability and correctness/reversal fields separate.
3. Treat sample value as conditional on the current learner and training process.
4. Preserve easy/prototypical coverage when constructing a hard or ambiguous replay set.
5. Model set composition and interaction; an individually low-leverage sample can prevent optimization failure of a difficult set.
6. Treat low-confidence hard tails as a mixture of useful difficulty, ambiguity and noise risk.
7. Measure cross-seed coordinate and rank stability before promoting a candidate rule.
8. Capture every inexpensive epoch because burn-out accuracy is empirical and schedule-dependent.

## What It Does Not Support

1. It does not support `high variability = high replay value`.
2. It does not support a numeric Stage1 replay percentage, stop epoch or guard share.
3. It does not show improvement on `TN_at_FN95`, `FN_at_TN68253` or an `FN <= 95` raw safety frontier.
4. It does not study additive duplicate replay or cumulative exposure.
5. It does not establish fixed-selection stability across initialization seeds.
6. It does not show that one average validation objective protects a weak-defect tail.
7. It does not justify using test predictions for selection or fitting.
8. It does not make checkpoint OOF trajectories equivalent to in-training presentation dynamics.

## Transfer Boundary

Stage1 already has 10-fold, 200-epoch held-out OOF checkpoint probabilities. Those describe generalization trajectories under a fixed evaluation protocol. Cartography's public implementation records train-mode, pre-update presentations under changing within-epoch model states.

The final schema must preserve at least three observables:

```text
train_presentation_dynamic:
  pre-update prediction on the actual stochastic training presentation

fixed_view_train_probe:
  common checkpoint inference on a frozen canonical view of selected train probes

oof_heldout_epoch_dynamic:
  held-out fold prediction under each checkpoint
```

Agreement among them is a hypothesis to measure. No one field may silently substitute for another.

## Concrete Field Requirements

For presentation dynamics:

- `sample_id`, `epoch`, `global_step`, `batch_index`, `position_in_epoch` and `occurrence_in_epoch`;
- `is_replay`, replay rule, replay slot type and cumulative exposure before presentation;
- pre-update gold probability, prediction, correctness, loss, margin and entropy;
- augmentation identity/seed and dropout/RNG provenance where feasible;
- previous-presentation deltas and reversal indicators.

For fixed-view trajectories:

- epoch, checkpoint SHA, model-state identity and probe-view identity;
- gold probability, defect probability, margin, correctness and rank/quantile within class;
- trajectory mean, standard deviation, slope, curvature, reversal count and adverse-drop/recovery measures;
- per-seed coordinate/rank agreement, not only pooled averages.

For set composition:

- coverage in feature and gradient space;
- easy/prototype share, ambiguous share and hard/noise-risk share;
- video/fold/source concentration and duplicate exposure;
- pairwise and set-level gradient cancellation or complementarity.

## Concrete Experiment Consequence

- Do not add a Cartography-only Treatment arm to the large campaign.
- Use confidence, variability and correctness reversals as pre-specified strata for mechanism analysis.
- In a small pilot, compare a hard/ambiguous-only replay set with the same set in which a controlled share of slots is replaced by representative easy/prototype samples; keep total replay dose and timing identical.
- Separate sample-composition effects from replay-timing and cumulative-dose effects.
- Test map/rank stability across unseen seeds before using any trajectory region as a frozen selection rule.
- Preserve canonical Stage1 hyperparameters; only preregistered replay schedule, ratio and composition may change.

## Reproduction Notes And Missing Information

- The official PDF and code are sufficient to reconstruct the core coordinate computation and subset-writing pipeline after adapting dependencies.
- Exact paper-result reproduction still requires historical task preprocessing, old model artifacts, original random selections and environment details that are not fully content-hashed.
- `REPLICATION_DEPTH` records a full method/code/artifact audit; no NLP benchmark rerun was performed here.

## Stage1 Decision Status

- Evidence status: `REPLICATION_DEPTH`
- Claim eligibility: yes, with explicit learner/observation dependence and selection-versus-replay boundaries
- Direct support for full-trajectory and set-composition fields: yes
- Direct support for Cartography-only replay, numeric Stage1 parameters or checkpoint-OOF equivalence: no
- Reviewed at: 2026-08-07
