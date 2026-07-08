# Stage-1 OOF Gap-Value Sample Selection Experiment Design

This document is the handoff design for the next sample-value experiment. It is
written for a human who did not participate in the discussion and needs to know
what to run, what to measure, and what would count as success.

## 中文接手摘要

这轮实验不是继续证明 HN 和 RN 谁更好。120-run 已经说明：
单次高置信误报不能稳定定义高价值样本。下一步要验证的是：
训练集 OOF 动态能不能提前识别“能扩大 normal 高风险尾部和 defect
低置信尾部间隔”的样本。

接手者只需要记住四件事：

```text
1. 输入是 12 万训练侧样本的严格 10-fold OOF 预测动态。
2. 每折只允许用该折没参与训练的 checkpoint 去预测该折 holdout。
3. 输出是一张 sample_value_table.csv，再由它生成回流 manifest。
4. test 预测只能做最终评价，不能参与训练样本选择。
```

核心实验组是：

```text
confidence_fp
boundary_fp
persistent_fp
learnable_hard_fp
gap_critical
gap_critical_diverse
```

最终评价只看：

```text
TN_at_FN95
FN_at_TN68253
gap_q68_q050
selected_TN
selected_FN
```

如果 `gap_critical_diverse` 明显优于 `confidence_fp` 和 `HN-old`，
说明“高价值样本”不是单纯 hard，而是与目标 score gap 同向、持续、
且不冗余的样本。如果它没有提升，就说明当前 OOF 动态代理仍不足，
需要转向梯度对齐、影响函数或人工质量复核。

## 1. Five-Minute Summary

The 120-run HN/RN study showed that single-shot hard-negative selection is not a
stable definition of valuable samples. The strongest signal was not the HN/RN
label. It was score separation:

```text
corr(TN_at_FN95, gap_q68_q050) ~= 0.936
corr(TN_at_FN95, FN_at_TN68253) ~= -0.933
```

In plain terms, the best runs did two things at the same time:

```text
1. normal high-risk scores moved lower
2. defect low-confidence scores did not move lower, and ideally moved higher
```

The next experiment therefore asks:

```text
Can OOF training dynamics identify samples whose behavior is aligned with
better score separation before we use the final test set?
```

The core proposal is:

```text
Use strict 10-fold OOF predictions over 200 checkpoints to build one
per-sample value table, then use that table to create replay manifests.
```

The first candidate method to prioritize is:

```text
Epoch-GapCritical + Diversity
```

This means:

```text
Select normal samples whose OOF scores are high in bad epochs and low in good
epochs, then remove near-duplicate or over-concentrated samples.
```

## 2. Why This Experiment Exists

The previous replay experiments answered an important negative question:

```text
High-confidence false positive normal samples are not automatically high-value
training samples.
```

The failure mode is expected from first principles. A sample with high loss or
large gradient has influence, but that influence can point in a good direction
or a bad direction.

For this project, the target is not generic accuracy. The operational target is:

```text
FN <= 95 on the 120k test protocol
maximize TN / specificity
```

So a useful replay sample must help the operating curve. It should improve the
working separation between:

```text
normal high-risk tail
defect low-confidence tail
```

The test-set 120-run analysis can diagnose this phenomenon, but it must not be
used to select training samples. The selection policy must be built from OOF or
train-side evidence, then evaluated on the final test protocol.

## 3. Data And Evidence Sources

### 3.1 Previous 120-Run Diagnostic Evidence

Use the final 120-run HTML reports only as diagnostic evidence:

```text
C:\baidunetdiskdownload\stage1_phase1_hn_band_120runs\html_reports_20260708\
```

Important files:

```text
index.html
csv\run_probability_distribution_summary_120k_all_120_runs.csv
csv\top20_TN_at_FN95_120k.csv
csv\top20_gap_q68_q050_120k.csv
```

The diagnostic evidence says:

```text
best Pareto runs: RN1A-10 and RN1B-14
TN_at_FN95 is strongly explained by gap_q68_q050
coarse HN/RN labels are weaker than score-gap behavior
```

Do not use test predictions to choose replay samples.

### 3.2 Strict OOF Base

The OOF candidate pool is the existing 120k training-side OOF set:

```text
60,000 defect-side training samples
60,000 normal-side training samples
```

Existing best-checkpoint OOF evidence lives under:

```text
artifacts/stage1_oof_predictions_calop_20260621/merged_10fold_20260622/
```

The dynamic experiment should not stop at best checkpoint OOF. It should use all
200 checkpoints per fold.

### 3.3 Active Experiment Directory

All native outputs for this experiment must stay under:

```text
artifacts/stage1_sample_value_experiments/experiments/oof_dynamics_gap_value_20260708/
```

Do not create a parallel root. The fixed layout is:

```text
00_registry/
01_oof_dynamics/
02_sample_value_tables/
03_replay_manifests/
04_training_runs/
05_eval/
06_reports/
```

The existing technical runbook is:

```text
docs/stage1_sample_value_oof_dynamics_20260708.md
```

This design document explains the science and experiment logic. The runbook
explains the directory layout and current scripts.

## 4. Definitions

### 4.1 Working Threshold

The working threshold is the threshold chosen to keep defect misses within the
allowed budget.

On the final 120k test protocol:

```text
FN budget = 95
positive_n = 20,000
negative_n = 100,000
```

For OOF folds, use the equivalent fold-level budget:

```text
fold_FN_budget = floor_or_round(0.005 * fold_positive_n)
```

The exact rounding rule must be written into the value-table manifest.

### 4.2 Dangerous Normal

A dangerous normal is a normal sample whose OOF defect score is close to or
above the working threshold.

These are the normal samples that cause false positives or reduce TN at the
required recall.

### 4.3 Dangerous Defect

A dangerous defect is a defect sample whose OOF defect score is close to or
below the working threshold.

These are the defect samples that threaten the FN budget.

### 4.4 High-Value Normal

A high-value normal is not merely a high-confidence false positive. It is a
normal sample whose training evidence suggests:

```text
training on this kind of sample should lower normal high-risk scores
without lowering defect low-confidence scores
```

### 4.5 High-Value Defect

A high-value defect is a defect sample whose training evidence suggests:

```text
training on this kind of sample should raise defect low-confidence scores
without raising normal high-risk scores
```

### 4.6 Gap

The previous reports used:

```text
gap_q68_q050 = defect_q0.5% - normal_q68%
```

Human translation:

```text
defect low-confidence tail minus normal high-risk tail
```

Bigger is better. It means dangerous normal scores and dangerous defect scores
are better separated.

## 5. OOF Dynamics To Collect

For each fold `k`, predict only fold `k` holdout samples using fold `k`
checkpoints.

Do not predict all 120k samples with every fold model. That would not be strict
OOF and would leak train-seen behavior into the sample-value table.

Expected full scale:

```text
10 folds * 200 checkpoints * about 12k holdout samples ~= 24M prediction rows
```

Each epoch prediction row should contain:

```text
sample_id
oof_fold
human_fold
epoch
y_true
p_defect_raw
raw_logit
raw_margin_signed
raw_abs_margin
raw_cross_entropy
raw_uncertainty
raw_correct
Filename
canonical_image_relpath
image_path
checkpoint_path
checkpoint_sha256
```

After summarization, each sample should have:

```text
sample_id
y_true
oof_fold
epoch_count
p_defect_start
p_defect_end
p_defect_trend
mean_p_defect
std_p_defect
min_p_defect
max_p_defect
mean_loss
loss_auc_mean
max_loss
mean_margin_signed
mean_abs_margin
min_margin_signed
correct_rate
final_correct
first_learned_epoch
last_wrong_epoch
forgetting_count
dynamic_bucket
```

The summary is valid only if every sample has all expected epochs.

## 6. Epoch-Level Gap Labels

The key new idea is to label epochs as good or bad using OOF fold performance.

For each fold and epoch, compute:

```text
fold_TN_at_FN_budget
fold_FN_at_baseline_TN_rate
fold_gap
```

Then define:

```text
Good epochs = top 20% epochs by fold_TN_at_FN_budget or fold_gap
Bad epochs = bottom 20% epochs by the same metric
```

Use the same metric consistently inside one value-table build. The first build
should use:

```text
primary epoch quality metric = fold_TN_at_FN_budget
secondary diagnostic metric = fold_gap
```

The value-table manifest must record:

```text
good_epoch_percentile = top 20%
bad_epoch_percentile = bottom 20%
epoch_quality_metric
fold_FN_budget
```

## 7. Sample-Value Metrics

The output of this stage is one table:

```text
02_sample_value_tables/sample_value_table.csv
```

Each row is one sample. Each method gets a score column.

### 7.1 Confidence-FP

Purpose: old confidence-only baseline.

For normal samples only:

```text
confidence_fp_score = mean_p_defect
```

High score means the normal sample was often predicted as defect.

Expected role:

```text
negative control
```

If this wins, the old HN idea was probably sufficient. Current evidence says it
probably will not win reliably.

### 7.2 Boundary-FP

Purpose: test whether threshold relevance beats extreme confidence.

For normal samples:

```text
boundary_fp_score = -abs(mean_p_defect - fold_working_threshold)
```

High score means the normal sample sits near the operating boundary.

Expected role:

```text
threshold-aware control
```

### 7.3 Persistent-FP

Purpose: identify stable false-positive normal samples.

For normal samples:

```text
persistent_fp_score =
  fraction of epochs where p_defect >= epoch_working_threshold
```

High score means the sample is persistently on the wrong side of the operating
boundary.

Expected role:

```text
stability-aware control
```

### 7.4 Learnable-Hard-FP

Purpose: separate learnable hard samples from permanently strange samples.

For normal samples:

```text
learnable_hard_fp_score =
  early_wrong_rate
  + positive_improvement_trend
  - late_wrong_rate_penalty
```

Plain meaning:

```text
This normal sample was hard early, but the model learned to push it down later.
```

This is different from a persistent outlier. A persistent outlier may be noisy,
ambiguous, mislabeled, or harmful.

### 7.5 Epoch-GapCritical-FP

Purpose: primary proposed method.

For normal samples:

```text
gap_critical_score =
  mean_p_defect_bad_epochs(x) - mean_p_defect_good_epochs(x)
```

High score means:

```text
When the fold model is bad, this normal sample scores high.
When the fold model is good, this normal sample scores low.
```

This is a direct OOF proxy for:

```text
sample is associated with lowering the normal high-risk tail
```

This is not test leakage because good and bad epochs are defined inside OOF
fold dynamics, not from final test predictions.

### 7.6 GapGuard-D

Purpose: protect defect low-confidence tail.

For defect samples:

```text
gap_guard_score =
  mean_p_defect_good_epochs(x) - mean_p_defect_bad_epochs(x)
```

High score means:

```text
When the fold model is good, this defect sample scores higher.
When the fold model is bad, this defect sample scores lower.
```

This is the defect-side counterpart of GapCritical-FP.

Do not mix this into the first replay stage unless the defect budget is
explicitly chosen.

### 7.7 Diversity Filter

Purpose: avoid selecting many repeated video frames or the same visual pattern.

Apply after scoring, not before scoring.

Minimum first implementation:

```text
same video or same source stem limit
canonical path grouping limit
optional pHash or embedding cluster limit if available
```

If no visual embedding exists yet, use a conservative path-level and filename
proximity limiter first. Record the limitation clearly in the manifest.

## 8. Dynamic Buckets

Each sample should be assigned a diagnostic bucket. These buckets are not final
selection methods by themselves. They explain why a sample was selected or
rejected.

Recommended buckets:

```text
easy_stable
learnable_hard
persistent_boundary
persistent_wrong
unstable
suspected_noise_or_outlier
```

Suggested rules:

```text
easy_stable:
  correct_rate high, loss low, far from threshold

learnable_hard:
  early wrong or high loss, later correct, p_defect trend moves in correct direction

persistent_boundary:
  often near working threshold, moderate variability

persistent_wrong:
  wrong across many epochs, high loss, little improvement

unstable:
  high variability and multiple forgetting events

suspected_noise_or_outlier:
  extreme loss or extreme score, persistent wrong, and not helped by good epochs
```

For replay, prefer:

```text
learnable_hard
persistent_boundary
gap-critical normal with non-outlier behavior
```

Be cautious with:

```text
persistent_wrong
suspected_noise_or_outlier
```

These may have large gradients but harmful direction.

## 9. Replay Experiment Matrix

The experiment should run in two stages.

### Stage A: Method Screening

Use one budget first:

```text
normal replay budget = 3000
```

Methods:

```text
HN-old
confidence_fp
boundary_fp
persistent_fp
learnable_hard_fp
gap_critical
gap_critical_diverse
```

Purpose:

```text
Find whether OOF dynamics beats confidence-only selection.
```

Expected outcome if the theory is right:

```text
gap_critical_diverse > gap_critical >= persistent/boundary > confidence_fp ~= HN-old
```

Do not add defect guard in Stage A. Keep the first test clean.

### Stage B: Budget Validation

Keep only the strongest Stage A methods.

Budgets:

```text
600
3000
6000
```

Candidate methods:

```text
boundary_fp
persistent_fp
gap_critical
gap_critical_diverse
```

Purpose:

```text
Check whether the method scales with replay budget or only works at one point.
```

### Stage C: Defect Guard

Only after Stage A and B show that normal replay can improve the gap, add a
defect guard.

Initial policy:

```text
normal replay : defect guard = 5 : 1
```

Examples:

```text
3000 normal + 600 defect
6000 normal + 1200 defect
```

Methods:

```text
gap_critical_diverse
gap_critical_diverse + gap_guard_d
```

Purpose:

```text
Verify whether defect guard protects FN without destroying normal filtering.
```

## 10. Evaluation Metrics

Use the same final evaluation metrics as the 120-run reports:

```text
TN_at_FN95
FN_at_TN68253
gap_q68_q050
selected_TN
selected_FN
```

Primary success:

```text
TN_at_FN95 increases
FN_at_TN68253 decreases or does not increase materially
gap_q68_q050 increases
```

A method is probably only moving the threshold if:

```text
selected_TN changes
selected_FN worsens
gap_q68_q050 does not improve
TN_at_FN95 does not improve
```

That would not count as evidence of high-value sample discovery.

## 11. Leakage Rules

The following are allowed:

```text
Use final test to diagnose why previous HN/RN experiments behaved as they did.
Use OOF fold dynamics to build sample-value scores.
Use final test only once a replay policy has been fixed.
```

The following are not allowed:

```text
Use test predictions to select replay samples.
Tune sample-value weights directly on final test outcomes.
Pick samples because they were gap-critical on test.
Replace missing OOF folds with train-seen predictions.
Mix 40k/debug eval outputs into 120k evaluation.
```

## 12. Required Artifacts

Before replay training, the handoff package for this experiment should contain:

```text
01_oof_dynamics/summary/sample_dynamics_summary.csv
01_oof_dynamics/summary/summary_validation.json
02_sample_value_tables/sample_value_table.csv
02_sample_value_tables/sample_value_manifest.json
03_replay_manifests/<method>/budget_xxxxx/<run_id>/selection_manifest.csv
03_replay_manifests/<method>/budget_xxxxx/<run_id>/train_manifest.csv
03_replay_manifests/<method>/budget_xxxxx/<run_id>/normal_train_manifest.csv
```

The value table must include enough columns to audit each selected sample:

```text
sample_id
y_true
oof_fold
Filename
canonical_image_relpath
mean_p_defect
p_defect_trend
std_p_defect
mean_loss
loss_auc_mean
correct_rate
forgetting_count
dynamic_bucket
confidence_fp_score
boundary_fp_score
persistent_fp_score
learnable_hard_fp_score
gap_critical_score
gap_guard_score
diversity_group
diversity_selected_rank
```

## 13. Human Review Before Training

Before launching replay training, manually inspect samples from each selected
pool:

```text
top 50 gap_critical normal
top 50 gap_critical_diverse normal
top 50 persistent_wrong normal
top 50 gap_guard defect
random 50 easy_stable normal
```

For each group, check:

```text
duplicate frames
black or overexposed images
ambiguous labels
normal images visually similar to subtle defects
defect images with tiny or hard-to-see defects
obvious label mistakes
```

The purpose is not to hand-label the whole pool. The purpose is to know whether
the metric is selecting plausible learning signal or just artifacts.

## 14. Operational Run Order

Recommended sequence for the next operator:

```text
1. Confirm 10 fold checkpoint archives are present and complete.
2. Run dry-run checkpoint planning for all folds.
3. Export OOF checkpoint dynamics, split across machines by fold.
4. Summarize dynamics and validate fold/epoch completeness.
5. Build sample_value_table.csv.
6. Generate Stage A replay manifests for budget 3000.
7. Train Stage A replay runs with yolo11l and the existing formal pipeline.
8. Evaluate on the accepted 120k protocol.
9. Compare only the operational metrics in Section 10.
10. If Stage A succeeds, run Stage B budgets.
11. If Stage B succeeds, add defect guard in Stage C.
```

## 15. What Would Falsify The Idea

The OOF sample-value idea is weakened if:

```text
gap_critical_diverse does not beat confidence_fp or boundary_fp
selected samples are mostly duplicates or obvious artifacts
gap_q68_q050 does not improve after replay
TN_at_FN95 and FN_at_TN68253 stay on the same old trade-off curve
defect guard improves FN only by destroying TN
```

That would mean OOF dynamics is still not capturing the causal sample value we
need, and the next direction should shift toward gradient alignment or influence
estimation rather than more confidence or dynamics proxies.

## 16. What Would Support The Idea

The idea is supported if:

```text
gap_critical_diverse improves TN_at_FN95 over confidence_fp
gap_q68_q050 increases
FN_at_TN68253 is stable or lower
selected samples are visually diverse and plausible
budget response is monotonic or at least not erratic
```

The strongest result would be:

```text
gap_critical_diverse at 3000 or 6000 produces a new Pareto point against the
120-run reference set.
```

## 17. One-Sentence Handoff

This experiment tests whether strict OOF 200-epoch training dynamics can find
normal replay samples that are not merely high-confidence false positives, but
are associated with the model states that actually widen the operational score
gap.
