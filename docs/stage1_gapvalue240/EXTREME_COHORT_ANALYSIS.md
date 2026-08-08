# Stage1 GapValue 240-Run 极端条件集分析

## 1. 目的

本分析不是继续按方法平均值寻找“唯一最佳排序”，也不是只挑一两个漂亮 run 讲故事。它以 80 个完整 `T/R1/R2` triad 为总体，先按冻结的业务结果把实验划分为若干表现档位，再比较好组、次好组、混合组和明确有害组在以下方面的共同规律：

- 选中样本的 OOF 训练动态与组成；
- 训练早、中、后期的损失和准确率分叉；
- normal 尾部、weak-defect 尾部、工作阈值和 AUROC 的变化；
- 固定 selection 在不同 training seed 下是否发生结果翻转；
- 规律是否同时面对 R1、R2 成立，是否受到机器、阶段或高重合对照的限制。

本分析是现有 `gapvalue240_pattern_analysis_20260728_v2` 的只读二阶段分析。它不修改 240 份 selection CSV，不重新训练，不重写 v2 报告，也不在源结果目录内生成文件。

## 2. 冻结的结果分档

对每个 triad 分别计算 Treatment 相对 R1、R2 的差值：

```text
delta_TN = TN_at_FN95(T) - TN_at_FN95(control)       越大越好
delta_FN = FN_at_TN68253(T) - FN_at_TN68253(control) 越小越好
```

R1 和 R2 始终分别判定，不能合并、平均或任选其一。五档定义如下：

| 档位 | 机器判定规则 | 解释 |
| --- | --- | --- |
| S | 对 R1、R2 均满足 `delta_TN >= 300` 且 `delta_FN <= 0` | 强正向且高性价比 |
| A | 对 R1、R2 均满足 `delta_TN > 0` 且 `delta_FN <= 0`，但至少一个对照的 `delta_TN < 300` | 安全改善，但 TN 増益较小 |
| B | 对 R1、R2 均满足 `delta_TN >= 300` 且 `delta_FN <= 2`，但不满足 S | TN 明显改善，允许最多多漏 2 张 |
| H | 对 R1、R2 均满足 `delta_TN < 0` 且 `delta_FN > 0` | 明确有害 |
| M | 不属于 S/A/B/H 的其余结果 | 混合、方向不一致或只赢一个对照 |

当前冻结回归计数必须为：

```text
S = 12
A = 3
B = 1
M = 41
H = 23
总计 = 80 triads
```

这些档位是结果后的研究分组，不是新的选样标签，也不能反向写入训练输入。

## 3. 统计单位

不同问题使用不同统计单位，禁止把大量相关行误当成独立样本：

- 业务结果和训练曲线对照：一个 `T-control` 配对；每个 triad 产生一条 T−R1 和一条 T−R2 比较。
- 档位判定：一个完整 triad；R1、R2 必须同时满足相应规则。
- seed 稳定性：同一个冻结 treatment selection 在不同 training seed 下形成的多个 triad。
- 选样组成：一个唯一 treatment selection digest；同一 selection 被多个 seed 复用时不能重复计权为多个独立样本集。
- OOF 动态：单个训练样本跨 200 epoch 的相关轨迹；200 个 epoch 不是 200 个独立重复实验。
- 条件级规律：以 condition/selection 为单位汇总，同时保留逐 seed 结果，不用图片行数虚增统计功效。

同一 selection 跨 seed 可能从好档翻转到坏档。具体 selection、档位组成和翻转数量以输出的 `selection_set_outcomes.csv` 与 `fixed_selection_seed_flips.csv` 为准，不能把单一 seed 的成功解释为该 selection 的固定价值。

## 4. 分析内容

### 4.1 极端结果对照

主对照为 S 对 H，并用 A、B 观察较弱的正向模式，用 M 检查规律是否只存在于极端样本中。分析同时保留逐 triad 原始值、T−R1、T−R2 和同机/跨机标记。

### 4.2 训练阶段特征

训练曲线按以下窗口冻结：

```text
epoch 1-40
epoch 41-120
epoch 121-160
epoch 161-200
```

后期训练损失使用三种互相校验的定义：

- 端点变化：`delta_train_loss(epoch121) - delta_train_loss(epoch200)`；
- 稳健窗口变化：epoch 121-130 均值减 epoch 191-200 均值；
- epoch 121-200 的 OLS slope。

任何“后期继续压低训练损失是否有害”的判断都必须同时查看 R1、R2、逐 seed 和不同定义，不能只依赖某一个端点。

### 4.3 OOF 工作阈值动态

OOF 动态由冻结的 `200 x 120000` float64 概率矩阵重新计算。业务阈值规则为：

```text
score >= threshold 时预测为 defect
```

每个 epoch 使用 `FN <= 285` 的全 OOF 比例约束计算工作点，并按完整同分组处理 threshold。当前数据的回归锚点为最佳 `TN_at_FN285` 出现在 epoch 149；实际结果必须由程序重算并验收，不能从文档硬编码为分析输入。

fold_01 的逻辑 epoch 178 来自已记录的 checkpoint 修复，与 epoch 177 重复。该来源必须进入 provenance 和影响样本审计，不能把这两个单元视为两份独立证据。

### 4.4 选样组成与结果机制

针对唯一 treatment selection，比较：

- dynamic bucket、fold 和候选池组成；当前冻结 selection 未提供可稳定连接的 group 字段，因此本报告不猜测 group；
- OOF 工作阈值下的错误率、学会时间、遗忘、后期持续误报和方向变化；
- treatment 结果中的工作阈值、AUROC、normal/defect 尾部与训练曲线变化；
- 同一 selection 跨 seed 的档位分布和方向翻转；
- R2 overlap、Jaccard 和有效独特对比比例。

工作阈值和 AUROC 是训练后结果/机制指标，不是可直接用于训练前选样的分数。

## 5. 正式运行命令

在仓库根目录 `C:\GitHub\YOLO-CV` 执行：

```powershell
uv run python scripts/stage1_gapvalue240/analyze_extreme_cohorts.py `
  --v2-report-dir artifacts\stage1_sample_value_experiments\experiments\oof_dynamics_gap_value_20260708\06_reports\gapvalue240_pattern_analysis_20260728_v2 `
  --expert-package-root C:\Users\28898\Desktop\待上传\Stage1_GapValue_240Run_ExpertAnalysis_Reconciliation_20260729 `
  --selection-root artifacts\stage1_sample_value_experiments\contracts\gapvalue240_v1_1\generated\selections `
  --output-dir artifacts\stage1_sample_value_experiments\experiments\oof_dynamics_gap_value_20260708\06_reports\gapvalue240_extreme_cohort_analysis_v3
```

所有路径都可以替换为内容等价的实际位置，但输入文件的 SHA-256、大小、schema 和冻结身份必须通过校验。输出目录及其同名 `.inprogress` 目录必须不存在；程序拒绝覆盖旧报告。

## 6. 输入

### 6.1 v2 结果报告

`--v2-report-dir` 必须包含通过 manifest 校验的 v2 报告，至少使用：

```text
manifest.json
tables/triad_control_deltas.csv
tables/canonical_run_metrics.csv
tables/calibration_diagnostics.csv
tables/paired_epoch_differences.csv
tables/training_curve_summary.csv
tables/r2_overlap_power_audit.csv
tables/prediction_tail_detail.csv
tables/prediction_tail_summary.csv
tables/selection_composition.csv
tables/selection_value_effects.csv
```

该层提供 240 个 canonical run、80 个 triad、160 个 T-control 比较及逐 epoch 配对结果。

### 6.2 专家 reconciliation 包

`--expert-package-root` 必须含 `FILE_MANIFEST.csv`，并至少包含：

```text
inputs/oof_probabilities_float64.mmap
inputs/oof_probabilities_metadata.json
inputs/sample_ids.csv
inputs/epoch_gap_metrics.csv
```

程序逐文件核验 size 和 SHA-256 后，才允许读取 OOF memmap。

### 6.3 冻结 selection

`--selection-root` 指向 240-run 合同生成的只读 selection 目录。分析使用 selection index、run 映射和内容 digest 验证 treatment 身份，不重新抽样、不修改 CSV。

## 7. 输出

成功后，`--output-dir` 至少包含：

```text
FINAL_REPORT_CN.md
README.md
index.html
analysis_contract.yaml
manifest.json
audit/
charts/
tables/
```

核心表包括：

```text
triad_performance_tiers.csv
extreme_triads_shortlist.csv
tier_composition_audit.csv
tier_method_composition.csv
condition_seed_tier_matrix.csv
training_window_features.csv
training_extreme_contrasts.csv
training_stratified_extreme_contrasts.csv
training_leave_one_group_out.csv
late_loss_definition_audit.csv
oof_epoch_operational_fn285.csv
oof_operational_sample_dynamics.csv
selection_run_operational_summary.csv
selection_feature_pair_deltas.csv
selection_extreme_contrasts.csv
selection_stratified_extreme_contrasts.csv
selection_leave_one_group_out.csv
defect_selection_extreme_contrasts.csv
treatment_selection_sets.csv
selection_set_outcomes.csv
fixed_selection_seed_flips.csv
selection_set_feature_summary.csv
outcome_mechanism_pairs.csv
outcome_mechanism_tier_summary.csv
outcome_extreme_contrasts.csv
outcome_stratified_extreme_contrasts.csv
outcome_leave_one_group_out.csv
r2_overlap_power_audit.csv
prediction_tail_detail.csv
prediction_tail_extreme_contrasts.csv
prediction_tail_summary.csv
training_curve_summary.csv
candidate_pattern_registry.csv
```

核心图包括：

```text
triad_performance_quadrants_zoomed.png
training_late_loss_s_h_zoomed.png
selection_set_tier_flips.png
prediction_tail_mechanism_zoomed.png
```

图中的缩放轴和零参考线必须显式标注，图旁必须能够追溯到对应 CSV。程序先写 sibling `.inprogress` 目录，全部验收成功后再原子改名为正式输出目录。

## 8. 解释边界

- 所有业务结论限定在冻结的 `val_op` 上；没有 blind/external test，不能宣称外部泛化已经确认。
- R1 是不重合的全局随机基线；R2 是高重合、低功效的近 Treatment 机制对照。两者必须分开报告，R2 旁必须显示独特样本比例。
- Phase A 的同机配对是主要内部证据。Phase B 多数比较存在 arm 与机器混杂，只能作为 exploratory 结果，不能直接升级为因果结论。
- Phase C 是 A02 固定 selection 在新 seeds 下的压力测试，不代表总体方法成功率；其跨机比较同样需要显式标记。
- 同一 selection 在不同 seed 下可能方向翻转，因此不能给单张图片或固定样本集赋予与模型状态无关的永久因果价值。
- 没有 no-replay arm，不能回答 replay 是否优于完全不 replay。
- 没有最终 run 的逐 epoch `val_op` 预测，不能绘制或推断逐 epoch TN/FN/gap 曲线；OOF 200-epoch 动态与最终回流 run 的训练曲线是两条不同证据链。
- 工作阈值、AUROC、tail 指标是训练后诊断量；相关性或分档差异不自动构成训练前可用的排序公式。
- epoch 行、图片行和高重合 R2 对照均不是额外独立重复。显著性和置信区间必须以正确的 triad/selection/seed 单位计算。
- 报告中的规律状态仅使用 `REPEATED_PATTERN`、`PARTIAL_PATTERN`、`COUNTEREXAMPLE_FOUND`、`INSUFFICIENT_EVIDENCE`；它们是候选规律审计，不等价于外部验证后的定律。

## 9. 验收方法

### 9.1 自动测试

```powershell
uv run --extra dev python -m pytest -q `
  tests/stage1_gapvalue240/test_extreme_cohorts.py `
  tests/stage1_gapvalue240/test_extreme_pipeline.py `
  tests/stage1_gapvalue240/test_extreme_reporting.py
```

### 9.2 数据与统计验收

正式报告必须同时满足：

1. 精确识别 240 个 canonical run、80 个 triad 和 160 个 T-control 比较。
2. 档位回归计数精确为 `S12/A3/B1/M41/H23`，且总数为 80。
3. R1、R2 的差值列和判断逻辑始终分开。
4. 每个配对具有完整 epoch 1-200 记录；窗口统计不得缺轮或重复。
5. OOF 元数据精确匹配 `200 x 120000`、float64；sample ID、label、fold 均通过一致性检查。
6. FN285 工作点采用 `score >= threshold` 和完整 tie group；epoch 149 回归锚点通过。
7. fold_01 epoch 178 修复的受影响样本数、排除单元数和来源记录完整。
8. 每份 treatment selection 的 hash、run 映射和唯一 selection digest 可追溯。
9. 同一 selection 跨 seed 的档位组成与翻转由独立表输出，不重复计权。
10. Phase B/Phase C 的机器混杂、R2 overlap 和无 blind test 边界在表和报告正文中可见。
11. 所有正式输出均列入 `manifest.json` 并通过 SHA-256 校验；报告没有读取或链接 `.inprogress` 半成品。
12. 源 v2 报告、专家包和 selection 目录在运行前后保持只读且 hash 不变。

### 9.3 文档和报告人工复核

- 从 `index.html` 的结论追溯到图、表和逐 triad 原始值。
- 抽查 S、A、B、M、H 各至少一个 triad，手算 T−R1、T−R2 与档位规则一致。
- 抽查一个跨 seed 复用 selection，确认 selection digest 相同但各 seed 档位没有被合并隐藏。
- 检查所有结论是否明确区分“数字上更好”“重复候选规律”“存在反例”和“不能作因果结论”。
- 确认没有把 val_op 结果写成 blind-test 结论，也没有把训练后工作阈值写成训练前选样特征。
