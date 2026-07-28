# Stage1 GapValue 240-Run 全面深度分析

本报告仅支持 val_op 内部发现与复现；没有 blind/external test，不能宣称外部泛化已经确认。

## 分析边界

- 图中的 treatment effect 使用配对差值；R1 与 R2 必须分别解释。
- 所有标记为 zoomed 的纵轴都不从零开始，原始数值以 CSV 表为准。
- 机器、resume、snapshot 等混杂必须结合上游敏感性表判断。

## 数据概览

- `validated_runs`: 240
- `triads`: 80
- `paired_comparisons`: 160
- `epoch_rows`: 48000
- `control_consensus_runs`: 160
- `prediction_identity_audits`: 480
- `val_op_prediction_files_verified`: 240
- `val_cal_prediction_files_verified`: 240
- `input_snapshots`: 2
- `resumed_runs`: 15
- `full_val_op_recompute`: True
- `semantic_snapshot_exception_authorized`: True
- `a02_vs_r1_status`: NOT_SUPPORTED
- `a02_vs_r2_status`: NOT_SUPPORTED
- `primary_result`: A02 GapCritical-Strict B3000 未通过预注册双对照成功门槛：R1: n=8, mean ΔFN=+5.875, FN upper95=+9.499, worst ΔFN=+11.000, mean ΔTN=-789.500, TN lower95=-1461.006；R2: n=8, mean ΔFN=+5.000, FN upper95=+11.592, worst ΔFN=+19.000, mean ΔTN=-571.625, TN lower95=-1532.313。因此当前 240-run val_op 证据不支持把 A02 作为优于随机对照的主方法。
- `conclusion_boundary`: val_op internal discovery/replication only; no blind/external claim
- `package_scope`: Git-lightweight; four large regenerable mechanism tables remain in the local full report

## 核心结论

A02 GapCritical-Strict B3000 未通过预注册双对照成功门槛：R1: n=8, mean ΔFN=+5.875, FN upper95=+9.499, worst ΔFN=+11.000, mean ΔTN=-789.500, TN lower95=-1461.006；R2: n=8, mean ΔFN=+5.000, FN upper95=+11.592, worst ΔFN=+19.000, mean ΔTN=-571.625, TN lower95=-1532.313。因此当前 240-run val_op 证据不支持把 A02 作为优于随机对照的主方法。

## 三层 A02 结果

Discovery-3：R1: n=3, mean ΔFN=+2.000, FN upper95=+13.801, worst ΔFN=+7.000, mean ΔTN=-673.667, TN lower95=-2500.142；R2: n=3, mean ΔFN=+1.333, FN upper95=+21.727, worst ΔFN=+15.000, mean ΔTN=-345.667, TN lower95=-3671.881。Confirmation-5：R1: n=5, mean ΔFN=+8.200, FN upper95=+11.012, worst ΔFN=+11.000, mean ΔTN=-859.000, TN lower95=-1883.951；R2: n=5, mean ΔFN=+7.200, FN upper95=+15.738, worst ΔFN=+19.000, mean ΔTN=-707.200, TN lower95=-1910.154。Pooled-8 是合同统计，但包含机器混杂，不能消除 Phase C 的执行差异。

## 分析边界

本报告检验训练动态选样的内部发现与复现证据，不把单个最高分 run 当作结论，也不作 blind/external 泛化声明。

## 机器混杂

Phase C 五个确认 triad 的 treatment 与 controls 分处不同机器；pooled-8 只作为预注册算术结果，不能消除机器混杂。

## R2 对照解释

R2 是高重合、低功效的近 treatment 机制对照；R1 是完全不重合的主要随机基线，两者必须分别解释。本批 R2 的中位有效独特对比比例为 7.65%。

## BottomGap 负对照

A13 实际包含 2844 个负分样本和 156 个非负分样本；A13 数字方向有害，但 A02 本身也未形成预期正向结果，因此不能宣称已经建立正负方向因果对照。

## Defect guard

Phase B 仅有 3 seeds，且主要 guard treatment-control 比较存在机器混杂。报告保留剂量和策略的数值差异，但将 guard 结论标为 INCONCLUSIVE。

## 子组

子组字段只使用冻结 val_op manifests 中真实存在的字段；primary_defect_class 缺失时明确标记不可用，不作推断。

## 假设证据状态

- `SUPPORTED`: 0
- `NOT_SUPPORTED`: 6
- `INCONCLUSIVE`: 2
- `NOT_TESTABLE`: 0

## 产物

- 分析合同：`analysis_contract.yaml`
- 审计：`audit/completeness.json`
- 审计：`audit/metric_recompute.csv`
- 审计：`audit/prediction_identity.csv`
- 审计：`audit/selection_sha.csv`
- 审计：`audit/source_boundaries.json`
- 审计：`audit/subgroup_field_availability.json`
- 审计：`audit/training_contract.csv`
- 表：`tables/a02_discovery_confirmation_combined.csv`
- 表：`tables/a02_extended_sensitivity.csv`
- 表：`tables/a02_leave_one_seed_out.csv`
- 表：`tables/a02_shift_subgroups.csv`
- 表：`tables/a02_threshold_frontier.csv`
- 表：`tables/a02_threshold_frontier_detail.csv`
- 表：`tables/a02_training_curves.csv`
- 表：`tables/bottom_gap_signs.csv`
- 表：`tables/budget_response.csv`
- 表：`tables/calibration_diagnostics.csv`
- 表：`tables/canonical_run_metrics.csv`
- 表：`tables/condition_control_summaries.csv`
- 表：`tables/cross_method_seed_paired.csv`
- 表：`tables/direct_treatment_comparisons.csv`
- 表：`tables/guard_policy_contrasts.csv`
- 表：`tables/guard_vs_a02_normal_only_direct.csv`
- 表：`tables/historical_failed_attempts.csv`
- 表：`tables/metric_recompute_audit.csv`
- 表：`tables/prediction_identity_audit.csv`
- 表：`tables/prediction_tail_detail.csv`
- 表：`tables/prediction_tail_summary.csv`
- 表：`tables/r2_overlap_power_audit.csv`
- 表：`tables/reliability_by_machine.csv`
- 表：`tables/resume_sensitivity.csv`
- 表：`tables/run_execution_reliability.csv`
- 表：`tables/selection_composition.csv`
- 表：`tables/selection_feature_smd.csv`
- 表：`tables/selection_identity_quality.csv`
- 表：`tables/selection_method_overlap.csv`
- 表：`tables/selection_run_audit.csv`
- 表：`tables/selection_sha_audit.csv`
- 表：`tables/selection_triad_overlap.csv`
- 表：`tables/sensitivity_results.csv`
- 表：`tables/snapshot_sensitivity.csv`
- 表：`tables/subgroup_field_availability.csv`
- 表：`tables/training_contract_audit.csv`
- 表：`tables/training_curve_summary.csv`
- 表：`tables/triad_control_deltas.csv`
- 表：`tables/hypothesis_registry.csv`
- 图：`charts/hypothesis_status.png`；来源表：`tables/hypothesis_registry.csv`；四级证据判定数量，计数轴从零开始。
- 图：`charts/condition_effects_zoomed.png`；来源表：`tables/condition_control_summaries.csv`；TN 与 FN 使用独立缩放纵轴，精确值见来源表。
- 图：`charts/a02_seed_forest_zoomed.png`；来源表：`tables/triad_control_deltas.csv`；逐 seed、逐对照展示 A02 配对效应，横轴为缩放轴。
- 图：`charts/condition_pareto_zoomed.png`；来源表：`tables/condition_control_summaries.csv`；左上方向更优；横纵轴均按观察范围缩放。
- 图：`charts/r2_effective_unique_contrast.png`；来源表：`tables/r2_overlap_power_audit.csv`；R2 与 treatment 高重合时，独特对比比例决定机制检验功效。
- 图：`charts/budget_response_zoomed.png`；来源表：`tables/budget_response.csv`；600/3000/6000 的响应曲线使用缩放纵轴。
- 图：`charts/guard_response_zoomed.png`；来源表：`tables/guard_policy_contrasts.csv`；guard 比例响应使用缩放纵轴，机器混杂需结合审计解释。
- 图：`charts/tail_shift_zoomed.png`；来源表：`tables/prediction_tail_summary.csv`；normal 与 defect 固定尾部的原始分数移动，纵轴缩放。
- 图：`charts/a02_threshold_frontier_zoomed.png`；来源表：`tables/a02_threshold_frontier.csv`；展示阈值扫描的 FN/TN 前沿，双轴按观察范围缩放。
- 图：`charts/a02_training_curves_zoomed.png`；来源表：`tables/a02_training_curves.csv`；top1 与 val loss 使用独立缩放纵轴，不能替代逐 epoch operational 指标。
- 图：`charts/resume_machine_reliability.png`；来源表：`tables/canonical_run_metrics.csv`；机器运行量和 native resume 次数均使用从零开始的计数轴。
