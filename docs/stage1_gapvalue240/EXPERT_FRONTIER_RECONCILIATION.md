# 专家相对结果与绝对性能前沿交叉分析

## 目的

本分析把专家定义的相对R1/R2强正向组，与零回流 `yolo11l best.pt` 的raw-score同FN性能前沿统一起来。专家分组保留为证据列，但不替代绝对性能判定。

最终区分：完整安全前沿提升、局部绝对Pareto提升、少量FN代价的次级提升、只胜随机对照、绝对共同有害以及混合结果。

## 科学口径

- 相同FN上限下比较最大TN，完整tie group不可拆分；
- 240-run Treatment必须分别保留R1和R2比较；
- raw score决定性能前沿，校准阈值只作结果机制；
- LateOverfit主定义固定为epoch 121至指定cutoff的Treatment额外训练损失下降；
- 统计单位是triad，交叉验证按完整condition或seed留出；
- 没有逐epoch val_op预测，因此不推断逐epoch TN/FN；
- 没有blind/external test，不宣称外部泛化。

## 运行入口

```powershell
uv run python scripts/stage1_gapvalue240/analyze_expert_frontier_reconciliation.py `
  --expert-zip C:\Users\28898\Downloads\Stage1_GapValue_240Run_GoodCohort_Patterns_20260802.zip `
  --v5-report-dir artifacts\stage1_sample_value_experiments\experiments\oof_dynamics_gap_value_20260708\06_reports\gapvalue_performance_frontier_analysis_20260801_v5 `
  --v3-report-dir artifacts\stage1_sample_value_experiments\experiments\oof_dynamics_gap_value_20260708\06_reports\gapvalue240_extreme_cohort_analysis_v3 `
  --full-analysis-dir artifacts\stage1_sample_value_experiments\experiments\oof_dynamics_gap_value_20260708\06_reports\gapvalue240_full_analysis_20260728_v1 `
  --output-dir artifacts\stage1_sample_value_experiments\experiments\oof_dynamics_gap_value_20260708\06_reports\gapvalue_expert_frontier_reconciliation_20260802_v3
```

输入按只读处理。输出先写入同级 `.inprogress`，验收后原子改名；已存在的报告拒绝覆盖。

## 主要输出

- 80个triad统一结果和专家到绝对结果映射；
- epoch 140/150/160/180/200的LateOverfit复算；
- condition和seed留组交叉验证逐折预测；
- weak-defect与normal尾部机制；
- 统计与假设注册表；
- 下一轮replay后期降权预注册建议；
- 中文HTML、图表源CSV和SHA-256 manifest。
