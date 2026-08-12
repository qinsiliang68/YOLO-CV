# Stage1 240/40/120 Run 性能前沿分析

## 目的

本分析回答“哪些训练方式真正让误报和漏检总体向好”，而不是寻找某个固定阈值下的最高数字。分析同时使用：

1. 零回流 `yolo11l best.pt` 基线，用于判断训练后模型是否获得绝对性能提升；
2. 同条件随机对照，用于判断提升是否可归因于设计选样，而不是 replay、训练 seed 或样本数量本身。

240-run 的 Treatment 必须分别比较 R1 全局随机与 R2 匹配随机，二者不可合并。40-run 使用同预算 HN/RN 配对。120-run 使用三个 RN 对照及其中位参考，但只具备 FN=70～120、步长 5 的已审计前沿。

## 核心判定

所有模型均在相同 FN 上限下比较其最大 TN，预测规则固定为 `score >= threshold`，相同分数作为完整 tie group 处理。因此，单纯移动置信度阈值不会被认定为模型性能提升。

结果分为：

- `ROBUST_SAFE_DOUBLE_GATE`：相对零回流基线和随机对照，在整个预定义安全 FN 区间均不劣，且至少一个位置严格改善；
- `LOCAL_PARETO_DOUBLE_GATE`：在有效业务点出现 FN 不增且 TN 增加，或 FN 下降且 TN 不减，并同时胜过随机对照，但不能覆盖整个安全区间；
- `ABSOLUTE_ONLY`：优于零回流基线，但没有同时胜过随机对照；
- `SECONDARY_CONTROLLED`：允许少量额外 FN 后换得明显 TN，但仅属于兜底结果；
- `JOINTLY_HARMFUL`：在基线业务 FN 处相对零回流基线及随机对照均减少 TN；
- `MIXED_OR_INCONCLUSIVE`：其余交叉或证据不足结果。

“稳健”与“局部”必须分开陈述。单个 seed 的好结果也不能升级成可复用训练规律。

## 数据范围与边界

- 240-run：240 个 canonical run、80 个 T/R1/R2 triad；使用完整 `val_op` 原始预测；
- 40-run：40 个完整预测、20 个 HN/RN 配对；使用历史 development benchmark；
- 120-run：111 个有效结果、9 个缺失；没有本地完整逐样本原始预测，只使用已审计的粗粒度 FN 前沿；
- 分析只用于内部发现，没有 blind/external test，不能宣称最终外部泛化；
- 240-run 相对零回流基线的绝对比较存在 seed 差异，方法归因主要依赖同 seed 的 R1/R2；
- raw score 是完整前沿的主依据；Platt 校准分数仅用于核查，因为概率取整会改变 tie group。

## 正式入口

```powershell
uv run python scripts/stage1_gapvalue240/analyze_performance_frontiers.py `
  --extracted-root C:\baidunetdiskdownload\stage1_gapvalue240_all_uploads_extracted `
  --inventory C:\baidunetdiskdownload\stage1_gapvalue240_extract_audit_20260728\GLOBAL_VALIDATED_RUN_INVENTORY.csv `
  --matrix artifacts\stage1_sample_value_experiments\contracts\gapvalue240_v1_1\generated\frozen_experiment_matrix.csv `
  --baseline-root artifacts\stage1_cls_eval_1to5_20260617 `
  --run40-root artifacts\stage1_phase1_hn_rn_40runs_20260630_redownload `
  --run120-root artifacts\stage1_phase1_hn_band_20260628_120runs_20260707_review `
  --v3-report-dir artifacts\stage1_sample_value_experiments\experiments\oof_dynamics_gap_value_20260708\06_reports\gapvalue240_extreme_cohort_analysis_v3 `
  --output-dir artifacts\stage1_sample_value_experiments\experiments\oof_dynamics_gap_value_20260708\06_reports\gapvalue_performance_frontier_analysis_20260801_v5
```

输出目录不可覆盖，先写同级 `.inprogress`，验收后原子改名。原始训练结果、selection CSV 和冻结矩阵均按只读输入处理。

## 主要输出

- `FINAL_REPORT_CN.md` 与 `index.html`：结论和可视化；
- `tables/all_run_baseline_dominance.csv`：全部 391 个可用候选相对零回流基线；
- `tables/paired_control_frontier_deltas.csv`：Treatment 与随机对照的同 FN 前沿差值；
- `tables/designed_method_double_gates.csv`：设计方法双门判定；
- `tables/method_repeatability_ranking.csv`：跨 seed 重复性；
- `tables/hypothesis_registry.csv`：逐条研究命题判定；
- `tables/*mechanism*.csv`：训练后期损失、弱缺陷尾部和选样组成机制；
- `manifest.json`：报告内永久文件 SHA-256。

## 当前结果解释原则

先报告是否存在稳健安全前沿，再报告局部 Pareto 成功，最后才分析机制。训练后工作阈值、AUROC 和尾部分数可解释为何成功，但不能反过来充当训练前样本价值分数。任何候选规律只有在不同 seed、随机对照和留出条件上保持方向一致，才能升级为下一轮预注册选样规则。
