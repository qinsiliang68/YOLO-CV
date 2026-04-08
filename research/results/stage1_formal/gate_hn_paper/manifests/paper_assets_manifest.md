# paper_assets_manifest

## table_hn_m_main_ranking

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/tables/table_hn_m_main_ranking.csv`
  - `research/results/stage1_formal/gate_hn_paper/tables/table_hn_m_main_ranking.md`
  - `research/results/stage1_formal/gate_hn_paper/captions/table_hn_m_main_ranking.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
- Rule: Rows are ranked by Spec@R99.5, Spec@R99.0, Prec@R99.0, and PTR@R99.0.
- Key finding: The current formal winner is hn14, indicating a non-monotonic sweet spot rather than a monotonic ratio effect.
- Limitation: This table operates on formal summary rows rather than raw PT checkpoint archives.

## table_hn_m_delta_vs_hn00_compact

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/tables/table_hn_m_delta_vs_hn00_compact.csv`
  - `research/results/stage1_formal/gate_hn_paper/tables/table_hn_m_delta_vs_hn00_compact.md`
  - `research/results/stage1_formal/gate_hn_paper/captions/table_hn_m_delta_vs_hn00_compact.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
- Rule: All deltas are computed as absolute differences against hn00.
- Key finding: The delta view makes the HN gain profile interpretable without repeating the raw values.
- Limitation: Positive delta on PTR indicates a worse pass-through rate because PTR is lower-is-better.

## table_hn_baseline_vs_hn_anchor_compare

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/tables/table_hn_baseline_vs_hn_anchor_compare.csv`
  - `research/results/stage1_formal/gate_hn_paper/tables/table_hn_baseline_vs_hn_anchor_compare.md`
  - `research/results/stage1_formal/gate_hn_paper/captions/table_hn_baseline_vs_hn_anchor_compare.md`
- Sources:
  - `research/results/stage1_formal/gate_capacity/binary_gate_capacity_summary.csv`
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
  - `research/results/stage1_formal/gate_hn_x_crosscheck/hn_sweep_summary.csv`
- Rule: Baseline rows come from the completed formal capacity scan; HN rows come from the formal HN sweep summaries.
- Key finding: This bridge table shows whether HN improves the same model relative to its own hn00 anchor instead of reopening the backbone selection question.
- Limitation: yolo11x is only a light cross-capacity check because only hn00 and hn02 are available.

## table_hn_x_crosscheck_main

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/tables/table_hn_x_crosscheck_main.csv`
  - `research/results/stage1_formal/gate_hn_paper/tables/table_hn_x_crosscheck_main.md`
  - `research/results/stage1_formal/gate_hn_paper/captions/table_hn_x_crosscheck_main.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_x_crosscheck/hn_sweep_summary.csv`
- Rule: This table is a light cross-capacity validation and does not claim a full sweep on yolo11x.
- Key finding: The yolo11x rows show only hn00 and hn02, which is sufficient for a directionality check but not for a full sweet-spot search.
- Limitation: No full x-side ratio sweep is available in the current working set.

## fig_hn_m_ratio_metric_curves_panel

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_ratio_metric_curves_panel.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_ratio_metric_curves_panel.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_m_ratio_metric_curves_panel.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
- Rule: Formal ranking follows Spec@R99.5 > Spec@R99.0 > Prec@R99.0 > PTR@R99.0.
- Key finding: The sweep exhibits a non-monotonic profile with hn14 as the current formal winner.
- Limitation: This panel summarizes per-ratio winners and does not visualize per-epoch trajectories.

## fig_hn_m_ratio_metric_curves_spec995

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_ratio_metric_curves_spec995.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_ratio_metric_curves_spec995.png`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_spec995_delta.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_spec995_delta.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_m_ratio_metric_curves_spec995.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
- Rule: Rows are interpreted under the formal gate-aware rule; PTR remains lower-is-better.
- Key finding: Spec@R99.5 does not improve monotonically, which supports the sweet-spot interpretation.
- Limitation: This figure isolates one metric and should be interpreted together with the other three formal metrics.

## fig_hn_m_ratio_metric_curves_spec995_delta

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_spec995_delta.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_spec995_delta.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_m_ratio_metric_curves_spec995_delta.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
- Rule: The hn00 row is treated as the anchor and all values are absolute differences.
- Key finding: The delta view shows the gain pattern more clearly than the raw-value plot.
- Limitation: Only the metric-specific delta is shown here; formal ranking still uses the full four-metric rule.

## fig_hn_m_ratio_metric_curves_spec990

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_ratio_metric_curves_spec990.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_ratio_metric_curves_spec990.png`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_spec990_delta.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_spec990_delta.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_m_ratio_metric_curves_spec990.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
- Rule: Rows are interpreted under the formal gate-aware rule; PTR remains lower-is-better.
- Key finding: Spec@R99.0 does not improve monotonically, which supports the sweet-spot interpretation.
- Limitation: This figure isolates one metric and should be interpreted together with the other three formal metrics.

## fig_hn_m_ratio_metric_curves_spec990_delta

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_spec990_delta.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_spec990_delta.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_m_ratio_metric_curves_spec990_delta.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
- Rule: The hn00 row is treated as the anchor and all values are absolute differences.
- Key finding: The delta view shows the gain pattern more clearly than the raw-value plot.
- Limitation: Only the metric-specific delta is shown here; formal ranking still uses the full four-metric rule.

## fig_hn_m_ratio_metric_curves_prec990

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_ratio_metric_curves_prec990.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_ratio_metric_curves_prec990.png`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_prec990_delta.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_prec990_delta.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_m_ratio_metric_curves_prec990.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
- Rule: Rows are interpreted under the formal gate-aware rule; PTR remains lower-is-better.
- Key finding: Prec@R99.0 does not improve monotonically, which supports the sweet-spot interpretation.
- Limitation: This figure isolates one metric and should be interpreted together with the other three formal metrics.

## fig_hn_m_ratio_metric_curves_prec990_delta

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_prec990_delta.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_prec990_delta.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_m_ratio_metric_curves_prec990_delta.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
- Rule: The hn00 row is treated as the anchor and all values are absolute differences.
- Key finding: The delta view shows the gain pattern more clearly than the raw-value plot.
- Limitation: Only the metric-specific delta is shown here; formal ranking still uses the full four-metric rule.

## fig_hn_m_ratio_metric_curves_ptr990

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_ratio_metric_curves_ptr990.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_ratio_metric_curves_ptr990.png`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_ptr990_delta.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_ptr990_delta.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_m_ratio_metric_curves_ptr990.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
- Rule: Rows are interpreted under the formal gate-aware rule; PTR remains lower-is-better.
- Key finding: PTR@R99.0 does not improve monotonically, which supports the sweet-spot interpretation.
- Limitation: This figure isolates one metric and should be interpreted together with the other three formal metrics.

## fig_hn_m_ratio_metric_curves_ptr990_delta

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_ptr990_delta.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_m_ratio_metric_curves_ptr990_delta.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_m_ratio_metric_curves_ptr990_delta.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
- Rule: The hn00 row is treated as the anchor and all values are absolute differences.
- Key finding: The delta view shows the gain pattern more clearly than the raw-value plot.
- Limitation: Only the metric-specific delta is shown here; formal ranking still uses the full four-metric rule.

## fig_hn_m_epoch_dynamics_spec995

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_epoch_dynamics_spec995.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_epoch_dynamics_spec995.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_m_epoch_dynamics_spec995.md`
- Sources:
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn00/epoch_gate_summary.csv`
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn14/epoch_gate_summary.csv`
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn20/epoch_gate_summary.csv`
- Rule: Representative ratios are hn00, the formal winner, and a high-ratio tail setting.
- Key finding: The epoch dynamics show that HN changes the training trajectory itself rather than only shifting the final selected checkpoint.
- Limitation: Only three representative ratios are shown here to keep the comparison interpretable.

## fig_hn_m_epoch_dynamics_spec990

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_epoch_dynamics_spec990.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_epoch_dynamics_spec990.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_m_epoch_dynamics_spec990.md`
- Sources:
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn00/epoch_gate_summary.csv`
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn14/epoch_gate_summary.csv`
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn20/epoch_gate_summary.csv`
- Rule: Representative ratios are hn00, the formal winner, and a high-ratio tail setting.
- Key finding: The epoch dynamics show that HN changes the training trajectory itself rather than only shifting the final selected checkpoint.
- Limitation: Only three representative ratios are shown here to keep the comparison interpretable.

## fig_hn_m_epoch_dynamics_prec990

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_epoch_dynamics_prec990.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_epoch_dynamics_prec990.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_m_epoch_dynamics_prec990.md`
- Sources:
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn00/epoch_gate_summary.csv`
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn14/epoch_gate_summary.csv`
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn20/epoch_gate_summary.csv`
- Rule: Representative ratios are hn00, the formal winner, and a high-ratio tail setting.
- Key finding: The epoch dynamics show that HN changes the training trajectory itself rather than only shifting the final selected checkpoint.
- Limitation: Only three representative ratios are shown here to keep the comparison interpretable.

## fig_hn_m_epoch_dynamics_ptr990

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_epoch_dynamics_ptr990.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_epoch_dynamics_ptr990.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_m_epoch_dynamics_ptr990.md`
- Sources:
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn00/epoch_gate_summary.csv`
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn14/epoch_gate_summary.csv`
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn20/epoch_gate_summary.csv`
- Rule: Representative ratios are hn00, the formal winner, and a high-ratio tail setting.
- Key finding: The epoch dynamics show that HN changes the training trajectory itself rather than only shifting the final selected checkpoint.
- Limitation: Only three representative ratios are shown here to keep the comparison interpretable.

## fig_hn_m_epoch_dynamics_panel

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_epoch_dynamics_panel.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_m_epoch_dynamics_panel.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_m_epoch_dynamics_panel.md`
- Sources:
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn00/epoch_gate_summary.csv`
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn14/epoch_gate_summary.csv`
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn20/epoch_gate_summary.csv`
- Rule: Representative ratios are selected as hn00, the formal winner, and a high-ratio tail condition.
- Key finding: The representative-ratio panel summarizes how HN affects the training dynamics beyond the final best-epoch snapshot.
- Limitation: The panel is limited to representative ratios and does not replace the full ratio-level summaries.

## fig_hn_cross_capacity_slope_spec995

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_cross_capacity_slope_spec995.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_cross_capacity_slope_spec995.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_cross_capacity_slope_spec995.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_overview/table_hn_cross_capacity_compare.csv`
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
  - `research/results/stage1_formal/gate_hn_x_crosscheck/hn_sweep_summary.csv`
- Rule: The comparison is restricted to hn00 and hn02 under the same formal gate-aware rule.
- Key finding: The cross-capacity slope view shows whether HN behaves directionally similarly on the main model and the second model.
- Limitation: yolo11x only has hn00 and hn02, so this is a light validation rather than a full sweep.

## fig_hn_cross_capacity_slope_spec990

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_cross_capacity_slope_spec990.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_cross_capacity_slope_spec990.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_cross_capacity_slope_spec990.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_overview/table_hn_cross_capacity_compare.csv`
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
  - `research/results/stage1_formal/gate_hn_x_crosscheck/hn_sweep_summary.csv`
- Rule: The comparison is restricted to hn00 and hn02 under the same formal gate-aware rule.
- Key finding: The cross-capacity slope view shows whether HN behaves directionally similarly on the main model and the second model.
- Limitation: yolo11x only has hn00 and hn02, so this is a light validation rather than a full sweep.

## fig_hn_cross_capacity_slope_prec990

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_cross_capacity_slope_prec990.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_cross_capacity_slope_prec990.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_cross_capacity_slope_prec990.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_overview/table_hn_cross_capacity_compare.csv`
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
  - `research/results/stage1_formal/gate_hn_x_crosscheck/hn_sweep_summary.csv`
- Rule: The comparison is restricted to hn00 and hn02 under the same formal gate-aware rule.
- Key finding: The cross-capacity slope view shows whether HN behaves directionally similarly on the main model and the second model.
- Limitation: yolo11x only has hn00 and hn02, so this is a light validation rather than a full sweep.

## fig_hn_cross_capacity_slope_ptr990

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_cross_capacity_slope_ptr990.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_cross_capacity_slope_ptr990.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_cross_capacity_slope_ptr990.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_overview/table_hn_cross_capacity_compare.csv`
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
  - `research/results/stage1_formal/gate_hn_x_crosscheck/hn_sweep_summary.csv`
- Rule: The comparison is restricted to hn00 and hn02 under the same formal gate-aware rule.
- Key finding: The cross-capacity slope view shows whether HN behaves directionally similarly on the main model and the second model.
- Limitation: yolo11x only has hn00 and hn02, so this is a light validation rather than a full sweep.

## fig_hn_cross_capacity_slope_panel

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_cross_capacity_slope_panel.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_cross_capacity_slope_panel.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_cross_capacity_slope_panel.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_overview/table_hn_cross_capacity_compare.csv`
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
  - `research/results/stage1_formal/gate_hn_x_crosscheck/hn_sweep_summary.csv`
- Rule: The comparison uses the existing cross-capacity summary without any new inference run.
- Key finding: The panel supports a directional cross-capacity validation rather than a claim of x-side full-sweep optimality.
- Limitation: Only hn00 and hn02 are available for yolo11x in the current working set.

## fig_hn_best_epoch_vs_ratio

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_best_epoch_vs_ratio.csv`
  - `research/results/stage1_formal/gate_hn_paper/figures/fig_hn_best_epoch_vs_ratio.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_best_epoch_vs_ratio.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
- Rule: Best epochs are taken from the formal HN sweep summary after gate-aware checkpoint selection.
- Key finding: The best-epoch position shifts with ratio, which indicates that HN changes the training regime rather than only the final score.
- Limitation: This figure tracks only the selected gate-best epoch and not the full per-epoch trajectory.

## table_hn_best_checkpoint_registry_clean

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/appendix/table_hn_best_checkpoint_registry_clean.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/table_hn_best_checkpoint_registry_clean.md`
  - `research/results/stage1_formal/gate_hn_paper/captions/table_hn_best_checkpoint_registry_clean.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_overview/hn_best_checkpoint_registry.csv`
- Rule: This table is a cleaned view of the formal HN checkpoint registry.
- Key finding: The registry provides a compact mapping from model and ratio to the selected formal checkpoint.
- Limitation: Checkpoint paths point to archived locations and may not all exist in the current local repo working set.

## table_hn_ratio_coverage_manifest

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/appendix/table_hn_ratio_coverage_manifest.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/table_hn_ratio_coverage_manifest.md`
  - `research/results/stage1_formal/gate_hn_paper/captions/table_hn_ratio_coverage_manifest.md`
- Sources:
  - `research/materials/stage1_formal/gate_hn_m_sweep`
  - `research/materials/stage1_formal/gate_hn_x_crosscheck`
  - `research/results/stage1_formal/gate_hn_overview/ingest_manifest.json`
- Rule: Coverage is derived from the current repo paths, with source-archive provenance carried over only in the note field.
- Key finding: The table reports repo working-set availability rather than the fuller external source archive.
- Limitation: PT checkpoints and raw per-epoch trees are intentionally tracked as unavailable in this repo working set.

## fig_hn_overview_heatmap_raw

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_overview_heatmap_raw.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_overview_heatmap_raw.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_overview_heatmap_raw.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
- Rule: Rows are ratios and columns are formal summary metrics plus best epoch.
- Key finding: The raw heatmap provides a compact whole-sweep view of the HN landscape.
- Limitation: The metrics have different scales, so the panel is best used as a qualitative scan rather than a numeric substitute for the tables.

## fig_hn_overview_heatmap_delta

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_overview_heatmap_delta.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_overview_heatmap_delta.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_overview_heatmap_delta.md`
- Sources:
  - `research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv`
- Rule: Rows are ratios and all values are absolute deltas relative to hn00.
- Key finding: The delta heatmap highlights the sweet-spot region more clearly than the raw-value view.
- Limitation: PTR remains lower-is-better, so a negative delta is favorable.

## table_hn_hard_normal_selection_stats

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/appendix/table_hn_hard_normal_selection_stats.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/table_hn_hard_normal_selection_stats.md`
  - `research/results/stage1_formal/gate_hn_paper/captions/table_hn_hard_normal_selection_stats.md`
- Sources:
  - `research/materials/stage1_formal/gate_hn_assets/yolo11m_train_normal_scores/train_normal_scores.csv`
  - `research/materials/stage1_formal/gate_hn_assets/yolo11m_train_normal_scores/top_false_positive_normals.csv`
  - `research/materials/stage1_formal/gate_hn_assets/yolo11m_train_normal_scores/summary.json`
- Rule: The selected hard normals are summarized using the existing score assets rather than any new inference run.
- Key finding: The table reports selected-pool score statistics, while the threshold column records the minimum selected score.
- Limitation: Heuristic labels in the source CSV are not used here because the key evidence is the score-tail behavior.

## fig_hn_hard_normal_score_distribution

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_hard_normal_score_distribution.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_hard_normal_score_distribution.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_hard_normal_score_distribution.md`
- Sources:
  - `research/materials/stage1_formal/gate_hn_assets/yolo11m_train_normal_scores/train_normal_scores.csv`
  - `research/materials/stage1_formal/gate_hn_assets/yolo11m_train_normal_scores/top_false_positive_normals.csv`
- Rule: The threshold is taken from the minimum score among the selected top false-positive normals.
- Key finding: The selected HN pool comes from the high-score tail rather than ad hoc manual picking.
- Limitation: This provenance view does not itself evaluate whether the selected normals are semantically diverse.

## fig_hn_hardest_normal_gallery_panel

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_hardest_normal_gallery_panel.csv`
  - `research/results/stage1_formal/gate_hn_paper/appendix/fig_hn_hardest_normal_gallery_panel.png`
  - `research/results/stage1_formal/gate_hn_paper/captions/fig_hn_hardest_normal_gallery_panel.md`
- Sources:
  - `research/materials/stage1_formal/gate_hn_assets/yolo11m_train_normal_scores/hardest_normal_gallery`
  - `research/materials/stage1_formal/gate_hn_assets/yolo11m_train_normal_scores/top_false_positive_normals.csv`
- Rule: The panel uses the highest-ranked gallery images already exported by the HN asset builder.
- Key finding: The gallery provides visual evidence that the HN source pool is tied to concrete difficult normal cases.
- Limitation: This panel is illustrative and should be interpreted together with the formal score-distribution statistics.

## table_hn_top1_vs_gatebest_by_ratio_unavailable

- Outputs:
  - `research/results/stage1_formal/gate_hn_paper/appendix/table_hn_top1_vs_gatebest_by_ratio_unavailable.md`
  - `research/results/stage1_formal/gate_hn_paper/captions/table_hn_top1_vs_gatebest_by_ratio_unavailable.md`
- Sources:
  - `research/materials/stage1_formal/gate_hn_m_sweep/hn02/all_checkpoints_index.csv`
  - `research/materials/stage1_formal/gate_hn_x_crosscheck/hn02/all_checkpoints_index.csv`
- Rule: Generation is conditional on the presence of reliable trainer-side top1 fields in the ratio-level checkpoint index.
- Key finding: The current HN working set does not include those trainer-side fields, so no ratio-level top1 mismatch table is emitted.
- Limitation: This absence is a material-level limitation rather than a modeling claim.
