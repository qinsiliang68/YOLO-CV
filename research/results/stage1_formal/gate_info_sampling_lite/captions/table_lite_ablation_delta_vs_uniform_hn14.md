# table_lite_ablation_delta_vs_uniform_hn14

1. This asset answers: Whether the weighted variants beat the current uniform HN14 anchor under the same replay budget.
2. Source files:
   - research/results/stage1_formal/gate_info_sampling_lite/derived/ablation_delta_vs_uniform_hn14.csv
3. Ranking/selection rule: All deltas are absolute differences against A1/uniform_hn14.
4. Key finding: This table directly tests whether pool-internal value heterogeneity matters under a fixed HN14 budget.
5. Limitation: The comparison is restricted to the yolo11m mainline and does not revisit backbone choice.
