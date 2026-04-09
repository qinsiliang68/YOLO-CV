# fig_budget_concentration_tripanel

1. This asset answers: Whether the weighted variants are redistributing the replay budget smoothly or collapsing it onto a very small subset of the hard pool.
2. Source files:
   - research/materials/stage1_formal/gate_info_sampling_lite/score_inputs/table_score_component_stats.csv
3. Ranking/selection rule: The tripanel compares the fixed top250 pool under the same one-shot hn00 teacher and budget.
4. Key finding: A3 and A4 concentrate much more heavily than A2, with substantially lower effective counts and much higher top10 cumulative replay mass.
5. Limitation: These are pool-level concentration diagnostics and do not by themselves prove which weighting policy should win downstream.
