# table_score_component_stats

1. This asset answers: How concentrated the score and replay-probability distributions are for the lite scoring variants.
2. Source files:
   - research/materials/stage1_formal/gate_info_sampling_lite/score_inputs/table_score_component_stats.csv
3. Ranking/selection rule: Scores come from the one-shot hn00 teacher and the fixed top250 pool.
4. Key finding: Higher concentration indicates that a small subset of hard negatives receives a larger share of the fixed replay budget.
5. Limitation: These statistics operate on the candidate pool only and do not by themselves prove downstream formal gains.
