# fig_uniform_vs_weighted_overlap

1. This asset answers: How much the weighted replay selections overlap with the existing uniform_hn14 replay set.
2. Source files:
   - research/materials/stage1_formal/gate_info_sampling_lite/score_inputs/uniform_hn14_reference.csv
   - research/materials/stage1_formal/gate_info_sampling_lite/score_inputs/A2_candidate_pool_scores.csv
   - research/materials/stage1_formal/gate_info_sampling_lite/score_inputs/A3_candidate_pool_scores.csv
   - research/materials/stage1_formal/gate_info_sampling_lite/score_inputs/A4_candidate_pool_scores.csv
3. Ranking/selection rule: Uniform HN14 is compared against the selected unique samples from each weighted variant.
4. Key finding: The overlap view indicates whether weighted replay primarily reorders the same pool or surfaces a different subset of hard normals.
5. Limitation: The figure summarizes unique-sample overlap rather than replay-count overlap.
