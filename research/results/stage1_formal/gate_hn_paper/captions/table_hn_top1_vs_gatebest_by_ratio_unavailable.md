# table_hn_top1_vs_gatebest_by_ratio_unavailable

1. This asset answers: Whether HN preserves the top1-best versus gate-best mismatch across ratios.
2. Source files:
   - research/materials/stage1_formal/gate_hn_m_sweep/hn02/all_checkpoints_index.csv
   - research/materials/stage1_formal/gate_hn_x_crosscheck/hn02/all_checkpoints_index.csv
3. Ranking/selection rule: Generation is conditional on the presence of reliable trainer-side top1 fields in the ratio-level checkpoint index.
4. Key finding: The current HN working set does not include those trainer-side fields, so no ratio-level top1 mismatch table is emitted.
5. Limitation: This absence is a material-level limitation rather than a modeling claim.
