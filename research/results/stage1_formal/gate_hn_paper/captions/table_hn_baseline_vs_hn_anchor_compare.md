# table_hn_baseline_vs_hn_anchor_compare

1. This asset answers: How the HN results connect back to the already-completed baseline capacity scan.
2. Source files:
   - research/results/stage1_formal/gate_capacity/binary_gate_capacity_summary.csv
   - research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv
   - research/results/stage1_formal/gate_hn_x_crosscheck/hn_sweep_summary.csv
3. Ranking/selection rule: Baseline rows come from the completed formal capacity scan; HN rows come from the formal HN sweep summaries.
4. Key finding: This bridge table shows whether HN improves the same model relative to its own hn00 anchor instead of reopening the backbone selection question.
5. Limitation: yolo11x is only a light cross-capacity check because only hn00 and hn02 are available.
