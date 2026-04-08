# table_hn_m_main_ranking

1. This asset answers: Which HN ratio is the formal winner for yolo11m under the recall-constrained stage-1 gate objective.
2. Source files:
   - research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv
3. Ranking/selection rule: Rows are ranked by Spec@R99.5, Spec@R99.0, Prec@R99.0, and PTR@R99.0.
4. Key finding: The current formal winner is hn02, indicating a non-monotonic sweet spot rather than a monotonic ratio effect.
5. Limitation: This table operates on formal summary rows rather than raw PT checkpoint archives.
