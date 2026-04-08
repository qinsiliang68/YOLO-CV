# table_hn_m_delta_vs_hn00_compact

1. This asset answers: How much each HN ratio changes the formal gate metrics relative to the no-HN anchor.
2. Source files:
   - research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv
3. Ranking/selection rule: All deltas are computed as absolute differences against hn00.
4. Key finding: The delta view makes the HN gain profile interpretable without repeating the raw values.
5. Limitation: Positive delta on PTR indicates a worse pass-through rate because PTR is lower-is-better.
