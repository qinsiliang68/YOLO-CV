# SUMMARY_gate_info_sampling_lite

## Experiment Groups
- A0: hn00 baseline (reused).
- A1: uniform_hn14 (reused).
- A2: weighted_hn14_risk_only (new training).
- A3: weighted_hn14_risk_consistency (new training).
- A4: weighted_hn14_risk_consistency_density (new training).

## Score Definition
- alpha: `2.00`
- kappa: `2.00`
- density_k: `15`
- A2 score: `S = R`
- A3 score: `S = sqrt(R * C)`
- A4 score: `S = (R * C * D)^(1/3)`

## Budget Alignment
- fixed pool: `250` hard-normal candidates
- fixed replay budget: `151` extra normal replays, matched to uniform_hn14

## Key Result
- best lite setting: `uniform_hn14` (formal rank `1`)
- A4 vs A1 delta Spec@R99.5: `-0.0833`
- A4 vs A1 delta Spec@R99.0: `-0.0714`
- A4 vs A1 delta Prec@R99.0: `-0.0124`
- A4 vs A1 delta PTR@R99.0: `+0.0119`

## Interpretation
- A4 better than uniform_hn14 on Spec@R99.5 with limited Spec@R99.0 damage: `False`
- A3/A4 outperform or match A2 on the primary metric: `True`
- overlap reference file: `research/results/stage1_formal/gate_info_sampling_lite/appendix/table_uniform_vs_weighted_overlap.csv`

## Next Step Recommendation
- L3: revisit teacher choice, fixed-pool width, or whether consistency/density over-suppressed true hard negatives before attempting the full version.
