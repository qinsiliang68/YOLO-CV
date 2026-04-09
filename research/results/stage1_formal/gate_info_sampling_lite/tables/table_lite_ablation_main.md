# table_lite_ablation_main

| setting | teacher | pool | budget_anchor | best_epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | formal_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1_uniform_hn14 | reused uniform HN14 | fixed top250 hard-normal pool | hn14 (151 extras) | 74 | 0.6429 | 0.6548 | 0.9350 | 0.8849 | 1 |
| A0_hn00 | reused hn00 teacher | none | hn14 (151 extras) | 78 | 0.5952 | 0.6548 | 0.9348 | 0.8829 | 2 |
| A4_weighted_hn14_risk_consistency_density | hn00 best checkpoint | fixed top250 hard-normal pool | hn14 (151 extras) | 48 | 0.5595 | 0.5833 | 0.9226 | 0.8968 | 3 |
| A2_weighted_hn14_risk_only | hn00 best checkpoint | fixed top250 hard-normal pool | hn14 (151 extras) | 43 | 0.5595 | 0.5714 | 0.9204 | 0.8968 | 4 |
| A3_weighted_hn14_risk_consistency | hn00 best checkpoint | fixed top250 hard-normal pool | hn14 (151 extras) | 80 | 0.5238 | 0.6429 | 0.9327 | 0.8849 | 5 |

Notes:
- A0 and A1 are reused anchors from the completed formal HN sweep.
- A2/A3/A4 reuse the fixed top250 pool and the hn14-equivalent budget, and only change pool-internal replay probability.
