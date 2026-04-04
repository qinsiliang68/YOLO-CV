# Stage-1 PTSG Summary

- Temperature: `2.41607888`
- Alpha/Beta/Gamma: `1.0 / 1.0 / 0.5`
- Best variant: `P3` (p_abnormal + trust + uncertainty)

| Variant | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | ECE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | calibrated p_abnormal | 0.464286 | 0.583333 | 0.922395 | 0.894841 | 0.04544 | 0.060808 |
| P1 | p_abnormal + uncertainty | 0.464286 | 0.583333 | 0.922395 | 0.894841 | 0.40078 | 0.243328 |
| P2 | p_abnormal + trust | 0.452381 | 0.583333 | 0.922395 | 0.894841 | 0.497717 | 0.341276 |
| P3 | p_abnormal + trust + uncertainty | 0.47619 | 0.559524 | 0.918322 | 0.89881 | 0.495626 | 0.333456 |
| P4 | P3 + HN-aware normal bank | 0.47619 | 0.559524 | 0.918322 | 0.89881 | 0.495815 | 0.333647 |
