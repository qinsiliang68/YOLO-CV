# Stage-1 PTSG Summary

- Temperature: `2.75167123`
- Alpha/Beta/Gamma: `1.0 / 1.0 / 0.5`
- Best variant: `P0` (calibrated p_abnormal)

| Variant | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | ECE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | calibrated p_abnormal | 0.52381 | 0.583333 | 0.922395 | 0.894841 | 0.023273 | 0.058354 |
| P1 | p_abnormal + uncertainty | 0.511905 | 0.583333 | 0.922395 | 0.894841 | 0.405649 | 0.242556 |
| P2 | p_abnormal + trust | 0.464286 | 0.559524 | 0.918322 | 0.89881 | 0.502279 | 0.337258 |
| P3 | p_abnormal + trust + uncertainty | 0.464286 | 0.583333 | 0.922395 | 0.894841 | 0.497062 | 0.33195 |
| P4 | P3 + HN-aware normal bank | 0.464286 | 0.583333 | 0.922395 | 0.894841 | 0.497419 | 0.332337 |
