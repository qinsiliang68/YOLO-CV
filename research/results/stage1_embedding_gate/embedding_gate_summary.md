# Stage-1 Strong-Embedding Gate Summary

| Group | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | TN@R99.5 | FN@R99.5 | TN@R99.0 | FN@R99.0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 | current best yolo11l-cls + hn02 + P2 | 0.52381 | 0.559524 | 0.918322 | 0.89881 | 44 | 2 | 47 | 4 |
| H1 | contrastive backbone + calibration + plain score | 0.47619 | 0.559524 | 0.918322 | 0.89881 | 40 | 1 | 47 | 4 |
| H2 | contrastive backbone + calibration + trust gate | 0.452381 | 0.52381 | 0.912281 | 0.904762 | 38 | 2 | 44 | 4 |

- Best group: `H0`
- Verdict: `no_clear_gain`
- Detail: H2 does not clearly beat H0; stage-1 strong-embedding route is not yet worth extending.
