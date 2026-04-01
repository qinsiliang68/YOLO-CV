# Stage-1 PTSG Summary

- Temperature: `2.30029358`
- Alpha/Beta/Gamma: `1.0 / 1.0 / 0.5`
- Best variant: `P0` (calibrated p_abnormal)

| Variant | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | ECE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | calibrated p_abnormal | 0.47619 | 0.559524 | 0.918322 | 0.89881 | 0.02527 | 0.053315 |
| P1 | p_abnormal + uncertainty | 0.464286 | 0.559524 | 0.918322 | 0.89881 | 0.407344 | 0.240547 |
| P2 | p_abnormal + trust | 0.452381 | 0.52381 | 0.912281 | 0.904762 | 0.501874 | 0.335654 |
| P3 | p_abnormal + trust + uncertainty | 0.464286 | 0.511905 | 0.910284 | 0.906746 | 0.494204 | 0.328779 |
| P4 | P3 + HN-aware normal bank | 0.464286 | 0.511905 | 0.910284 | 0.906746 | 0.494588 | 0.329197 |
