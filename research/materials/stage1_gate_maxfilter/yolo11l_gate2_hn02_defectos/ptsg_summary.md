# Stage-1 PTSG Summary

- Temperature: `2.24338869`
- Alpha/Beta/Gamma: `1.0 / 1.0 / 0.5`
- Best variant: `P3` (p_abnormal + trust + uncertainty)

| Variant | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | ECE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | calibrated p_abnormal | 0.380952 | 0.559524 | 0.918322 | 0.89881 | 0.024104 | 0.05359 |
| P1 | p_abnormal + uncertainty | 0.369048 | 0.547619 | 0.9163 | 0.900794 | 0.40469 | 0.240992 |
| P2 | p_abnormal + trust | 0.380952 | 0.47619 | 0.904348 | 0.912698 | 0.498518 | 0.335147 |
| P3 | p_abnormal + trust + uncertainty | 0.404762 | 0.535714 | 0.914286 | 0.902778 | 0.492779 | 0.329825 |
| P4 | P3 + HN-aware normal bank | 0.404762 | 0.535714 | 0.914286 | 0.902778 | 0.493174 | 0.330246 |
