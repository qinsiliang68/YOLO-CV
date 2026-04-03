# Stage-1 PTSG Summary

- Temperature: `1.98526164`
- Alpha/Beta/Gamma: `1.0 / 1.0 / 0.5`
- Best variant: `P2` (p_abnormal + trust)

| Variant | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | ECE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | calibrated p_abnormal | 0.428571 | 0.535714 | 0.914286 | 0.902778 | 0.035266 | 0.05646 |
| P1 | p_abnormal + uncertainty | 0.428571 | 0.547619 | 0.9163 | 0.900794 | 0.408438 | 0.238188 |
| P2 | p_abnormal + trust | 0.440476 | 0.535714 | 0.914286 | 0.902778 | 0.50523 | 0.339878 |
| P3 | p_abnormal + trust + uncertainty | 0.404762 | 0.511905 | 0.910284 | 0.906746 | 0.494836 | 0.327972 |
| P4 | P3 + HN-aware normal bank | 0.404762 | 0.511905 | 0.910284 | 0.906746 | 0.495196 | 0.328368 |
