# Stage-1 PTSG Summary

- Temperature: `2.08121187`
- Alpha/Beta/Gamma: `1.0 / 1.0 / 0.5`
- Best variant: `P2` (p_abnormal + trust)

| Variant | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | ECE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | calibrated p_abnormal | 0.5 | 0.547619 | 0.916484 | 0.902778 | 0.029802 | 0.059915 |
| P1 | p_abnormal + uncertainty | 0.5 | 0.547619 | 0.9163 | 0.900794 | 0.406394 | 0.241167 |
| P2 | p_abnormal + trust | 0.52381 | 0.559524 | 0.918322 | 0.89881 | 0.498581 | 0.334955 |
| P3 | p_abnormal + trust + uncertainty | 0.52381 | 0.559524 | 0.918322 | 0.89881 | 0.488623 | 0.324886 |
| P4 | P3 + HN-aware normal bank | 0.52381 | 0.559524 | 0.918322 | 0.89881 | 0.488834 | 0.3251 |
