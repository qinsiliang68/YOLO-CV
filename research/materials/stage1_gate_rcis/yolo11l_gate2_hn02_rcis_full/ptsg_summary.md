# Stage-1 PTSG Summary

- Temperature: `2.47993453`
- Alpha/Beta/Gamma: `1.0 / 1.0 / 0.5`
- Best variant: `P1` (p_abnormal + uncertainty)

| Variant | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | ECE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | calibrated p_abnormal | 0.52381 | 0.547619 | 0.916484 | 0.902778 | 0.020857 | 0.053123 |
| P1 | p_abnormal + uncertainty | 0.52381 | 0.571429 | 0.920354 | 0.896825 | 0.408045 | 0.241216 |
| P2 | p_abnormal + trust | 0.369048 | 0.547619 | 0.9163 | 0.900794 | 0.500357 | 0.337403 |
| P3 | p_abnormal + trust + uncertainty | 0.452381 | 0.547619 | 0.9163 | 0.900794 | 0.491087 | 0.331442 |
| P4 | P3 + HN-aware normal bank | 0.452381 | 0.547619 | 0.9163 | 0.900794 | 0.491269 | 0.331643 |
