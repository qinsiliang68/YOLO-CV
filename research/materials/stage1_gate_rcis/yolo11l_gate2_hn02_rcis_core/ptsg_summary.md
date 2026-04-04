# Stage-1 PTSG Summary

- Temperature: `2.42709267`
- Alpha/Beta/Gamma: `1.0 / 1.0 / 0.5`
- Best variant: `P1` (p_abnormal + uncertainty)

| Variant | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | ECE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | calibrated p_abnormal | 0.511905 | 0.607143 | 0.926667 | 0.892857 | 0.021566 | 0.054661 |
| P1 | p_abnormal + uncertainty | 0.52381 | 0.607143 | 0.926503 | 0.890873 | 0.412136 | 0.241338 |
| P2 | p_abnormal + trust | 0.47619 | 0.595238 | 0.924444 | 0.892857 | 0.502763 | 0.336728 |
| P3 | p_abnormal + trust + uncertainty | 0.511905 | 0.607143 | 0.926503 | 0.890873 | 0.496347 | 0.331442 |
| P4 | P3 + HN-aware normal bank | 0.511905 | 0.607143 | 0.926503 | 0.890873 | 0.496535 | 0.331646 |
