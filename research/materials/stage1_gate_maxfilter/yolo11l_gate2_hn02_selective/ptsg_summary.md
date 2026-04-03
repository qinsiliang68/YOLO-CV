# Stage-1 PTSG Summary

- Temperature: `2.35745985`
- Alpha/Beta/Gamma: `1.0 / 1.0 / 0.5`
- Best variant: `P1` (p_abnormal + uncertainty)

| Variant | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | ECE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | calibrated p_abnormal | 0.488095 | 0.52381 | 0.912281 | 0.904762 | 0.020921 | 0.053629 |
| P1 | p_abnormal + uncertainty | 0.488095 | 0.52381 | 0.912473 | 0.906746 | 0.410645 | 0.241409 |
| P2 | p_abnormal + trust | 0.464286 | 0.511905 | 0.910284 | 0.906746 | 0.50431 | 0.336309 |
| P3 | p_abnormal + trust + uncertainty | 0.488095 | 0.5 | 0.908297 | 0.90873 | 0.492735 | 0.330754 |
| P4 | P3 + HN-aware normal bank | 0.488095 | 0.5 | 0.908297 | 0.90873 | 0.493084 | 0.331121 |
