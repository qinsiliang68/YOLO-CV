# Stage-1 PTSG Summary

- Temperature: `2.35837716`
- Alpha/Beta/Gamma: `1.0 / 1.0 / 0.5`
- Best variant: `P0` (calibrated p_abnormal)

| Variant | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | ECE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | calibrated p_abnormal | 0.488095 | 0.559524 | 0.918322 | 0.89881 | 0.020109 | 0.053107 |
| P1 | p_abnormal + uncertainty | 0.47619 | 0.547619 | 0.9163 | 0.900794 | 0.409981 | 0.241101 |
| P2 | p_abnormal + trust | 0.440476 | 0.535714 | 0.914286 | 0.902778 | 0.504429 | 0.336183 |
| P3 | p_abnormal + trust + uncertainty | 0.464286 | 0.52381 | 0.912281 | 0.904762 | 0.492804 | 0.330467 |
| P4 | P3 + HN-aware normal bank | 0.464286 | 0.52381 | 0.912281 | 0.904762 | 0.493157 | 0.330846 |
