# Stage-1 Max-Filter Suite Summary

| Label | Best Variant | Best Spec@R99.5 | Best Spec@R99.0 | Best Prec@R99.0 | Best PTR@R99.0 | P0 Spec@R99.5 | P2 Spec@R99.5 | TN@R99.5 | FN@R99.5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 current best hn02 + P2 | P2 | 0.52381 | 0.559524 | 0.918322 | 0.89881 | 0.5 | 0.52381 | 44 | 2 |
| Selective / recall-constrained loss | P1 | 0.488095 | 0.52381 | 0.912473 | 0.906746 | 0.488095 | 0.464286 | 41 | 1 |
| Hard positive + hard normal mining | P0 | 0.52381 | 0.583333 | 0.922395 | 0.894841 | 0.52381 | 0.464286 | 44 | 2 |
| Weighted BCE | P0 | 0.488095 | 0.559524 | 0.918322 | 0.89881 | 0.488095 | 0.440476 | 41 | 2 |
| Focal BCE | P2 | 0.440476 | 0.535714 | 0.914286 | 0.902778 | 0.428571 | 0.440476 | 37 | 2 |
| Defect oversampling | P3 | 0.404762 | 0.535714 | 0.914286 | 0.902778 | 0.380952 | 0.380952 | 34 | 2 |

- Best row: `Hard positive + hard normal mining`
- Verdict: `new_experiment_wins`
