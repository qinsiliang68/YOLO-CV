# Stage-1 RCIS Suite Summary

| Label | Best Variant | Best Spec@R99.5 | Best Spec@R99.0 | Best Prec@R99.0 | Best PTR@R99.0 | P0 Spec@R99.5 | P2 Spec@R99.5 | TN@R99.5 | FN@R99.5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G4 current best HardMix + P0 | P0 | 0.52381 | 0.583333 | 0.922395 | 0.894841 | 0.52381 | 0.464286 | 44 | 2 |
| RCIS boundary-only sampling | P3 | 0.47619 | 0.559524 | 0.918322 | 0.89881 | 0.464286 | 0.452381 | 40 | 2 |
| RCIS core linear information sampling | P1 | 0.52381 | 0.607143 | 0.926503 | 0.890873 | 0.511905 | 0.47619 | 44 | 2 |
| RCIS full exploratory information sampling | P1 | 0.52381 | 0.571429 | 0.920354 | 0.896825 | 0.52381 | 0.369048 | 44 | 2 |

- Best row: `RCIS core linear information sampling`
- Verdict: `rcis_wins`
