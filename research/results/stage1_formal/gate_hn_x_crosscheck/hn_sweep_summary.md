# formal yolo11x gate HN cross-capacity check

| Ratio | Backflow Count | Best Epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | Tau@R99.5 | Tau@R99.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hn00 | 0 | 125 | 0.547619 | 0.595238 | 0.924444 | 0.892857 | 0.2900 | 0.3600 |
| hn02 | 22 | 1 | 0.523810 | 0.547619 | 0.916300 | 0.900794 | 0.3100 | 0.3300 |

- Best ratio: `hn00`
- Best epoch: `125`
- Ranking rule: `Spec@R99.5 descending -> Spec@R99.0 descending -> Prec@R99.0 descending -> PTR@R99.0 ascending`
