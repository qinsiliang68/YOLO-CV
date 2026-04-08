# formal yolo11x gate HN cross-capacity check

| Ratio | Backflow Count | Best Epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | Tau@R99.5 | Tau@R99.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hn00 | 0 | 125 | 0.547619 | 0.595238 | 0.924444 | 0.892857 | 0.2900 | 0.3600 |
| hn02 | 22 | 30 | 0.571429 | 0.571429 | 0.920705 | 0.900794 | 0.2000 | 0.2000 |

- Best ratio: `hn02`
- Best epoch: `30`
- Ranking rule: `Spec@R99.5 descending -> Spec@R99.0 descending -> Prec@R99.0 descending -> PTR@R99.0 ascending`
