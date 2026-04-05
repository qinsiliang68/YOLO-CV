# Formal Binary Gate Capacity Summary

| Model | Best Epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | Tau@R99.5 | Tau@R99.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo11l-cls | 54 | 0.52381 | 0.571429 | 0.92053 | 0.89881 | 0.28 | 0.3 |
| yolo11m-cls | 78 | 0.595238 | 0.654762 | 0.934831 | 0.882937 | 0.28 | 0.31 |
| yolo11n-cls | 76 | 0.511905 | 0.583333 | 0.922395 | 0.894841 | 0.25 | 0.36 |
| yolo11s-cls | 77 | 0.52381 | 0.535714 | 0.914286 | 0.902778 | 0.25 | 0.26 |
| yolo11x-cls | 125 | 0.547619 | 0.595238 | 0.924444 | 0.892857 | 0.29 | 0.36 |

- Best formal gate model: `yolo11m-cls`
- Best epoch: `78`
- Ranking rule: `Spec@R99.5 -> Spec@R99.0 -> Prec@R99.0 -> PTR@R99.0 ascending`
