# Table: Stage-1 Direct Binary Gate Formal Capacity Scan

| Model | Best Epoch | Spec@R99.5 (up) | Spec@R99.0 (up) | Prec@R99.0 (up) | PTR@R99.0 (down) | tau_R99.5 | tau_R99.0 | Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo11m-cls | 78 | **0.5952** | **0.6548** | **0.9348** | **0.8829** | 0.2800 | 0.3100 | 1 |
| yolo11x-cls | 125 | _0.5476_ | _0.5952_ | _0.9244_ | _0.8929_ | 0.2900 | 0.3600 | 2 |
| yolo11l-cls | 54 | 0.5238 | 0.5714 | 0.9205 | 0.8988 | 0.2800 | 0.3000 | 3 |
| yolo11s-cls | 77 | 0.5238 | 0.5357 | 0.9143 | 0.9028 | 0.2500 | 0.2600 | 4 |
| yolo11n-cls | 76 | 0.5119 | 0.5833 | 0.9224 | 0.8948 | 0.2500 | 0.3600 | 5 |

Note:
Binary gate is the primary stage-1 selection view.
Ranking follows `Spec@R99.5 > Spec@R99.0 > Prec@R99.0 > PTR@R99.0`.
