# table_hn_baseline_vs_hn_anchor_compare

| setting | best_epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | delta_vs_same_model_hn00 |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11m baseline (hn00) | 78 | 0.5952 | 0.6548 | 0.9348 | 0.8829 | baseline anchor |
| yolo11m + hn14 | 74 | 0.6429 | 0.6548 | 0.9350 | 0.8849 | delta Spec@R99.5=0.0476; delta Spec@R99.0=0.0000; delta Prec@R99.0=0.0001; delta PTR@R99.0=0.0020 |
| yolo11x baseline (hn00) | 125 | 0.5476 | 0.5952 | 0.9244 | 0.8929 | baseline anchor |
| yolo11x + hn02 | 1 | 0.5238 | 0.5476 | 0.9163 | 0.9008 | delta Spec@R99.5=-0.0238; delta Spec@R99.0=-0.0476; delta Prec@R99.0=-0.0081; delta PTR@R99.0=0.0079 |
