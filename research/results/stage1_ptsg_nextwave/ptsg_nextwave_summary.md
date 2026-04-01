# Stage-1 PTSG Next Wave Summary

- Temperature: `2.08121187`
- Alpha/Beta/Delta: `1.0 / 1.0 / 0.5`
- Best variant: `P2` (single-prototype trust)
- Verdict: `stop_at_stage1`
- Detail: P5/P6 do not clearly beat P2; multi-prototype trust does not add enough value in this round.

| Variant | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | TN@R99.5 | FN@R99.5 | TN@R99.0 | FN@R99.0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | single-prototype trust | 0.52381 | 0.559524 | 0.918322 | 0.89881 | 44 | 2 | 47 | 4 |
| P5a | K4 multi-prototype trust | 0.464286 | 0.571429 | 0.920354 | 0.896825 | 39 | 2 | 48 | 4 |
| P5b | K8 multi-prototype trust | 0.511905 | 0.559524 | 0.918502 | 0.900794 | 43 | 2 | 47 | 3 |
| P6a | K4 multi-prototype + margin trust | 0.488095 | 0.571429 | 0.920354 | 0.896825 | 41 | 2 | 48 | 4 |
| P6b | K8 multi-prototype + margin trust | 0.488095 | 0.559524 | 0.918502 | 0.900794 | 41 | 2 | 47 | 3 |

## Relative To P2

- `P5a`: Spec@R99.5 -0.0595, Spec@R99.0 +0.0119, Prec@R99.0 +0.0020, PTR@R99.0 -0.0020, TN@R99.5 -5, FN@R99.5 +0, TN@R99.0 +1, FN@R99.0 +0
- `P5b`: Spec@R99.5 -0.0119, Spec@R99.0 +0.0000, Prec@R99.0 +0.0002, PTR@R99.0 +0.0020, TN@R99.5 -1, FN@R99.5 +0, TN@R99.0 +0, FN@R99.0 -1
- `P6a`: Spec@R99.5 -0.0357, Spec@R99.0 +0.0119, Prec@R99.0 +0.0020, PTR@R99.0 -0.0020, TN@R99.5 -3, FN@R99.5 +0, TN@R99.0 +1, FN@R99.0 +0
- `P6b`: Spec@R99.5 -0.0357, Spec@R99.0 +0.0000, Prec@R99.0 +0.0002, PTR@R99.0 +0.0020, TN@R99.5 -3, FN@R99.5 +0, TN@R99.0 +0, FN@R99.0 -1
