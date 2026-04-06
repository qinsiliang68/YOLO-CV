# Stage-1 Gate Top1 vs Gate-Best Summary

| model | top1 best epoch | top1 best value | gate best epoch | same as gate best |
| --- | ---: | ---: | ---: | --- |
| yolo11n | 120 | 0.92500 | 76 | no |
| yolo11s | 88 | 0.92083 | 77 | no |
| yolo11m | 197 | 0.93472 | 78 | no |
| yolo11l | 103 | 0.93056 | 54 | no |
| yolo11x | 120 | 0.93194 | 125 | no |

- Same count: `0`
- Different count: `5`
- Conclusion: trainer `top1` and formal gate-aware best checkpoint are inconsistent for all five models.
