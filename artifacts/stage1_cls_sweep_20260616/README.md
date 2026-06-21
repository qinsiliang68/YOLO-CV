# Stage 1 Classification Sweep Artifacts - 2026-06-16

Status: `SUPPORTING_TRAIN_RUNS`

Use this directory as training-run evidence for the current baseline models.
Do not cite it as the formal model comparison or final evaluation result. For
current baseline metrics, use `artifacts/stage1_cls_eval_1to5_20260617/`.

Collected from LAN training nodes after the full YOLO11 classification sweep
completed.

Policy:

- checkpoint weights are intentionally excluded: no `*.pt` files are stored here
- large `train_log.txt` files are compressed as `train_log.txt.zip`
- raw run folders are grouped by node IP under `node-192.168.100.xx/`

Included runs:

| node | model | run | final epoch | final top1 | best epoch | best top1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 192.168.100.13 | yolo11l-cls | `full_yolo11l_cls_20260615-123305` | 200 | 0.94463 | 193 | 0.94488 |
| 192.168.100.18 | yolo11m-cls | `full_yolo11m_cls_20260615-180049` | 200 | 0.94292 | 193 | 0.94367 |
| 192.168.100.18 | yolo11n-cls | `full_yolo11n_cls_20260614-190411` | 200 | 0.93979 | 200 | 0.93979 |
| 192.168.100.18 | yolo11s-cls | `full_yolo11s_cls_20260615-063433` | 200 | 0.94292 | 200 | 0.94292 |
| 192.168.100.13 | yolo11x-cls | `full_yolo11x_cls_20260614-185818` | 200 | 0.94446 | 196 | 0.94483 |

Index files:

- `metrics_summary.csv`: one row per full model run
- `file_manifest.csv`: committed non-weight artifacts and file sizes

Source nodes:

- `192.168.100.13`: yolo11x, yolo11l
- `192.168.100.18`: yolo11n, yolo11s, yolo11m
