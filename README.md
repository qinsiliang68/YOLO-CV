# YOLO-CV

YOLO-CV is the parent academic research workspace for comparing, modifying, and testing multiple YOLO generations side by side.

## Active Research Folders

- `YOLOv11/`
- `YOLOv12/`
- `YOLOv13/`
- `YOLOv26/`

Each version folder includes:

- original upstream source snapshot
- `scripts/setup.ps1`
- `scripts/train.ps1`
- `scripts/val.ps1`
- `scripts/test.ps1`
- `scripts/predict.ps1`
- `configs/datasets/`
- `configs/runtime/*.json`
- `datasets/`
- `weights/`
- `runs/`

## Recommended Workflow

1. Open the version folder you want to study.
2. Run `.\scripts\setup.ps1`.
3. Put your dataset into that version folder's `datasets/`.
4. Create your own dataset YAML under `configs/datasets/`.
5. Start with `.\scripts\train.ps1`, then use `val/test/predict` as needed.

## Upstream Pinning

- `YOLOv11`: `ultralytics/ultralytics` at `v8.3.0`
- `YOLOv12`: `sunsmarterjie/yolov12` main snapshot
- `YOLOv13`: `iMoonLab/yolov13` main snapshot
- `YOLOv26`: `ultralytics/ultralytics` at `v8.4.0`

See each version folder's `UPSTREAM.md` for pinned commit details.
