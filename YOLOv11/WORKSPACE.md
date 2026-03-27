# YOLOv11 Research Workspace

This directory is the active training version inside `YOLO-CV`.
The environment is managed by the root `uv` project instead of `venv + pip`.

## Quick Start

```powershell
..\scripts\setup.ps1 -Backend cu128
.\scripts\train.ps1
.\scripts\val.ps1
.\scripts\test.ps1
.\scripts\predict.ps1
```

## Fixed Runtime Defaults

- Detection train config: `configs/runtime/train_detect_struct6_reviewed.json`
- Detection validation config: `configs/runtime/val_detect_struct6_reviewed.json`
- Detection prediction config: `configs/runtime/predict_detect_struct6_reviewed.json`
- Detection dataset YAMLs:
  - `configs/datasets/struct6_detect_reviewed.yaml`
  - `configs/datasets/struct6_detect_pseudo.yaml`

## Local-Only Data Paths

- `datasets/sewerml_hla_cls3_focus/`
- `datasets/sewerml_hla_cls6_focus/`
- `datasets/struct6_cls_target/`
- `datasets/struct6_det_pseudo/`
- `datasets/struct6_det_reviewed/`

These folders are ignored by Git. Move the data into place on the training machine, then run the scripts.
