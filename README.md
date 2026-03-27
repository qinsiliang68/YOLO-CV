# YOLO-CV

This repository is the training workspace for the YOLOv11 sewer-defect research line.
Only code, configs, scripts, and notes are tracked in Git. Images, datasets, weights,
and run outputs stay local.

## Layout

- `YOLOv11/`: local YOLOv11 source tree plus runtime configs
- `scripts/`: root workflow entrypoints built around `uv`
- `research/`: label alignment, pipeline specs, and training-machine notes
- `data/`: local-only raw data and intermediate results, ignored by Git

## Training Machine Flow

1. `git pull`
2. `.\scripts\setup.ps1 -Backend cu128`
3. `.\scripts\check.ps1`
4. Move datasets into the fixed local-only paths described in `research/training_machine_runbook.md`
5. Start classification pretraining, CAM export, or detector training

## Main Entry Points

- Source classification pretraining: `.\scripts\cls_pretrain.ps1`
- Target classification fine-tuning: `.\scripts\cls_finetune_target.ps1`
- CAM export: `.\scripts\export_cam.ps1`
- CAM to pseudo boxes: `.\scripts\cam_to_pseudobox.ps1`
- Detector training: `.\YOLOv11\scripts\train.ps1`
- Detector validation: `.\YOLOv11\scripts\val.ps1`
- Detector test: `.\YOLOv11\scripts\test.ps1`
- Detector predict: `.\YOLOv11\scripts\predict.ps1`

See `research/training_machine_runbook.md` for the fixed dataset paths and recommended commands.
