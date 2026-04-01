# YOLO-CV

This repository is the training workspace for the YOLOv11 sewer-defect research line.
Only code, configs, scripts, and notes are tracked in Git. Images, datasets, weights,
and run outputs stay local.

## Layout

- `YOLOv11/`: local YOLOv11 source tree plus runtime configs
- `scripts/`: workflow entrypoints, sync helpers, calibration and analysis scripts
- `research/`: experiment materials, result summaries, runbook and pipeline notes
- `essay/`: thesis sources, figures and generated PDF
- `data/`: local-only raw data and intermediate results, ignored by Git

## Synchronization Workflow

This repository uses a **single main branch** for both the local working machine and the training machine.

- Local working machine:
  - updates code, configs, scripts and thesis
  - pushes curated changes to `main`
- Training machine:
  - syncs from `main`
  - runs experiments
  - pushes only `research/materials` and `research/results`

Helper scripts:

- sync local machine to latest main  
  `powershell -ExecutionPolicy Bypass -File .\scripts\git_sync_main.ps1`
- push only experiment outputs  
  `powershell -ExecutionPolicy Bypass -File .\scripts\git_push_results_main.ps1`

## Training Machine Flow

1. `powershell -ExecutionPolicy Bypass -File .\scripts\git_sync_main.ps1`
2. `.\scripts\setup.ps1 -Backend cu128`
3. `.\scripts\check.ps1`
4. Move datasets into the fixed local-only paths described in `research/training_machine_runbook.md`
5. Run `uv run main.py --rerun`

## Main Entry Points

- Current root training entrypoint for the uniform five-scale CLS6 sweep: `uv run main.py --rerun`
- Compatibility wrapper for the same sweep: `uv run main_cls6_sweep.py --rerun`
- Source classification pretraining: `.\scripts\cls_pretrain.ps1`
- Extract the only active 3000-image source set directly from raw SewerML: `.\scripts\extract_sewerml_cls6_train3000.ps1 -Clean`
- Target classification fine-tuning: `.\scripts\cls_finetune_target.ps1`
- CAM export: `.\scripts\export_cam.ps1`
- CAM to pseudo boxes: `.\scripts\cam_to_pseudobox.ps1`
- Research figures: `.\scripts\plot_research_figures.ps1 -Mode sewerml`
- Detector training: `.\YOLOv11\scripts\train.ps1`
- Detector validation: `.\YOLOv11\scripts\val.ps1`
- Detector test: `.\YOLOv11\scripts\test.ps1`
- Detector predict: `.\YOLOv11\scripts\predict.ps1`

See `research/training_machine_runbook.md` for the fixed dataset paths and recommended commands.
