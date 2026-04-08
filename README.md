# YOLO-CV

This repository is the training workspace for the YOLOv11 sewer-defect research line.
Only code, configs, scripts, and notes are tracked in Git. Images, datasets, weights,
and run outputs stay local.

## Layout

- `PROJECT_MEMORY.md`: long-term project memory and stable decisions
- `YOLOv11/`: local YOLOv11 source tree plus runtime configs
- `scripts/`: workflow entrypoints, sync helpers, calibration and analysis scripts
- `research/`: experiment materials, result summaries, runbook, project memory and pipeline notes
- `essay/`: thesis sources, figures and generated PDF
- `data/`: local-only raw data and intermediate results, ignored by Git

## Project Memory

Before editing code, thesis, configs, or long-running experiment scripts, read:

- `PROJECT_MEMORY.md`
- `research/project_memory/stage1_formal_protocol.md`
- `research/project_memory/decision_log.md`
- `research/archive/stage1_preformal_legacy/archive_manifest.md`

These files store the current thesis direction, the formal stage-1 rules, stable decisions, and the traceable location of archived legacy notes.

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
5. Run `uv run main.py --task ...` for the required formal task

## Main Entry Points

- Unified launcher: `uv run main.py`
- Formal binary gate capacity scan: `uv run main.py --task stage1_formal_gate_capacity`
- Formal six-class source capacity scan: `uv run main.py --task stage1_formal_cls6_capacity`
- Formal HN sweep on `yolo11m`: `uv run main.py --task stage1_formal_gate_hn_m_sweep`
- Formal HN cross-check on `yolo11x`: `uv run main.py --task stage1_formal_gate_hn_x_crosscheck`
- Combined formal HN launcher: `uv run main.py --task stage1_formal_gate_hn_all`
- Formal RCD-Lite from the `yolo11m + hn14` anchor: `uv run main.py --task stage1_formal_gate_rcd_lite`
- Run the uniform five-scale CLS6 sweep explicitly: `uv run main.py --task cls6_sweep --rerun`
- Active task selector: `YOLOv11/configs/runtime/main_entry.json`
- Current committed default task: `stage1_formal_gate_rcd_lite`
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
