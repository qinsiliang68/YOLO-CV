# Stage1 Phase1 HN/RN Node24 Deployment Record - 2026-06-23

Branch: `push-info-sampling-lite`

Deployed commit: `08cc6bb Tighten phase1 manifest value validation`

Formal entry scripts:

- `scripts/build_stage1_phase1_hn_rn_manifests_20260623.py`
- `scripts/validate_stage1_phase1_hn_rn_manifests_20260623.py`
- `scripts/run_stage1_phase1_hn_rn_pipeline_20260623.py`

## Node

- SSH alias: `node24`
- Training user: `ASUS`
- GPU: RTX 3090, CUDA torch environment checked
- Existing training process before deployment: none found
- Dataset path: `C:\Users\ASUS\Desktop\ssh\AI\datasets\final_sewerml_dataset`
- Dataset storage rule: real `C:\` path, not a reparse point
- Repo path: `C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV`
- Repo storage: junction to `D:\ssh\AI\repos\YOLO-CV`
- Formal phase root: `D:\ssh\AI\artifacts\stage1_phase1_hn_rn_20260623`
- Formal work root: `C:\Users\ASUS\Desktop\ssh\AI\phase1_workdirs_c\stage1_phase1_hn_rn_20260623`
- Formal runs root: `D:\ssh\AI\runs\stage1_phase1_hn_rn_20260623`
- Formal eval root: `D:\ssh\AI\artifacts\stage1_phase1_hn_rn_20260623\eval`

## Dataset Counts After Repair

Node24 was missing several normal-class splits. Missing normal images were transferred directly from node18 to node24 over LAN, without staging the dataset through the local workstation.

- `train`: 60000
- `normal_train`: 60000
- `val_model`: 12000
- `normal_val_model`: 12000
- `val_cal`: 20000
- `normal_val_cal`: 100000
- `val_op`: 20000
- `normal_val_op`: 100000
- `test`: 20000
- `normal_test`: 100000

## Weight

- Weight file: `yolo11l-cls.pt`
- Source: node18
- Destination: `C:\Users\ASUS\Desktop\ssh\AI\repos\YOLO-CV\yolo11l-cls.pt`
- SHA256: `6b56513a5d8bdae6b8f0a36dacaf01b26d5a522ba1b34197c3bac9fa6463366c`

## Formal Manifest Validation

Validation was run after dataset repair.

- Checked runs: 41
- Failed runs: 0
- Validation CSV: `D:\ssh\AI\artifacts\stage1_phase1_hn_rn_20260623\validation_summary_pretrain_after_sync.csv`
- Status: `PRETRAIN_VALIDATION_OK`

## Smoke Test

Smoke command used a one-epoch HN-01 run with evaluator limited to 64 images per class per split.

- Smoke root: `D:\ssh\AI\artifacts\stage1_phase1_hn_rn_smoke_node24_20260623_1ep_v2`
- Smoke work root: `C:\Users\ASUS\Desktop\ssh\AI\phase1_workdirs_c\stage1_phase1_hn_rn_smoke_node24_20260623_1ep_v2`
- Smoke runs root: `D:\ssh\AI\runs\stage1_phase1_hn_rn_smoke_node24_20260623_1ep_v2`
- Smoke eval root: `D:\ssh\AI\artifacts\stage1_phase1_hn_rn_smoke_node24_20260623_1ep_v2\eval`
- Train output: `D:\ssh\AI\runs\stage1_phase1_hn_rn_smoke_node24_20260623_1ep_v2\HN-01\full_yolo11l_cls_20260623-211308`
- Required weights produced: `weights\best.pt`, `weights\last.pt`
- Evaluator splits produced: `val_cal`, `val_op`, `test`
- Evaluator run dir: `D:\ssh\AI\artifacts\stage1_phase1_hn_rn_smoke_node24_20260623_1ep_v2\eval\HN-01\eval_HN-01_best`
- Reproducible outputs: prediction CSVs, artifact manifest CSV/JSON, metrics CSV
- Status: `SMOKE_OK`

## Formal Assignment

Node24 is assigned first as node-index 1 unless reassigned before launch:

- `HN-01`
- `RN-01`
- `HN-02`
- `RN-02`

Formal run requirements:

- fixed model: `yolo11l`
- evaluator splits: `val_cal,val_op,test`
- threshold split: `val_op`
- score column: `p_defect_operational`
- retain every `best.pt` and `last.pt`
- preserve CSV/JSON/log manifests; avoid collecting or copying large image outputs

