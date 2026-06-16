# Stage-1 CLS Gate Evaluation

weights: `D:\ssh\AI\runs\stage1_cls_sweep\full_yolo11l_cls_20260615-123305\weights\best.pt`
dataset_root: `C:\Users\ASUS\Desktop\ssh\AI\datasets\final_sewerml_dataset`
splits: `val_model,val_cal,val_op,test`
seed: `20260606`
imgsz: `224`
batch: `128`
device: `0`
target_recall: `0.995`
deployment_defect_prevalence: `0.1`
selected_threshold_column: `p_defect_operational`
selected_threshold: `0.0034749854`

Files:

- `predictions_*.csv`: per-image predictions with manifest identifiers.
- `calibration.json`: Platt calibration parameters fitted on val_cal.
- `threshold.json`: threshold selected on val_op.
- `metrics_at_selected_threshold.csv`: metrics for each split at the selected threshold.
- `run_config.json`: reproducibility snapshot for this evaluation run.
- `artifact_manifest.csv/json`: output file inventory with size and SHA-256.
