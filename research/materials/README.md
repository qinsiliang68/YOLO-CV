# Classification Raw Materials

This directory stores pushed experiment materials for the YOLO11 classification baselines.

Formal stage-1 thesis-facing materials now belong under:

- `research/materials/stage1_formal/`

Legacy exploratory materials should migrate into:

- `research/archive/stage1_preformal_legacy/`

Included:
- `run_master.csv`: one-line summary per baseline run
- `yolo11*_cls6_train7200/`: per-run raw materials and copied artifacts

Each run directory keeps:
- `run_manifest.json`: config snapshot, paths, class names, collection metadata
- `env_info.json` and `pip_freeze.txt`: environment snapshot
- `epoch_metrics.csv`: epoch-level metrics and runtime fields
- `val_predictions.csv`: per-image validation predictions with full probabilities
- `val_summary.json`: aggregate metrics, per-class metrics, confusion data
- `threshold_sweep.csv` and `threshold_operating_points.json`: binary gate threshold analysis
- `roc_curve.csv`, `pr_curve.csv`, `calibration_curve.csv`: curve raw data
- `fp_normal.csv`, `fn_abnormal.csv`, `misclassified_samples.csv`, `hard_examples_topk.csv`: error-analysis materials
- `val_embeddings.npy` and `val_embeddings_index.csv`: validation embedding exports
- `raw_run_artifacts/`: copied training outputs such as `args.yaml`, `results.csv`, `results.png`, confusion matrices, batch previews, and `training_runtime.json`

Some gate runs may also include a `calibration_ts/` subdirectory with:
- `temperature_scaling.json`: fitted temperature and split metadata
- `val_cal.csv` and `val_op.csv`: stratified calibration/evaluation split records
- `val_op_predictions_calibrated.csv`: calibrated evaluation probabilities
- `threshold_sweep_before.csv` and `threshold_sweep_calibrated.csv`
- `threshold_operating_points_before.json` and `threshold_operating_points_calibrated.json`
- `calibration_curve_before.csv` and `calibration_curve_after.csv`
- exported comparison plots for reliability, ECE/Brier, and threshold sweep

Excluded from Git:
- original datasets
- large model weights such as `*.pt`
- local temporary or auth-diagnostic files
