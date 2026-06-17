# Stage-1 CLS Gate Evaluation Artifacts

Collected from the two Windows training nodes after running `scripts/evaluate_stage1_cls_gate.py` at commit `78ab07a`.

Contents:

- `metrics_summary.csv`: combined metrics for all models and splits.
- `file_manifest.csv`: local file inventory with SHA-256 hashes.
- `node-*/eval_full_*/predictions_*.csv`: per-image prediction outputs for val_model, val_cal, val_op, and test.
- `node-*/eval_full_*/calibration.json`: Platt calibration parameters fitted on val_cal.
- `node-*/eval_full_*/threshold.json`: selected operational threshold from val_op.
- `node-*/eval_full_*/metrics_at_selected_threshold.csv`: per-run metrics.
- `node-*/eval_full_*/artifact_manifest.csv/json`: per-run artifact inventories produced on the training node.

No `.pt` weight files are included.
