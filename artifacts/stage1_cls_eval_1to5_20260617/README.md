# Stage-1 CLS Gate 1:5 Evaluation Artifacts

Status: `CURRENT_BASELINE`

Use this directory for the current paper-facing Stage-1 1:5 baseline metrics.
This is the formal model comparison and thresholded evaluation artifact for the
current research line.

Generated on 2026-06-17 from the final 1:5 normal evaluation manifests
introduced in commit `d841de3`.

This is an evaluation-only rerun. No model weights were retrained. The trained
`yolo11n/s/m/l/x` checkpoints were reused, while `val_cal`, `val_op`, and
`test` were rerun with the corrected evaluation ratio:

- defect: 20,000 images per split
- normal: 100,000 images per split
- total: 120,000 images per split

Contents:

- `metrics_summary.csv`: combined metrics for all models and all rerun splits
- `test_summary.csv`: test-only model comparison table
- `status.json`: runner status, dataset count checks, model run names, and exit codes
- `file_manifest.csv`: local artifact inventory with SHA-256 hashes
- `node-192.168.100.18/eval_1to5_full_*/predictions_*.csv`: per-image predictions for `val_cal`, `val_op`, and `test`
- `node-192.168.100.18/eval_1to5_full_*/calibration.json`: Platt calibration parameters fitted on 1:5 `val_cal`
- `node-192.168.100.18/eval_1to5_full_*/threshold.json`: selected operational threshold from 1:5 `val_op`
- `node-192.168.100.18/eval_1to5_full_*/metrics_at_selected_threshold.csv`: per-run metrics
- `logs/`: runner logs, summaries, and PowerShell runner scripts used on the evaluation node

All five model jobs completed with exit code `0`. Each prediction CSV has
120,001 lines including the header.

Test split headline metrics:

| model | threshold | recall | specificity | precision | fp | tn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo11n | 0.0036587383 | 0.99500 | 0.64114 | 0.35672 | 35,886 | 64,114 |
| yolo11s | 0.0028321807 | 0.99645 | 0.63036 | 0.35029 | 36,964 | 63,036 |
| yolo11m | 0.0035858426 | 0.99505 | 0.68340 | 0.38597 | 31,660 | 68,340 |
| yolo11l | 0.0035432747 | 0.99525 | 0.68253 | 0.38537 | 31,747 | 68,253 |
| yolo11x | 0.0030557117 | 0.99535 | 0.67087 | 0.37688 | 32,913 | 67,087 |

Historical note: old 1:1 evaluation artifacts are non-current and should not be
used for final calibration, operating threshold selection, or final test
comparison.
