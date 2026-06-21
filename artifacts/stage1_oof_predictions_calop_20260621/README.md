# Stage-1 OOF Predictions, Cal/Op Required

Status: `CURRENT_OOF_PREDICTIONS_TARGET`

This directory is reserved for regenerated Stage-1 OOF prediction artifacts.
The previous raw-only directory `artifacts/stage1_oof_predictions_20260621/`
is invalid for confidence, sample-value, difficulty, or threshold conclusions.

Required rule:

```text
Any confidence/difficulty result must pass through cal and op.
Raw probabilities may remain as audit inputs, but raw confidence/difficulty
columns must not be used as conclusions.
```

## Node Commands

On `192.168.100.18` for folds 1-4:

```powershell
uv run python scripts\predict_stage1_oof_folds_20260621.py --folds 1-4 --fold-base 1 --dataset-root data\final_sewerml_dataset --oof-root artifacts\stage1_oof_folds_10fold_20260617 --runs-root D:\ssh\AI\runs\YOLOv11\stage1_oof_10fold --output-root artifacts\stage1_oof_predictions_calop_20260621\node-192.168.100.18 --device 0 --batch 64 --exist-ok

uv run python scripts\validate_stage1_oof_predictions_calop_20260621.py --prediction-root artifacts\stage1_oof_predictions_calop_20260621\node-192.168.100.18 --expected-folds 1-4 --fold-base 1
```

On `192.168.100.13` for folds 5-8:

```powershell
uv run python scripts\predict_stage1_oof_folds_20260621.py --folds 5-8 --fold-base 1 --dataset-root data\final_sewerml_dataset --oof-root artifacts\stage1_oof_folds_10fold_20260617 --runs-root F:\ssh\AI\runs\YOLOv11\stage1_oof_10fold --output-root artifacts\stage1_oof_predictions_calop_20260621\node-192.168.100.13 --device 0 --batch 64 --exist-ok

uv run python scripts\validate_stage1_oof_predictions_calop_20260621.py --prediction-root artifacts\stage1_oof_predictions_calop_20260621\node-192.168.100.13 --expected-folds 5-8 --fold-base 1
```

On `192.168.100.15` for folds 9-10:

```powershell
uv run python scripts\predict_stage1_oof_folds_20260621.py --folds 9-10 --fold-base 1 --dataset-root data\final_sewerml_dataset --oof-root artifacts\stage1_oof_folds_10fold_20260617 --runs-root D:\ssh\AI\runs\YOLOv11\stage1_oof_10fold --output-root artifacts\stage1_oof_predictions_calop_20260621\node-192.168.100.15 --device 0 --batch 64 --exist-ok

uv run python scripts\validate_stage1_oof_predictions_calop_20260621.py --prediction-root artifacts\stage1_oof_predictions_calop_20260621\node-192.168.100.15 --expected-folds 9-10 --fold-base 1
```

## Expected Files

Each node output should contain:

| File | Meaning |
| --- | --- |
| `predictions_fold_XX.csv` | OOF holdout predictions with cal/op confidence and operational difficulty columns. |
| `calibration_fold_XX.json` | Calibration fitted on global `val_cal`. |
| `threshold_fold_XX.json` | Operational threshold selected on global `val_op`. |
| `difficulty_summary_operational.csv` | Operational difficulty bucket counts. |
| `wrong_confidence_operational_hist.png` | Operational sample-value distribution. |
| `artifact_manifest.csv` / `.json` | File inventory and hashes. |

Do not push node outputs until validation prints `VALID_CAL_OP_OOF_PREDICTIONS`.
