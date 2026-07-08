# Artifacts Scope

This directory mixes current paper-facing artifacts and supporting run evidence.
Use the status column before citing a result.

| Directory | Status | Meaning |
| --- | --- | --- |
| `stage1_cls_eval_1to5_20260617/` | CURRENT_BASELINE | Formal current 1:5 Stage-1 gate evaluation. Use this for current baseline metrics, threshold curves, and model comparison. |
| `stage1_cls_sweep_20260616/` | SUPPORTING_TRAIN_RUNS | Training outputs that produced the model weights used by the baseline. Useful for audit, but not the formal result table. |
| `stage1_oof_200epoch_archives_20260621/` | CURRENT_OOF_ARCHIVE_INDEX | Lightweight indexes for completed 200-epoch OOF folds 1-8. This proves training completion and archive integrity. It does not contain per-image difficulty predictions. |
| `stage1_oof_predictions_calop_20260621/` | CURRENT_OOF_PREDICTIONS_TARGET | Expected regenerated OOF prediction outputs after cal/op scoring with `scripts/predict_stage1_oof_folds_20260621.py`. |
| `stage1_oof_predictions_20260621/` | INVALID_RAW_ONLY | Raw-only OOF prediction outputs. Retain for audit only; do not cite for confidence, difficulty, sample value, or threshold conclusions. |
| `stage1_sample_value_experiments/` | CURRENT_SAMPLE_VALUE_EXPERIMENT_FAMILY | Grouped registry for sample-value exploration. It registers legacy 40-run/120-run replay studies as parallel experiments and keeps the active OOF dynamics experiment under a fixed stage/method/budget/run tree. |

Historical or non-current artifacts belong under `_recycle_bin/`, not here.

## Citation Rule

- For current model baseline numbers, cite `stage1_cls_eval_1to5_20260617/`.
- For OOF completion status, cite `stage1_oof_200epoch_archives_20260621/`.
- For OOF sample value or difficulty, cite only cal/op-validated outputs under
  `stage1_oof_predictions_calop_20260621/`.
- For the OOF training-dynamics sample-value line, use
  `stage1_sample_value_experiments/experiments/oof_dynamics_gap_value_20260708/`
  and follow `docs/stage1_sample_value_oof_dynamics_20260708.md`.
- Before pushing or citing OOF prediction outputs, run
  `scripts/validate_stage1_oof_predictions_calop_20260621.py`.
- Do not cite `stage1_cls_sweep_20260616/` as the final evaluation result.
