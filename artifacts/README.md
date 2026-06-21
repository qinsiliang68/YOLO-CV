# Artifacts Scope

This directory mixes current paper-facing artifacts and supporting run evidence.
Use the status column before citing a result.

| Directory | Status | Meaning |
| --- | --- | --- |
| `stage1_cls_eval_1to5_20260617/` | CURRENT_BASELINE | Formal current 1:5 Stage-1 gate evaluation. Use this for current baseline metrics, threshold curves, and model comparison. |
| `stage1_cls_sweep_20260616/` | SUPPORTING_TRAIN_RUNS | Training outputs that produced the model weights used by the baseline. Useful for audit, but not the formal result table. |
| `stage1_oof_200epoch_archives_20260621/` | CURRENT_OOF_ARCHIVE_INDEX | Lightweight indexes for completed 200-epoch OOF folds 1-8. This proves training completion and archive integrity. It does not contain per-image difficulty predictions. |
| `stage1_oof_predictions_20260621/` | CURRENT_OOF_PREDICTIONS | Expected OOF prediction outputs after running `scripts/predict_stage1_oof_folds_20260621.py`. If absent, OOF difficulty histograms have not yet been exported. |

Historical or non-current artifacts belong under `_recycle_bin/`, not here.

## Citation Rule

- For current model baseline numbers, cite `stage1_cls_eval_1to5_20260617/`.
- For OOF completion status, cite `stage1_oof_200epoch_archives_20260621/`.
- For OOF sample difficulty, cite `stage1_oof_predictions_20260621/` only after
  the prediction exporter has been run and committed.
- Do not cite `stage1_cls_sweep_20260616/` as the final evaluation result.
