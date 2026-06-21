# Current Research Scope

This file is the source-of-truth map for the active research line.  If another
file conflicts with this scope map, use this file first and then verify the
specific artifact.

## Active Research Line

Current work is the final SewerML Stage-1 gate study:

- final sampled dataset under `data/final_sewerml_dataset/`
- target defect labels: `PF / DE / FS / RB / AF / OB`
- binary Stage-1 gate view: `target_defect` vs `no_target`
- formal baseline protocol: 1:5 defect-to-normal evaluation
- main model choice for follow-up experiments: `yolo11l`
- active sample-value direction: 10-fold OOF prediction, difficulty scoring,
  sample weighting, and resampling experiments

## Current Dataset Scope

Use these counts when discussing the current study:

| Scope | Images | Meaning |
| --- | ---: | --- |
| Effective local source pool | 759,586 | Local images available under `C:\Sewer-ML\sewerml_train_images`. |
| Final sampled dataset | 504,000 | All current train, validation, calibration, operation, and test images. |
| OOF training pool | 120,000 | `train` 60,000 defect images plus `normal_train` 60,000 normal images. |
| Unused source remainder | 255,586 | Effective source images not copied into the final sampled dataset. |

Size references from the local workspace:

| Scope | Size |
| --- | ---: |
| Final sampled image files | 100.006 GiB |
| Final sampled dataset directory | 100.181 GiB |
| OOF training image files | 25.607 GiB |
| Unused source remainder | 55.175 GiB |

## Current Artifacts

| Path | Status | Use |
| --- | --- | --- |
| `artifacts/stage1_cls_eval_1to5_20260617/` | CURRENT_BASELINE | Formal 1:5 Stage-1 gate baseline. Use for current model comparison and paper-facing baseline numbers. |
| `artifacts/stage1_cls_sweep_20260616/` | SUPPORTING_TRAIN_RUNS | Training-run evidence for baseline models. Do not treat this as the formal result table. |
| `artifacts/stage1_oof_200epoch_archives_20260621/` | CURRENT_OOF_ARCHIVE_INDEX | Completed OOF training archive indexes for folds 1-8. This proves folds trained; it is not the per-image difficulty table. |
| `artifacts/stage1_oof_predictions_calop_20260621/` | CURRENT_OOF_PREDICTIONS_TARGET | Expected output folder for regenerated OOF held-out prediction CSVs after cal/op scoring with `scripts/predict_stage1_oof_folds_20260621.py`. |
| `artifacts/stage1_oof_predictions_20260621/` | INVALID_RAW_ONLY | Raw-only OOF prediction artifact. Do not use for sample value, confidence, difficulty, threshold, or paper-facing conclusions. |

## Current Documentation

| Path | Status | Use |
| --- | --- | --- |
| `docs/stage1_oof_10fold.md` | CURRENT_OOF_PLAN | How the 10 OOF folds were built and how fold training is launched. |
| `docs/stage1_oof_200epoch_archives_20260621.md` | CURRENT_OOF_ARCHIVES | What has finished on the training nodes and how to export OOF predictions. |

## Current Scripts

| Path | Status | Use |
| --- | --- | --- |
| `scripts/build_stage1_oof_folds.py` | CURRENT_OOF_SPLIT_BUILDER | Builds group-disjoint OOF fold manifests. |
| `scripts/run_stage1_oof_folds_20260617.py` | CURRENT_OOF_TRAIN_WRAPPER | Runs OOF fold training jobs. |
| `scripts/predict_stage1_oof_folds_20260621.py` | CURRENT_OOF_PREDICT_EXPORTER | Predicts each fold's held-out images after fitting cal on `val_cal` and selecting op threshold on `val_op`; writes cal/op difficulty coordinates. |
| `scripts/validate_stage1_oof_predictions_calop_20260621.py` | CURRENT_OOF_PREDICT_VALIDATOR | Fails raw-only or incomplete OOF prediction outputs before they are pushed or cited. |
| `scripts/evaluate_stage1_cls_gate.py` | CURRENT_BASELINE_EVALUATOR | Formal 1:5 baseline evaluator for val_cal, val_op, and test. |

## Historical Or Non-Current Materials

These materials are retained for audit only.  Do not use them as current
results unless the user explicitly asks for history.

| Path | Status | Rule |
| --- | --- | --- |
| `_recycle_bin/pre_final_dataset_reset/` | HISTORICAL | Old v1/v3/stage-1 work before the final dataset reset. |
| `_recycle_bin/non_current_stage1_results_20260617-1905/` | HISTORICAL | Old non-current Stage-1 evaluation results, including older 1:1 style material. |
| `_recycle_bin/research_before_training_reset_20260612-152123/` | HISTORICAL | Pre-reset research material. |

## Local-Only Materials

These should stay local and should not be staged unless a later decision says
otherwise:

- raw source images: `C:\Sewer-ML\sewerml_train_images\`
- raw source annotation CSVs: `YOLOv11/datasets/sewerml_annotations/*.csv`
- final sampled image files under `data/final_sewerml_dataset/Det/images/`
- model checkpoints and large run directories unless represented by committed
  manifest/index files

## Answering Rule

When answering project questions, default to the current research scope above.
If using historical material, say so explicitly.  If a number can refer to more
than one scope, name the scope before giving the number.
