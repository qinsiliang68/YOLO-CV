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
- active sample-value direction: conditional sample value under training state,
  dynamic replay scheduling, weak-defect tail protection, and seed-stability
  experiments
- active campaign deadline: `2026-09-10`, when access to the ten training
  machines ends

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
| `artifacts/stage1_sample_value_experiments/experiments/oof_dynamics_gap_value_20260708/` | CURRENT_SOURCE_EVIDENCE | Complete 10-fold, 200-epoch OOF dynamics plus the prior 240-run replay evidence. Use as immutable source evidence, not as the output root of the new campaign. |
| `artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/` | ACTIVE_CAMPAIGN | Isolated field audit, literature review, preregistration, queue, process exports, evaluation, and reports for the pre-September-10 campaign. |
| `artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/03_preregistration_v2/` | ACTIVE_PREREGISTRATION | Frozen 30-seed, four-cycle protocol. Cycle 1/2 contain 80 executable logical runs and 296 physical segment jobs; Cycle 3/4 remain gated templates. |
| `artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/04_run_queue_v2/` | ACTIVE_RUN_QUEUE | Validated v2 queue consumed by all formal runtime entrypoints. The unversioned `03_preregistration/` and `04_run_queue/` siblings are immutable v1 evidence only. |

## Current Documentation

| Path | Status | Use |
| --- | --- | --- |
| `docs/stage1_oof_10fold.md` | CURRENT_OOF_PLAN | How the 10 OOF folds were built and how fold training is launched. |
| `docs/stage1_oof_200epoch_archives_20260621.md` | CURRENT_OOF_ARCHIVES | What has finished on the training nodes and how to export OOF predictions. |
| `docs/stage1_gapvalue240/DYNAMIC_REPLAY_CAMPAIGN_FIELD_AUDIT.md` | ACTIVE_CAMPAIGN_AUDIT | Field sufficiency contract, storage accounting rules, generation commands, and launch gates for the new campaign. |
| `docs/stage1_gapvalue240/DYNAMIC_REPLAY_CAMPAIGN_OPERATOR_BRIEFING_V2.md` | ACTIVE_OPERATOR_BRIEFING | End-to-end scientific background, frozen cycle matrix, run outputs, monitoring checklist, and operator stop conditions for the ten-machine campaign. |
| `docs/stage1_gapvalue240/DYNAMIC_REPLAY_CAMPAIGN_OPERATIONS_V2.md` | ACTIVE_OPERATIONS | Formal v2 generation, release, assignment, single-job worker, recovery, and ten-machine canary procedure. |
| `docs/stage1_gapvalue240/RUN_QUEUE_V2_README.md` | ACTIVE_QUEUE_CONTRACT | Queue-v2 identity, release, assignment, and historical-v1 separation rules. |

## Current Scripts

| Path | Status | Use |
| --- | --- | --- |
| `scripts/build_stage1_oof_folds.py` | CURRENT_OOF_SPLIT_BUILDER | Builds group-disjoint OOF fold manifests. |
| `scripts/run_stage1_oof_folds_20260617.py` | CURRENT_OOF_TRAIN_WRAPPER | Runs OOF fold training jobs. |
| `scripts/predict_stage1_oof_folds_20260621.py` | CURRENT_OOF_PREDICT_EXPORTER | Predicts each fold's held-out images after fitting cal on `val_cal` and selecting op threshold on `val_op`; writes cal/op difficulty coordinates. |
| `scripts/validate_stage1_oof_predictions_calop_20260621.py` | CURRENT_OOF_PREDICT_VALIDATOR | Fails raw-only or incomplete OOF prediction outputs before they are pushed or cited. |
| `scripts/evaluate_stage1_cls_gate.py` | CURRENT_BASELINE_EVALUATOR | Formal 1:5 baseline evaluator for val_cal, val_op, and test. |
| `scripts/stage1_gapvalue240/build_dynamic_replay_preregistration.py` | ACTIVE_PREREGISTRATION_BUILDER | Freezes the full GapCritical-Strict normal ranking into the percentage-based four-cycle protocol under `03_preregistration_v2/`. |
| `scripts/stage1_gapvalue240/build_dynamic_campaign_run_queue.py` | ACTIVE_QUEUE_BUILDER | Compiles the active preregistration into the validated `04_run_queue_v2/` physical job graph. |
| `scripts/stage1_gapvalue240/dynamic_campaign_train_worker.py` | ACTIVE_SINGLE_JOB_WORKER | Sole formal training process entrypoint. It requires exactly one released `--job-id` and consumes only queue v2. |
| `scripts/stage1_gapvalue240/build_dynamic_campaign_assignment.py` | ACTIVE_ASSIGNMENT_BUILDER | Produces versioned placement and one standalone command per released physical job. |
| `scripts/stage1_gapvalue240/run_dynamic_campaign_controller.py` | OPTIONAL_CONTROLLER | Dispatch convenience layer over the same release and assignment; it is not the only training entrypoint. |

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
