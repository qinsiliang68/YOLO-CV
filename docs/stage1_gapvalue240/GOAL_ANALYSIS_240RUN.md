# Stage1 GapValue 240-run goal analysis

## Scope

This analysis uses exactly the 240 canonical `VALIDATED` runs named by the
global inventory: 80 complete `T/R1/R2` triads and 48,000 epoch rows. It does
not mix legacy 40-run or 120-run studies, canaries, debug jobs, failed attempts,
or noncanonical retries.

The purpose is to test whether any observable selection or training-dynamics
rule can predict a true two-ended gain on unseen training seeds: more normal
images intercepted while defect misses do not increase, relative to both R1 and
R2. Fixed operating points are necessary but not sufficient; the final evidence
also audits raw-score FN=0–95 frontiers and fixed weak-defect/difficult-normal
tails so that threshold sliding is not called model improvement.

## Finalization command

Run only after all component analyses and the refined field ledger have passed:

```powershell
uv run python scripts/stage1_gapvalue240/finalize_goal_analysis.py `
  --report-root artifacts\stage1_sample_value_experiments\experiments\oof_dynamics_gap_value_20260708\06_reports\gapvalue240_goal_analysis_20260806_v1.inprogress `
  --inventory C:\baidunetdiskdownload\stage1_gapvalue240_extract_audit_20260728\GLOBAL_VALIDATED_RUN_INVENTORY.csv
```

The command refuses to overwrite an existing final report. It verifies the
numeric gates, field-usage gates, local HTML links, and SHA manifest before
atomically renaming the `.inprogress` directory.

## Evidence layers

- `audit/DATA_USAGE_LEDGER_REFINED.csv` and
  `audit/FIELD_VALUE_PROFILES.csv` classify and profile every discovered field.
- `tables/unified_triad_feature_matrix.csv` and the feature registries preserve
  availability time and prevent outcome, checkpoint, resource, lineage, and
  confound fields from leaking into pre-outcome prediction.
- `tables/joint_prediction_*` contains LOSO-seed, leave-selection-digest,
  double-exclusion, and Phase-C external-falsification results.
- `tables/reversal_*` holds exact-selection cross-seed reversal analyses.
- `tables/raw_frontier_*` and `tables/raw_cohort_mechanism_*` separate raw score
  mechanism evidence from Platt-calibrated deployment diagnostics.
- `tables/checkpoint_*`, `tables/training_*`, and
  `tables/confound_sensitivity_*` cover parameter drift, all 200 epochs,
  effective optimizer/LR/exposure, machine, resume, snapshot, and budget.
- `tables/LITERATURE_RESULT_MATRIX.csv` joins each preregistered primary source
  hypothesis to observed or missing capabilities and the final result boundary.

## Non-negotiable boundaries

- No no-replay arm exists, so replay versus no replay is not testable.
- No blind/external test exists; conclusions are internal to `val_op`.
- Per-sample gradients, intermediate checkpoints around epoch 150, per-step
  minibatch order, augmentation realizations, and per-epoch `val_op`
  predictions were not retained. They are documented as not testable rather
  than inferred from loss curves.
- R1 and R2 remain separate. R2 is a high-overlap near-treatment control and its
  effective unique contrast is always reported.
- AUC is a discrimination diagnostic, not an 80% success probability.

## Asset lifecycle

The extracted machine uploads and frozen selection files remain read-only. The
final report is a reproducible experiment output and contains only derived
tables, charts, documents, audits, and hashes. Raw predictions and checkpoints
are not copied into Git.
