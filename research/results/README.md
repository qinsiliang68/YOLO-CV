# Results Layout

- Keep only compact artifacts that are useful to sync across machines.
- Do not commit raw datasets, checkpoints, large logs, or full `YOLOv11/runs` directories.
- Store each completed experiment under its own subdirectory, for example `research/results/source_cls6_train7200/`.

Formal stage-1 thesis-facing summaries now belong under:

- `research/results/stage1_formal/`

Preformal exploratory summaries should be archived under:

- `research/archive/stage1_preformal_legacy/`

Recommended contents per run:

- `summary.md`: short human-readable result summary
- `metrics.json`: machine-readable key metrics
- `train_curves.png`: compact training curves figure
- `confusion_matrix_normalized.png`: compact normalized confusion matrix
- `threshold_sweep.txt`: threshold scan with TP/FN/FP/TN and derived metrics
