# Data schema and lineage

The package carries frozen copies of:

- `train_oof_assignments.csv` — authoritative mapping from `canonical_image_relpath` to label, fold, group, primary class, and source manifest;
- `sample_dynamics_summary.csv` — 120,000 sample-level summaries across 200 epochs;
- `sample_value_table.csv` — candidate value scores with class-specific NaN semantics;
- `epoch_gap_metrics.csv` — 200 global epoch metrics;
- `summary_input_manifest.csv` — the 2,000 raw OOF file list;
- `group_summary.csv` and validation metadata.

`master_sample_index.csv` is materialized from the authoritative assignment table. It is not manually maintained and cannot become a second identity source.

`oof_fold` is always read as a string and normalized to `00`–`09`. Epochs are one-based. The raw probability column is `p_defect_raw`. Column guessing is forbidden.

The raw 24M-row OOF files are not duplicated in this delivery; machine configuration binds their root. They are scanned once into a 200×120,000 float64 memmap, then all dynamic variants reuse that cache.
