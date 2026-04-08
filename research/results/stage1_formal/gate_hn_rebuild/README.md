# Stage-1 Formal HN Rebuild Materials

This directory stores the minimum structured materials needed to rebuild the formal HN dataset views without keeping `YOLOv11/datasets/stage1_formal_gate_hn` on disk.

## What is tracked here

- `duplication_csvs/m/*.csv`: `yolo11m` formal HN sweep duplicate lists
- `duplication_csvs/x/*.csv`: `yolo11x` formal HN cross-check duplicate lists
- `stage1_hn_rebuild_registry.csv/json/md`: registry for every tracked ratio
- `stage1_hn_rebuild_validation.json/md`: validation summary for a full rebuild test
- `scripts/rebuild_stage1_formal_hn_dataset.py`: rebuild entry point

## Rebuild rule

Each formal HN dataset view is defined as:

1. Mirror the full base dataset `YOLOv11/datasets/sewerml_gate2_train7200`
2. Read the corresponding duplication CSV
3. For every row, copy or hardlink `source_rel_path` to `target_rel_path`

The duplication CSVs already encode the exact selected hard-normal samples and the exact `_hn1` target filenames.

## Example

Rebuild `yolo11m hn14`:

```powershell
python scripts/rebuild_stage1_formal_hn_dataset.py `
  --source-dataset YOLOv11/datasets/sewerml_gate2_train7200 `
  --duplication-csv research/results/stage1_formal/gate_hn_rebuild/duplication_csvs/m/yolo11m_gate2_formal_hn14_duplications_only.csv `
  --output-dataset H:\rebuild_test\yolo11m_gate2_formal_hn14 `
  --link-mode copy
```

## Validation status

On April 8, 2026, every tracked HN dataset view was rebuilt from the base dataset plus these duplication CSVs and matched the formal `dataset_inventory.csv` exactly:

- missing files: `0`
- extra files: `0`
- file-size mismatches: `0`

See `stage1_hn_rebuild_validation.json` / `stage1_hn_rebuild_validation.md` for the per-ratio checks.
