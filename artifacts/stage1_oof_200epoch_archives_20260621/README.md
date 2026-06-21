# Stage-1 OOF 200-Epoch Archive Indexes

Status: `CURRENT_OOF_ARCHIVE_INDEX`

This directory records completed OOF training archives for folds 1-8. It proves
that the fold runs reached 200 epochs and that archive packages were created.

It does not contain per-image OOF prediction scores. For sample difficulty,
first run the OOF prediction exporter.

On `192.168.100.18` for folds 1-4:

```powershell
uv run python scripts\predict_stage1_oof_folds_20260621.py --folds 1-4 --fold-base 1 --dataset-root data\final_sewerml_dataset --oof-root artifacts\stage1_oof_folds_10fold_20260617 --runs-root D:\ssh\AI\runs\YOLOv11\stage1_oof_10fold --output-root artifacts\stage1_oof_predictions_20260621\node-192.168.100.18 --device 0 --batch 64 --exist-ok
```

On `192.168.100.13` for folds 5-8:

```powershell
uv run python scripts\predict_stage1_oof_folds_20260621.py --folds 5-8 --fold-base 1 --dataset-root data\final_sewerml_dataset --oof-root artifacts\stage1_oof_folds_10fold_20260617 --runs-root F:\ssh\AI\runs\YOLOv11\stage1_oof_10fold --output-root artifacts\stage1_oof_predictions_20260621\node-192.168.100.13 --device 0 --batch 64 --exist-ok
```

Important files:

| File | Meaning |
| --- | --- |
| `stage1_oof_200epoch_archive_index_20260621.csv` | One row per completed fold archive. |
| `stage1_oof_200epoch_archive_sha256_20260621.csv` | SHA-256 index for archive packages. |
| `stage1_oof_200epoch_archive_sources_20260621.csv` | Expanded source inventory for each archive. |
| `raw_node_summaries/` | Raw summary CSVs copied from training nodes. |
| `raw_archive_manifests/` | Per-archive manifest TXT files. |
| `raw_sha256/` | Raw SHA-256 sidecar files. |

Large `.tar` archives are intentionally not committed.
