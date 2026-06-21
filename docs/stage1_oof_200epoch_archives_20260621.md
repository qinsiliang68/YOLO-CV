# Stage-1 OOF 200-Epoch Run Archives, Folds 1-8

Created: 2026-06-21

This note records the completed Stage-1 OOF 200-epoch runs for folds 1 through
8, the archive packages created from those runs, and the training/recovery
scripts that were actually present on the nodes.

The public archive labels use human fold numbers starting at 1. Paths under
`runs`, `checkpoint_archive`, and `artifacts/stage1_oof_folds_10fold_20260617`
still show the original training directory names because those are the exact
on-disk paths used by the training scripts.

Fold ownership:

| Fold | Node |
| ---: | --- |
| 1 | `192.168.100.18` |
| 2 | `192.168.100.18` |
| 3 | `192.168.100.18` |
| 4 | `192.168.100.18` |
| 5 | `192.168.100.13` |
| 6 | `192.168.100.13` |
| 7 | `192.168.100.13` |
| 8 | `192.168.100.13` |

All eight checked runs have `results.csv` with 200 rows, `last_epoch=200`,
`weights/best.pt`, and `weights/last.pt`. No matching training process was
running on either node at the time of inspection.

## Archive Policy

Archives were written only to non-C disks:

| Node | Archive root | Free space after archive |
| --- | --- | --- |
| `192.168.100.18` | `D:\ssh\AI\run_archives\stage1_oof_10fold_200epoch` | `D: 542.96 GB`, `C: 8.95 GB` |
| `192.168.100.13` | `F:\ssh\AI\run_archives\stage1_oof_10fold_200epoch` | `F: 117.51 GB`, `C: 41.93 GB` |

The archives are uncompressed `.tar` files for speed. Each archive includes:

- the completed run directory;
- the external `checkpoint_archive` directory for folds whose `epoch*.pt`
  checkpoints were moved during the C-drive incident;
- the fold-specific OOF manifest directory;
- global OOF fold metadata files when present;
- the relevant scripts, docs, and tests present on the node;
- a per-archive manifest with source paths, code hashes, epoch counts, and git
  status.

`tar.exe` printed `Removing leading drive letter from member names`; this is
the normal Windows bsdtar behavior when archiving absolute paths.

## Run And Archive Table

| Fold | Node | Run path | Run epoch ckpts | External epoch ckpts | Archive size GB | SHA256 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | `192.168.100.18` | `D:\ssh\AI\runs\YOLOv11\stage1_oof_10fold\fold_00\full_yolo11l_cls_20260617-214804` | 0 | 200 | 14.581 | `ACE3458A5352218EC8A0185D11383DC5EEAB32569E3D31AC7368E175EA4A61FB` |
| 2 | `192.168.100.18` | `D:\ssh\AI\runs\YOLOv11\stage1_oof_10fold\fold_01\full_yolo11l_cls_20260618-082829` | 22 | 178 | 14.507 | `A19B6B1B62029DFF1B034617E8F8B1A6361987DD78A961CD71FAC10242AFFE7C` |
| 3 | `192.168.100.18` | `D:\ssh\AI\runs\YOLOv11\stage1_oof_10fold\fold_02\full_yolo11l_cls_20260619-221302` | 200 | 0 | 14.581 | `EB050F2EC0489B07D8FA133BA4052E406849FAE786B3392B8BC98D0B182841DD` |
| 4 | `192.168.100.18` | `D:\ssh\AI\runs\YOLOv11\stage1_oof_10fold\fold_03\full_yolo11l_cls_20260620-085259` | 200 | 0 | 14.581 | `758230726F487A629E84388F41FFE5FA3F3221D4F90D0EBE57966A99EEA77D3B` |
| 5 | `192.168.100.13` | `F:\ssh\AI\runs\YOLOv11\stage1_oof_10fold\fold_04\full_yolo11l_cls_20260617-205648` | 0 | 200 | 14.581 | `CD2DE32055D229E8845274A487010E457DB7E76EE979D2DC81EFF663A8A45B9F` |
| 6 | `192.168.100.13` | `F:\ssh\AI\runs\YOLOv11\stage1_oof_10fold\fold_05\full_yolo11l_cls_20260618-075627` | 10 | 190 | 14.560 | `63122FE6BC608A00B481FB493CFFDD9959C3D43AAD20C3814FA3C9552E41BC8A` |
| 7 | `192.168.100.13` | `F:\ssh\AI\runs\YOLOv11\stage1_oof_10fold\fold_06\full_yolo11l_cls_20260619-213452` | 200 | 0 | 14.581 | `4CF7E2DA8A191B6554A42F01131C7010AE6DE0987E7D23B2347031E7CE707683` |
| 8 | `192.168.100.13` | `F:\ssh\AI\runs\YOLOv11\stage1_oof_10fold\fold_07\full_yolo11l_cls_20260620-084047` | 200 | 0 | 14.581 | `EC66BC6278735C4CBC418B6B68A1E5F7C0964C42A4D3FF5E64907695EE344BD5` |

Archive filenames:

```text
D:\ssh\AI\run_archives\stage1_oof_10fold_200epoch\stage1_oof_200epoch_192.168.100.18_fold_1_full_yolo11l_cls_20260617-214804.tar
D:\ssh\AI\run_archives\stage1_oof_10fold_200epoch\stage1_oof_200epoch_192.168.100.18_fold_2_full_yolo11l_cls_20260618-082829.tar
D:\ssh\AI\run_archives\stage1_oof_10fold_200epoch\stage1_oof_200epoch_192.168.100.18_fold_3_full_yolo11l_cls_20260619-221302.tar
D:\ssh\AI\run_archives\stage1_oof_10fold_200epoch\stage1_oof_200epoch_192.168.100.18_fold_4_full_yolo11l_cls_20260620-085259.tar
F:\ssh\AI\run_archives\stage1_oof_10fold_200epoch\stage1_oof_200epoch_192.168.100.13_fold_5_full_yolo11l_cls_20260617-205648.tar
F:\ssh\AI\run_archives\stage1_oof_10fold_200epoch\stage1_oof_200epoch_192.168.100.13_fold_6_full_yolo11l_cls_20260618-075627.tar
F:\ssh\AI\run_archives\stage1_oof_10fold_200epoch\stage1_oof_200epoch_192.168.100.13_fold_7_full_yolo11l_cls_20260619-213452.tar
F:\ssh\AI\run_archives\stage1_oof_10fold_200epoch\stage1_oof_200epoch_192.168.100.13_fold_8_full_yolo11l_cls_20260620-084047.tar
```

Each `.tar` has a sibling `.sha256` file in the same archive directory.

## Committed Small Materials

The large `.tar` files are not committed. The lightweight index files copied
from the archive step are committed under:

```text
artifacts/stage1_oof_200epoch_archives_20260621/
```

Key committed CSV files:

| File | Purpose |
| --- | --- |
| `stage1_oof_200epoch_archive_index_20260621.csv` | One row per completed fold archive, including archive path, size, SHA256, run path, epoch counts, and `best.pt` / `last.pt` checks. |
| `stage1_oof_200epoch_archive_sha256_20260621.csv` | SHA256 index copied from the sibling `.sha256` files. |
| `stage1_oof_200epoch_archive_sources_20260621.csv` | Expanded per-archive source list showing every run, checkpoint archive, manifest folder, metadata file, and code file included in each tar. |

Raw per-node summary CSVs, per-archive manifest TXT files, and raw `.sha256`
files are also committed in subdirectories under the same artifact folder.

## Node Code Audit

Both training nodes reported their local repo as branch
`push-info-sampling-lite` at git commit `97c5746`, with a dirty worktree. This
is expected: later repair/resume files were copied directly to the nodes because
node GitHub access was not reliable. The actual file hashes on both nodes match
the local committed files at `308f3c2 Fix OOF node layout and resume validation`.

Relevant code hashes observed on both nodes and locally:

| File | SHA256 |
| --- | --- |
| `scripts/build_stage1_oof_folds.py` | `89A2C5F209C5980E612AB7F46543462BCB694EF30586295C9362373AE4ACD0A5` |
| `scripts/run_stage1_oof_folds_20260617.py` | `9DAD28D74F3C7DDACD01E46094732DC0687E35271B8C751B2AC4F16CF827ABFD` |
| `scripts/continue_stage1_oof_node_20260619.py` | `9F686753F57C6A71AD1E3EF3299FB8FDF9A2576E6D8076A4752582304B9EC28A` |
| `scripts/train_stage1_cls_sweep.py` | `83FE5E15DAF98CD1CB3564DFFFCEE930077EC706C95CD369B7CCEE295A9E2846` |
| `scripts/repair_stage1_oof_node_layout_20260619.ps1` | `9EDFAE0BECFDAC909F46CCE776973A0BCD83DCB1D7A4DA9C46D67BD1927EBB8E` |
| `scripts/relocate_stage1_code_env_20260619.ps1` | `6F96FDD01D085E57E4998D120A06DCF826F43697CC926C468773F42BAA830C7A` |
| `scripts/inspect_stage1_oof_node_layout_20260619.ps1` | `EAD0AAF0A3BBAB1AEF8BAA2F59E1122FBABC2CE3DAE9F6BA232D8A35E8CDFEB7` |
| `scripts/validate_stage1_oof_continue_20260619.ps1` | `38C79FBDEFF5A51ABA9A64A8500514002EA54BE43315138980BC6EFF6AD3E9F8` |
| `scripts/archive_stage1_oof_runs_20260621.ps1` | `192647268C0C501EB6268424B70CC1141E9E27EDBB6A43A49623B8B8FB1D23C7` |

## Follow-Up

The next technical step is to run OOF prediction aggregation: load each fold's
best checkpoint, predict the held-out manifests, merge the held-out prediction
tables, and score difficult samples. This can be run before all 10 folds finish;
for the currently completed folds 1-8, run:

```powershell
uv run python scripts\predict_stage1_oof_folds_20260621.py --folds 1-8 --fold-base 1 --dataset-root data\final_sewerml_dataset --oof-root artifacts\stage1_oof_folds_10fold_20260617 --runs-root YOLOv11\runs\stage1_oof_10fold --output-root artifacts\stage1_oof_predictions_20260621 --device 0 --batch 64 --exist-ok
```

When folds 9-10 finish, either run only the new folds with `--folds 9-10
--fold-base 1` and a separate output directory, or rerun folds 1-10 into one
merged output directory. The important outputs are:

| File | Purpose |
| --- | --- |
| `predictions_fold_XX.csv` | Per-image OOF predictions for one held-out fold. |
| `oof_predictions_merged.csv` | Merged per-image OOF predictions across selected folds. |
| `difficulty_summary.csv` | Counts by difficulty bucket and fold. |
| `wrong_confidence_hist.png` | Histogram where `0.4-0.6` is the decision-boundary band and `>=0.9` is confidently wrong. |

The difficulty coordinate is computed from raw class probabilities, not
deployment-adjusted operational probabilities:

```text
true_confidence_raw =
  p_defect_raw if y_true=1
  p_normal_raw if y_true=0

wrong_confidence_raw = 1 - true_confidence_raw
```

`192.168.100.18` has very low C free space after archival (`8.95 GB`). This
document only records the state; no cleanup was performed during archival.
