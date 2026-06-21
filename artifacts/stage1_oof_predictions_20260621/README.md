# Stage-1 OOF Predictions, Folds 1-10

Status: `INVALID_RAW_ONLY`

This artifact set is kept only as an audit record. Do not use it for sample
value, confidence, difficulty, threshold, or paper-facing conclusions. It was
generated before the cal/op requirement was enforced: `p_defect_cal` and
`p_defect_operational` are empty, and the difficulty columns are raw-only.

Regenerate with:

```powershell
uv run python scripts\predict_stage1_oof_folds_20260621.py --folds <range> --fold-base 1 --dataset-root data\final_sewerml_dataset --oof-root artifacts\stage1_oof_folds_10fold_20260617 --runs-root <runs-root> --output-root artifacts\stage1_oof_predictions_calop_20260621\node-<ip> --device 0 --batch 64 --exist-ok

uv run python scripts\validate_stage1_oof_predictions_calop_20260621.py --prediction-root artifacts\stage1_oof_predictions_calop_20260621\node-<ip>
```

Created: 2026-06-21

This archive contains per-image out-of-fold predictions for the completed
Stage-1 OOF folds 1 through 10.

The prediction exporter used each fold's `weights/best.pt` to predict only that
fold's held-out manifests.  Difficulty is based on raw class probabilities:

```text
true_confidence_raw = p_defect_raw when y_true=1, otherwise p_normal_raw
wrong_confidence_raw = 1 - true_confidence_raw
```

The top-level `predictions_fold_01.csv` through `predictions_fold_10.csv` use
human fold numbers.  The node subdirectories preserve the raw exporter outputs,
whose filenames use zero-based code fold numbers.

## Node Sources

| Human folds | Node | Source output directory |
| ---: | --- | --- |
| 1-4 | `192.168.100.18` | `D:\ssh\AI\repos\YOLO-CV\artifacts\stage1_oof_predictions_20260621\node-192.168.100.18` |
| 5-8 | `192.168.100.13` | `F:\ssh\AI\repos\YOLO-CV\artifacts\stage1_oof_predictions_20260621\node-192.168.100.13` |
| 9-10 | `192.168.100.15` | `D:\ssh\AI\repos\YOLO-CV\artifacts\stage1_oof_predictions_20260621\node-192.168.100.15` |

## Row Counts

| Fold | Rows |
| ---: | ---: |
| 1 | 12036 |
| 2 | 12003 |
| 3 | 12039 |
| 4 | 12004 |
| 5 | 11987 |
| 6 | 11959 |
| 7 | 11997 |
| 8 | 11945 |
| 9 | 12001 |
| 10 | 12029 |
| Total | 120000 |

## Key Files

| File | Purpose |
| --- | --- |
| `predictions_fold_01.csv` ... `predictions_fold_10.csv` | Per-image OOF predictions for each completed human fold. |
| `oof_predictions_merged_10fold.csv` | Merged 10-fold prediction table. |
| `difficulty_summary_10fold.csv` | Counts by fold and raw difficulty bucket. |
| `wrong_confidence_hist_10fold.png` | Histogram using `wrong_confidence_raw`; the `0.4-0.6` band is the raw decision-boundary region. |
| `artifact_manifest_10fold.csv` / `.json` | SHA256 manifest for the archived prediction artifacts. |

The earlier 8-fold merged files are preserved for continuity:

```text
oof_predictions_merged_8fold.csv
difficulty_summary_8fold.csv
wrong_confidence_hist_8fold.png
artifact_manifest_8fold.csv
artifact_manifest_8fold.json
```

## Validation

```text
uv run --with pytest --with numpy --with scikit-learn --no-project python -m pytest tests/test_predict_stage1_oof_folds_20260621.py
4 passed

merged_rows=120000
summary_total=120000
expected_total=120000
```
