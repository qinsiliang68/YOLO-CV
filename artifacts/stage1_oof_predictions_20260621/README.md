# Stage-1 OOF Predictions, Folds 1-8

Created: 2026-06-21

This archive contains per-image out-of-fold predictions for the completed
Stage-1 OOF folds 1 through 8.

The prediction exporter used each fold's `weights/best.pt` to predict only that
fold's held-out manifests.  Difficulty is based on raw class probabilities:

```text
true_confidence_raw = p_defect_raw when y_true=1, otherwise p_normal_raw
wrong_confidence_raw = 1 - true_confidence_raw
```

The top-level `predictions_fold_01.csv` through `predictions_fold_08.csv` use
human fold numbers.  The node subdirectories preserve the raw exporter outputs,
whose filenames use zero-based code fold numbers.

## Node Sources

| Human folds | Node | Source output directory |
| ---: | --- | --- |
| 1-4 | `192.168.100.18` | `D:\ssh\AI\repos\YOLO-CV\artifacts\stage1_oof_predictions_20260621\node-192.168.100.18` |
| 5-8 | `192.168.100.13` | `F:\ssh\AI\repos\YOLO-CV\artifacts\stage1_oof_predictions_20260621\node-192.168.100.13` |

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
| Total | 95970 |

## Key Files

| File | Purpose |
| --- | --- |
| `predictions_fold_01.csv` ... `predictions_fold_08.csv` | Per-image OOF predictions for each completed human fold. |
| `oof_predictions_merged_8fold.csv` | Merged 8-fold prediction table. |
| `difficulty_summary_8fold.csv` | Counts by fold and raw difficulty bucket. |
| `wrong_confidence_hist_8fold.png` | Histogram using `wrong_confidence_raw`; the `0.4-0.6` band is the raw decision-boundary region. |
| `artifact_manifest_8fold.csv` / `.json` | SHA256 manifest for the archived prediction artifacts. |

## Validation

```text
uv run --with pytest --with numpy --with scikit-learn --no-project python -m pytest tests/test_predict_stage1_oof_folds_20260621.py
4 passed

merged_rows=95970
summary_total=95970
expected_total=95970
```
