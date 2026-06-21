from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predict_stage1_oof_folds_20260621 import (  # noqa: E402
    FoldJob,
    add_difficulty_columns,
    difficulty_bucket,
    metric_columns_for_rows,
    parse_fold_spec,
)


def test_parse_fold_spec_accepts_zero_based_ranges() -> None:
    assert parse_fold_spec("0-3,7", base=0) == [0, 1, 2, 3, 7]


def test_parse_fold_spec_accepts_human_one_based_ranges() -> None:
    assert parse_fold_spec("1-4,8", base=1) == [0, 1, 2, 3, 7]


def test_difficulty_bucket_boundaries() -> None:
    assert difficulty_bucket(0.95) == "confidently_wrong"
    assert difficulty_bucket(0.7) == "wrong_not_confident"
    assert difficulty_bucket(0.5) == "decision_boundary"
    assert difficulty_bucket(0.2) == "correct_not_confident"
    assert difficulty_bucket(0.05) == "confidently_correct"


def test_add_difficulty_columns_uses_cal_and_operational_confidence(tmp_path: Path) -> None:
    job = FoldJob(
        fold=3,
        manifest_dir=tmp_path / "manifests",
        weights=tmp_path / "best.pt",
        run_dir=tmp_path / "run",
    )
    rows = [
        {"y_true": "1", "p_defect_cal": "0.8000000000", "p_defect_operational": "0.7000000000"},
        {"y_true": "0", "p_defect_cal": "0.3000000000", "p_defect_operational": "0.9700000000"},
    ]

    enriched = add_difficulty_columns(rows, job, operational_threshold=0.6000000000)

    assert enriched[0]["wrong_confidence_cal"] == "0.2000000000"
    assert enriched[0]["difficulty_bucket_cal"] == "correct_not_confident"
    assert enriched[0]["wrong_confidence_operational"] == "0.3000000000"
    assert enriched[0]["difficulty_bucket_operational"] == "correct_not_confident"
    assert enriched[0]["y_pred_operational"] == "1"
    assert enriched[0]["operational_correct"] == "1"
    assert enriched[1]["wrong_confidence_cal"] == "0.3000000000"
    assert enriched[1]["wrong_confidence_operational"] == "0.9700000000"
    assert enriched[1]["difficulty_bucket_operational"] == "confidently_wrong"
    assert enriched[1]["y_pred_operational"] == "1"
    assert enriched[1]["operational_correct"] == "0"
    assert "wrong_confidence_raw" not in enriched[0]
    assert enriched[1]["oof_fold"] == "03"
    assert enriched[1]["human_fold"] == "4"


def test_add_difficulty_columns_rejects_missing_cal_op_scores(tmp_path: Path) -> None:
    job = FoldJob(
        fold=0,
        manifest_dir=tmp_path / "manifests",
        weights=tmp_path / "best.pt",
        run_dir=tmp_path / "run",
    )

    try:
        add_difficulty_columns([{"y_true": "1", "p_defect_cal": "", "p_defect_operational": ""}], job, operational_threshold=0.5)
    except ValueError as exc:
        assert "cal/op" in str(exc)
    else:
        raise AssertionError("missing cal/op scores should fail")


def test_metric_columns_for_rows_preserves_metrics_for_split_keys() -> None:
    rows = [
        {
            "split": "fold_00_oof_holdout",
            "score_column": "p_defect_operational",
            "threshold": "0.6000000000",
            "deployment_defect_prevalence": "0.100000",
            "n": "2",
            "positive_n": "1",
            "negative_n": "1",
            "tp": "1",
            "fp": "0",
            "tn": "1",
            "fn": "0",
            "recall": "1.0000000000",
            "specificity": "1.0000000000",
            "precision": "1.0000000000",
            "accuracy": "1.0000000000",
            "f1": "1.0000000000",
            "fpr": "0.0000000000",
            "predicted_positive_rate": "0.5000000000",
            "weighted_precision": "1.0000000000",
            "weighted_accuracy": "1.0000000000",
            "weighted_f1": "1.0000000000",
            "weighted_pass_through_rate": "0.1000000000",
        }
    ]

    columns = metric_columns_for_rows(rows)

    assert "n" in columns
    assert "positive_n" in columns
    assert "negative_n" in columns
    assert "predicted_positive_rate" in columns
    assert "weighted_precision" in columns
    assert "weighted_pass_through_rate" in columns
    assert "pass_through_rate" not in columns
    assert "review_rate" not in columns
