from __future__ import annotations

from scripts.validate_stage1_oof_predictions_calop_20260621 import parse_expected_folds, validate_rows


VALID_COLUMNS = [
    "human_fold",
    "y_true",
    "p_defect_cal",
    "p_defect_operational",
    "true_confidence_cal",
    "wrong_confidence_cal",
    "difficulty_bucket_cal",
    "true_confidence_operational",
    "wrong_confidence_operational",
    "difficulty_bucket_operational",
    "operational_threshold",
    "y_pred_operational",
    "operational_correct",
]


def test_validate_rows_accepts_cal_op_predictions() -> None:
    rows = [
        {
            "human_fold": "1",
            "y_true": "1",
            "p_defect_cal": "0.8000000000",
            "p_defect_operational": "0.7000000000",
            "true_confidence_cal": "0.8000000000",
            "wrong_confidence_cal": "0.2000000000",
            "difficulty_bucket_cal": "correct_not_confident",
            "true_confidence_operational": "0.7000000000",
            "wrong_confidence_operational": "0.3000000000",
            "difficulty_bucket_operational": "correct_not_confident",
            "operational_threshold": "0.6000000000",
            "y_pred_operational": "1",
            "operational_correct": "1",
        },
        {
            "human_fold": "1",
            "y_true": "0",
            "p_defect_cal": "0.3000000000",
            "p_defect_operational": "0.9700000000",
            "true_confidence_cal": "0.7000000000",
            "wrong_confidence_cal": "0.3000000000",
            "difficulty_bucket_cal": "correct_not_confident",
            "true_confidence_operational": "0.0300000000",
            "wrong_confidence_operational": "0.9700000000",
            "difficulty_bucket_operational": "confidently_wrong",
            "operational_threshold": "0.6000000000",
            "y_pred_operational": "1",
            "operational_correct": "0",
        },
    ]

    assert validate_rows(VALID_COLUMNS, rows) == []


def test_validate_rows_rejects_raw_only_legacy_predictions() -> None:
    columns = [
        "human_fold",
        "y_true",
        "p_defect_raw",
        "true_confidence_raw",
        "wrong_confidence_raw",
        "difficulty_bucket_raw",
        "p_defect_cal",
        "p_defect_operational",
    ]
    rows = [
        {
            "human_fold": "1",
            "y_true": "1",
            "p_defect_raw": "0.9000000000",
            "true_confidence_raw": "0.9000000000",
            "wrong_confidence_raw": "0.1000000000",
            "difficulty_bucket_raw": "confidently_correct",
            "p_defect_cal": "",
            "p_defect_operational": "",
        }
    ]

    errors = validate_rows(columns, rows)

    assert any("Missing required columns" in error for error in errors)
    assert any("legacy raw confidence" in error for error in errors)
    assert any("empty required cal/op" in error for error in errors)


def test_validate_rows_checks_operational_prediction_consistency() -> None:
    bad_row = {
        "human_fold": "2",
        "y_true": "0",
        "p_defect_cal": "0.2000000000",
        "p_defect_operational": "0.7000000000",
        "true_confidence_cal": "0.8000000000",
        "wrong_confidence_cal": "0.2000000000",
        "difficulty_bucket_cal": "correct_not_confident",
        "true_confidence_operational": "0.3000000000",
        "wrong_confidence_operational": "0.7000000000",
        "difficulty_bucket_operational": "wrong_not_confident",
        "operational_threshold": "0.6000000000",
        "y_pred_operational": "0",
        "operational_correct": "1",
    }

    errors = validate_rows(VALID_COLUMNS, [bad_row])

    assert any("y_pred_operational mismatch" in error for error in errors)
    assert any("operational_correct mismatch" in error for error in errors)


def test_validate_rows_rejects_missing_expected_fold() -> None:
    row = {
        "human_fold": "9",
        "y_true": "1",
        "p_defect_cal": "0.8000000000",
        "p_defect_operational": "0.7000000000",
        "true_confidence_cal": "0.8000000000",
        "wrong_confidence_cal": "0.2000000000",
        "difficulty_bucket_cal": "correct_not_confident",
        "true_confidence_operational": "0.7000000000",
        "wrong_confidence_operational": "0.3000000000",
        "difficulty_bucket_operational": "correct_not_confident",
        "operational_threshold": "0.6000000000",
        "y_pred_operational": "1",
        "operational_correct": "1",
    }

    errors = validate_rows(VALID_COLUMNS, [row], expected_human_folds={"9", "10"})

    assert any("Expected human folds" in error for error in errors)


def test_parse_expected_folds_accepts_human_ranges() -> None:
    assert parse_expected_folds("9-10", base=1) == {"9", "10"}


def test_parse_expected_folds_accepts_zero_based_ranges() -> None:
    assert parse_expected_folds("8-9", base=0) == {"9", "10"}
