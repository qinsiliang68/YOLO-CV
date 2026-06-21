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


def test_add_difficulty_columns_uses_raw_true_label_confidence(tmp_path: Path) -> None:
    job = FoldJob(
        fold=3,
        manifest_dir=tmp_path / "manifests",
        weights=tmp_path / "best.pt",
        run_dir=tmp_path / "run",
    )
    rows = [
        {"y_true": "1", "p_defect_raw": "0.9200000000", "p_normal_raw": "0.0800000000"},
        {"y_true": "0", "p_defect_raw": "0.9700000000", "p_normal_raw": "0.0300000000"},
    ]

    enriched = add_difficulty_columns(rows, job)

    assert enriched[0]["wrong_confidence_raw"] == "0.0800000000"
    assert enriched[0]["difficulty_bucket_raw"] == "confidently_correct"
    assert enriched[1]["wrong_confidence_raw"] == "0.9700000000"
    assert enriched[1]["difficulty_bucket_raw"] == "confidently_wrong"
    assert enriched[1]["oof_fold"] == "03"
    assert enriched[1]["human_fold"] == "4"
