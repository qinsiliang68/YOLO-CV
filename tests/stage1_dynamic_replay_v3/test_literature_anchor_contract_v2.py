from __future__ import annotations

import csv
from pathlib import Path

import pytest

from stage1_dynamic_replay_v3.literature_anchor_contract_v2 import (
    AnchorContractError,
    build_anchor_source_expected_rows,
    load_validated_anchor_work_ids,
    validate_anchor_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CONTRACT = REPO_ROOT / (
    "artifacts/stage1_sample_value_experiments/experiments/"
    "dynamic_replay_budget_efficiency_20260807/02_literature/"
    "review_500_300_100_v2/discovery/CORE_METHOD_ANCHORS_v2.csv"
)
REAL_BROAD_MEMBERSHIP = REPO_ROOT / (
    "artifacts/stage1_sample_value_experiments/experiments/"
    "dynamic_replay_budget_efficiency_20260807/02_literature/"
    "review_500_300_100_v2/staging/broad_freeze_v2/BROAD_500.csv"
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate(path: Path):
    return validate_anchor_contract(
        path,
        expected_count=40,
        broad_membership_path=REAL_BROAD_MEMBERSHIP,
    )


def test_real_anchor_contract_has_exact_auditable_40_row_structure() -> None:
    result = _validate(REAL_CONTRACT)

    assert result.status == "PASS"
    assert result.anchor_count == 40
    assert result.status_counts == {
        "BROAD_V2_ELIGIBLE": 19,
        "DISCOVERED_NOT_BROAD_SCREENED": 9,
        "EXTERNAL_PRIMARY_IDENTITY_VERIFIED_PENDING_SCREEN": 12,
    }
    assert result.formal_broad_increment == 0
    assert result.formal_screened_increment == 0
    assert result.formal_deep_increment == 0


def test_anchor_contract_rejects_duplicate_queue_identity(tmp_path: Path) -> None:
    rows = _read_rows(REAL_CONTRACT)
    rows[1]["queue_id"] = rows[0]["queue_id"]
    path = tmp_path / "anchors.csv"
    _write_rows(path, rows)

    with pytest.raises(AnchorContractError, match="duplicate queue_id"):
        _validate(path)


def test_anchor_contract_rejects_canonical_id_not_derived_from_identity(
    tmp_path: Path,
) -> None:
    rows = _read_rows(REAL_CONTRACT)
    target = next(row for row in rows if row["current_status"] != "BROAD_V2_ELIGIBLE")
    target["canonical_work_id"] = "CW00000000000000000000"
    path = tmp_path / "anchors.csv"
    _write_rows(path, rows)

    with pytest.raises(AnchorContractError, match="canonical_work_id mismatch"):
        _validate(path)


def test_existing_broad_anchor_must_match_frozen_broad_membership(
    tmp_path: Path,
) -> None:
    rows = _read_rows(REAL_CONTRACT)
    target = next(row for row in rows if row["current_status"] == "BROAD_V2_ELIGIBLE")
    target["canonical_work_id"] = "CW00000000000000000000"
    path = tmp_path / "anchors.csv"
    _write_rows(path, rows)

    with pytest.raises(AnchorContractError, match="frozen BROAD membership mismatch"):
        _validate(path)


def test_anchor_contract_rejects_non_primary_or_insecure_source(tmp_path: Path) -> None:
    rows = _read_rows(REAL_CONTRACT)
    rows[0]["primary_url"] = "http://example.invalid/secondary-summary"
    path = tmp_path / "anchors.csv"
    _write_rows(path, rows)

    with pytest.raises(AnchorContractError, match="primary_url"):
        _validate(path)


def test_anchor_contract_requires_contiguous_frozen_anchor_ids(tmp_path: Path) -> None:
    rows = _read_rows(REAL_CONTRACT)
    rows[-1]["anchor_id"] = "A099"
    path = tmp_path / "anchors.csv"
    _write_rows(path, rows)

    with pytest.raises(AnchorContractError, match="anchor_id sequence"):
        _validate(path)


def test_anchor_contract_does_not_grant_reading_credit() -> None:
    rows = _read_rows(REAL_CONTRACT)

    assert not any(
        field in row
        for row in rows
        for field in ("broad_credit", "screened_credit", "deep_credit")
    )


def test_anchor_source_expected_rows_include_only_missing_source_strata() -> None:
    rows = build_anchor_source_expected_rows(
        REAL_CONTRACT,
        broad_membership_path=REAL_BROAD_MEMBERSHIP,
    )

    assert len(rows) == 21
    assert {row["paper_id"] for row in rows} == {
        row["queue_id"]
        for row in _read_rows(REAL_CONTRACT)
        if row["current_status"] != "BROAD_V2_ELIGIBLE"
    }
    assert all(set(row) == {"paper_id", "title"} for row in rows)


def test_validated_anchor_work_ids_are_exact_and_ordered() -> None:
    rows = _read_rows(REAL_CONTRACT)

    work_ids = load_validated_anchor_work_ids(
        REAL_CONTRACT,
        expected_count=40,
        broad_membership_path=REAL_BROAD_MEMBERSHIP,
    )

    assert work_ids == tuple(row["canonical_work_id"] for row in rows)
    assert len(work_ids) == len(set(work_ids)) == 40
