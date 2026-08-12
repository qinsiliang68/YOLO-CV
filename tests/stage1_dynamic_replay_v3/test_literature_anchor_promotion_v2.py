from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from stage1_dynamic_replay_v3.literature_anchor_promotion_v2 import (
    AnchorPromotionError,
    build_anchor_promotion_rows,
    promote_anchor_batch,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / (
    "artifacts/stage1_sample_value_experiments/experiments/"
    "dynamic_replay_budget_efficiency_20260807/02_literature/"
    "review_500_300_100_v2"
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _local_anchor_source_files_available() -> bool:
    inventory_path = CORPUS_ROOT / "discovery" / "ANCHOR_SOURCE_INVENTORY_v2.csv"
    if not inventory_path.is_file():
        return False
    return all(
        (CORPUS_ROOT / row["path"]).is_file()
        for row in _read(inventory_path)
    )


def test_real_anchor_evidence_promotes_to_exact_auditable_batch_24(
    tmp_path: Path,
) -> None:
    if not _local_anchor_source_files_available():
        pytest.skip("local evidence integration test requires 21 anchor source files")

    result = promote_anchor_batch(
        CORPUS_ROOT,
        output_discovery_root=tmp_path / "discovery",
        batch_number=24,
    )

    review_rows = _read(result.review_input_path)
    source_rows = _read(result.source_validation_path)
    decision_rows = _read(result.decision_path)
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))

    assert result.status == "PASS"
    assert result.promoted_count == 21
    assert len(review_rows) == len(source_rows) == len(decision_rows) == 21
    assert {row["queue_id"] for row in review_rows} == {
        row["paper_id"] for row in source_rows
    } == {row["queue_id"] for row in decision_rows}
    assert all(row["decision"] == "ELIGIBLE_BROAD" for row in decision_rows)
    assert receipt["formal_broad_increment"] == 0
    assert receipt["reading_credit_granted"] is False
    assert receipt["output_count"] == 21


def test_promotion_rejects_missing_source_identity() -> None:
    contract = [{"queue_id": "AX0000000000000000"}]
    queue = [{"queue_id": "AX0000000000000000"}]
    decisions = [{"queue_id": "AX0000000000000000"}]

    with pytest.raises(AnchorPromotionError, match="identity sets differ"):
        build_anchor_promotion_rows(
            contract_rows=contract,
            queue_rows=queue,
            source_rows=[],
            decision_rows=decisions,
        )


def test_promotion_does_not_overwrite_existing_batch(tmp_path: Path) -> None:
    if not _local_anchor_source_files_available():
        pytest.skip("local evidence integration test requires 21 anchor source files")

    output = tmp_path / "discovery"
    first = promote_anchor_batch(
        CORPUS_ROOT,
        output_discovery_root=output,
        batch_number=24,
    )
    assert first.review_input_path.is_file()

    with pytest.raises(AnchorPromotionError, match="already exists"):
        promote_anchor_batch(
            CORPUS_ROOT,
            output_discovery_root=output,
            batch_number=24,
        )
