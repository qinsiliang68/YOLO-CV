from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from stage1_dynamic_replay_v3.literature_deep_queue_v4 import (
    build_deep_review_queue,
)
from stage1_dynamic_replay_v3.literature_tier_freeze_v2 import TierSelectionPolicy


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "corpus"
    canonical: list[dict[str, object]] = []
    screened: list[dict[str, object]] = []
    for index in range(1, 11):
        paper_id = f"P{index:04d}"
        work_id = f"CW{index:04d}"
        source = root / "staging" / "broad" / "sources" / f"{paper_id}.pdf"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"%PDF-1.4\nfixture {paper_id}\n".encode())
        digest = hashlib.sha256(source.read_bytes()).hexdigest().upper()
        rq = f"RQ{min(index, 8)}"
        canonical.append(
            {
                "paper_id": paper_id,
                "canonical_work_id": work_id,
                "title": f"Study {index}",
                "authors": f"Author {index}",
                "year": 2020 + index,
                "doi": f"10.1234/study-{index}",
            }
        )
        screened.append(
            {
                "paper_id": paper_id,
                "queue_id": f"RG{index:04d}",
                "canonical_work_id": work_id,
                "title": f"Study {index}",
                "quota_rq": rq,
                "secondary_rqs": "NOT_APPLICABLE_WITH_REASON:no secondary RQ",
                "relevance_class": "DIRECT_MECHANISM",
                "effect_relation": "SUPPORTED",
                "method_source_path": f"sources/{paper_id}.pdf",
                "method_source_sha256": digest,
                "method_source_bytes": source.stat().st_size,
            }
        )

    canonical_path = root / "staging" / "broad" / "canonical.csv"
    screened_path = root / "screened.csv"
    anchors_path = root / "anchors.csv"
    legacy_path = root / "legacy.csv"
    _write_csv(canonical_path, canonical)
    _write_csv(screened_path, screened)
    _write_csv(
        anchors_path,
        [{"anchor_id": "A001", "canonical_work_id": "CW0001"}],
    )
    legacy_note = root / "legacy_notes" / "P003.md"
    legacy_note.parent.mkdir(parents=True)
    legacy_note.write_text("# inherited full-text review\n", encoding="utf-8")
    _write_csv(
        legacy_path,
        [
            {
                "paper_id": "P003",
                "title": "Study 3",
                "pdf_sha256": screened[2]["method_source_sha256"],
                "note_path": legacy_note.relative_to(root).as_posix(),
                "reading_status": "REPLICATION_DEPTH",
            }
        ],
    )
    reviews = root / "reviews"
    reviews.mkdir()
    (reviews / "P0002.json").write_text(
        json.dumps(
            {
                "paper_id": "P0002",
                "canonical_work_id": "CW0002",
                "title": "Study 2",
                "method_source": {
                    "sha256": screened[1]["method_source_sha256"],
                    "bytes": screened[1]["method_source_bytes"],
                },
            }
        ),
        encoding="utf-8",
    )
    return root, {
        "canonical": canonical_path,
        "screened": screened_path,
        "anchors": anchors_path,
        "legacy": legacy_path,
        "reviews": reviews,
    }


def test_deep_queue_freezes_mandatory_and_hash_bound_inherited_evidence(
    tmp_path: Path,
) -> None:
    root, paths = _fixture(tmp_path)
    result = build_deep_review_queue(
        root,
        canonical_registry_path=paths["canonical"],
        screened_primary_path=paths["screened"],
        core_anchors_path=paths["anchors"],
        screened_review_dir=paths["reviews"],
        legacy_fulltext_ledger_path=paths["legacy"],
        output_relative=Path("staging/deep_queue_v4"),
        policy=TierSelectionPolicy(
            total=8,
            minimum_per_rq=1,
            maximum_per_rq=2,
            maximum_transfer=2,
            mandatory_canonical_work_ids=(),
            tier_label="DEEP",
            frozen_seed="deep-queue-test",
        ),
    )

    assert result.selected_count == 8
    assert result.reserve_count == 2
    assert result.mandatory_count == 3
    assert result.current_review_count == 1
    assert result.byte_identical_legacy_count == 1
    assert result.ready_union_count == 2

    with (result.output_root / "DEEP_PRIMARY_100.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_work = {row["canonical_work_id"]: row for row in rows}
    assert {"CW0001", "CW0002", "CW0003"}.issubset(by_work)
    assert by_work["CW0002"]["current_screened_review"] == "HASH_BOUND_PRESENT"
    assert by_work["CW0003"]["legacy_deep_note"] == "BYTE_IDENTICAL_PRESENT"
    assert by_work["CW0001"]["method_source_path"].startswith("staging/broad/")
    assert all(row["formal_deep_credit"] == "false" for row in rows)

    receipt = json.loads(
        (result.output_root / "DEEP_REVIEW_QUEUE_RECEIPT.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "PASS"
    assert receipt["formal_deep_increment"] == 0
    assert receipt["formal_training_started_by_this_builder"] is False
    assert receipt["engineering_gate_generated_by_this_builder"] is False
    assert receipt["blind_holdout_opened_by_this_builder"] is False
    assert receipt["global_runtime_state_assessed"] is False
    assert "engineering_gate_generated" not in receipt


def test_deep_queue_does_not_reuse_legacy_note_when_pdf_hash_changed(
    tmp_path: Path,
) -> None:
    root, paths = _fixture(tmp_path)
    rows = list(
        csv.DictReader(paths["legacy"].open(encoding="utf-8-sig", newline=""))
    )
    rows[0]["pdf_sha256"] = "0" * 64
    _write_csv(paths["legacy"], rows)

    result = build_deep_review_queue(
        root,
        canonical_registry_path=paths["canonical"],
        screened_primary_path=paths["screened"],
        core_anchors_path=paths["anchors"],
        screened_review_dir=paths["reviews"],
        legacy_fulltext_ledger_path=paths["legacy"],
        output_relative=Path("staging/deep_queue_v4"),
        policy=TierSelectionPolicy(
            total=8,
            minimum_per_rq=1,
            maximum_per_rq=2,
            maximum_transfer=2,
            mandatory_canonical_work_ids=(),
            tier_label="DEEP",
            frozen_seed="deep-queue-test",
        ),
    )

    assert result.byte_identical_legacy_count == 0
    assert result.ready_union_count == 1


def test_deep_queue_cli_starts_without_manual_pythonpath() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/stage1_dynamic_replay_v3/build_literature_deep_queue_v4.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--legacy-note-dir" in result.stdout
