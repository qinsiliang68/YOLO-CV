from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
import sys

import pytest

from stage1_dynamic_replay_v3.literature_broad_staging_v2 import build_broad_staging
from stage1_dynamic_replay_v3.literature_screening_queue_v2 import (
    ScreeningQueueError,
    build_screening_queue,
)
from stage1_dynamic_replay_v3.literature_tier_freeze_v2 import TierSelectionPolicy
from test_literature_broad_staging_v2 import (
    _fixture_corpus,
    _write_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_screening_queue_is_nested_exact_and_non_credit(tmp_path: Path) -> None:
    corpus = _fixture_corpus(tmp_path)
    policy_source = corpus / "discovery" / "SCREENED_POLICY.csv"
    policy_source.write_text("policy_id\nfixture-screened-policy\n", encoding="utf-8")
    broad = build_broad_staging(
        corpus,
        batch_numbers=(1,),
        policy=TierSelectionPolicy(
            total=10,
            minimum_per_rq=1,
            maximum_per_rq=3,
            maximum_transfer=2,
            frozen_seed="screen-queue-broad",
        ),
        output_relative=Path("staging/broad"),
    )
    result = build_screening_queue(
        broad.output_root,
        output_root=corpus / "staging" / "screening_queue",
        policy=TierSelectionPolicy(
            total=8,
            minimum_per_rq=1,
            maximum_per_rq=2,
            maximum_transfer=2,
            tier_label="SCREENED",
            frozen_seed="screen-queue",
        ),
        reserve_read_count=2,
        policy_source_paths=(policy_source,),
    )

    assert result.status == "PASS"
    assert result.primary_count == 8
    assert result.reserve_count == 2
    assert result.reading_queue_count == 10
    assert result.formal_screened_increment == 0
    assert all(count >= 1 for count in result.quota_counts.values())

    with (result.output_root / "SCREENED_READING_QUEUE.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    reserve_rows = [row for row in rows if row["selection_role"] == "RESERVE"]
    assert len(reserve_rows) == 2
    assert all(
        row["quota_rq"].startswith("NOT_APPLICABLE_WITH_REASON:")
        for row in reserve_rows
    )
    assert all(row["screened_credit"] == "NOT_ASSESSED_AT_BROAD_LEVEL" for row in rows)
    assert all(len(row["broad_source_sha256"]) == 64 for row in rows)
    assert all(int(row["broad_source_bytes"]) > 0 for row in rows)
    receipt = json.loads(
        (result.output_root / "SCREENING_QUEUE_RECEIPT.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["selection_policy"]["total"] == 8
    assert receipt["selection_policy"]["mandatory_canonical_work_ids"] == []
    policy_identity = next(
        row
        for row in receipt["input_artifacts"]
        if row["input_role"] == "SELECTION_POLICY_SOURCE"
    )
    assert policy_identity["path"] == "discovery/SCREENED_POLICY.csv"
    assert policy_identity["sha256"] == hashlib.sha256(
        policy_source.read_bytes()
    ).hexdigest().upper()


def test_screening_queue_cli_starts_without_manual_pythonpath() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "stage1_dynamic_replay_v3"
                / "build_literature_screening_queue_v2.py"
            ),
            "--help",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--broad-staging-root" in result.stdout
    assert "--minimum-counterevidence-per-rq" in result.stdout
    assert "--required-source-format" in result.stdout


def test_screening_queue_source_gate_fails_when_full_text_pool_is_too_small(
    tmp_path: Path,
) -> None:
    corpus = _fixture_corpus(tmp_path)
    broad = build_broad_staging(
        corpus,
        batch_numbers=(1,),
        policy=TierSelectionPolicy(
            total=10,
            minimum_per_rq=1,
            maximum_per_rq=3,
            maximum_transfer=2,
            frozen_seed="screen-source-gate-broad",
        ),
        output_relative=Path("staging/broad"),
    )

    with pytest.raises(ScreeningQueueError, match="PDF.*0.*required 8"):
        build_screening_queue(
            broad.output_root,
            output_root=corpus / "staging" / "screening_queue_pdf",
            policy=TierSelectionPolicy(
                total=8,
                minimum_per_rq=1,
                maximum_per_rq=2,
                maximum_transfer=2,
                tier_label="SCREENED",
                frozen_seed="screen-source-gate",
            ),
            reserve_read_count=0,
            required_source_format="PDF",
        )


def test_screening_queue_uses_hash_validated_method_source_override(
    tmp_path: Path,
) -> None:
    corpus = _fixture_corpus(tmp_path)
    broad = build_broad_staging(
        corpus,
        batch_numbers=(1,),
        policy=TierSelectionPolicy(
            total=10,
            minimum_per_rq=1,
            maximum_per_rq=3,
            maximum_transfer=2,
            frozen_seed="screen-override-broad",
        ),
        output_relative=Path("staging/broad"),
    )
    with (broad.output_root / "CANONICAL_WORKS.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        registry = list(csv.DictReader(handle))
    target = registry[0]
    method_pdf = corpus / "sources" / "method" / "override.pdf"
    method_pdf.parent.mkdir(parents=True)
    method_pdf.write_bytes(b"%PDF-1.4\nfixture method source\n%%EOF\n")
    method_sha = hashlib.sha256(method_pdf.read_bytes()).hexdigest().upper()
    method_receipt = method_pdf.with_suffix(".pdf.receipt.json")
    method_receipt.write_text(
        json.dumps(
            {
                "paper_id": target["canonical_work_id"],
                "artifact_role": "METHOD_SOURCE",
                "sha256": method_sha,
                "ledger_row": {
                    "path": method_pdf.relative_to(corpus).as_posix(),
                    "sha256": method_sha,
                },
            }
        ),
        encoding="utf-8",
    )
    override = corpus / "discovery" / "METHOD_SOURCE_OVERRIDES.csv"
    _write_csv(
        override,
        [
            {
                "canonical_work_id": target["canonical_work_id"],
                "title": target["title"],
                "path": method_pdf.relative_to(corpus).as_posix(),
                "bytes": method_pdf.stat().st_size,
                "sha256": method_sha,
                "source_authority": "PRIMARY_PUBLISHER",
                "source_url": "https://example.org/method.pdf",
                "receipt_path": method_receipt.relative_to(corpus).as_posix(),
                "override_reason": "fixture verified PDF replaces landing HTML",
                "reading_credit_granted": "False",
            }
        ],
    )
    result = build_screening_queue(
        broad.output_root,
        output_root=corpus / "staging" / "screening_queue_override",
        policy=TierSelectionPolicy(
            total=8,
            minimum_per_rq=1,
            maximum_per_rq=2,
            maximum_transfer=2,
            mandatory_canonical_work_ids=(target["canonical_work_id"],),
            tier_label="SCREENED",
            frozen_seed="screen-override",
        ),
        reserve_read_count=0,
        method_source_override_path=override,
    )

    rows = _read_queue(result.output_root / "SCREENED_PRIMARY.csv")
    selected = next(
        row for row in rows if row["canonical_work_id"] == target["canonical_work_id"]
    )
    assert selected["broad_source_format"] == "HTML"
    assert selected["method_source_format"] == "PDF"
    assert selected["method_source_origin"] == "VERIFIED_OVERRIDE"
    assert selected["method_source_sha256"] == method_sha


def _read_queue(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
