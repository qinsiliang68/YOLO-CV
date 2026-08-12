from __future__ import annotations

import subprocess
from pathlib import Path
import sys

import pytest

from stage1_dynamic_replay_v3.literature_screened_source_plan_v2 import (
    ScreenedSourcePlanError,
    build_screened_source_plan_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _screening_row(
    paper_id: str,
    queue_id: str,
    *,
    source_format: str,
) -> dict[str, str]:
    return {
        "paper_id": paper_id,
        "queue_id": queue_id,
        "title": f"Study {paper_id}",
        "selection_role": "PRIMARY",
        "broad_source_path": f"sources/{paper_id}.{source_format.lower()}",
        "broad_source_format": source_format,
    }


def test_screened_source_plan_distinguishes_pdf_landing_and_reuse() -> None:
    screening = [
        _screening_row("P0001", "Q1", source_format="PDF"),
        _screening_row("P0002", "Q2", source_format="HTML"),
        _screening_row("P0003", "Q3", source_format="HTML"),
    ]
    discovery = [
        {
            "queue_id": "Q1",
            "primary_url": "https://example.org/one",
            "full_text_url": "https://example.org/one.pdf",
            "doi": "10.1/one",
        },
        {
            "queue_id": "Q2",
            "primary_url": "https://publisher.example/two",
            "full_text_url": "https://publisher.example/two/paper.pdf",
            "doi": "10.1/two",
        },
        {
            "queue_id": "Q3",
            "primary_url": "https://publisher.example/three",
            "full_text_url": "https://doi.org/10.1/three",
            "doi": "10.1/three",
        },
    ]

    plan = build_screened_source_plan_rows(
        screening,
        discovery,
        source_subdir="sources/screened_method_v2",
    )

    assert [row["source_action"] for row in plan] == [
        "REUSE_VERIFIED_BROAD_PDF",
        "ACQUIRE_DIRECT_PDF",
        "DISCOVERY_REQUIRED",
    ]
    assert plan[1]["destination"] == "sources/screened_method_v2/P0002.pdf"
    assert plan[2]["destination"].startswith("NOT_APPLICABLE_WITH_REASON:")
    assert all(row["screened_credit"] == "NOT_ASSESSED_AT_BROAD_LEVEL" for row in plan)


def test_screened_source_plan_prefers_verified_method_source_override() -> None:
    row = _screening_row("P0001", "Q1", source_format="HTML")
    row.update(
        {
            "method_source_path": "sources/method/P0001.pdf",
            "method_source_format": "PDF",
            "method_source_sha256": "A" * 64,
            "method_source_bytes": "1234",
            "method_source_origin": "VERIFIED_OVERRIDE",
        }
    )

    plan = build_screened_source_plan_rows(
        [row],
        [
            {
                "queue_id": "Q1",
                "primary_url": "https://example.org/landing",
                "full_text_url": "NOT_REPORTED_BY_SOURCE",
                "doi": "NOT_REPORTED_BY_SOURCE",
            }
        ],
        source_subdir="sources/screened_method_v2",
    )

    assert plan[0]["source_action"] == "REUSE_VERIFIED_METHOD_PDF"
    assert plan[0]["destination"] == "sources/method/P0001.pdf"
    assert plan[0]["method_source_origin"] == "VERIFIED_OVERRIDE"


def test_screened_source_plan_rejects_missing_discovery_identity() -> None:
    with pytest.raises(ScreenedSourcePlanError, match="Q1.*discovery"):
        build_screened_source_plan_rows(
            [_screening_row("P0001", "Q1", source_format="HTML")],
            [],
            source_subdir="sources/screened_method_v2",
        )


def test_screened_source_plan_cli_starts_without_manual_pythonpath() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "stage1_dynamic_replay_v3"
                / "build_literature_screened_source_plan_v2.py"
            ),
            "--help",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--screening-queue" in result.stdout
    assert "--discovery-glob" in result.stdout
    assert "--discovery-path" in result.stdout
