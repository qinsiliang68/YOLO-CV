from __future__ import annotations

import subprocess
from pathlib import Path
import sys

import pytest

from stage1_dynamic_replay_v3.literature_screened_text_ledger_v2 import (
    ScreenedTextLedgerError,
    build_screened_text_ledger_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_screened_text_ledger_preserves_source_identity() -> None:
    rows = build_screened_text_ledger_rows(
        [
            {
                "reading_rank": "1",
                "selection_role": "PRIMARY",
                "paper_id": "P0001",
                "title": "A useful paper",
                "broad_source_path": "sources/P0001.pdf",
                "broad_source_format": "PDF",
                "broad_source_sha256": "A" * 64,
                "broad_source_bytes": "1234",
            }
        ],
        broad_staging_relative="staging/broad_freeze_v2",
    )

    assert rows == [
        {
            "paper_id": "P0001",
            "title": "A useful paper",
            "path": "staging/broad_freeze_v2/sources/P0001.pdf",
            "bytes": "1234",
            "sha256": "A" * 64,
            "source_format": "PDF",
            "selection_role": "PRIMARY",
            "reading_rank": "1",
            "reading_credit_granted": "False",
        }
    ]


def test_screened_text_ledger_rejects_non_pdf() -> None:
    with pytest.raises(ScreenedTextLedgerError, match="verified PDF"):
        build_screened_text_ledger_rows(
            [
                {
                    "reading_rank": "1",
                    "selection_role": "PRIMARY",
                    "paper_id": "P0001",
                    "title": "Landing page only",
                    "broad_source_path": "sources/P0001.html",
                    "broad_source_format": "HTML",
                    "broad_source_sha256": "A" * 64,
                    "broad_source_bytes": "1234",
                }
            ],
            broad_staging_relative="staging/broad_freeze_v2",
        )


def test_screened_text_ledger_uses_corpus_relative_method_override() -> None:
    rows = build_screened_text_ledger_rows(
        [
            {
                "reading_rank": "1",
                "selection_role": "PRIMARY",
                "paper_id": "P0001",
                "title": "A useful paper",
                "broad_source_path": "sources/P0001.html",
                "broad_source_format": "HTML",
                "broad_source_sha256": "B" * 64,
                "broad_source_bytes": "222",
                "method_source_path": "sources/method/P0001.pdf",
                "method_source_format": "PDF",
                "method_source_sha256": "A" * 64,
                "method_source_bytes": "1234",
                "method_source_origin": "VERIFIED_OVERRIDE",
            }
        ],
        broad_staging_relative="staging/broad_freeze_v2",
    )

    assert rows[0]["path"] == "sources/method/P0001.pdf"
    assert rows[0]["sha256"] == "A" * 64
    assert rows[0]["bytes"] == "1234"


def test_screened_text_ledger_cli_starts_without_manual_pythonpath() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "stage1_dynamic_replay_v3"
                / "build_literature_screened_text_ledger_v2.py"
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
    assert "--broad-staging-relative" in result.stdout
