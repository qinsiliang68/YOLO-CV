from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from stage1_dynamic_replay_v3.literature_screening_packet_v2 import (
    ScreeningPacketError,
    scan_screening_text,
    verify_extracted_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_scan_screening_text_returns_page_and_line_locators() -> None:
    text = (
        "Abstract\nWe study useful samples.\n"
        "\f2 Method\nWe select a fixed budget of 100 examples.\nAlgorithm 1\n"
        "\f3 Experiments\nWe compare with random sampling over five seeds.\n"
        "3.2 Ablation Study\nThe method does not outperform random at low budget.\n"
        "\f4 Limitations\nThe setting covers one dataset only.\n4 Conclusion\n"
    )

    scan = scan_screening_text(text)

    assert scan.page_count == 4
    assert ("METHOD", 2, 1, "2 Method") in scan.headings
    assert any(item[0] == "EXPERIMENT" and item[1] == 3 for item in scan.headings)
    assert any(item[0] == "ABLATION" and item[1] == 3 for item in scan.headings)
    assert any(item[0] == "LIMITATION" and item[1] == 4 for item in scan.headings)
    assert any(item[0] == "BUDGET" and item[1] == 2 for item in scan.evidence_candidates)
    assert any(item[0] == "RANDOM_BASELINE" and item[1] == 3 for item in scan.evidence_candidates)
    assert any(item[0] == "SEED" and item[1] == 3 for item in scan.evidence_candidates)
    assert any(item[0] == "NEGATIVE_RESULT" and item[1] == 3 for item in scan.evidence_candidates)


def test_verify_extracted_text_rejects_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "P0001.txt"
    source.write_text("full extracted paper text", encoding="utf-8")
    expected = hashlib.sha256(source.read_bytes()).hexdigest().upper()
    source.write_text("changed extracted paper text", encoding="utf-8")

    with pytest.raises(ScreeningPacketError, match="text SHA"):
        verify_extracted_text(source, expected_sha256=expected, expected_bytes=None)


def test_screening_packet_cli_starts_without_manual_pythonpath() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "stage1_dynamic_replay_v3"
                / "build_literature_screening_packets_v2.py"
            ),
            "--help",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--extraction-ledger" in result.stdout
    assert "--screening-queue" in result.stdout
