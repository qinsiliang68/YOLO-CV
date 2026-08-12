import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from stage1_dynamic_replay_v3.review_claim_ledger import (
    ReviewLedgerError,
    load_expert_findings,
    validate_line_evidence_matrix,
    validate_v3_assessments,
    write_expert_findings_ledger,
    write_merged_review_ledger,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _finding(finding_id: str, severity: str) -> dict[str, object]:
    return {
        "id": finding_id,
        "severity": severity,
        "title": f"Title {finding_id}",
        "locations": ["src/example.py:1-2"],
        "impact": "Impact",
        "evidence": "Evidence",
        "required_fix": "Required fix",
    }


def _write_findings(path: Path, findings: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"findings": findings}, indent=2),
        encoding="utf-8",
    )


def test_expert_findings_ledger_preserves_all_claims_and_source_lines(
    tmp_path: Path,
) -> None:
    source = tmp_path / "findings.json"
    _write_findings(
        source,
        [
            _finding("P0-01", "P0"),
            _finding("H-01", "High"),
            _finding("M-01", "Moderate"),
        ],
    )

    findings = load_expert_findings(source)
    output = tmp_path / "ledger.csv"
    write_expert_findings_ledger(findings, output)

    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["finding_id"] for row in rows] == ["P0-01", "H-01", "M-01"]
    assert all(int(row["source_line"]) > 0 for row in rows)
    assert rows[0]["expert_locations"] == "src/example.py:1-2"


def test_expert_findings_mark_unsupplied_optional_narrative(tmp_path: Path) -> None:
    source = tmp_path / "findings.json"
    finding = _finding("P0-01", "P0")
    finding["impact"] = ""
    finding["evidence"] = ""
    _write_findings(source, [finding])

    findings = load_expert_findings(source)
    output = tmp_path / "ledger.csv"
    write_expert_findings_ledger(findings, output)

    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["impact"] == "NOT_SUPPLIED_IN_EXPERT_JSON"
    assert row["expert_evidence"] == "NOT_SUPPLIED_IN_EXPERT_JSON"
    assert row["impact_supplied"] == "false"
    assert row["expert_evidence_supplied"] == "false"


@pytest.mark.parametrize(
    "findings",
    [
        [_finding("P0-01", "P0"), _finding("P0-01", "P0")],
        [_finding("H-01", "P0")],
        [_finding("NOT-AN-ID", "High")],
    ],
)
def test_expert_findings_reject_duplicate_or_inconsistent_identity(
    tmp_path: Path,
    findings: list[dict[str, object]],
) -> None:
    source = tmp_path / "findings.json"
    _write_findings(source, findings)

    with pytest.raises(ReviewLedgerError):
        load_expert_findings(source)


def test_v3_assessments_require_exact_ids_and_existing_line_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "findings.json"
    _write_findings(source, [_finding("P0-01", "P0")])
    findings = load_expert_findings(source)
    code = tmp_path / "module.py"
    code.write_text("first\nsecond\n", encoding="utf-8")

    rows = [
        {
            "finding_id": "P0-01",
            "v3_status": "PARTIALLY_MITIGATED",
            "v3_source_refs": "module.py:2",
            "local_reproduction": "NOT_RUN_SOURCE_MISSING",
            "observed_result": "Observed",
            "remaining_risk": "Risk",
            "required_action": "Action",
        }
    ]
    validate_v3_assessments(findings, rows, repo_root=tmp_path)

    rows[0]["v3_source_refs"] = "module.py:3"
    with pytest.raises(ReviewLedgerError, match="outside"):
        validate_v3_assessments(findings, rows, repo_root=tmp_path)


def test_v3_assessments_reject_missing_claims(tmp_path: Path) -> None:
    source = tmp_path / "findings.json"
    _write_findings(
        source,
        [_finding("P0-01", "P0"), _finding("H-01", "High")],
    )
    findings = load_expert_findings(source)

    with pytest.raises(ReviewLedgerError, match="assessment IDs"):
        validate_v3_assessments(
            findings,
            [
                {
                    "finding_id": "P0-01",
                    "v3_status": "CONFIRMED_PRESENT",
                    "v3_source_refs": "test_review_claim_ledger.py:1",
                    "local_reproduction": "NOT_RUN",
                    "observed_result": "Observed",
                    "remaining_risk": "Risk",
                    "required_action": "Action",
                }
            ],
            repo_root=Path(__file__).parent,
        )


def test_claim_ledger_cli_starts_directly_without_pythonpath() -> None:
    command = [
        sys.executable,
        str(
            REPO_ROOT
            / "scripts"
            / "stage1_dynamic_replay_v3"
            / "build_review_claim_ledger.py"
        ),
        "--help",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--findings-json" in result.stdout


def test_merged_review_ledger_is_complete_and_immutable(tmp_path: Path) -> None:
    source = tmp_path / "findings.json"
    _write_findings(source, [_finding("P0-01", "P0")])
    findings = load_expert_findings(source)
    code = tmp_path / "module.py"
    code.write_text("first\nsecond\n", encoding="utf-8")
    assessments = [
        {
            "finding_id": "P0-01",
            "v3_status": "PARTIALLY_MITIGATED",
            "v3_source_refs": "module.py:2",
            "local_reproduction": "NOT_RUN_SOURCE_MISSING",
            "observed_result": "Observed",
            "remaining_risk": "Risk",
            "required_action": "Action",
        }
    ]
    output = tmp_path / "merged.csv"

    digest = write_merged_review_ledger(
        findings,
        assessments,
        output,
        repo_root=tmp_path,
    )
    repeated = write_merged_review_ledger(
        findings,
        assessments,
        output,
        repo_root=tmp_path,
    )

    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert repeated == digest
    assert row["finding_id"] == "P0-01"
    assert row["v3_status"] == "PARTIALLY_MITIGATED"
    assert row["source_json"] == "findings.json"

    changed = [dict(assessments[0], observed_result="Different")]
    with pytest.raises(ReviewLedgerError, match="immutable"):
        write_merged_review_ledger(findings, changed, output, repo_root=tmp_path)


def test_line_evidence_matrix_requires_exact_ids_and_valid_references(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.md"
    source.write_text("one\ntwo\n", encoding="utf-8")
    rows = [
        {
            "finding_id": "F01",
            "status": "CONFIRMED_ABSENT",
            "expert_review_ref": "source.md:1",
            "expert_source_refs": "source.md:2",
            "v3_source_refs": "source.md:1-2",
        }
    ]
    validate_line_evidence_matrix(
        rows,
        expected_ids=["F01"],
        repo_root=tmp_path,
        reference_fields=("expert_review_ref", "expert_source_refs", "v3_source_refs"),
    )

    rows[0]["expert_source_refs"] = "source.md:3"
    with pytest.raises(ReviewLedgerError, match="outside"):
        validate_line_evidence_matrix(
            rows,
            expected_ids=["F01"],
            repo_root=tmp_path,
            reference_fields=("expert_review_ref", "expert_source_refs", "v3_source_refs"),
        )
