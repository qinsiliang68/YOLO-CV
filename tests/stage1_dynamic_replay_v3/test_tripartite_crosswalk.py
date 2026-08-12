from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from stage1_dynamic_replay_v3.tripartite_crosswalk import (
    ALLOWED_STATUSES,
    CROSSWALK_FIELDS,
    SOURCE_MISSING_REFERENCE,
    validate_tripartite_crosswalks,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "stage1_sample_value_experiments"
    / "experiments"
    / "dynamic_replay_budget_efficiency_20260807"
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    requirement_id: str,
    *,
    expert_source_status: str,
    expert_source_refs: str,
) -> dict[str, str]:
    return {
        "requirement_id": requirement_id,
        "overall_status": "NOT_TESTABLE_SOURCE_MISSING"
        if expert_source_status == "NOT_TESTABLE_SOURCE_MISSING"
        else "PARTIALLY_MITIGATED",
        "expert_source_status": expert_source_status,
        "expert_claim_refs": "evidence/claim.md:1",
        "expert_source_refs": expert_source_refs,
        "v3_status": "PARTIALLY_MITIGATED",
        "v3_source_refs": "code/module.py:1",
        "reproduction_command": (
            "uv run pytest tests/test_contract.py::test_contract -q"
        ),
        "exit_code": "0",
        "result_artifact_sha": "A" * 64,
        "observed_result": "Observed result",
        "remaining_risk": "Remaining risk",
        "required_action": "Required action",
    }


def _write_inventory(path: Path, *, budgeted_source_missing: bool) -> None:
    status = (
        "REPORT_ONLY_SOURCE_MISSING"
        if budgeted_source_missing
        else "PRESENT_AND_VERIFIED"
    )
    _write_csv(
        path,
        ["artifact_id", "evidence_role", "required_source", "status"],
        [
            {
                "artifact_id": "EXPERT-001",
                "evidence_role": "budgeted_replay_source_tar",
                "required_source": "true",
                "status": status,
            }
        ],
    )


def _write_reference_tree(root: Path) -> None:
    for relative, text in (
        ("evidence/claim.md", "claim\n"),
        ("evidence/source.py", "source\n"),
        ("code/module.py", "module\n"),
        ("tests/test_contract.py", "def test_contract():\n    pass\n"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_crosswalk_contract_accepts_only_complete_hash_bound_rows(
    tmp_path: Path,
) -> None:
    _write_reference_tree(tmp_path)
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, budgeted_source_missing=True)
    budgeted = tmp_path / "budgeted.csv"
    dynamic = tmp_path / "dynamic.csv"
    _write_csv(
        budgeted,
        list(CROSSWALK_FIELDS),
        [
            _row(
                "BUDGETED:P0-01",
                expert_source_status="NOT_TESTABLE_SOURCE_MISSING",
                expert_source_refs=SOURCE_MISSING_REFERENCE,
            )
        ],
    )
    _write_csv(
        dynamic,
        list(CROSSWALK_FIELDS),
        [
            _row(
                "DYNAMIC:F01",
                expert_source_status="CONFIRMED_PRESENT",
                expert_source_refs="evidence/source.py:1",
            )
        ],
    )

    report = validate_tripartite_crosswalks(
        repo_root=tmp_path,
        budgeted_matrix=budgeted,
        dynamic_matrix=dynamic,
        expert_inventory=inventory,
        expected_budgeted_rows=1,
        expected_dynamic_rows=1,
    )

    assert report["status"] == "PASS"
    assert report["error_count"] == 0
    assert report["total_rows"] == 2
    assert report["budgeted_source_missing"] is True
    assert report["reproduction_commands_executed"] is False
    assert report["contract"]["required_fields"] == list(CROSSWALK_FIELDS)
    assert set(report["contract"]["allowed_statuses"]) == ALLOWED_STATUSES


def test_crosswalk_contract_lists_row_level_failures_without_executing_commands(
    tmp_path: Path,
) -> None:
    _write_reference_tree(tmp_path)
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, budgeted_source_missing=True)
    budgeted = tmp_path / "budgeted.csv"
    dynamic = tmp_path / "dynamic.csv"
    bad = _row(
        "BUDGETED:P0-01",
        expert_source_status="CONFIRMED_PRESENT",
        expert_source_refs="evidence/source.py",
    )
    bad["overall_status"] = "NOT_APPLICABLE_CAPABILITY_ABSENT"
    bad["v3_status"] = "NOT_APPLICABLE_CAPABILITY_ABSENT"
    bad["reproduction_command"] = "uv run pytest <fill-me>"
    bad["exit_code"] = "unknown"
    bad["result_artifact_sha"] = "short"
    _write_csv(budgeted, list(CROSSWALK_FIELDS), [bad])
    _write_csv(
        dynamic,
        list(CROSSWALK_FIELDS),
        [
            _row(
                "DYNAMIC:F01",
                expert_source_status="CONFIRMED_PRESENT",
                expert_source_refs="evidence/source.py:1",
            )
        ],
    )

    report = validate_tripartite_crosswalks(
        repo_root=tmp_path,
        budgeted_matrix=budgeted,
        dynamic_matrix=dynamic,
        expert_inventory=inventory,
        expected_budgeted_rows=1,
        expected_dynamic_rows=1,
    )

    assert report["status"] == "FAIL"
    errors = {
        (error["requirement_id"], error["field"], error["code"])
        for error in report["errors"]
    }
    assert (
        "BUDGETED:P0-01",
        "expert_source_status",
        "BUDGETED_SOURCE_MISSING_STATUS_REQUIRED",
    ) in errors
    assert (
        "BUDGETED:P0-01",
        "expert_source_refs",
        "SOURCE_MISSING_REFERENCE_REQUIRED",
    ) in errors
    assert (
        "BUDGETED:P0-01",
        "overall_status",
        "UNSUPPORTED_STATUS",
    ) in errors
    assert (
        "BUDGETED:P0-01",
        "v3_status",
        "UNSUPPORTED_STATUS",
    ) in errors
    assert (
        "BUDGETED:P0-01",
        "reproduction_command",
        "INVALID_REPRODUCTION_COMMAND",
    ) in errors
    assert ("BUDGETED:P0-01", "exit_code", "INVALID_EXIT_CODE") in errors
    assert (
        "BUDGETED:P0-01",
        "result_artifact_sha",
        "INVALID_SHA256",
    ) in errors
    assert report["reproduction_commands_executed"] is False


def test_crosswalk_contract_requires_exact_schema_and_line_references(
    tmp_path: Path,
) -> None:
    _write_reference_tree(tmp_path)
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, budgeted_source_missing=False)
    budgeted = tmp_path / "budgeted.csv"
    dynamic = tmp_path / "dynamic.csv"
    fields = [field for field in CROSSWALK_FIELDS if field != "reproduction_command"]
    row = _row(
        "BUDGETED:P0-01",
        expert_source_status="CONFIRMED_PRESENT",
        expert_source_refs="evidence/source.py:1",
    )
    row.pop("reproduction_command")
    _write_csv(budgeted, fields, [row])
    _write_csv(
        dynamic,
        list(CROSSWALK_FIELDS),
        [
            _row(
                "DYNAMIC:F01",
                expert_source_status="CONFIRMED_PRESENT",
                expert_source_refs="evidence/source.py:99",
            )
        ],
    )

    report = validate_tripartite_crosswalks(
        repo_root=tmp_path,
        budgeted_matrix=budgeted,
        dynamic_matrix=dynamic,
        expert_inventory=inventory,
        expected_budgeted_rows=1,
        expected_dynamic_rows=1,
    )

    assert any(
        error["code"] == "SCHEMA_FIELDS_MISMATCH"
        and error["matrix_role"] == "budgeted"
        for error in report["errors"]
    )
    assert any(
        error["code"] == "MISSING_FIELD"
        and error["field"] == "reproduction_command"
        for error in report["errors"]
    )
    assert any(
        error["code"] == "LINE_REFERENCE_OUT_OF_BOUNDS"
        and error["requirement_id"] == "DYNAMIC:F01"
        for error in report["errors"]
    )


def test_current_crosswalk_cli_fails_closed_after_v3_runtime_retirement(
    tmp_path: Path,
) -> None:
    output = tmp_path / "TRIPARTITE_CROSSWALK_VALIDATION_v2.json"
    command = [
        sys.executable,
        str(
            REPO_ROOT
            / "scripts"
            / "stage1_dynamic_replay_v3"
            / "validate_tripartite_crosswalk.py"
        ),
        "--repo-root",
        str(REPO_ROOT),
        "--budgeted-matrix",
        str(
            EXPERIMENT_ROOT
            / "01_field_audit"
            / "expert_review_reproductions"
            / "expert_vs_v3_tripartite_v2.csv"
        ),
        "--dynamic-matrix",
        str(
            EXPERIMENT_ROOT
            / "01_field_audit"
            / "expert_review_reproductions"
            / "dynamic_review_vs_v3_tripartite_v2.csv"
        ),
        "--expert-inventory",
        str(
            EXPERIMENT_ROOT
            / "01_field_audit"
            / "expert_delivery_audit_v3"
            / "expert_v1_inventory.csv"
        ),
        "--output",
        str(output),
    ]

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["total_rows"] == 46
    assert report["budgeted_source_missing"] is True
    assert report["reproduction_commands_executed"] is False
    assert report["error_count"] > 0
    assert {error["code"] for error in report["errors"]} == {
        "INVALID_REPRODUCTION_COMMAND",
        "LINE_REFERENCE_FILE_MISSING",
    }
