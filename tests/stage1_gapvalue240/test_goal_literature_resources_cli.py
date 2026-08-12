from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from stage1_gapvalue240.resource_reliability import ResourceReliabilityResult


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts"
    / "stage1_gapvalue240"
    / "analyze_goal_literature_resources.py"
)


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "gapvalue_goal_literature_resources_cli", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_result() -> ResourceReliabilityResult:
    return ResourceReliabilityResult(
        runs=pd.DataFrame([{"run_slot": "RUN_001", "validated_complete": True}]),
        machines=pd.DataFrame([{"machine_id": "machine_01", "run_count": 1}]),
        triads=pd.DataFrame([{"triad_id": "TRIAD_001", "run_count": 3}]),
        summary={"canonical_runs": 1, "triads": 1},
    )


def test_publisher_requires_inprogress_and_atomically_writes_all_tables(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_cli()
    monkeypatch.setattr(
        module,
        "build_literature_evidence_matrix",
        lambda: pd.DataFrame([{"evidence_id": "E1", "topic": "test"}]),
    )
    monkeypatch.setattr(module, "analyze_canonical_resources", lambda *a, **k: _fake_result())
    inventory = tmp_path / "inventory.csv"
    ledger = tmp_path / "ledger.csv"
    inventory.write_text("run_slot\nRUN_001\n", encoding="utf-8")
    ledger.write_text("run_slot\nRUN_001\n", encoding="utf-8")
    output = tmp_path / "report.inprogress"

    receipt = module.publish_literature_resources(
        inventory_path=inventory,
        source_ledger_path=ledger,
        output_dir=output,
        expected_runs=1,
        expected_triads=1,
    )

    expected = {
        "tables/literature_evidence_matrix.csv",
        "tables/resource_reliability_runs.csv",
        "tables/resource_reliability_machines.csv",
        "tables/resource_reliability_triads.csv",
        "audit/resource_reliability_summary.json",
        "audit/literature_resource_publish_receipt.json",
    }
    assert {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()} == expected
    assert receipt["status"] == "PASS"
    assert receipt["row_counts"] == {
        "literature_evidence_matrix": 1,
        "resource_reliability_runs": 1,
        "resource_reliability_machines": 1,
        "resource_reliability_triads": 1,
    }
    assert all(item["sha256"] for item in receipt["files"])
    persisted = json.loads(
        (output / "audit/literature_resource_publish_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted == receipt
    assert not list(output.rglob("*.tmp"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.publish_literature_resources(
            inventory_path=inventory,
            source_ledger_path=ledger,
            output_dir=output,
            expected_runs=1,
            expected_triads=1,
        )


def test_publisher_rejects_non_inprogress_output(tmp_path: Path) -> None:
    module = _load_cli()
    with pytest.raises(ValueError, match="must end with .inprogress"):
        module.publish_literature_resources(
            inventory_path=tmp_path / "inventory.csv",
            source_ledger_path=tmp_path / "ledger.csv",
            output_dir=tmp_path / "final_report",
        )


def test_cli_prints_receipt_and_forwards_overwrite(monkeypatch, tmp_path, capsys) -> None:
    module = _load_cli()
    calls = []

    def fake_publish(**kwargs):
        calls.append(kwargs)
        return {"status": "PASS", "row_counts": {"runs": 240}}

    monkeypatch.setattr(module, "publish_literature_resources", fake_publish)
    inventory = tmp_path / "inventory.csv"
    ledger = tmp_path / "ledger.csv"
    output = tmp_path / "report.inprogress"
    exit_code = module.main(
        [
            "--inventory",
            str(inventory),
            "--source-ledger",
            str(ledger),
            "--output-dir",
            str(output),
            "--max-workers",
            "3",
            "--overwrite",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "inventory_path": inventory.resolve(),
            "source_ledger_path": ledger.resolve(),
            "output_dir": output.resolve(),
            "expected_runs": 240,
            "expected_triads": 80,
            "max_workers": 3,
            "overwrite": True,
        }
    ]
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_direct_script_entrypoint_can_import_repo_package() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--source-ledger" in completed.stdout
