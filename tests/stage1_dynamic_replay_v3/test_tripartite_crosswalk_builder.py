from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "stage1_sample_value_experiments"
    / "experiments"
    / "dynamic_replay_budget_efficiency_20260807"
)
REPRODUCTION_ROOT = EXPERIMENT_ROOT / "01_field_audit" / "expert_review_reproductions"
def test_builder_refuses_to_rebind_retired_v3_runtime_sources(tmp_path: Path) -> None:
    budgeted = tmp_path / "budgeted_v2.csv"
    dynamic = tmp_path / "dynamic_v2.csv"
    results = tmp_path / "results"
    receipt = tmp_path / "receipt.json"
    command = [
        sys.executable,
        str(
            REPO_ROOT
            / "scripts"
            / "stage1_dynamic_replay_v3"
            / "build_tripartite_crosswalk_v2.py"
        ),
        "--repo-root",
        str(REPO_ROOT),
        "--budgeted-source",
        str(REPRODUCTION_ROOT / "expert_vs_v3_full_matrix_v1.csv"),
        "--dynamic-source",
        str(REPRODUCTION_ROOT / "dynamic_review_vs_v3_matrix_v1.csv"),
        "--budgeted-output",
        str(budgeted),
        "--dynamic-output",
        str(dynamic),
        "--result-dir",
        str(results),
        "--receipt",
        str(receipt),
    ]

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "not present in active tree after runtime retirement" in result.stderr
    assert not receipt.exists()
    assert not budgeted.exists()
    assert not dynamic.exists()
