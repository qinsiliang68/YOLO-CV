from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.performance_reporting import build_performance_report


def _tables() -> dict[str, pd.DataFrame]:
    return {
        "baseline_frontier_val_op": pd.DataFrame(
            {"fn_budget": [0, 1], "TN": [10, 20], "evaluation_split": ["val_op"] * 2}
        ),
        "baseline_frontier_test": pd.DataFrame(
            {"fn_budget": [0, 1], "TN": [9, 19], "evaluation_split": ["test"] * 2}
        ),
        "all_run_baseline_dominance": pd.DataFrame(
            {
                "experiment_family": ["240", "40"],
                "run_id": ["RUN_001", "HN-01"],
                "performance_class": ["DUAL_GAIN", "DOMINATED"],
                "delta_TN_at_baseline_fn": [100, -50],
                "safe_min_delta_TN": [0, -80],
                "safe_max_delta_TN": [120, 0],
            }
        ),
        "paired_control_frontier_deltas": pd.DataFrame(
            {"experiment_family": ["240"], "comparison_id": ["T1"], "real_gain": [True]}
        ),
        "designed_method_double_gates": pd.DataFrame(
            {
                "experiment_family": ["240", "40"],
                "run_id": ["RUN_001", "HN-01"],
                "outcome_cohort": ["STRONG_DOUBLE_GATE", "JOINTLY_HARMFUL"],
                "delta_TN_at_baseline_fn": [100, -50],
            }
        ),
        "method_repeatability_ranking": pd.DataFrame(
            {
                "experiment_family": ["240"],
                "condition_id": ["A01"],
                "double_gate_rate": [2 / 3],
                "repeatability_class": ["REPEATABLE_STRONG"],
            }
        ),
        "hypothesis_registry": pd.DataFrame(
            {
                "hypothesis_id": ["H01"],
                "hypothesis": ["A repeatably robust method exists"],
                "status": ["NOT_SUPPORTED"],
                "evidence": ["none found"],
                "boundary": ["synthetic test"],
            }
        ),
    }


def test_report_is_atomic_manifested_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "report"
    result = build_performance_report(
        output,
        tables=_tables(),
        metadata={
            "analysis_id": "test",
            "candidate_runs": 2,
            "run_counts": {"240": 1, "40": 1},
            "baseline_fn": {"val_op": 1, "test": 1},
            "scientific_boundaries": ["test only"],
        },
    )

    assert result == output
    assert not output.with_name("report.inprogress").exists()
    assert (output / "index.html").is_file()
    assert (output / "FINAL_REPORT_CN.md").is_file()
    assert "性能前沿" in (output / "FINAL_REPORT_CN.md").read_text(encoding="utf-8")
    assert "假设判定" in (output / "index.html").read_text(encoding="utf-8")
    assert (output / "tables/hypothesis_registry.csv").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert any(row["relative_path"] == "index.html" for row in manifest["files"])
    with pytest.raises(FileExistsError):
        build_performance_report(output, tables=_tables(), metadata={"analysis_id": "x"})
