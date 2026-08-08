from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.reconciliation_reporting import build_reconciliation_report


def _tables() -> dict[str, pd.DataFrame]:
    return {
        "unified_triad_outcomes": pd.DataFrame(
            {
                "triad_id": ["T1", "T2"],
                "condition_id": ["C1", "C2"],
                "training_seed": [1, 2],
                "unified_outcome": ["LOCAL_ABSOLUTE_PARETO", "JOINTLY_HARMFUL"],
                "delta_TN_at_baseline_fn": [100, -50],
                "safe_min_delta_TN": [-10, -100],
                "strong_positive": [True, False],
            }
        ),
        "expert_to_absolute_crosswalk": pd.DataFrame(
            {"expert_group": ["strong_positive"], "unified_outcome": ["LOCAL_ABSOLUTE_PARETO"], "count": [1]}
        ),
        "late_overfit_timing": pd.DataFrame(
            {"cutoff_epoch": [140, 160, 200], "strict_good_mean": [0.0, 0.1, 0.1], "harmful_mean": [0.2, 0.4, 0.5]}
        ),
        "weak_defect_vs_normal_tail": pd.DataFrame(
            {"unified_outcome": ["LOCAL_ABSOLUTE_PARETO", "JOINTLY_HARMFUL"], "label": ["defect", "defect"], "mean_shift": [0.1, -0.1]}
        ),
        "control_strength_audit": pd.DataFrame({"triad_id": ["T1"], "control_weakness": [0.2]}),
        "clustered_models": pd.DataFrame({"analysis": ["late"], "status": ["PASS"]}),
        "cross_validation_fold_predictions": pd.DataFrame({"triad_id": ["T1"], "score": [0.9], "leakage_free": [True]}),
        "hypothesis_registry": pd.DataFrame(
            {"hypothesis_id": ["H01"], "status": ["EXPLORATORY_SUPPORTED"], "evidence": ["synthetic"]}
        ),
    }


def test_report_is_atomic_manifested_and_has_readable_chinese(tmp_path: Path) -> None:
    output = tmp_path / "report"
    result = build_reconciliation_report(
        output,
        tables=_tables(),
        metadata={"analysis_id": "test", "source_policy": "read-only"},
    )
    assert result == output
    assert "交叉分析" in (output / "FINAL_RECONCILIATION_REPORT_CN.md").read_text(encoding="utf-8")
    assert "专家相对结果" in (output / "index.html").read_text(encoding="utf-8")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert not output.with_name(output.name + ".inprogress").exists()
    with pytest.raises(FileExistsError):
        build_reconciliation_report(output, tables=_tables(), metadata={})
