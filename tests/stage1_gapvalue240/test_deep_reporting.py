from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
import yaml

from stage1_gapvalue240.deep_reporting import build_deep_report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _fixtures() -> tuple[dict[str, pd.DataFrame], dict[str, object], pd.DataFrame]:
    condition_summary = pd.DataFrame(
        {
            "condition_id": ["A01", "A02", "A13", "B04"],
            "control": ["R1", "R1", "R1", "R2"],
            "mean_delta_TN": [18.0, 41.0, -22.0, 7.0],
            "mean_delta_FN": [-0.5, -1.0, 3.0, 0.5],
            "n": [3, 8, 3, 3],
        }
    )
    hypotheses = pd.DataFrame(
        {
            "hypothesis_id": ["H01", "H02", "H03", "H04"],
            "claim": [
                "GapCritical beats R1",
                "GapCritical beats R2",
                "BottomGap is harmful",
                "Guard protects FN",
            ],
            "status": ["SUPPORTED", "INCONCLUSIVE", "SUPPORTED", "NOT_TESTABLE"],
            "note": ["same-machine", "high overlap", "negative control", "confounded"],
        }
    )
    tables = {
        "canonical_run_metrics": pd.DataFrame(
            {
                "run_slot": ["RUN_001", "RUN_002"],
                "TN_at_FN95": [68270, 68262],
                "FN_at_TN68253": [91, 94],
            }
        ),
        "condition_control_summaries": condition_summary,
    }
    metadata = {
        "validated_runs": 240,
        "triads": 80,
        "comparisons": 160,
        "conclusion_boundary": "val_op only; no blind/external claim",
    }
    return tables, metadata, hypotheses


def test_build_deep_report_writes_readable_self_contained_package(tmp_path: Path) -> None:
    tables, metadata, hypotheses = _fixtures()
    output = tmp_path / "gapvalue240_report"

    result = build_deep_report(
        output,
        tables=tables,
        metadata=metadata,
        hypothesis_registry=hypotheses,
    )

    assert result == output
    assert output.is_dir()
    assert not output.with_name(output.name + ".inprogress").exists()
    assert (output / "index.html").exists()
    assert (output / "FINAL_REPORT_CN.md").exists()
    assert (output / "README.md").exists()
    assert (output / "manifest.json").exists()
    assert "本报告仅支持 val_op" in (output / "FINAL_REPORT_CN.md").read_text(encoding="utf-8")

    chart_paths = sorted((output / "charts").glob("*.png"))
    assert len(chart_paths) >= 2
    for path in chart_paths:
        with Image.open(path) as image:
            image.verify()
        assert path.stat().st_size > 1_000

    index = (output / "index.html").read_text(encoding="utf-8")
    assert 'href="tables/canonical_run_metrics.csv"' in index
    assert 'src="charts/condition_effects_zoomed.png"' in index
    assert "Zoomed y-axes" in index
    assert "GapCritical beats R1" in index

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "stage1_gapvalue240_deep_report_v1"
    assert manifest["counts"]["tables"] == 3
    assert manifest["counts"]["charts"] >= 2
    assert manifest["metadata"] == metadata
    paths = {row["relative_path"] for row in manifest["files"]}
    assert "manifest.json" not in paths
    assert "index.html" in paths
    for row in manifest["files"]:
        path = output / row["relative_path"]
        assert path.stat().st_size == row["size_bytes"]
        assert _sha256(path) == row["sha256"]


def test_build_deep_report_refuses_to_overwrite_final_or_partial_directory(tmp_path: Path) -> None:
    tables, metadata, hypotheses = _fixtures()
    output = tmp_path / "gapvalue240_report"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_deep_report(
            output,
            tables=tables,
            metadata=metadata,
            hypothesis_registry=hypotheses,
        )
    assert sentinel.read_text(encoding="utf-8") == "user data"

    output.rmdir() if not any(output.iterdir()) else None
    sentinel.unlink()
    output.rmdir()
    partial = output.with_name(output.name + ".inprogress")
    partial.mkdir()
    partial_sentinel = partial / "evidence.txt"
    partial_sentinel.write_text("partial evidence", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_deep_report(
            output,
            tables=tables,
            metadata=metadata,
            hypothesis_registry=hypotheses,
        )
    assert partial_sentinel.read_text(encoding="utf-8") == "partial evidence"


def test_build_deep_report_writes_contract_audits_narrative_and_all_available_charts(
    tmp_path: Path,
) -> None:
    _, metadata, hypotheses = _fixtures()
    seed_rows = []
    for seed, stage in [(1, "discovery"), (2, "discovery"), (3, "confirmation")]:
        for control, delta_tn, delta_fn in [("R1", 20 + seed, -seed), ("R2", 5 + seed, seed / 2)]:
            seed_rows.append(
                {
                    "training_seed": seed,
                    "analysis_cohort": stage,
                    "condition_slot": "A02",
                    "control": control,
                    "delta_TN": delta_tn,
                    "delta_FN": delta_fn,
                }
            )
    tables = {
        "condition_control_summaries": pd.DataFrame(
            {
                "condition_id": ["A01", "A02", "A13", "B04"],
                "control": ["R1", "R1", "R1", "R2"],
                "mean_delta_TN": [18.0, 41.0, -22.0, 7.0],
                "mean_delta_FN": [-0.5, -1.0, 3.0, 0.5],
                "n": [3, 8, 3, 3],
            }
        ),
        "triad_control_deltas": pd.DataFrame(seed_rows),
        "a02_discovery_confirmation_combined": pd.DataFrame(
            {
                "condition_slot": ["A02", "A02"],
                "analysis_cohort": ["discovery", "confirmation"],
                "control": ["R1", "R1"],
                "n": [3, 5],
                "mean_delta_TN": [20.0, 15.0],
                "mean_delta_FN": [-1.0, 0.5],
            }
        ),
        "r2_overlap_power_audit": pd.DataFrame(
            {
                "condition_id": ["A01", "A02", "A02", "A03"],
                "effective_unique_contrast_rate": [0.105, 0.087, 0.081, 0.071],
            }
        ),
        "budget_response": pd.DataFrame(
            {
                "reference_condition": ["A01", "A01", "A02"],
                "comparator_condition": ["A02", "A03", "A03"],
                "control": ["R1", "R1", "R1"],
                "mean_diff_delta_TN": [18.0, 41.0, 23.0],
                "mean_diff_delta_FN": [-0.5, -1.0, 0.5],
            }
        ),
        "guard_policy_contrasts": pd.DataFrame(
            {
                "reference_condition": ["B03", "B03", "B04"],
                "comparator_condition": ["B04", "B05", "B05"],
                "control": ["R1", "R1", "R1"],
                "mean_diff_delta_TN": [8.0, 14.0, 3.0],
                "mean_diff_delta_FN": [0.5, -1.0, -1.5],
            }
        ),
        "prediction_tail_summary": pd.DataFrame(
            {
                "condition_id": ["A01", "A02", "A13"],
                "delta_normal_tail_score": [-0.01, -0.03, 0.02],
                "delta_defect_tail_score": [0.005, 0.012, -0.008],
            }
        ),
        "a02_threshold_frontier": pd.DataFrame(
            {
                "arm": ["T", "T", "R1", "R1", "R2", "R2"],
                "FN": [90, 95, 90, 95, 90, 95],
                "TN": [68240, 68275, 68220, 68255, 68235, 68262],
            }
        ),
        "a02_training_curves": pd.DataFrame(
            {
                "arm": ["T", "T", "R1", "R1", "R2", "R2"],
                "epoch": [1, 200, 1, 200, 1, 200],
                "top1": [0.80, 0.94, 0.80, 0.92, 0.80, 0.93],
                "val_loss": [0.50, 0.18, 0.50, 0.23, 0.50, 0.21],
            }
        ),
        "canonical_run_metrics": pd.DataFrame(
            {
                "run_slot": ["RUN_001", "RUN_002", "RUN_003", "RUN_004"],
                "machine_id": ["M01", "M01", "M02", "M03"],
                "resume_count": [0, 0, 1, 2],
            }
        ),
        "selection_value_effect_associations": pd.DataFrame(
            {
                "analysis_scope": ["phase_a_normal_gapcritical"] * 4,
                "control": ["R1", "R1", "R2", "R2"],
                "predictor": ["treatment_mean_gap_critical_score"] * 4,
                "outcome": ["delta_TN", "delta_FN", "delta_TN", "delta_FN"],
                "n_conditions": [8, 8, 8, 8],
                "spearman_rho": [0.42, -0.31, 0.18, -0.11],
                "p_value": [0.2, 0.3, 0.6, 0.7],
            }
        ),
        "raw_calibrated_operational_sensitivity": pd.DataFrame(
            {
                "triad_id": ["TRIAD_004", "TRIAD_004"],
                "condition_slot": ["A02", "A02"],
                "control": ["R1", "R2"],
                "delta_TN": [21.0, 7.0],
                "delta_FN": [-1.0, 0.5],
                "delta_raw_TN_at_FN95": [21.0, 7.0],
                "delta_raw_FN_at_TN68253": [-1.0, 0.5],
                "integer_effects_equal": [True, True],
            }
        ),
        "paired_epoch_differences": pd.DataFrame(
            [
                {
                    "condition_slot": "A02",
                    "control": control,
                    "training_seed": seed,
                    "epoch": epoch,
                    "delta_top1": 0.001 * seed + 0.0001 * epoch,
                    "delta_val_loss": -0.002 * seed - 0.00005 * epoch,
                }
                for control in ["R1", "R2"]
                for seed in [1, 2]
                for epoch in [1, 100, 200]
            ]
        ),
        "paired_epoch_effect_summary": pd.DataFrame(
            {
                "condition_slot": ["A02", "A02"],
                "condition_id": ["A02_gapcritical", "A02_gapcritical"],
                "phase": ["A", "A"],
                "control": ["R1", "R2"],
                "seed_count": [3, 3],
                "final_delta_top1": [0.012, 0.008],
                "final_delta_val_loss": [-0.009, -0.004],
            }
        ),
    }
    output = tmp_path / "rich_report"
    contract = {
        "schema_version": "gapvalue240_analysis_contract_v1",
        "conclusion_scope": "val_op",
        "safety_ci_upper_margin": 2,
    }
    audits = {
        "metric_recompute_audit": pd.DataFrame(
            {"run_slot": ["RUN_001"], "exact_match": [True]}
        ),
        "source_provenance": {"canonical_attempts": 240, "source_read_only": True},
    }
    narrative = {
        "核心结论": "训练动态价值必须同时通过安全门和双随机对照。",
        "重要限制": "Phase C 的处理臂与机器完全混杂，不能单独作因果确认。",
    }

    build_deep_report(
        output,
        tables=tables,
        metadata=metadata,
        hypothesis_registry=hypotheses,
        analysis_contract=contract,
        audits=audits,
        narrative=narrative,
    )

    assert yaml.safe_load((output / "analysis_contract.yaml").read_text(encoding="utf-8")) == contract
    assert (output / "audit" / "metric_recompute_audit.csv").exists()
    assert json.loads(
        (output / "audit" / "source_provenance.json").read_text(encoding="utf-8")
    ) == audits["source_provenance"]
    for text_path in [
        output / "index.html",
        output / "FINAL_REPORT_CN.md",
        output / "README.md",
        output / "analysis_contract.yaml",
    ]:
        text = text_path.read_text(encoding="utf-8")
        assert "\ufffd" not in text
    assert "训练动态价值必须同时通过安全门和双随机对照" in (
        output / "index.html"
    ).read_text(encoding="utf-8")

    expected_charts = {
        "hypothesis_status.png",
        "condition_effects_zoomed.png",
        "a02_seed_forest_zoomed.png",
        "condition_pareto_zoomed.png",
        "r2_effective_unique_contrast.png",
        "budget_response_zoomed.png",
        "guard_response_zoomed.png",
        "tail_shift_zoomed.png",
        "a02_threshold_frontier_zoomed.png",
        "a02_training_curves_zoomed.png",
        "resume_machine_reliability.png",
        "selection_value_effect_associations.png",
        "raw_calibrated_operational_sensitivity.png",
        "paired_epoch_effects_zoomed.png",
    }
    assert expected_charts == {path.name for path in (output / "charts").glob("*.png")}
    index = (output / "index.html").read_text(encoding="utf-8")
    for chart in expected_charts:
        assert f'src="charts/{chart}"' in index
    assert index.index('src="charts/a02_seed_forest_zoomed.png"') < index.index(
        'href="tables/triad_control_deltas.csv"',
        index.index('src="charts/a02_seed_forest_zoomed.png"'),
    )
    assert index.index('src="charts/paired_epoch_effects_zoomed.png"') < index.index(
        'href="tables/paired_epoch_differences.csv"',
        index.index('src="charts/paired_epoch_effects_zoomed.png"'),
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifested = {row["relative_path"] for row in manifest["files"]}
    assert "analysis_contract.yaml" in manifested
    assert "audit/metric_recompute_audit.csv" in manifested
    assert "audit/source_provenance.json" in manifested
    assert manifest["counts"]["audits"] == 2


def test_build_deep_report_rejects_question_mark_runs_from_encoding_loss(
    tmp_path: Path,
) -> None:
    tables, metadata, hypotheses = _fixtures()
    output = tmp_path / "corrupted_report"

    with pytest.raises(UnicodeError, match="question-mark run"):
        build_deep_report(
            output,
            tables=tables,
            metadata={
                **metadata,
                "primary_result": "A02 ?????? R1/R2",
            },
            hypothesis_registry=hypotheses,
            narrative={"核心结论": "??????"},
        )

    assert not output.exists()
    assert not output.with_name(output.name + ".inprogress").exists()
