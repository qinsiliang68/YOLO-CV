from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import pytest
import yaml

import stage1_gapvalue240.extreme_reporting as extreme_reporting
from stage1_gapvalue240.extreme_reporting import build_extreme_report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _fixtures() -> tuple[dict[str, pd.DataFrame], dict[str, object], list[dict[str, object]]]:
    tiers = pd.DataFrame(
        {
            "triad_id": ["TRIAD_001", "TRIAD_002", "TRIAD_003", "TRIAD_004"],
            "cohort_code": ["S", "H", "M", "A"],
            "cohort_label": ["特别好", "明确有害", "混合", "安全改善"],
            "delta_TN_R1": [3807, -1400, 80, 260],
            "delta_FN_R1": [-18, 11, -1, 0],
            "delta_TN_R2": [3664, -900, -40, 120],
            "delta_FN_R2": [-22, 8, 2, -1],
        }
    )
    training = pd.DataFrame(
        {
            "triad_id": ["TRIAD_001", "TRIAD_001", "TRIAD_002", "TRIAD_002"],
            "control": ["R1", "R2", "R1", "R2"],
            "cohort_code": ["S", "S", "H", "H"],
            "train_loss_extra_drop_epoch121_to_200": [0.0005, -0.0003, 0.0034, 0.0007],
            "train_loss_robust_drop_121_130_to_191_200": [0.0003, -0.0002, 0.0028, 0.0005],
            "train_loss_slope_121_200": [-0.00001, 0.00001, -0.00005, -0.00002],
        }
    )
    selection_sets = pd.DataFrame(
        {
            "sample_set_digest": ["A02_TOP3000", "A08_BOUNDARY3000"],
            "cohort_codes": ["S|M|H", "H"],
            "triad_count": [8, 3],
            "exceptional_count": [1, 0],
            "harmful_count": [4, 3],
            "spans_exceptional_and_harmful": [True, False],
        }
    )
    prediction_tail = pd.DataFrame(
        {
            "label": ["normal", "normal", "defect", "defect"],
            "tail_scope": ["operational"] * 4,
            "score_type": ["raw"] * 4,
            "analysis_scope": ["all"] * 4,
            "control": ["R1", "R2", "R1", "R2"],
            "feature": ["mean_shift"] * 4,
            "exceptional_mean": [-0.001, 0.002, 0.0002, 0.00025],
            "harmful_mean": [-0.008, -0.005, -0.0001, -0.00008],
        }
    )
    tables = {
        "triad_performance_tiers": tiers,
        "training_window_features": training,
        "selection_set_outcomes": selection_sets,
        "prediction_tail_extreme_contrasts": prediction_tail,
        "counterexample_registry": pd.DataFrame(
            {"pattern": ["late loss"], "counterexample": ["seed-dependent reversal"]}
        ),
    }
    metadata = {
        "analysis_id": "stage1_gapvalue240_extreme_cohort_analysis_v3",
        "triads": 80,
        "cohort_counts": {"S": 12, "A": 3, "B": 1, "M": 41, "H": 23},
        "conclusion_boundary": "val_op internal; descriptive extreme-cohort analysis",
    }
    findings = [
        {
            "status": "PARTIAL_PATTERN",
            "title": "后期损失候选规律",
            "evidence": "S组后期额外下降通常小于H组，但存在seed反例。",
            "boundary": "不能解释为单样本因果价值。",
        }
    ]
    return tables, metadata, findings


def test_build_extreme_report_publishes_complete_atomic_package(tmp_path: Path) -> None:
    tables, metadata, findings = _fixtures()
    output = tmp_path / "extreme_report"

    result = build_extreme_report(
        output,
        tables=tables,
        metadata=metadata,
        findings=findings,
    )

    assert result == output
    assert output.is_dir()
    assert not output.with_name(output.name + ".inprogress").exists()
    required_files = {
        "FINAL_REPORT_CN.md",
        "README.md",
        "index.html",
        "analysis_contract.yaml",
        "manifest.json",
        "audit/report_inputs.json",
        "charts/triad_performance_quadrants_zoomed.png",
        "charts/training_late_loss_s_h_zoomed.png",
        "charts/selection_set_tier_flips.png",
        "charts/prediction_tail_mechanism_zoomed.png",
    }
    assert required_files.issubset(
        {
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file()
        }
    )
    for table_name, expected in tables.items():
        written = pd.read_csv(output / "tables" / f"{table_name}.csv")
        pd.testing.assert_frame_equal(written, expected, check_dtype=False)

    for path in sorted((output / "charts").glob("*.png")):
        with Image.open(path) as image:
            image.verify()
        assert path.stat().st_size > 1_000

    html = (output / "index.html").read_text(encoding="utf-8")
    assert 'href="tables/triad_performance_tiers.csv"' in html
    assert 'src="charts/triad_performance_quadrants_zoomed.png"' in html
    assert 'src="charts/training_late_loss_s_h_zoomed.png"' in html
    assert 'src="charts/selection_set_tier_flips.png"' in html
    assert 'src="charts/prediction_tail_mechanism_zoomed.png"' in html
    assert "后期损失候选规律" in html
    assert "Zoomed axes" in html

    report = (output / "FINAL_REPORT_CN.md").read_text(encoding="utf-8")
    assert "极端条件集对照分析" in report
    assert "后期损失候选规律" in report
    contract = yaml.safe_load(
        (output / "analysis_contract.yaml").read_text(encoding="utf-8")
    )
    assert contract["report_id"] == "stage1_gapvalue240_extreme_report_v1"
    assert set(contract["required_tables"]) == {
        "triad_performance_tiers",
        "training_window_features",
        "selection_set_outcomes",
        "prediction_tail_extreme_contrasts",
    }

    audit = json.loads(
        (output / "audit" / "report_inputs.json").read_text(encoding="utf-8")
    )
    assert audit["table_rows"]["triad_performance_tiers"] == 4
    assert audit["finding_count"] == 1

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "stage1_gapvalue240_extreme_report_v1"
    assert manifest["metadata"] == metadata
    manifested = {row["relative_path"] for row in manifest["files"]}
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert manifested == actual
    assert "manifest.json" not in manifested
    for row in manifest["files"]:
        path = output / row["relative_path"]
        assert path.stat().st_size == row["size_bytes"]
        assert _sha256(path) == row["sha256"]


def test_build_extreme_report_refuses_existing_final_or_partial(tmp_path: Path) -> None:
    tables, metadata, findings = _fixtures()
    output = tmp_path / "extreme_report"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("user evidence", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_extreme_report(output, tables=tables, metadata=metadata, findings=findings)
    assert sentinel.read_text(encoding="utf-8") == "user evidence"

    sentinel.unlink()
    output.rmdir()
    partial = output.with_name(output.name + ".inprogress")
    partial.mkdir()
    partial_sentinel = partial / "keep.txt"
    partial_sentinel.write_text("partial evidence", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_extreme_report(output, tables=tables, metadata=metadata, findings=findings)
    assert partial_sentinel.read_text(encoding="utf-8") == "partial evidence"


@pytest.mark.parametrize(
    ("table_name", "missing_column"),
    [
        ("triad_performance_tiers", None),
        ("training_window_features", None),
        ("selection_set_outcomes", None),
        ("prediction_tail_extreme_contrasts", None),
        ("triad_performance_tiers", "delta_TN_R2"),
        ("training_window_features", "train_loss_slope_121_200"),
        ("selection_set_outcomes", "spans_exceptional_and_harmful"),
        ("prediction_tail_extreme_contrasts", "exceptional_mean"),
    ],
)
def test_build_extreme_report_fails_on_missing_required_table_or_column(
    tmp_path: Path, table_name: str, missing_column: str | None
) -> None:
    tables, metadata, findings = _fixtures()
    if missing_column is None:
        tables.pop(table_name)
    else:
        tables[table_name] = tables[table_name].drop(columns=missing_column)

    output = tmp_path / f"missing_{table_name}_{missing_column}"
    with pytest.raises(ValueError, match=table_name):
        build_extreme_report(output, tables=tables, metadata=metadata, findings=findings)
    assert not output.exists()
    assert not output.with_name(output.name + ".inprogress").exists()


def test_build_extreme_report_never_publishes_final_on_render_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tables, metadata, findings = _fixtures()
    output = tmp_path / "failed_report"

    def fail_renderer(frame: pd.DataFrame, output_path: Path) -> None:
        raise RuntimeError("synthetic chart failure")

    monkeypatch.setattr(
        extreme_reporting, "_save_training_window_features", fail_renderer
    )
    with pytest.raises(RuntimeError, match="synthetic chart failure"):
        build_extreme_report(output, tables=tables, metadata=metadata, findings=findings)

    assert not output.exists()
    partial = output.with_name(output.name + ".inprogress")
    assert partial.is_dir()
    assert (partial / "tables" / "triad_performance_tiers.csv").is_file()
