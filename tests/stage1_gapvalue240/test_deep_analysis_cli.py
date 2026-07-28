from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage1_gapvalue240" / "analyze_results.py"


def _load_cli(monkeypatch, result):
    calls = []
    pipeline = types.ModuleType("stage1_gapvalue240.deep_pipeline")

    def run_deep_analysis(**kwargs):
        calls.append(kwargs)
        return result

    pipeline.run_deep_analysis = run_deep_analysis
    monkeypatch.setitem(sys.modules, "stage1_gapvalue240.deep_pipeline", pipeline)
    spec = importlib.util.spec_from_file_location("gapvalue_deep_analysis_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, calls


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--extracted-root",
        str(tmp_path / "extracted"),
        "--inventory",
        str(tmp_path / "inventory.csv"),
        "--completeness-audit",
        str(tmp_path / "completeness.json"),
        "--matrix",
        str(tmp_path / "matrix.csv"),
        "--output-dir",
        str(tmp_path / "report"),
    ]


def test_cli_forwards_paths_and_recomputes_predictions_by_default(
    monkeypatch, tmp_path, capsys
):
    result = {"status": "PASS", "validated_runs": 240}
    module, calls = _load_cli(monkeypatch, result)
    args = _required_args(tmp_path) + [
        "--selection-root",
        str(tmp_path / "selections"),
        "--value-table",
        str(tmp_path / "values.csv"),
    ]

    assert module.main(args) == 0

    assert calls == [
        {
            "extracted_root": (tmp_path / "extracted").resolve(),
            "inventory_path": (tmp_path / "inventory.csv").resolve(),
            "completeness_audit_path": (tmp_path / "completeness.json").resolve(),
            "matrix_path": (tmp_path / "matrix.csv").resolve(),
            "output_dir": (tmp_path / "report").resolve(),
            "selection_root": (tmp_path / "selections").resolve(),
            "value_table": (tmp_path / "values.csv").resolve(),
            "recompute_predictions": True,
        }
    ]
    assert json.loads(capsys.readouterr().out) == result


def test_cli_skip_switch_and_path_result_are_json_serializable(
    monkeypatch, tmp_path, capsys
):
    result_path = tmp_path / "report" / "index.html"
    module, calls = _load_cli(monkeypatch, result_path)

    assert module.main(_required_args(tmp_path) + ["--skip-prediction-recompute"]) == 0

    assert calls[0]["recompute_predictions"] is False
    assert calls[0]["selection_root"] is None
    assert calls[0]["value_table"] is None
    assert json.loads(capsys.readouterr().out) == {
        "status": "PASS",
        "output": str(result_path.resolve()),
    }
