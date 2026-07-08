from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate_stage1_cls_gate as eval_gate  # noqa: E402


class DummyModel:
    pass


def test_predict_records_has_tempfile_dependency_imported() -> None:
    assert eval_gate.tempfile.NamedTemporaryFile


def raw_row(split: str, y_true: int, p_defect: float, filename: str) -> dict[str, str]:
    y_pred = 1 if p_defect >= 0.5 else 0
    return {
        "eval_split": split,
        "source_split": split,
        "y_true": str(y_true),
        "y_pred_raw": str(y_pred),
        "raw_correct": str(int(y_pred == y_true)),
        "p_defect_raw": f"{p_defect:.10f}",
        "p_normal_raw": f"{1.0 - p_defect:.10f}",
        "raw_logit": f"{eval_gate.logit(p_defect):.10f}",
        "Filename": filename,
        "image_path": f"C:/fake/{filename}",
    }


def write_raw_predictions(path: Path, split: str, scores: tuple[tuple[int, float], ...]) -> None:
    rows = [raw_row(split, y_true, p_defect, f"{split}_{index}.png") for index, (y_true, p_defect) in enumerate(scores)]
    eval_gate.write_csv(path, rows, eval_gate.PREDICTION_COLUMNS)


def test_write_calibrated_outputs_reads_raw_prediction_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_paths = {
        "val_cal": raw_dir / "val_cal.csv",
        "val_op": raw_dir / "val_op.csv",
        "test": raw_dir / "test.csv",
    }
    write_raw_predictions(raw_paths["val_cal"], "val_cal", ((1, 0.90), (1, 0.80), (0, 0.20), (0, 0.10)))
    write_raw_predictions(raw_paths["val_op"], "val_op", ((1, 0.85), (1, 0.75), (0, 0.25), (0, 0.15)))
    write_raw_predictions(raw_paths["test"], "test", ((1, 0.70), (0, 0.30)))
    cfg = SimpleNamespace(
        splits=("val_cal", "val_op", "test"),
        target_recall=1.0,
        deployment_defect_prevalence=0.10,
    )

    result = eval_gate.write_calibrated_outputs_from_raw_predictions(tmp_path, raw_paths, cfg)

    assert result["threshold_json"]["selection_split"] == "val_op"
    assert result["threshold_json"]["score_column"] == "p_defect_operational"
    assert [row["split"] for row in result["metrics_rows"]] == ["val_cal", "val_op", "test"]
    for split in cfg.splits:
        output_path = tmp_path / f"predictions_{split}.csv"
        rows = eval_gate.read_csv_rows(output_path)
        assert rows
        assert all(row["p_defect_cal"] for row in rows)
        assert all(row["p_defect_operational"] for row in rows)
    assert (tmp_path / "metrics_at_selected_threshold.csv").exists()
    assert (tmp_path / "calibration.json").exists()
    assert (tmp_path / "threshold.json").exists()


def test_predict_raw_directory_is_removed_after_success(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    cfg = SimpleNamespace(splits=("val_cal", "val_op", "test"))
    monkeypatch.setattr(eval_gate, "load_split_records", lambda paths, split, cfg: [{"split": split}])
    monkeypatch.setattr(eval_gate, "predict_records", lambda model, records, cfg: [raw_row("val_cal", 1, 0.90, "x.png")])
    monkeypatch.setattr(
        eval_gate,
        "write_calibrated_outputs_from_raw_predictions",
        lambda run_dir, raw_paths, cfg: {"threshold": 0.5},
    )

    result = eval_gate.predict_and_calibrate_from_persistent_raw(run_dir, DummyModel(), object(), cfg)

    assert result["threshold"] == 0.5
    assert not list(run_dir.glob("raw_predictions_*"))


def test_predict_raw_directory_is_kept_after_calibration_failure(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    cfg = SimpleNamespace(splits=("val_cal", "val_op", "test"))
    monkeypatch.setattr(eval_gate, "load_split_records", lambda paths, split, cfg: [{"split": split}])
    monkeypatch.setattr(eval_gate, "predict_records", lambda model, records, cfg: [raw_row("val_cal", 1, 0.90, "x.png")])

    def fail_calibration(run_dir, raw_paths, cfg):
        raise RuntimeError("calibration failed")

    monkeypatch.setattr(eval_gate, "write_calibrated_outputs_from_raw_predictions", fail_calibration)

    try:
        eval_gate.predict_and_calibrate_from_persistent_raw(run_dir, DummyModel(), object(), cfg)
    except RuntimeError as exc:
        assert "calibration failed" in str(exc)
    else:
        raise AssertionError("expected calibration failure")

    raw_dirs = list(run_dir.glob("raw_predictions_*"))
    assert len(raw_dirs) == 1
    assert list(raw_dirs[0].glob("predictions_*_raw.csv"))
