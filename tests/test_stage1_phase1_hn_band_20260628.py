from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_stage1_phase1_hn_band_manifests_20260628 as builder  # noqa: E402
from scripts import run_stage1_phase1_hn_band_pipeline_20260628 as pipeline  # noqa: E402


DATASET_ROOT = REPO_ROOT / "data" / "final_sewerml_dataset"
OOF_PATH = (
    REPO_ROOT
    / "artifacts"
    / "stage1_oof_predictions_calop_20260621"
    / "merged_10fold_20260622"
    / "oof_predictions_merged.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def selected_set(phase_root: Path, run_id: str) -> set[str]:
    rows = read_csv(phase_root / "manifests" / run_id / "selection_manifest.csv")
    return {
        row["source_canonical_image_relpath"].replace("\\", "/")
        for row in rows
        if row["role"] == "selected"
    }


@pytest.mark.skipif(not DATASET_ROOT.exists() or not OOF_PATH.exists(), reason="local dataset/OOF artifacts missing")
def test_hn_band_builder_creates_non_cumulative_30_run_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / "hn_band_smoke"
    args = argparse.Namespace(
        dataset_root=str(DATASET_ROOT),
        oof_predictions=str(OOF_PATH),
        output_root=str(output_root),
        replay_mode="append",
        normal_slots=100,
        defect_slots=100,
        val_defect_slots=20,
        val_normal_slots=20,
        seed=builder.SEED,
        force=True,
    )

    builder.build_manifests(args)

    run_matrix = read_csv(output_root / "run_matrix.csv")
    repro_expected = read_csv(output_root / "repro_manifest_expected.csv")
    assert [row["run_id"] for row in run_matrix] == [
        *(f"HN1-{index:02d}" for index in range(1, 21)),
        *(f"HN2-{index:02d}" for index in range(1, 11)),
    ]
    assert len(run_matrix) == 30
    assert len(repro_expected) == 30
    assert repro_expected[0]["run_id"] == "HN1-01"
    assert repro_expected[0]["fixed_model"] == "yolo11l-cls"
    assert repro_expected[0]["selected_queue_filter"] == "selection_manifest.csv where role=selected"
    assert repro_expected[0]["normal_train_queue_csv"].endswith("HN1-01\\normal_train_manifest.csv") or repro_expected[
        0
    ]["normal_train_queue_csv"].endswith("HN1-01/normal_train_manifest.csv")

    matrix_by_id = {row["run_id"]: row for row in run_matrix}
    for index in range(1, 21):
        row = matrix_by_id[f"HN1-{index:02d}"]
        assert row["band_start_percent"] == str(index - 1)
        assert row["band_end_percent"] == str(index)
        assert row["band_width_percent"] == "1"
        assert row["band_rank_start"] == str(index - 1)
        assert row["band_rank_end_exclusive"] == str(index)
        assert row["selected_unique"] == "1"
        assert row["replay_duplicate_slots"] == "1"
        assert row["final_normal_rows"] == "101"

    for index in range(1, 11):
        row = matrix_by_id[f"HN2-{index:02d}"]
        assert row["band_start_percent"] == str((index - 1) * 2)
        assert row["band_end_percent"] == str(index * 2)
        assert row["band_width_percent"] == "2"
        assert row["selected_unique"] == "2"
        assert row["replay_duplicate_slots"] == "2"
        assert row["final_normal_rows"] == "102"

    hn1_sets = [selected_set(output_root, f"HN1-{index:02d}") for index in range(1, 21)]
    assert sum(len(item) for item in hn1_sets) == len(set().union(*hn1_sets)) == 20
    for index in range(1, 11):
        expected = selected_set(output_root, f"HN1-{2 * index - 1:02d}") | selected_set(
            output_root, f"HN1-{2 * index:02d}"
        )
        assert selected_set(output_root, f"HN2-{index:02d}") == expected

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_stage1_phase1_hn_band_manifests_20260628.py"),
            "--phase-root",
            str(output_root),
            "--dataset-root",
            str(DATASET_ROOT),
            "--oof-predictions",
            str(OOF_PATH),
            "--skip-workdir",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_hn_band_pipeline_exposes_30_entries_and_10_node_groups() -> None:
    assert len(pipeline.BAND_ENTRYPOINTS) == 30

    entry_runs = [run_id for func in pipeline.BAND_ENTRYPOINTS.values() for run_id in func()]
    assert entry_runs == [
        *(f"HN1-{index:02d}" for index in range(1, 21)),
        *(f"HN2-{index:02d}" for index in range(1, 11)),
    ]

    node_runs = [run_id for run_ids in pipeline.NODE_PLAN.values() for run_id in run_ids]
    assert len(pipeline.NODE_PLAN) == 10
    assert len(node_runs) == 30
    assert len(set(node_runs)) == 30
    assert node_runs == entry_runs


def test_hn_band_pipeline_refuses_to_reuse_run_outputs_without_exist_ok(tmp_path: Path) -> None:
    run_id = "HN1-01"
    paths = {
        "summary_root": tmp_path / "pipeline_summaries",
        "work_root": tmp_path / "workdirs",
        "runs_root": tmp_path / "runs",
        "eval_root": tmp_path / "eval",
    }
    (paths["summary_root"]).mkdir(parents=True)
    (paths["summary_root"] / f"{run_id}.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        pipeline.assert_run_outputs_safe(run_id, SimpleNamespace(exist_ok=False), paths)

    pipeline.assert_run_outputs_safe(run_id, SimpleNamespace(exist_ok=True), paths)


def test_hn_band_pipeline_rejects_cpu_training_by_default(tmp_path: Path) -> None:
    phase_root = tmp_path / "phase"
    (phase_root / "manifests" / "HN1-01").mkdir(parents=True)
    with (phase_root / "run_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "replay_mode",
                "group",
                "q_percent",
                "manifest_dir",
                "normal_slots",
                "defect_slots",
                "selected_unique",
                "replay_duplicate_slots",
                "displaced_unique",
                "kept_unselected",
                "final_normal_rows",
                "final_defect_rows",
                "selected_actual_oof_fp",
                "band_start_percent",
                "band_end_percent",
                "band_width_percent",
                "band_rank_start",
                "band_rank_end_exclusive",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "HN1-01",
                "replay_mode": "append",
                "group": "HN1",
                "q_percent": "1",
                "manifest_dir": str(phase_root / "manifests" / "HN1-01"),
                "normal_slots": "100",
                "defect_slots": "100",
                "selected_unique": "1",
                "replay_duplicate_slots": "1",
                "displaced_unique": "0",
                "kept_unselected": "99",
                "final_normal_rows": "101",
                "final_defect_rows": "100",
                "selected_actual_oof_fp": "1",
                "band_start_percent": "0",
                "band_end_percent": "1",
                "band_width_percent": "1",
                "band_rank_start": "0",
                "band_rank_end_exclusive": "1",
            }
        )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_stage1_phase1_hn_band_pipeline_20260628.py"),
            "--phase-root",
            str(phase_root),
            "--run-id",
            "HN1-01",
            "--train-device",
            "cpu",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "CPU smoke training" in result.stderr


def test_hn_band_pipeline_writes_per_run_repro_manifest(tmp_path: Path) -> None:
    run_id = "HN1-01"
    manifest_dir = tmp_path / "manifests" / run_id
    manifest_dir.mkdir(parents=True)
    for name in [
        "selection_manifest.csv",
        "selection_summary.json",
        "normal_train_manifest.csv",
        "train_manifest.csv",
        "val_model_manifest.csv",
        "normal_val_model_manifest.csv",
    ]:
        (manifest_dir / name).write_text("x\n", encoding="utf-8")
    eval_run_dir = tmp_path / "eval" / run_id / f"eval_{run_id}_best"
    eval_run_dir.mkdir(parents=True)
    for name in [
        "predictions_val_cal.csv",
        "predictions_val_op.csv",
        "predictions_test.csv",
        "calibration.json",
        "threshold.json",
        "metrics_at_selected_threshold.csv",
        "artifact_manifest.csv",
        "artifact_manifest.json",
        "run_config.json",
    ]:
        (eval_run_dir / name).write_text("x\n", encoding="utf-8")
    weight_dir = tmp_path / "runs" / run_id / "full_yolo11l_cls_fake" / "weights"
    weight_dir.mkdir(parents=True)
    best_pt = weight_dir / "best.pt"
    last_pt = weight_dir / "last.pt"
    best_pt.write_text("best", encoding="utf-8")
    last_pt.write_text("last", encoding="utf-8")
    (weight_dir.parent / "results.csv").write_text("epoch\n", encoding="utf-8")
    (tmp_path / "runs" / run_id / "summary.csv").write_text("run\n", encoding="utf-8")

    paths = {
        "repo_root": REPO_ROOT,
        "phase_root": tmp_path,
        "eval_root": tmp_path / "eval",
        "runs_root": tmp_path / "runs",
        "summary_root": tmp_path / "pipeline_summaries",
        "repro_root": tmp_path / "repro_runs",
    }
    payload = {
        "run_id": run_id,
        "status": "ok",
        "error": "",
        "model": "l",
        "epochs": 200,
        "seed": 20260606,
        "batch": 128,
        "eval_batch": 64,
        "imgsz": 224,
        "workers": 4,
        "train_device": "0",
        "eval_device": "cpu",
        "save_period": -1,
        "target_recall": 0.995,
        "eval_splits": "val_cal,val_op,test",
        "best_weight": str(best_pt),
        "preflight_csv": str(tmp_path / "validation" / "preflight" / f"{run_id}.csv"),
        "preflight_json": str(tmp_path / "validation" / "preflight" / f"{run_id}.json"),
        "post_train_validation_csv": str(tmp_path / "validation" / "post_train" / f"{run_id}.csv"),
        "post_train_validation_json": str(tmp_path / "validation" / "post_train" / f"{run_id}.json"),
        "work_root": str(tmp_path / "workdirs" / run_id),
        "eval_root": str(tmp_path / "eval" / run_id),
        "eval_verification": {
            "prediction_rows": {"val_cal": 4, "val_op": 4, "test": 4},
            "selected_threshold": 0.0035,
            "threshold_selection_split": "val_op",
            "threshold_score_column": "p_defect_operational",
        },
        "started_at": "2026-06-28T00:00:00",
        "ended_at": "2026-06-28T00:01:00",
        "duration_sec": 60,
        "log_path": str(tmp_path / "pipeline_logs" / f"{run_id}.log"),
    }
    expectation = {
        "replay_mode": "append",
        "group": "HN1",
        "q_percent": "1",
        "band_start_percent": "0",
        "band_end_percent": "1",
        "band_width_percent": "1",
        "band_rank_start": "0",
        "band_rank_end_exclusive": "600",
        "normal_slots": "60000",
        "defect_slots": "60000",
        "selected_unique": "600",
        "selected_actual_oof_fp": "600",
        "selection_policy": "global_oof_p_defect_operational_normal_band_00_01_percent",
    }

    row = pipeline.write_run_repro_manifest(run_id, payload, paths, manifest_dir, expectation)

    assert Path(row["repro_run_csv"]).exists()
    assert Path(row["repro_run_json"]).exists()
    repro_rows = read_csv(Path(row["repro_run_csv"]))
    assert repro_rows[0]["run_id"] == run_id
    assert repro_rows[0]["selection_manifest_csv"].endswith("selection_manifest.csv")
    assert repro_rows[0]["predictions_test_csv"].endswith("predictions_test.csv")
    assert repro_rows[0]["best_pt"].endswith("best.pt")
    assert repro_rows[0]["last_pt"].endswith("last.pt")
    assert repro_rows[0]["prediction_rows_test"] == "4"
