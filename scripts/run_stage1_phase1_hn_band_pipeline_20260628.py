# -*- coding: utf-8 -*-
"""Run one or more phase-1 HN band replay experiments end to end.

Each run-id is a logical entry point:
    HN1-01 ... HN1-20  one-percent bands
    HN2-01 ... HN2-10  two-percent bands

The launcher always runs the calibrated evaluator after training. The final
evidence is metrics/predictions CSV from scripts/evaluate_stage1_cls_gate.py,
not YOLO raw validation output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SEED = 20260606
DEFAULT_PHASE_ROOT = Path("artifacts") / "stage1_phase1_hn_band_20260628"
DEFAULT_DATASET_ROOT = Path("data") / "final_sewerml_dataset"
DEFAULT_OOF_PREDICTIONS = (
    Path("artifacts")
    / "stage1_oof_predictions_calop_20260621"
    / "merged_10fold_20260622"
    / "oof_predictions_merged.csv"
)
DEFAULT_YOLO_ROOT = Path("YOLOv11")
FORMAL_MODEL = "l"
FORMAL_EVAL_SPLITS = ("val_cal", "val_op", "test")
REQUIRED_PREDICTION_COLUMNS = ("p_defect_cal", "p_defect_operational")


def HN1_01() -> list[str]:
    return ["HN1-01"]


def HN1_02() -> list[str]:
    return ["HN1-02"]


def HN1_03() -> list[str]:
    return ["HN1-03"]


def HN1_04() -> list[str]:
    return ["HN1-04"]


def HN1_05() -> list[str]:
    return ["HN1-05"]


def HN1_06() -> list[str]:
    return ["HN1-06"]


def HN1_07() -> list[str]:
    return ["HN1-07"]


def HN1_08() -> list[str]:
    return ["HN1-08"]


def HN1_09() -> list[str]:
    return ["HN1-09"]


def HN1_10() -> list[str]:
    return ["HN1-10"]


def HN1_11() -> list[str]:
    return ["HN1-11"]


def HN1_12() -> list[str]:
    return ["HN1-12"]


def HN1_13() -> list[str]:
    return ["HN1-13"]


def HN1_14() -> list[str]:
    return ["HN1-14"]


def HN1_15() -> list[str]:
    return ["HN1-15"]


def HN1_16() -> list[str]:
    return ["HN1-16"]


def HN1_17() -> list[str]:
    return ["HN1-17"]


def HN1_18() -> list[str]:
    return ["HN1-18"]


def HN1_19() -> list[str]:
    return ["HN1-19"]


def HN1_20() -> list[str]:
    return ["HN1-20"]


def HN2_01() -> list[str]:
    return ["HN2-01"]


def HN2_02() -> list[str]:
    return ["HN2-02"]


def HN2_03() -> list[str]:
    return ["HN2-03"]


def HN2_04() -> list[str]:
    return ["HN2-04"]


def HN2_05() -> list[str]:
    return ["HN2-05"]


def HN2_06() -> list[str]:
    return ["HN2-06"]


def HN2_07() -> list[str]:
    return ["HN2-07"]


def HN2_08() -> list[str]:
    return ["HN2-08"]


def HN2_09() -> list[str]:
    return ["HN2-09"]


def HN2_10() -> list[str]:
    return ["HN2-10"]


BAND_ENTRYPOINTS = {
    "HN1_01": HN1_01,
    "HN1_02": HN1_02,
    "HN1_03": HN1_03,
    "HN1_04": HN1_04,
    "HN1_05": HN1_05,
    "HN1_06": HN1_06,
    "HN1_07": HN1_07,
    "HN1_08": HN1_08,
    "HN1_09": HN1_09,
    "HN1_10": HN1_10,
    "HN1_11": HN1_11,
    "HN1_12": HN1_12,
    "HN1_13": HN1_13,
    "HN1_14": HN1_14,
    "HN1_15": HN1_15,
    "HN1_16": HN1_16,
    "HN1_17": HN1_17,
    "HN1_18": HN1_18,
    "HN1_19": HN1_19,
    "HN1_20": HN1_20,
    "HN2_01": HN2_01,
    "HN2_02": HN2_02,
    "HN2_03": HN2_03,
    "HN2_04": HN2_04,
    "HN2_05": HN2_05,
    "HN2_06": HN2_06,
    "HN2_07": HN2_07,
    "HN2_08": HN2_08,
    "HN2_09": HN2_09,
    "HN2_10": HN2_10,
}

NODE_PLAN: dict[int, list[str]] = {
    1: ["HN1-01", "HN1-02", "HN1-03"],
    2: ["HN1-04", "HN1-05", "HN1-06"],
    3: ["HN1-07", "HN1-08", "HN1-09"],
    4: ["HN1-10", "HN1-11", "HN1-12"],
    5: ["HN1-13", "HN1-14", "HN1-15"],
    6: ["HN1-16", "HN1-17", "HN1-18"],
    7: ["HN1-19", "HN1-20", "HN2-01"],
    8: ["HN2-02", "HN2-03", "HN2-04"],
    9: ["HN2-05", "HN2-06", "HN2-07"],
    10: ["HN2-08", "HN2-09", "HN2-10"],
}

REPRO_RUN_COLUMNS = (
    "run_id",
    "status",
    "error",
    "experiment",
    "hostname",
    "cwd",
    "argv",
    "model",
    "epochs",
    "seed",
    "batch",
    "eval_batch",
    "imgsz",
    "workers",
    "train_device",
    "eval_device",
    "save_period",
    "target_recall",
    "eval_splits",
    "replay_mode",
    "group",
    "q_percent",
    "band_start_percent",
    "band_end_percent",
    "band_width_percent",
    "band_rank_start",
    "band_rank_end_exclusive",
    "normal_slots",
    "defect_slots",
    "selected_unique",
    "selected_actual_oof_fp",
    "selection_policy",
    "phase_root",
    "manifest_dir",
    "selection_manifest_csv",
    "selection_summary_json",
    "normal_train_queue_csv",
    "defect_train_queue_csv",
    "val_model_defect_queue_csv",
    "val_model_normal_queue_csv",
    "selected_queue_filter",
    "kept_normal_queue_filter",
    "replay_duplicate_filter",
    "work_root",
    "train_run_root",
    "train_summary_csv",
    "best_pt",
    "last_pt",
    "results_csv",
    "preflight_validation_csv",
    "preflight_validation_json",
    "post_train_validation_csv",
    "post_train_validation_json",
    "eval_root",
    "eval_run_dir",
    "predictions_val_cal_csv",
    "predictions_val_op_csv",
    "predictions_test_csv",
    "calibration_json",
    "threshold_json",
    "metrics_at_selected_threshold_csv",
    "eval_artifact_manifest_csv",
    "eval_artifact_manifest_json",
    "eval_run_config_json",
    "selected_threshold",
    "threshold_selection_split",
    "threshold_score_column",
    "prediction_rows_val_cal",
    "prediction_rows_val_op",
    "prediction_rows_test",
    "pipeline_log",
    "pipeline_summary_json",
    "repro_run_csv",
    "repro_run_json",
    "started_at",
    "ended_at",
    "duration_sec",
)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def run_command(args: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n=== started {started} ===\n")
        log.write(" ".join(args) + "\n")
        log.flush()
        proc = subprocess.Popen(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                print(line, end="")
            except UnicodeEncodeError:
                encoding = sys.stdout.encoding or "utf-8"
                safe_line = line.encode(encoding, errors="replace").decode(encoding, errors="replace")
                print(safe_line, end="")
            log.write(line)
        proc.wait()
        ended = datetime.now().isoformat(timespec="seconds")
        log.write(f"=== ended {ended} exit={proc.returncode} ===\n")
        return int(proc.returncode)


def read_run_matrix(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing run_matrix.csv: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return {row["run_id"]: row for row in csv.DictReader(f)}


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_csv_header(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return next(reader)


def newest_best_weight(run_root: Path, model_key: str) -> Path:
    prefix = f"full_yolo11{model_key}_cls_"
    candidates = [
        path
        for path in run_root.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and (path / "weights" / "best.pt").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No best.pt found under {run_root} for model={model_key}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] / "weights" / "best.pt"


def resolve_run_ids(args: argparse.Namespace) -> list[str]:
    run_ids: list[str] = []
    for value in args.run_id or []:
        run_ids.extend(part.strip() for part in value.split(",") if part.strip())
    for value in args.entry or []:
        for part in [item.strip() for item in value.split(",") if item.strip()]:
            if part not in BAND_ENTRYPOINTS:
                raise ValueError(f"Unknown --entry {part}. Choices: {','.join(sorted(BAND_ENTRYPOINTS))}")
            run_ids.extend(BAND_ENTRYPOINTS[part]())
    if args.node_index is not None:
        if args.node_index not in NODE_PLAN:
            raise ValueError(f"--node-index must be 1..10, got {args.node_index}")
        run_ids.extend(NODE_PLAN[args.node_index])
    if not run_ids:
        raise ValueError("Pass --run-id HN1-01, --entry HN1_01, or --node-index 1")
    return list(dict.fromkeys(run_ids))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def assert_run_outputs_safe(run_id: str, args: argparse.Namespace, paths: dict[str, Path]) -> None:
    if args.exist_ok:
        return
    protected = {
        "summary": paths["summary_root"] / f"{run_id}.json",
        "work_root": paths["work_root"] / run_id,
        "train_run_root": paths["runs_root"] / run_id,
        "eval_run_root": paths["eval_root"] / run_id,
    }
    existing = {name: str(path) for name, path in protected.items() if path.exists()}
    if existing:
        raise FileExistsError(
            f"Refusing to reuse existing outputs for {run_id}. "
            "Use a new --phase-root for a clean run, or pass --exist-ok for an intentional resume/rerun. "
            f"Existing paths: {existing}"
        )


def run_expectation(row: dict[str, str]) -> dict[str, str]:
    keys = (
        "run_id",
        "replay_mode",
        "group",
        "q_percent",
        "band_start_percent",
        "band_end_percent",
        "band_width_percent",
        "band_rank_start",
        "band_rank_end_exclusive",
        "normal_slots",
        "defect_slots",
        "selected_unique",
        "replay_duplicate_slots",
        "displaced_unique",
        "final_normal_rows",
        "final_defect_rows",
        "selected_actual_oof_fp",
        "selection_policy",
    )
    return {key: row.get(key, "") for key in keys}


def maybe_path(path: Path) -> str:
    return str(path) if path.exists() else ""


def build_run_repro_row(
    run_id: str,
    payload: dict,
    paths: dict[str, Path],
    manifest_dir: Path,
    expectation: dict[str, str],
) -> dict[str, object]:
    repro_csv = paths["repro_root"] / f"{run_id}.csv"
    repro_json = paths["repro_root"] / f"{run_id}.json"
    eval_run_dir = paths["eval_root"] / run_id / f"eval_{run_id}_best"
    train_run_root = paths["runs_root"] / run_id
    best_pt = Path(payload["best_weight"]) if payload.get("best_weight") else None
    last_pt = best_pt.parent / "last.pt" if best_pt else None
    results_csv = best_pt.parent.parent / "results.csv" if best_pt else None
    prediction_rows = payload.get("eval_verification", {}).get("prediction_rows", {})
    return {
        "run_id": run_id,
        "status": payload.get("status", ""),
        "error": payload.get("error", ""),
        "experiment": "stage1_phase1_hn_band_20260628",
        "hostname": platform.node(),
        "cwd": str(paths["repo_root"]),
        "argv": " ".join(sys.argv),
        "model": payload.get("model", ""),
        "epochs": payload.get("epochs", ""),
        "seed": payload.get("seed", ""),
        "batch": payload.get("batch", ""),
        "eval_batch": payload.get("eval_batch", ""),
        "imgsz": payload.get("imgsz", ""),
        "workers": payload.get("workers", ""),
        "train_device": payload.get("train_device", ""),
        "eval_device": payload.get("eval_device", ""),
        "save_period": payload.get("save_period", ""),
        "target_recall": payload.get("target_recall", ""),
        "eval_splits": payload.get("eval_splits", ""),
        "replay_mode": expectation.get("replay_mode", ""),
        "group": expectation.get("group", ""),
        "q_percent": expectation.get("q_percent", ""),
        "band_start_percent": expectation.get("band_start_percent", ""),
        "band_end_percent": expectation.get("band_end_percent", ""),
        "band_width_percent": expectation.get("band_width_percent", ""),
        "band_rank_start": expectation.get("band_rank_start", ""),
        "band_rank_end_exclusive": expectation.get("band_rank_end_exclusive", ""),
        "normal_slots": expectation.get("normal_slots", ""),
        "defect_slots": expectation.get("defect_slots", ""),
        "selected_unique": expectation.get("selected_unique", ""),
        "selected_actual_oof_fp": expectation.get("selected_actual_oof_fp", ""),
        "selection_policy": expectation.get("selection_policy", ""),
        "phase_root": str(paths["phase_root"]),
        "manifest_dir": str(manifest_dir),
        "selection_manifest_csv": str(manifest_dir / "selection_manifest.csv"),
        "selection_summary_json": str(manifest_dir / "selection_summary.json"),
        "normal_train_queue_csv": str(manifest_dir / "normal_train_manifest.csv"),
        "defect_train_queue_csv": str(manifest_dir / "train_manifest.csv"),
        "val_model_defect_queue_csv": str(manifest_dir / "val_model_manifest.csv"),
        "val_model_normal_queue_csv": str(manifest_dir / "normal_val_model_manifest.csv"),
        "selected_queue_filter": "selection_manifest.csv where role=selected",
        "kept_normal_queue_filter": "selection_manifest.csv where role=kept_unselected",
        "replay_duplicate_filter": "normal_train_manifest.csv where replay_slot_type=replay_duplicate",
        "work_root": payload.get("work_root", ""),
        "train_run_root": str(train_run_root),
        "train_summary_csv": maybe_path(train_run_root / "summary.csv"),
        "best_pt": str(best_pt) if best_pt else "",
        "last_pt": maybe_path(last_pt) if last_pt else "",
        "results_csv": maybe_path(results_csv) if results_csv else "",
        "preflight_validation_csv": payload.get("preflight_csv", ""),
        "preflight_validation_json": payload.get("preflight_json", ""),
        "post_train_validation_csv": payload.get("post_train_validation_csv", ""),
        "post_train_validation_json": payload.get("post_train_validation_json", ""),
        "eval_root": payload.get("eval_root", ""),
        "eval_run_dir": maybe_path(eval_run_dir),
        "predictions_val_cal_csv": maybe_path(eval_run_dir / "predictions_val_cal.csv"),
        "predictions_val_op_csv": maybe_path(eval_run_dir / "predictions_val_op.csv"),
        "predictions_test_csv": maybe_path(eval_run_dir / "predictions_test.csv"),
        "calibration_json": maybe_path(eval_run_dir / "calibration.json"),
        "threshold_json": maybe_path(eval_run_dir / "threshold.json"),
        "metrics_at_selected_threshold_csv": maybe_path(eval_run_dir / "metrics_at_selected_threshold.csv"),
        "eval_artifact_manifest_csv": maybe_path(eval_run_dir / "artifact_manifest.csv"),
        "eval_artifact_manifest_json": maybe_path(eval_run_dir / "artifact_manifest.json"),
        "eval_run_config_json": maybe_path(eval_run_dir / "run_config.json"),
        "selected_threshold": payload.get("eval_verification", {}).get("selected_threshold", ""),
        "threshold_selection_split": payload.get("eval_verification", {}).get("threshold_selection_split", ""),
        "threshold_score_column": payload.get("eval_verification", {}).get("threshold_score_column", ""),
        "prediction_rows_val_cal": prediction_rows.get("val_cal", ""),
        "prediction_rows_val_op": prediction_rows.get("val_op", ""),
        "prediction_rows_test": prediction_rows.get("test", ""),
        "pipeline_log": payload.get("log_path", ""),
        "pipeline_summary_json": str(paths["summary_root"] / f"{run_id}.json"),
        "repro_run_csv": str(repro_csv),
        "repro_run_json": str(repro_json),
        "started_at": payload.get("started_at", ""),
        "ended_at": payload.get("ended_at", ""),
        "duration_sec": payload.get("duration_sec", ""),
    }


def write_run_repro_manifest(
    run_id: str,
    payload: dict,
    paths: dict[str, Path],
    manifest_dir: Path,
    expectation: dict[str, str],
) -> dict[str, object]:
    row = build_run_repro_row(run_id, payload, paths, manifest_dir, expectation)
    write_csv_rows(paths["repro_root"] / f"{run_id}.csv", [row], REPRO_RUN_COLUMNS)
    write_json(paths["repro_root"] / f"{run_id}.json", row)
    return row


def run_validator(
    run_id: str,
    stage: str,
    args: argparse.Namespace,
    paths: dict[str, Path],
    log_path: Path,
    skip_workdir: bool,
) -> tuple[int, Path, Path]:
    output_dir = paths["phase_root"] / "validation" / stage
    output_csv = output_dir / f"{run_id}.csv"
    output_json = output_dir / f"{run_id}.json"
    cmd = [
        sys.executable,
        str(paths["repo_root"] / "scripts" / "validate_stage1_phase1_hn_band_manifests_20260628.py"),
        "--phase-root",
        str(paths["phase_root"]),
        "--dataset-root",
        str(paths["dataset_root"]),
        "--oof-predictions",
        str(paths["oof_predictions"]),
        "--run-id",
        run_id,
        "--replay-mode",
        args.replay_mode,
        "--output-csv",
        str(output_csv),
        "--output-json",
        str(output_json),
    ]
    if skip_workdir:
        cmd.append("--skip-workdir")
    else:
        cmd.extend(["--work-root", str(paths["work_root"])])
    exit_code = run_command(cmd, paths["repo_root"], log_path)
    return exit_code, output_csv, output_json


def verify_eval_outputs(eval_run_dir: Path, eval_splits: str) -> dict[str, object]:
    required = [
        "calibration.json",
        "threshold.json",
        "metrics_at_selected_threshold.csv",
        "artifact_manifest.csv",
        "artifact_manifest.json",
        "run_config.json",
    ]
    missing = [name for name in required if not (eval_run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing evaluation outputs in {eval_run_dir}: {missing}")

    splits = [part.strip() for part in eval_splits.split(",") if part.strip()]
    if tuple(splits) != FORMAL_EVAL_SPLITS:
        raise ValueError(f"eval_splits must be {','.join(FORMAL_EVAL_SPLITS)}, got {eval_splits}")
    prediction_rows: dict[str, int] = {}
    for split in splits:
        prediction_path = eval_run_dir / f"predictions_{split}.csv"
        header = read_csv_header(prediction_path)
        missing_columns = [name for name in REQUIRED_PREDICTION_COLUMNS if name not in header]
        if missing_columns:
            raise RuntimeError(f"{prediction_path} missing calibrated columns: {missing_columns}")
        prediction_rows[split] = count_csv_rows(prediction_path)

    metrics_rows = read_csv_rows(eval_run_dir / "metrics_at_selected_threshold.csv")
    metric_by_split = {row["split"]: row for row in metrics_rows}
    mismatches = []
    for split, rows in prediction_rows.items():
        metric = metric_by_split.get(split)
        if metric is None:
            mismatches.append(f"{split}:missing_metrics_row")
            continue
        if int(metric["n"]) != rows:
            mismatches.append(f"{split}:prediction_rows={rows}:metrics_n={metric['n']}")
    if mismatches:
        raise RuntimeError(f"Evaluation row-count mismatch: {mismatches}")

    threshold = json.loads((eval_run_dir / "threshold.json").read_text(encoding="utf-8"))
    calibration = json.loads((eval_run_dir / "calibration.json").read_text(encoding="utf-8"))
    if threshold.get("selection_split") != "val_op":
        raise RuntimeError(f"threshold selection_split must be val_op, got {threshold.get('selection_split')}")
    if threshold.get("score_column") != "p_defect_operational":
        raise RuntimeError(f"threshold score_column must be p_defect_operational, got {threshold.get('score_column')}")
    return {
        "eval_run_dir": str(eval_run_dir),
        "prediction_rows": prediction_rows,
        "metrics_rows": len(metrics_rows),
        "selected_threshold": threshold.get("selected_threshold"),
        "threshold_score_column": threshold.get("score_column"),
        "threshold_selection_split": threshold.get("selection_split"),
        "calibration_source_prevalence": calibration.get("source_prevalence"),
        "calibration_deployment_prevalence": calibration.get("deployment_defect_prevalence"),
    }


def run_one(run_id: str, args: argparse.Namespace, paths: dict[str, Path], run_matrix: dict[str, dict[str, str]]) -> dict:
    if run_id not in run_matrix:
        raise ValueError(f"Unknown run_id={run_id}. Check run_matrix.csv")

    manifest_dir = paths["manifest_root"] / run_id
    if not manifest_dir.exists():
        raise FileNotFoundError(f"Missing manifest dir for {run_id}: {manifest_dir}")
    assert_run_outputs_safe(run_id, args, paths)

    work_root = paths["work_root"] / run_id
    train_run_root = paths["runs_root"] / run_id
    eval_root = paths["eval_root"] / run_id
    summary_path = paths["summary_root"] / f"{run_id}.json"
    log_path = paths["log_root"] / f"{run_id}.log"

    started = time.time()
    status = "ok"
    train_exit = None
    eval_exit = None
    preflight_exit = None
    post_train_validation_exit = None
    best_weight = ""
    error = ""
    preflight_csv = ""
    preflight_json = ""
    post_train_validation_csv = ""
    post_train_validation_json = ""
    eval_verification: dict[str, object] = {}
    expectation = run_expectation(run_matrix[run_id])

    try:
        print(f"run_expectation={json.dumps(expectation, ensure_ascii=False)}")
        preflight_exit, preflight_csv_path, preflight_json_path = run_validator(
            run_id, "preflight", args, paths, log_path, skip_workdir=True
        )
        preflight_csv = str(preflight_csv_path)
        preflight_json = str(preflight_json_path)
        if preflight_exit != 0:
            raise RuntimeError(f"Preflight manifest validation failed for {run_id}, exit={preflight_exit}")

        if not args.skip_train:
            train_cmd = [
                sys.executable,
                str(paths["repo_root"] / "scripts" / "train_stage1_cls_sweep.py"),
                "--mode",
                "full",
                "--models",
                args.model,
                "--epochs",
                str(args.epochs),
                "--seed",
                str(args.seed),
                "--batch",
                str(args.batch),
                "--imgsz",
                str(args.imgsz),
                "--workers",
                str(args.workers),
                "--device",
                args.train_device,
                "--save-period",
                str(args.save_period),
                "--manifest-dir",
                str(manifest_dir),
                "--work-root",
                str(work_root),
                "--runs-root",
                str(train_run_root),
                "--dataset-root",
                str(paths["dataset_root"]),
                "--yolo-root",
                str(paths["yolo_root"]),
            ]
            if args.exist_ok:
                train_cmd.append("--exist-ok")
            train_exit = run_command(train_cmd, paths["repo_root"], log_path)
            if train_exit != 0:
                raise RuntimeError(f"Training failed for {run_id}, exit={train_exit}")

        post_train_validation_exit, post_csv_path, post_json_path = run_validator(
            run_id, "post_train", args, paths, log_path, skip_workdir=False
        )
        post_train_validation_csv = str(post_csv_path)
        post_train_validation_json = str(post_json_path)
        if post_train_validation_exit != 0:
            raise RuntimeError(
                f"Post-training dataset validation failed for {run_id}, exit={post_train_validation_exit}"
            )

        if args.weights:
            best_weight_path = Path(args.weights).resolve()
        else:
            best_weight_path = newest_best_weight(train_run_root, args.model)
        best_weight = str(best_weight_path)

        if not args.skip_eval:
            eval_cmd = [
                sys.executable,
                str(paths["repo_root"] / "scripts" / "evaluate_stage1_cls_gate.py"),
                "--weights",
                str(best_weight_path),
                "--run-name",
                f"eval_{run_id}_best",
                "--splits",
                args.eval_splits,
                "--seed",
                str(args.seed),
                "--imgsz",
                str(args.imgsz),
                "--batch",
                str(args.eval_batch),
                "--device",
                args.eval_device,
                "--target-recall",
                str(args.target_recall),
                "--dataset-root",
                str(paths["dataset_root"]),
                "--yolo-root",
                str(paths["yolo_root"]),
                "--output-root",
                str(eval_root),
            ]
            if args.eval_limit_per_class is not None:
                eval_cmd.extend(["--limit-per-class", str(args.eval_limit_per_class)])
            if args.exist_ok:
                eval_cmd.append("--exist-ok")
            eval_exit = run_command(eval_cmd, paths["repo_root"], log_path)
            if eval_exit != 0:
                raise RuntimeError(f"Evaluation failed for {run_id}, exit={eval_exit}")
            eval_verification = verify_eval_outputs(eval_root / f"eval_{run_id}_best", args.eval_splits)
    except Exception as exc:
        status = "failed"
        error = repr(exc)
        raise
    finally:
        payload = {
            "run_id": run_id,
            "status": status,
            "error": error,
            "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "duration_sec": round(time.time() - started, 3),
            "manifest_dir": str(manifest_dir),
            "work_root": str(work_root),
            "train_run_root": str(train_run_root),
            "eval_root": str(eval_root),
            "best_weight": best_weight,
            "train_exit": train_exit,
            "eval_exit": eval_exit,
            "preflight_exit": preflight_exit,
            "post_train_validation_exit": post_train_validation_exit,
            "preflight_csv": preflight_csv,
            "preflight_json": preflight_json,
            "post_train_validation_csv": post_train_validation_csv,
            "post_train_validation_json": post_train_validation_json,
            "run_expectation": expectation,
            "eval_verification": eval_verification,
            "eval_splits": args.eval_splits,
            "eval_limit_per_class": args.eval_limit_per_class,
            "model": args.model,
            "epochs": args.epochs,
            "seed": args.seed,
            "batch": args.batch,
            "eval_batch": args.eval_batch,
            "imgsz": args.imgsz,
            "workers": args.workers,
            "train_device": args.train_device,
            "eval_device": args.eval_device,
            "save_period": args.save_period,
            "target_recall": args.target_recall,
            "log_path": str(log_path),
        }
        write_json(summary_path, payload)
        repro_row = write_run_repro_manifest(run_id, payload, paths, manifest_dir, expectation)
        payload["repro_run_csv"] = repro_row["repro_run_csv"]
        payload["repro_run_json"] = repro_row["repro_run_json"]
        write_json(summary_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-1 HN band training and calibrated evaluation.")
    parser.add_argument("--run-id", action="append", help="Run id such as HN1-01. Can be comma-separated or repeated.")
    parser.add_argument("--entry", action="append", help="Entry function such as HN1_01. Can be comma-separated.")
    parser.add_argument("--node-index", type=int, help="Run the 3-run assignment for node 1..10.")
    parser.add_argument("--phase-root", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--oof-predictions", default=None)
    parser.add_argument("--yolo-root", default=None)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--eval-root", default=None)
    parser.add_argument("--model", default=FORMAL_MODEL, choices=(FORMAL_MODEL,))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-device", default="0")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--target-recall", type=float, default=0.995)
    parser.add_argument("--eval-splits", default=",".join(FORMAL_EVAL_SPLITS))
    parser.add_argument("--eval-limit-per-class", type=int, default=None)
    parser.add_argument("--replay-mode", choices=("append", "fixed"), default="append")
    parser.add_argument("--weights", default=None, help="Evaluate this weight instead of the just-trained best.pt.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true", help="Disabled for formal phase-1 runs; kept only to fail loudly.")
    parser.add_argument(
        "--allow-cpu-train",
        action="store_true",
        help="Allow --train-device cpu for manual debugging. Formal runs should use a CUDA device.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Compatibility flag only. The archived training script is not modified, so this is not forwarded.",
    )
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_script()
    phase_root = Path(args.phase_root).resolve() if args.phase_root else repo_root / DEFAULT_PHASE_ROOT
    paths = {
        "repo_root": repo_root,
        "phase_root": phase_root,
        "manifest_root": phase_root / "manifests",
        "dataset_root": Path(args.dataset_root).resolve() if args.dataset_root else repo_root / DEFAULT_DATASET_ROOT,
        "oof_predictions": Path(args.oof_predictions).resolve() if args.oof_predictions else repo_root / DEFAULT_OOF_PREDICTIONS,
        "yolo_root": Path(args.yolo_root).resolve() if args.yolo_root else repo_root / DEFAULT_YOLO_ROOT,
        "work_root": Path(args.work_root).resolve() if args.work_root else phase_root / "workdirs",
        "runs_root": Path(args.runs_root).resolve() if args.runs_root else phase_root / "runs",
        "eval_root": Path(args.eval_root).resolve() if args.eval_root else phase_root / "eval",
        "summary_root": phase_root / "pipeline_summaries",
        "log_root": phase_root / "pipeline_logs",
        "repro_root": phase_root / "repro_runs",
    }
    run_matrix = read_run_matrix(phase_root / "run_matrix.csv")
    summaries = []
    run_ids = resolve_run_ids(args)
    if args.model != FORMAL_MODEL:
        raise ValueError(f"Phase-1 HN band experiments are locked to yolo11{FORMAL_MODEL}.")
    if tuple(part.strip() for part in args.eval_splits.split(",") if part.strip()) != FORMAL_EVAL_SPLITS:
        raise ValueError(f"Phase-1 evaluation is locked to {','.join(FORMAL_EVAL_SPLITS)}.")
    if args.skip_eval:
        raise ValueError("Phase-1 runs must execute calibrated val_cal,val_op,test evaluation; --skip-eval is disabled.")
    if not args.skip_train and args.train_device.strip().lower() == "cpu" and not args.allow_cpu_train:
        raise ValueError(
            "Formal HN band training should run on a CUDA device. "
            "CPU smoke training can trigger the archived training script's torch/cuDNN metadata bug; "
            "pass --allow-cpu-train only for intentional local debugging."
        )
    if args.weights and len(run_ids) != 1:
        raise ValueError("--weights may only be used with exactly one run to avoid cross-run checkpoint reuse")
    if args.weights and not args.skip_train:
        raise ValueError("--weights requires --skip-train; otherwise the trained checkpoint should be evaluated")

    for run_id in run_ids:
        print(f"=== phase1 HN band pipeline run_id={run_id} ===")
        summaries.append(run_one(run_id, args, paths, run_matrix))
    write_json(
        phase_root / "last_pipeline_batch.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_ids": [item["run_id"] for item in summaries],
            "summaries": summaries,
        },
    )
    batch_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}"
    batch_repro_rows = []
    for item in summaries:
        repro_value = item.get("repro_run_json", "")
        if not repro_value:
            continue
        repro_path = Path(repro_value)
        if repro_path.is_file():
            batch_repro_rows.append(json.loads(repro_path.read_text(encoding="utf-8")))
    write_json(
        phase_root / "repro_batches" / f"{batch_id}.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "batch_id": batch_id,
            "run_ids": [item["run_id"] for item in summaries],
            "repro_rows": batch_repro_rows,
        },
    )
    if batch_repro_rows:
        write_csv_rows(phase_root / "repro_batches" / f"{batch_id}.csv", batch_repro_rows, REPRO_RUN_COLUMNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
