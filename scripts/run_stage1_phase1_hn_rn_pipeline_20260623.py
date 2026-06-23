# -*- coding: utf-8 -*-
"""Run one or more phase-1 HN/RN replay experiments end to end.

Each run-id is a logical entry point:
    HN-01 ... HN-20
    RN-01 ... RN-20

The launcher always runs the calibrated evaluator after training. The final
evidence is metrics/predictions CSV from
scripts/evaluate_stage1_cls_gate.py, not YOLO raw validation output.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SEED = 20260606
DEFAULT_PHASE_ROOT = Path("artifacts") / "stage1_phase1_hn_rn_20260623"
DEFAULT_DATASET_ROOT = Path("data") / "final_sewerml_dataset"
DEFAULT_YOLO_ROOT = Path("YOLOv11")
FORMAL_MODEL = "l"
FORMAL_EVAL_SPLITS = ("val_cal", "val_op", "test")
REQUIRED_PREDICTION_COLUMNS = ("p_defect_cal", "p_defect_operational")

NODE_PLAN: dict[int, list[str]] = {
    1: ["HN-01", "RN-01", "HN-02", "RN-02"],
    2: ["HN-03", "RN-03", "HN-04", "RN-04"],
    3: ["HN-05", "RN-05", "HN-06", "RN-06"],
    4: ["HN-07", "RN-07", "HN-08", "RN-08"],
    5: ["HN-09", "RN-09", "HN-10", "RN-10"],
    6: ["HN-11", "RN-11", "HN-12", "RN-12"],
    7: ["HN-13", "RN-13", "HN-14", "RN-14"],
    8: ["HN-15", "RN-15", "HN-16", "RN-16"],
    9: ["HN-17", "RN-17", "HN-18", "RN-18"],
    10: ["HN-19", "RN-19", "HN-20", "RN-20"],
}


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
    if args.node_index is not None:
        if args.node_index not in NODE_PLAN:
            raise ValueError(f"--node-index must be 1..10, got {args.node_index}")
        run_ids.extend(NODE_PLAN[args.node_index])
    if not run_ids:
        raise ValueError("Pass --run-id HN-01 or --node-index 1")
    return list(dict.fromkeys(run_ids))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_expectation(row: dict[str, str]) -> dict[str, str]:
    keys = (
        "run_id",
        "replay_mode",
        "group",
        "q_percent",
        "normal_slots",
        "defect_slots",
        "selected_unique",
        "replay_duplicate_slots",
        "displaced_unique",
        "final_normal_rows",
        "final_defect_rows",
        "selected_actual_oof_fp",
    )
    return {key: row.get(key, "") for key in keys}


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
        str(paths["repo_root"] / "scripts" / "validate_stage1_phase1_hn_rn_manifests_20260623.py"),
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
    if run_id not in run_matrix and run_id != "BL-0":
        raise ValueError(f"Unknown run_id={run_id}. Check run_matrix.csv")

    manifest_dir = paths["manifest_root"] / run_id
    if not manifest_dir.exists():
        raise FileNotFoundError(f"Missing manifest dir for {run_id}: {manifest_dir}")

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
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-1 HN/RN training and calibrated evaluation.")
    parser.add_argument("--run-id", action="append", help="Run id such as HN-01. Can be comma-separated or repeated.")
    parser.add_argument("--node-index", type=int, help="Run the 4-run assignment for node 1..10.")
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
        "oof_predictions": Path(args.oof_predictions).resolve()
        if args.oof_predictions
        else repo_root
        / "artifacts"
        / "stage1_oof_predictions_calop_20260621"
        / "merged_10fold_20260622"
        / "oof_predictions_merged.csv",
        "yolo_root": Path(args.yolo_root).resolve() if args.yolo_root else repo_root / DEFAULT_YOLO_ROOT,
        "work_root": Path(args.work_root).resolve() if args.work_root else phase_root / "workdirs",
        "runs_root": Path(args.runs_root).resolve() if args.runs_root else phase_root / "runs",
        "eval_root": Path(args.eval_root).resolve() if args.eval_root else phase_root / "eval",
        "summary_root": phase_root / "pipeline_summaries",
        "log_root": phase_root / "pipeline_logs",
    }
    run_matrix = read_run_matrix(phase_root / "run_matrix.csv")
    summaries = []
    run_ids = resolve_run_ids(args)
    if args.model != FORMAL_MODEL:
        raise ValueError(f"Phase-1 HN/RN experiments are locked to yolo11{FORMAL_MODEL}.")
    if tuple(part.strip() for part in args.eval_splits.split(",") if part.strip()) != FORMAL_EVAL_SPLITS:
        raise ValueError(f"Phase-1 evaluation is locked to {','.join(FORMAL_EVAL_SPLITS)}.")
    if args.skip_eval:
        raise ValueError("Phase-1 runs must execute calibrated val_cal,val_op,test evaluation; --skip-eval is disabled.")
    if args.weights and len(run_ids) != 1:
        raise ValueError("--weights may only be used with exactly one --run-id to avoid cross-run checkpoint reuse")
    if args.weights and not args.skip_train:
        raise ValueError("--weights requires --skip-train; otherwise the trained checkpoint should be evaluated")

    for run_id in run_ids:
        print(f"=== phase1 pipeline run_id={run_id} ===")
        summaries.append(run_one(run_id, args, paths, run_matrix))
    write_json(
        phase_root / "last_pipeline_batch.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_ids": [item["run_id"] for item in summaries],
            "summaries": summaries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
