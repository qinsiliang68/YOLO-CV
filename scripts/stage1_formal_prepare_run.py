from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from collect_cls_raw_materials import (
    build_dataset_stats,
    build_dataset_stats_with_gate,
    build_env_info,
    collect_dataset_rows,
    write_csv,
    write_json,
)
from pipeline_common import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or update formal stage-1 run manifests.")
    parser.add_argument("--task-kind", choices=("gate", "cls6"), required=True, help="Formal task kind.")
    parser.add_argument("--task-name", required=True, help="Human-facing task name.")
    parser.add_argument("--config-path", required=True, help="Resolved runtime config path.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root for this run.")
    parser.add_argument("--run-dir", required=True, help="Training run directory.")
    parser.add_argument("--summary-dir", required=True, help="Formal material output directory.")
    parser.add_argument("--split-manifest", default="", help="Optional val-cal / val-op split manifest.")
    parser.add_argument("--normal-class", default="Normal", help="Normal class name for gate-style stats.")
    parser.add_argument("--batch", type=int, default=24, help="Formal training batch.")
    parser.add_argument("--epochs", type=int, default=200, help="Formal epoch target.")
    parser.add_argument(
        "--status",
        choices=("planned", "training_started", "training_completed", "evaluation_completed"),
        required=True,
        help="Lifecycle status to persist.",
    )
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
        )
    except Exception:
        return ""


def load_epoch_completion(run_dir: Path) -> int:
    checkpoint_index = run_dir / "all_checkpoints_index.csv"
    if checkpoint_index.exists():
        import csv

        with checkpoint_index.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        epochs = [int(float(row["epoch"])) for row in rows if row.get("epoch")]
        if epochs:
            return max(epochs)
    return 0


def build_dataset_manifest(args: argparse.Namespace, dataset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = (
        build_dataset_stats_with_gate(dataset_rows, args.normal_class)
        if args.task_kind == "gate"
        else build_dataset_stats(dataset_rows)
    )
    return {
        "task_kind": args.task_kind,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "split_manifest": str(Path(args.split_manifest).resolve()) if args.split_manifest else "",
        "normal_class": args.normal_class if args.task_kind == "gate" else "",
        "stats": stats,
    }


def update_run_manifest(args: argparse.Namespace, env_info: dict[str, Any]) -> dict[str, Any]:
    summary_dir = Path(args.summary_dir).resolve()
    manifest_path = summary_dir / "run_manifest.json"
    existing = read_json_if_exists(manifest_path)
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    manifest = {
        "task_kind": args.task_kind,
        "task_name": args.task_name,
        "config_path": str(Path(args.config_path).resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "run_dir": str(Path(args.run_dir).resolve()),
        "summary_dir": str(summary_dir),
        "split_manifest": str(Path(args.split_manifest).resolve()) if args.split_manifest else "",
        "formal_batch": args.batch,
        "target_epochs": args.epochs,
        "machine_name": socket.gethostname(),
        "commit_hash": git_commit(),
        "python_executable": sys.executable,
        "status": args.status,
        "epochs_indexed": load_epoch_completion(Path(args.run_dir).resolve()),
    }

    manifest["created_at"] = existing.get("created_at", now)
    manifest["updated_at"] = now
    if args.status == "training_started" and not existing.get("training_started_at"):
        manifest["training_started_at"] = now
    else:
        manifest["training_started_at"] = existing.get("training_started_at", "")
    if args.status == "training_completed":
        manifest["training_completed_at"] = now
    else:
        manifest["training_completed_at"] = existing.get("training_completed_at", "")
    if args.status == "evaluation_completed":
        manifest["evaluation_completed_at"] = now
    else:
        manifest["evaluation_completed_at"] = existing.get("evaluation_completed_at", "")

    if env_info.get("platform"):
        manifest["platform"] = env_info["platform"]
    return {**existing, **manifest}


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = collect_dataset_rows(Path(args.dataset_root).resolve(), skip_hash=True)
    dataset_manifest = build_dataset_manifest(args, dataset_rows)
    write_csv(summary_dir / "dataset_inventory.csv", list(dataset_rows[0].keys()), dataset_rows)
    write_json(summary_dir / "dataset_manifest.json", dataset_manifest)

    env_info, pip_freeze = build_env_info()
    env_snapshot = {
        **env_info,
        "machine_name": socket.gethostname(),
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(summary_dir / "env_snapshot.json", env_snapshot)
    (summary_dir / "pip_freeze.txt").write_text(pip_freeze, encoding="utf-8")

    run_manifest = update_run_manifest(args, env_info)
    write_json(summary_dir / "run_manifest.json", run_manifest)
    print_step("done", f"updated manifests under {summary_dir}")


if __name__ == "__main__":
    main()
