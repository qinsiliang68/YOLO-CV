from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline_common import YOLOV11_ROOT, load_json_config, resolve_relative_path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CONFIG = YOLOV11_ROOT / "configs" / "runtime" / "cls_source_cls6.json"
DEFAULT_MATRIX_CONFIG = YOLOV11_ROOT / "configs" / "runtime" / "cls_source_experiments.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch of source-domain classification experiments.")
    parser.add_argument("--base-config", default=str(DEFAULT_SOURCE_CONFIG), help="Base source training config JSON.")
    parser.add_argument("--matrix-config", default=str(DEFAULT_MATRIX_CONFIG), help="Experiment matrix JSON.")
    parser.add_argument(
        "--workspace",
        default="research/source_experiments",
        help="Workspace directory for generated configs and experiment summaries.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on the number of runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing them.")
    return parser.parse_args()


def resolve_repo_path(text: str | Path) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def sanitize_piece(value: Any) -> str:
    text = str(value).strip().replace("\\", "/").replace(".pt", "")
    chars = []
    for ch in text:
        chars.append(ch if ch.isalnum() else "_")
    return "".join(chars).strip("_").lower()


def load_matrix_config(path: Path) -> dict[str, Any]:
    matrix = load_json_config(path)
    if not isinstance(matrix, dict):
        raise SystemExit(f"Experiment matrix must be a JSON object: {path}")
    if not isinstance(matrix.get("grid"), dict):
        raise SystemExit(f"Experiment matrix must contain a 'grid' object: {path}")
    return matrix


def build_experiment_configs(base_cfg: dict[str, Any], matrix_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    common = matrix_cfg.get("common") or {}
    if not isinstance(common, dict):
        raise SystemExit("'common' must be a JSON object when provided.")

    grid = matrix_cfg["grid"]
    keys = list(grid.keys())
    value_lists: list[list[Any]] = []
    for key in keys:
        values = grid[key]
        if not isinstance(values, list) or not values:
            raise SystemExit(f"Experiment grid field '{key}' must be a non-empty array.")
        value_lists.append(values)

    experiments: list[dict[str, Any]] = []
    name_prefix = str(matrix_cfg.get("name_prefix") or "srcexp").strip() or "srcexp"
    project_value = matrix_cfg.get("project")
    if project_value:
        common["project"] = project_value

    for index, combo in enumerate(itertools.product(*value_lists), start=1):
        overrides = dict(common)
        overrides.update(dict(zip(keys, combo, strict=True)))
        cfg = dict(base_cfg)
        cfg.update(overrides)

        pieces = [name_prefix, f"{index:02d}"]
        for key, value in zip(keys, combo, strict=True):
            pieces.append(f"{sanitize_piece(key)}_{sanitize_piece(value)}")
        cfg["name"] = "_".join(piece for piece in pieces if piece)
        experiments.append(cfg)

    return experiments


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def run_training(config_path: Path, dry_run: bool) -> int:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "cls_pretrain.py"), "--config", str(config_path)]
    print("[run]", " ".join(f'"{part}"' if " " in part else part for part in cmd))
    if dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))
    if os.name == "nt":
        env.setdefault("PIN_MEMORY", "False")
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return completed.returncode


def results_csv_path(config: dict[str, Any]) -> Path:
    project = resolve_repo_path(resolve_relative_path(config.get("project"), YOLOV11_ROOT) or "")
    name = str(config.get("name") or "").strip()
    return project / name / "results.csv"


def weights_path(config: dict[str, Any]) -> Path:
    project = resolve_repo_path(resolve_relative_path(config.get("project"), YOLOV11_ROOT) or "")
    name = str(config.get("name") or "").strip()
    return project / name / "weights" / "best.pt"


def summarize_run(config: dict[str, Any], status: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "name": config.get("name", ""),
        "model": config.get("model", ""),
        "imgsz": config.get("imgsz", ""),
        "batch": config.get("batch", ""),
        "workers": config.get("workers", ""),
        "epochs": config.get("epochs", ""),
        "project": config.get("project", ""),
        "status": status,
        "best_epoch": "",
        "best_top1": "",
        "best_top5": "",
        "best_val_loss": "",
        "results_csv": str(results_csv_path(config)),
        "best_weights": str(weights_path(config)),
    }

    csv_path = results_csv_path(config)
    if not csv_path.exists():
        return summary

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return summary

    def top1_value(row: dict[str, str]) -> float:
        return float(row.get("metrics/accuracy_top1", "nan"))

    best_row = max(rows, key=top1_value)
    summary["best_epoch"] = best_row.get("epoch", "")
    summary["best_top1"] = best_row.get("metrics/accuracy_top1", "")
    summary["best_top5"] = best_row.get("metrics/accuracy_top5", "")
    summary["best_val_loss"] = best_row.get("val/loss", "")
    return summary


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "model",
        "imgsz",
        "batch",
        "workers",
        "epochs",
        "project",
        "status",
        "best_epoch",
        "best_top1",
        "best_top5",
        "best_val_loss",
        "results_csv",
        "best_weights",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_config_path = resolve_repo_path(args.base_config)
    matrix_config_path = resolve_repo_path(args.matrix_config)
    workspace = resolve_repo_path(args.workspace)
    generated_dir = workspace / "generated_configs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = workspace / "latest_summary.csv"

    base_cfg = load_json_config(base_config_path)
    matrix_cfg = load_matrix_config(matrix_config_path)
    experiments = build_experiment_configs(base_cfg, matrix_cfg)
    if args.limit > 0:
        experiments = experiments[: args.limit]

    summaries: list[dict[str, Any]] = []
    for index, config in enumerate(experiments, start=1):
        config_path = generated_dir / f"{index:02d}_{config['name']}.json"
        write_json(config_path, config)
        print(f"[experiment] {index}/{len(experiments)} {config['name']}")

        status = "dry_run"
        if not args.dry_run:
            return_code = run_training(config_path, dry_run=False)
            status = "ok" if return_code == 0 else f"failed:{return_code}"
        else:
            run_training(config_path, dry_run=True)

        summaries.append(summarize_run(config, status=status))
        write_summary(summary_path, summaries)

        if status.startswith("failed"):
            print(f"[stop] experiment failed: {config['name']}")
            break

    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
