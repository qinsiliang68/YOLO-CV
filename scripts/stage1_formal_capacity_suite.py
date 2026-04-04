from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_common import REPO_ROOT, YOLOV11_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal stage-1 capacity scan suite.")
    parser.add_argument("--config", required=True, help="Formal suite config path.")
    parser.add_argument("--task-kind", choices=("gate", "cls6"), required=True, help="Formal task kind.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned steps without executing them.")
    parser.add_argument("--rerun", action="store_true", help="Archive existing run/material dirs before rerunning.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")
    return payload


def resolve_path(value: str | None, *, base: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return base.resolve()
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def resolve_str(value: object, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def run_python(script: str, args: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, str(REPO_ROOT / script), *args]
    print_step("run", " ".join(f'"{part}"' if " " in part else part for part in cmd))
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))
    if os.name == "nt":
        env.setdefault("PIN_MEMORY", "False")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def archive_existing_path(target: Path, recycle_root: Path, dry_run: bool) -> None:
    if not target.exists():
        return
    destination = recycle_root / target.name
    print_step("archive", f"{target} -> {destination}")
    if dry_run:
        return
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), str(destination))


def model_stem(model_name: str) -> str:
    return Path(model_name).stem.replace("-cls", "")


def actual_epoch_from_checkpoint(path: Path) -> int | None:
    match = re.fullmatch(r"epoch(\d+)\.pt", path.name)
    if not match:
        return None
    return int(match.group(1)) + 1


def highest_saved_epoch(run_dir: Path) -> int:
    weights_dir = run_dir / "weights"
    if not weights_dir.exists():
        return 0
    epochs = [actual_epoch_from_checkpoint(path) for path in weights_dir.glob("epoch*.pt")]
    valid = [epoch for epoch in epochs if epoch is not None]
    return max(valid) if valid else 0


def find_auto_increment_candidate(run_dir: Path) -> Path | None:
    base_name = run_dir.name
    candidates = [
        path
        for path in run_dir.parent.glob(f"{base_name}*")
        if path.is_dir() and path.name != base_name and re.fullmatch(rf"{re.escape(base_name)}\d+", path.name)
    ]
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda path: (highest_saved_epoch(path), path.name), reverse=True)
    return ranked[0]


def looks_like_log_stub(run_dir: Path) -> bool:
    if not run_dir.exists():
        return True
    allowed = {"stdout.log", "stderr.log"}
    children = [path.name for path in run_dir.iterdir()]
    return all(name in allowed for name in children)


def normalize_run_dir(run_dir: Path, dry_run: bool) -> None:
    if highest_saved_epoch(run_dir) > 0:
        return
    candidate = find_auto_increment_candidate(run_dir)
    if candidate is None or not looks_like_log_stub(run_dir):
        return
    print_step("repair", f"{candidate} -> {run_dir}")
    if dry_run:
        return
    for log_name in ("stdout.log", "stderr.log"):
        stub_log = run_dir / log_name
        candidate_log = candidate / log_name
        if stub_log.exists() and not candidate_log.exists():
            shutil.move(str(stub_log), str(candidate_log))
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.move(str(candidate), str(run_dir))


def build_names(task_kind: str, model_name: str) -> tuple[str, str]:
    stem = model_stem(model_name)
    if task_kind == "gate":
        return f"{stem}_gate2_formal_200ep", f"{stem}_gate2_formal"
    return f"{stem}_cls6_formal_200ep", f"{stem}_cls6_formal"


def build_train_config(
    *,
    task_kind: str,
    suite_cfg: dict[str, Any],
    model_name: str,
    source_dataset: Path,
    project_dir: Path,
    run_name: str,
    resume_value: str | bool,
) -> dict[str, Any]:
    payload = {
        "task": "classify",
        "model": model_name,
        "data": str(source_dataset),
        "epochs": int(suite_cfg.get("epochs", 200) or 200),
        "imgsz": int(suite_cfg.get("imgsz", 640) or 640),
        "batch": int(suite_cfg.get("batch", 24) or 24),
        "device": resolve_str(suite_cfg.get("device"), "0"),
        "workers": int(suite_cfg.get("workers", 4) or 4),
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
        "pretrained": bool(suite_cfg.get("pretrained", True)),
        "patience": int(suite_cfg.get("patience", 0) or 0),
        "optimizer": resolve_str(suite_cfg.get("optimizer"), "auto"),
        "cache": bool(suite_cfg.get("cache", False)),
        "resume": resume_value,
        "save_period": int(suite_cfg.get("save_period", 1) or 1),
        "seed": int(suite_cfg.get("seed", 20260330) or 20260330),
    }
    if task_kind == "gate":
        return payload
    return payload


def maybe_prepare_run(
    *,
    task_kind: str,
    task_name: str,
    config_path: Path,
    dataset_root: Path,
    run_dir: Path,
    summary_dir: Path,
    split_manifest: Path | None,
    normal_class: str,
    batch: int,
    epochs: int,
    status: str,
    dry_run: bool,
) -> None:
    args = [
        "--task-kind",
        task_kind,
        "--task-name",
        task_name,
        "--config-path",
        str(config_path),
        "--dataset-root",
        str(dataset_root),
        "--run-dir",
        str(run_dir),
        "--summary-dir",
        str(summary_dir),
        "--normal-class",
        normal_class,
        "--batch",
        str(batch),
        "--epochs",
        str(epochs),
        "--status",
        status,
    ]
    if split_manifest is not None:
        args.extend(["--split-manifest", str(split_manifest)])
    run_python("scripts/stage1_formal_prepare_run.py", args, dry_run=dry_run)


def maybe_run_evaluator(
    *,
    task_kind: str,
    run_dir: Path,
    data_root: Path,
    summary_dir: Path,
    split_csv: Path | None,
    normal_class: str,
    device: str,
    eval_batch: int,
    imgsz: int,
    dry_run: bool,
) -> None:
    if not dry_run and not (run_dir / "weights").exists():
        return
    if task_kind == "gate":
        if split_csv is None:
            raise SystemExit("Formal gate evaluation requires --split-csv.")
        run_python(
            "scripts/stage1_formal_gate_epoch_eval.py",
            [
                "--run-dir",
                str(run_dir),
                "--data-root",
                str(data_root),
                "--summary-dir",
                str(summary_dir),
                "--split-csv",
                str(split_csv),
                "--normal-class",
                normal_class,
                "--device",
                device,
                "--batch",
                str(eval_batch),
                "--imgsz",
                str(imgsz),
            ],
            dry_run=dry_run,
        )
        return
    run_python(
        "scripts/stage1_formal_cls6_epoch_eval.py",
        [
            "--run-dir",
            str(run_dir),
            "--data-root",
            str(data_root),
            "--summary-dir",
            str(summary_dir),
            "--normal-class",
            normal_class,
            "--device",
            device,
            "--batch",
            str(eval_batch),
            "--imgsz",
            str(imgsz),
        ],
        dry_run=dry_run,
    )


def run_global_summary(
    *,
    task_kind: str,
    materials_root: Path,
    results_dir: Path,
    registry_csv: Path,
    dry_run: bool,
) -> None:
    script = (
        "scripts/stage1_formal_summarize_gate_capacity.py"
        if task_kind == "gate"
        else "scripts/stage1_formal_summarize_cls6_capacity.py"
    )
    run_python(
        script,
        [
            "--materials-root",
            str(materials_root),
            "--results-dir",
            str(results_dir),
            "--registry-csv",
            str(registry_csv),
        ],
        dry_run=dry_run,
    )


def run_suite(args: argparse.Namespace) -> None:
    config_path = resolve_path(args.config, base=YOLOV11_ROOT / "configs" / "runtime")
    suite_cfg = load_json(config_path)
    task_kind = args.task_kind
    print_step("task", f"{task_kind} formal capacity scan ({resolve_str(suite_cfg.get('label'), task_kind)})")

    models = suite_cfg.get("models") or []
    if not isinstance(models, list) or not models:
        raise SystemExit("Formal suite config must define a non-empty 'models' list.")

    source_dataset = resolve_path(
        suite_cfg.get("source_dataset"),
        base=YOLOV11_ROOT / "datasets" / ("sewerml_gate2_train7200" if task_kind == "gate" else "sewerml_cls6_train7200"),
    )
    project_dir = resolve_path(
        suite_cfg.get("project"),
        base=YOLOV11_ROOT / "runs" / ("stage1_formal_gate" if task_kind == "gate" else "stage1_formal_cls6"),
    )
    recycle_root = resolve_path(
        suite_cfg.get("recycle_root"),
        base=REPO_ROOT / "_recycle_bin" / ("stage1_formal_gate" if task_kind == "gate" else "stage1_formal_cls6"),
    )
    temp_config_dir = resolve_path(
        suite_cfg.get("temp_config_dir"),
        base=REPO_ROOT / "$out" / "generated_configs" / "stage1_formal" / ("gate_capacity" if task_kind == "gate" else "cls6_capacity"),
    )
    materials_root = resolve_path(
        suite_cfg.get("materials_root"),
        base=REPO_ROOT / "research" / "materials" / "stage1_formal" / ("gate_capacity" if task_kind == "gate" else "cls6_capacity"),
    )
    results_dir = resolve_path(
        suite_cfg.get("results_dir"),
        base=REPO_ROOT / "research" / "results" / "stage1_formal" / ("gate_capacity" if task_kind == "gate" else "cls6_capacity"),
    )
    registry_csv = resolve_path(
        suite_cfg.get("registry_csv"),
        base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "formal_capacity_scan_registry.csv",
    )
    split_csv = (
        resolve_path(
            suite_cfg.get("split_csv"),
            base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "val_cal_op_split.csv",
        )
        if task_kind == "gate"
        else None
    )
    normal_class = resolve_str(suite_cfg.get("normal_class"), "Normal")
    device = resolve_str(suite_cfg.get("device"), "0")
    imgsz = int(suite_cfg.get("imgsz", 640) or 640)
    batch = int(suite_cfg.get("batch", 24) or 24)
    eval_batch = int(suite_cfg.get("eval_batch", batch) or batch)
    epochs = int(suite_cfg.get("epochs", 200) or 200)

    for model_name in models:
        run_name, summary_name = build_names(task_kind, str(model_name))
        run_dir = project_dir / run_name
        summary_dir = materials_root / summary_name
        temp_config_path = temp_config_dir / f"{run_name}.json"
        stdout_log = run_dir / "stdout.log"
        stderr_log = run_dir / "stderr.log"

        if args.rerun:
            archive_existing_path(run_dir, recycle_root / "runs", dry_run=args.dry_run)
            archive_existing_path(summary_dir, recycle_root / "materials", dry_run=args.dry_run)

        normalize_run_dir(run_dir, dry_run=args.dry_run)

        maybe_prepare_run(
            task_kind=task_kind,
            task_name=f"stage1_formal_{task_kind}_capacity",
            config_path=temp_config_path,
            dataset_root=source_dataset,
            run_dir=run_dir,
            summary_dir=summary_dir,
            split_manifest=split_csv,
            normal_class=normal_class,
            batch=batch,
            epochs=epochs,
            status="planned",
            dry_run=args.dry_run,
        )

        maybe_run_evaluator(
            task_kind=task_kind,
            run_dir=run_dir,
            data_root=source_dataset,
            summary_dir=summary_dir,
            split_csv=split_csv,
            normal_class=normal_class,
            device=device,
            eval_batch=eval_batch,
            imgsz=imgsz,
            dry_run=args.dry_run,
        )
        run_global_summary(
            task_kind=task_kind,
            materials_root=materials_root,
            results_dir=results_dir,
            registry_csv=registry_csv,
            dry_run=args.dry_run,
        )

        max_epoch = 0 if args.dry_run else highest_saved_epoch(run_dir)
        if max_epoch >= epochs:
            print_step("skip", f"{run_name}: checkpoints already reach epoch {max_epoch}/{epochs}")
        else:
            resume_value: str | bool = False
            last_checkpoint = run_dir / "weights" / "last.pt"
            if bool(suite_cfg.get("resume", True)) and last_checkpoint.exists():
                resume_value = str(last_checkpoint)
                print_step("resume", f"{run_name}: resume from {last_checkpoint}")

            train_cfg = build_train_config(
                task_kind=task_kind,
                suite_cfg=suite_cfg,
                model_name=str(model_name),
                source_dataset=source_dataset,
                project_dir=project_dir,
                run_name=run_name,
                resume_value=resume_value,
            )
            if not args.dry_run:
                write_json(temp_config_path, train_cfg)

            maybe_prepare_run(
                task_kind=task_kind,
                task_name=f"stage1_formal_{task_kind}_capacity",
                config_path=temp_config_path,
                dataset_root=source_dataset,
                run_dir=run_dir,
                summary_dir=summary_dir,
                split_manifest=split_csv,
                normal_class=normal_class,
                batch=batch,
                epochs=epochs,
                status="training_started",
                dry_run=args.dry_run,
            )

            train_script = "scripts/stage1_gate_train.py" if task_kind == "gate" else "scripts/cls_pretrain.py"
            run_python(
                train_script,
                [
                    "--config",
                    str(temp_config_path),
                    "--stdout-log",
                    str(stdout_log),
                    "--stderr-log",
                    str(stderr_log),
                ],
                dry_run=args.dry_run,
            )
            maybe_prepare_run(
                task_kind=task_kind,
                task_name=f"stage1_formal_{task_kind}_capacity",
                config_path=temp_config_path,
                dataset_root=source_dataset,
                run_dir=run_dir,
                summary_dir=summary_dir,
                split_manifest=split_csv,
                normal_class=normal_class,
                batch=batch,
                epochs=epochs,
                status="training_completed",
                dry_run=args.dry_run,
            )

        maybe_run_evaluator(
            task_kind=task_kind,
            run_dir=run_dir,
            data_root=source_dataset,
            summary_dir=summary_dir,
            split_csv=split_csv,
            normal_class=normal_class,
            device=device,
            eval_batch=eval_batch,
            imgsz=imgsz,
            dry_run=args.dry_run,
        )
        maybe_prepare_run(
            task_kind=task_kind,
            task_name=f"stage1_formal_{task_kind}_capacity",
            config_path=temp_config_path,
            dataset_root=source_dataset,
            run_dir=run_dir,
            summary_dir=summary_dir,
            split_manifest=split_csv,
            normal_class=normal_class,
            batch=batch,
            epochs=epochs,
            status="evaluation_completed",
            dry_run=args.dry_run,
        )
        run_global_summary(
            task_kind=task_kind,
            materials_root=materials_root,
            results_dir=results_dir,
            registry_csv=registry_csv,
            dry_run=args.dry_run,
        )


def main() -> None:
    args = parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
