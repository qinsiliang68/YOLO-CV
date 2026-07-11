from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .errors import ExternalCommandError, ValidationError
from .subprocesses import run_logged
from .util import atomic_write_json, sha256_file


def inspect_checkpoint(
    path: str | Path,
    *,
    require_resume_state: bool,
    yolo_root: str | Path | None = None,
) -> dict:
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        if yolo_root is not None:
            root = Path(yolo_root).resolve()
            if not root.is_dir():
                raise FileNotFoundError(root)
            sys.path.insert(0, str(root))
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValidationError(f"Checkpoint cannot be loaded: {checkpoint_path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValidationError(f"Checkpoint root is not a mapping: {checkpoint_path}")
    epoch = int(checkpoint.get("epoch", -1))
    train_args = checkpoint.get("train_args")
    resume_fields = {"epochs", "data", "model", "batch", "imgsz", "seed"}
    has_model_state = checkpoint.get("ema") is not None or checkpoint.get("model") is not None
    resumable = (
        epoch >= 0
        and isinstance(checkpoint.get("optimizer"), dict)
        and isinstance(train_args, dict)
        and resume_fields.issubset(train_args)
        and has_model_state
    )
    if require_resume_state and not resumable:
        raise ValidationError(f"Checkpoint is not resumable: {checkpoint_path}")
    yolo_reload = None
    if yolo_root is not None:
        try:
            from ultralytics import YOLO

            model = YOLO(str(checkpoint_path))
            yolo_reload = {
                "task": str(getattr(model, "task", "")),
                "model_type": type(getattr(model, "model", None)).__name__,
            }
            if getattr(model, "model", None) is None:
                raise ValidationError("YOLO reload did not construct a model")
            del model
        except Exception as exc:
            raise ValidationError(f"Checkpoint failed local YOLO reload: {checkpoint_path}: {exc}") from exc
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "epoch": epoch,
        "resumable": resumable,
        "yolo_reload": yolo_reload,
    }


def run_checkpoint_probe_worker(
    *,
    python_executable: str,
    worker_script: str | Path,
    checkpoint: str | Path,
    result_json: str | Path,
    log_path: str | Path,
    cwd: str | Path,
    require_resume_state: bool,
    yolo_root: str | Path,
    timeout_seconds: float = 120,
) -> dict:
    result_path = Path(result_json).resolve()
    command = [
        str(python_executable), str(Path(worker_script).resolve()),
        "--checkpoint", str(Path(checkpoint).resolve()),
        "--result-json", str(result_path),
        "--yolo-root", str(Path(yolo_root).resolve()),
    ]
    if require_resume_state:
        command.append("--require-resume-state")
    try:
        run_logged(command, cwd, log_path, timeout=timeout_seconds)
    except ExternalCommandError:
        if result_path.is_file():
            return json.loads(result_path.read_text(encoding="utf-8"))
        raise
    return json.loads(result_path.read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reload and audit one YOLO checkpoint in an isolated process.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--yolo-root", required=True)
    parser.add_argument("--require-resume-state", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.time()
    try:
        report = inspect_checkpoint(
            args.checkpoint,
            require_resume_state=args.require_resume_state,
            yolo_root=args.yolo_root,
        )
        report["exit_code"] = 0
    except Exception as exc:
        report = {
            "status": "FAIL",
            "exit_code": 30,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    report.update({"pid": __import__("os").getpid(), "duration_seconds": time.time() - started})
    atomic_write_json(args.result_json, report, overwrite=True)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
