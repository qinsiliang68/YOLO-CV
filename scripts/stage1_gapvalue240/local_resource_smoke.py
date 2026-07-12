from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from stage1_gapvalue240.subprocesses import run_logged
from stage1_gapvalue240.util import atomic_write_json, sha256_file


def _gpu_memory_mib(nvidia_smi: str, gpu_id: str) -> int:
    output = subprocess.check_output([
        nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu_id)
    ], text=True).strip().splitlines()
    return int(output[0].strip())


def _link_rows(dataset_root: Path, manifest: Path, destination: Path, count: int) -> None:
    frame = pd.read_csv(manifest, nrows=count)
    destination.mkdir(parents=True, exist_ok=True)
    for row in frame.to_dict("records"):
        source = (dataset_root / str(row["canonical_image_relpath"])).resolve()
        target = destination / str(row["Filename"])
        if not source.is_file():
            raise FileNotFoundError(source)
        os.link(source, target)


def _worker(args: argparse.Namespace) -> int:
    from stage1_gapvalue240.formal_trainer import FormalTrainingSpec, run_formal_training

    spec = FormalTrainingSpec(
        dataset_dir=Path(args.dataset), checkpoint=Path(args.checkpoint), output_dir=Path(args.output),
        yolo_root=Path(args.yolo_root), epochs=args.epochs, batch=128, imgsz=224,
        seed=args.seed, device=args.gpu_id, workers=args.workers, expected_steps_per_epoch=1,
    )
    result = run_formal_training(spec)
    print(json.dumps({"status": "PASS", "audit": str(result.audit_path), "results": str(result.results_csv)}))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run two isolated non-scientific YOLO11l resource smoke jobs.")
    parser.add_argument("--machine-config")
    parser.add_argument("--output-root")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-per-class", type=int, default=24)
    parser.add_argument("--val-per-class", type=int, default=8)
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--yolo-root", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-id", default="0", help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=8, help=argparse.SUPPRESS)
    return parser


def _default_output_root(machine) -> Path:
    return machine.path_value("output_root") / "runtime_validation/local_resource_smoke"


def _temporary_parent(machine) -> Path:
    # Smoke images are hardlinks. They must be created on the dataset/staging
    # volume instead of the system TEMP volume, which may be redirected to D:.
    return machine.path_value("staging_root").parent / ".resource_smoke_tmp"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker:
        return _worker(args)
    if not args.machine_config:
        raise ValueError("--machine-config is required")
    from stage1_gapvalue240.machine import load_machine_config

    machine = load_machine_config(args.machine_config)
    repo = machine.path_value("repo_root")
    output_root = Path(args.output_root).resolve() if args.output_root else _default_output_root(machine)
    output_root.mkdir(parents=True, exist_ok=True)
    nvidia_smi = str(machine.data.get("nvidia_smi_path") or "nvidia-smi")
    gpu_id = str(machine.data["gpu_id"])
    baseline = _gpu_memory_mib(nvidia_smi, gpu_id)
    report = {"status": "RUNNING", "baseline_gpu_memory_mib": baseline, "runs": []}
    temporary_parent = _temporary_parent(machine)
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gapvalue240_resource_smoke_", dir=temporary_parent) as temp_name:
        tiny = Path(temp_name) / "dataset"
        _link_rows(machine.path_value("dataset_root"), machine.path_value("normal_train_manifest"), tiny / "train/no_target", args.train_per_class)
        _link_rows(machine.path_value("dataset_root"), machine.path_value("train_manifest"), tiny / "train/target_defect", args.train_per_class)
        _link_rows(machine.path_value("dataset_root"), machine.path_value("val_model_normal_manifest"), tiny / "val/no_target", args.val_per_class)
        _link_rows(machine.path_value("dataset_root"), machine.path_value("val_model_defect_manifest"), tiny / "val/target_defect", args.val_per_class)
        for index in range(1, args.runs + 1):
            run_dir = output_root / f"run_{index:02d}"
            if run_dir.exists():
                raise FileExistsError(run_dir)
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker", "--dataset", str(tiny),
                "--checkpoint", str(machine.path_value("base_checkpoint")), "--yolo-root", str(repo / "YOLOv11"),
                "--output", str(run_dir), "--epochs", str(args.epochs), "--seed", str(9000 + index),
                "--gpu-id", gpu_id, "--workers", str(int(machine.data["num_workers"])),
            ]
            started = time.time()
            run_logged(command, repo, output_root / f"run_{index:02d}.log", timeout=1800)
            time.sleep(3)
            post_memory = _gpu_memory_mib(nvidia_smi, gpu_id)
            audit = json.loads((run_dir / "training_execution_audit.json").read_text(encoding="utf-8"))
            results = pd.read_csv(run_dir / "trainer/results.csv")
            row = {
                "run": index, "duration_seconds": time.time() - started,
                "epochs": len(results), "steps_per_epoch": audit["observed_steps_per_epoch"],
                "post_gpu_memory_mib": post_memory, "gpu_memory_delta_mib": post_memory - baseline,
                "best_sha256": sha256_file(run_dir / "trainer/weights/best.pt"),
                "last_sha256": sha256_file(run_dir / "training_state/last.pt"),
            }
            report["runs"].append(row)
            if not args.keep_checkpoints:
                shutil.rmtree(run_dir / "trainer/weights")
                shutil.rmtree(run_dir / "training_state")
    report["status"] = "PASS" if all(
        row["epochs"] == args.epochs and row["steps_per_epoch"] == [1] * args.epochs
        for row in report["runs"]
    ) else "FAIL"
    report["orphan_python_process_check"] = "manual/AIOps process inventory required"
    atomic_write_json(output_root / "resource_smoke_report.json", report, overwrite=True)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 30


if __name__ == "__main__":
    raise SystemExit(main())
