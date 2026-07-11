from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stage1_gapvalue240.contract import Contract, load_contract
from stage1_gapvalue240.evaluation import finalize_evaluation
from stage1_gapvalue240.integration import trainer_command
from stage1_gapvalue240.machine import MachineConfig
from stage1_gapvalue240.manifests import build_replay_manifests
from stage1_gapvalue240.predictor import predict_split
from stage1_gapvalue240.runtime import ensure_ultralytics_runtime
from stage1_gapvalue240.subprocesses import run_logged
from stage1_gapvalue240.util import atomic_write_json, sha256_file


SMOKE_SPECS = (
    ("smoke_01_T_normal", "n", ("n", 0, 3)),
    ("smoke_02_R1_normal", "n", ("n", 3, 6)),
    ("smoke_03_R2_normal", "n", ("n", 6, 9)),
    ("smoke_04_guard", "n", ("guard", 0, 0)),
    ("smoke_05_formal_l", "l", ("n", 9, 12)),
)


def _read_head(path: Path, rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path, nrows=rows, dtype={"canonical_image_relpath": "string"})
    if len(frame) != rows:
        raise ValueError(f"Expected {rows} rows from {path}, got {len(frame)}")
    return frame


def _selection(run_slot: str, mode: tuple[str, int, int], normal: pd.DataFrame, defect: pd.DataFrame) -> pd.DataFrame:
    kind, start, end = mode
    if kind == "guard":
        chosen = pd.concat([normal.iloc[:2].assign(y_true=0), defect.iloc[:1].assign(y_true=1)], ignore_index=True)
        roles = ["normal_replay", "normal_replay", "defect_guard"]
    else:
        chosen = normal.iloc[start:end].copy().assign(y_true=0)
        roles = ["normal_replay"] * len(chosen)
    return pd.DataFrame(
        {
            "run_slot": run_slot,
            "rank": range(1, len(chosen) + 1),
            "sample_id": chosen.canonical_image_relpath.astype(str),
            "y_true": chosen.y_true.astype(int),
            "replay_role": roles,
        }
    )


def _smoke_contract(base: Contract, model_code: str, epochs: int, batch: int) -> Contract:
    data = copy.deepcopy(base.data)
    data["training"]["model_code"] = model_code
    data["training"]["trainer_cli_fixed_args"] = [
        "--mode", "full", "--models", model_code, "--epochs", str(epochs), "--imgsz", "224",
        "--batch", str(batch), "--save-period", "-1", "--keep-data",
    ]
    return Contract(base.path, data, "SMOKE_NONSCIENTIFIC")


def _unique(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one {name} under {root}, got {len(hits)}")
    return hits[0]


def run_smoke(output_root: Path, epochs: int = 3, batch: int = 8, device: str = "0", runs: int = 5) -> dict:
    if not 1 <= runs <= len(SMOKE_SPECS):
        raise ValueError(f"--runs must be in [1,{len(SMOKE_SPECS)}]")
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Smoke output must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = REPO_ROOT / "data/final_sewerml_dataset"
    source = dataset_root / "manifests"
    fixture = output_root / "fixture"
    fixture.mkdir()
    defect = _read_head(source / "train_manifest.csv", 12)
    normal = _read_head(source / "normal_train_manifest.csv", 12)
    val_defect = _read_head(source / "val_model_manifest.csv", 4)
    val_normal = _read_head(source / "normal_val_model_manifest.csv", 4)
    cal_defect = _read_head(source / "val_cal_manifest.csv", 8)
    cal_normal = _read_head(source / "normal_val_cal_manifest.csv", 12)
    op_defect = _read_head(source / "val_op_manifest.csv", 10)
    op_normal = _read_head(source / "normal_val_op_manifest.csv", 20)
    frames = {
        "train_manifest.csv": defect,
        "normal_train_manifest.csv": normal,
        "val_model_manifest.csv": val_defect,
        "normal_val_model_manifest.csv": val_normal,
        "val_cal_manifest.csv": cal_defect,
        "normal_val_cal_manifest.csv": cal_normal,
        "val_op_manifest.csv": op_defect,
        "normal_val_op_manifest.csv": op_normal,
    }
    for name, frame in frames.items():
        frame.to_csv(fixture / name, index=False)
    base_contract = load_contract(REPO_ROOT / "configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml")
    summaries = []
    for index, (run_name, model_code, selection_mode) in enumerate(SMOKE_SPECS[:runs], start=1):
        started = time.time()
        run_root = output_root / run_name
        manifests = run_root / "manifests"
        manifests.mkdir(parents=True)
        selection = _selection(run_name, selection_mode, normal, defect)
        selection_path = manifests / "selection_manifest.csv"
        selection.to_csv(selection_path, index=False)
        built = build_replay_manifests(
            fixture / "train_manifest.csv", fixture / "normal_train_manifest.csv", selection_path, manifests,
            expected_base_total=24,
        )
        shutil.copy2(fixture / "val_model_manifest.csv", manifests / "val_model_manifest.csv")
        shutil.copy2(fixture / "normal_val_model_manifest.csv", manifests / "normal_val_model_manifest.csv")
        contract = _smoke_contract(base_contract, model_code, epochs, batch)
        machine = MachineConfig(
            path=run_root / "machine_smoke.yaml",
            data={
                "machine_id": "local_smoke",
                "repo_root": str(REPO_ROOT),
                "dataset_root": str(dataset_root),
                "artifact_root": str(run_root),
                "output_root": str(run_root),
                "cache_root": str(run_root / "cache"),
                "gpu_id": device,
                "num_workers": 0,
                "python_executable": sys.executable,
                "base_checkpoint": str(REPO_ROOT / f"yolo11{model_code}-cls.pt"),
            },
        )
        trainer_root = run_root / "trainer"
        cmd = trainer_command(contract, machine, built.train_manifest, built.normal_train_manifest, trainer_root, 2026071100 + index)
        ensure_ultralytics_runtime(run_root / "cache")
        run_logged(
            cmd,
            REPO_ROOT,
            run_root / "train.log",
            env={"CUDA_VISIBLE_DEVICES": str(device), "YOLO_CONFIG_DIR": str(run_root / "cache")},
        )
        best = _unique(trainer_root, "best.pt")
        last = _unique(trainer_root, "last.pt")
        prediction_root = run_root / "predictions"
        prediction_root.mkdir()
        accepted = ["defect", "def", "1", "abnormal", "target_defect"]
        predict_split(
            best, dataset_root, fixture / "val_cal_manifest.csv", fixture / "normal_val_cal_manifest.csv",
            prediction_root / "val_cal_predictions.csv", device, batch, 0, 224, accepted, REPO_ROOT / "YOLOv11",
        )
        predict_split(
            best, dataset_root, fixture / "val_op_manifest.csv", fixture / "normal_val_op_manifest.csv",
            prediction_root / "val_op_predictions.csv", device, batch, 0, 224, accepted, REPO_ROOT / "YOLOv11",
        )
        metrics = finalize_evaluation(
            prediction_root / "val_cal_predictions.csv", prediction_root / "val_op_predictions.csv", run_root / "metrics",
        )
        summary = {
            "run": run_name,
            "model_code": model_code,
            "epochs": epochs,
            "base_rows": 24,
            "replay_rows": 3,
            "epoch_samples": 27,
            "selection_rows": len(selection),
            "best_checkpoint": str(best),
            "best_sha256": sha256_file(best),
            "last_checkpoint": str(last),
            "val_cal_prediction_rows": len(pd.read_csv(prediction_root / "val_cal_predictions.csv")),
            "val_op_prediction_rows": len(pd.read_csv(prediction_root / "val_op_predictions.csv")),
            "metric_version": metrics["metric_version"],
            "duration_seconds": time.time() - started,
            "status": "PASS",
        }
        atomic_write_json(run_root / "smoke_run_summary.json", summary)
        summaries.append(summary)
    report = {"status": "PASS", "runs": summaries, "run_count": len(summaries), "created_at": time.time()}
    atomic_write_json(output_root / "SMOKE_COMPLETE.json", report)
    pd.DataFrame(summaries).to_csv(output_root / "smoke_summary.csv", index=False)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run five real-image, three-epoch GapValue integration smoke jobs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args(argv)
    print(json.dumps(run_smoke(args.output_root, args.epochs, args.batch, args.device, args.runs), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
