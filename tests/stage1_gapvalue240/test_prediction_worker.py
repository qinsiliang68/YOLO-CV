from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.prediction_controller import (
    PredictionWorkerSpec,
    run_prediction_workers,
)
from stage1_gapvalue240.prediction_worker import PredictionJob, execute_prediction_job
from stage1_gapvalue240.util import atomic_write_bytes, sha256_file


def _job(tmp_path: Path, split: str = "val_cal") -> PredictionJob:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    yolo_root = tmp_path / "YOLOv11"
    yolo_root.mkdir()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    defect = tmp_path / f"{split}_defect.csv"
    normal = tmp_path / f"{split}_normal.csv"
    pd.DataFrame({"sample_id": ["d1", "d2"], "image_path": ["d1.png", "d2.png"]}).to_csv(defect, index=False)
    pd.DataFrame({"sample_id": ["n1"], "image_path": ["n1.png"]}).to_csv(normal, index=False)
    return PredictionJob(
        split=split,
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        defect_manifest=defect,
        normal_manifest=normal,
        output=tmp_path / f"{split}_predictions.csv",
        result_json=tmp_path / f"{split}_worker_result.json",
        yolo_root=yolo_root,
        gpu_id="0",
        batch=32,
        workers=0,
        imgsz=224,
        accepted_defect_names=("defect", "target_defect"),
    )


def test_campaign_causal_probe_is_an_allowed_prediction_split(tmp_path: Path) -> None:
    job = _job(tmp_path, split="causal_train_probe")

    def fake_predict(**kwargs):
        pd.DataFrame(
            {"sample_id": ["d1", "d2", "n1"], "y_true": [1, 1, 0], "score": [0.9, 0.8, 0.1]}
        ).to_csv(kwargs["output"], index=False)

    result = execute_prediction_job(job, predict_fn=fake_predict)

    assert result["status"] == "PASS"


def test_prediction_worker_writes_atomic_pass_result_with_exact_inputs(tmp_path):
    job = _job(tmp_path)

    def fake_predict(**kwargs):
        assert Path(kwargs["checkpoint"]).resolve() == job.checkpoint.resolve()
        assert Path(kwargs["yolo_root"]).resolve() == job.yolo_root.resolve()
        frame = pd.DataFrame(
            {"sample_id": ["d1", "d2", "n1"], "y_true": [1, 1, 0], "score": [0.9, 0.8, 0.1]}
        )
        return atomic_write_bytes(kwargs["output"], frame.to_csv(index=False).encode())

    result = execute_prediction_job(job, predict_fn=fake_predict)

    assert result["status"] == "PASS"
    assert result["exit_code"] == 0
    assert result["split"] == "val_cal"
    assert result["checkpoint_sha256"] == sha256_file(job.checkpoint)
    assert result["row_count"] == 3
    assert result["defect_count"] == 2
    assert result["normal_count"] == 1
    assert result["output_sha256"] == sha256_file(job.output)
    assert json.loads(job.result_json.read_text(encoding="utf-8")) == result
    assert not list(tmp_path.glob("*.tmp"))


def test_prediction_worker_failure_has_result_and_no_final_prediction(tmp_path):
    job = _job(tmp_path)

    def fail_predict(**_kwargs):
        raise RuntimeError("synthetic prediction failure")

    result = execute_prediction_job(job, predict_fn=fail_predict)

    assert result["status"] == "PREDICTION_FAILED"
    assert result["exit_code"] != 0
    assert result["error_type"] == "RuntimeError"
    assert "synthetic prediction failure" in result["error"]
    assert job.result_json.exists()
    assert not job.output.exists()


def test_prediction_worker_never_deletes_a_preexisting_output(tmp_path):
    job = _job(tmp_path)
    job.output.write_text("owner,evidence\nuser,keep\n", encoding="utf-8")
    original = job.output.read_bytes()

    result = execute_prediction_job(job, predict_fn=lambda **_kwargs: None)

    assert result["status"] == "INVALID_INPUT"
    assert result["exit_code"] == 2
    assert job.output.read_bytes() == original


def test_controller_runs_val_cal_then_val_op_in_distinct_worker_processes(tmp_path):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(
        """
import argparse, csv, hashlib, json, os
from pathlib import Path
p=argparse.ArgumentParser()
for name in ['split','checkpoint','dataset-root','defect-manifest','normal-manifest','output','result-json','yolo-root','gpu-id','batch','workers','imgsz']:
    p.add_argument('--'+name, required=True)
p.add_argument('--accepted-defect-name', action='append')
a=p.parse_args()
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text('sample_id,y_true,score\\n'+a.split+',1,0.9\\n',encoding='utf-8')
h=hashlib.sha256(out.read_bytes()).hexdigest().upper()
result={'schema_version':'stage1_gapvalue240_prediction_worker_v1','status':'PASS','exit_code':0,'split':a.split,'pid':os.getpid(),'output':str(out.resolve()),'output_sha256':h,'row_count':1,'defect_count':1,'normal_count':0}
Path(a.result_json).write_text(json.dumps(result),encoding='utf-8')
""".strip(),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"x")
    yolo_root = tmp_path / "YOLOv11"
    yolo_root.mkdir()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id,image_path\nx,x.png\n", encoding="utf-8")

    specs = []
    for split in ("val_cal", "val_op"):
        specs.append(
            PredictionWorkerSpec(
                split=split,
                defect_manifest=manifest,
                normal_manifest=manifest,
                output=tmp_path / f"{split}.csv",
                result_json=tmp_path / f"{split}.json",
                log_path=tmp_path / f"{split}.log",
            )
        )
    controller_result = tmp_path / "controller.json"

    report = run_prediction_workers(
        specs=specs,
        python_executable=sys.executable,
        worker_script=worker,
        cwd=tmp_path,
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        yolo_root=yolo_root,
        gpu_id="0",
        batch=16,
        workers=0,
        imgsz=224,
        accepted_defect_names=("defect",),
        controller_result_json=controller_result,
        timeout_seconds=10,
    )

    assert report["status"] == "PASS"
    assert [row["split"] for row in report["workers"]] == ["val_cal", "val_op"]
    assert len({row["pid"] for row in report["workers"]}) == 2
    assert all(spec.output.exists() and spec.result_json.exists() and spec.log_path.exists() for spec in specs)
    assert json.loads(controller_result.read_text(encoding="utf-8"))["status"] == "PASS"


def test_controller_rejects_noncanonical_split_order(tmp_path):
    spec = PredictionWorkerSpec("val_op", tmp_path / "d", tmp_path / "n", tmp_path / "o", tmp_path / "r", tmp_path / "l")
    with pytest.raises(ValueError, match="val_cal.*val_op"):
        run_prediction_workers(
            specs=[spec], python_executable=sys.executable, worker_script=tmp_path / "worker.py", cwd=tmp_path,
            checkpoint=tmp_path / "best.pt", dataset_root=tmp_path, yolo_root=tmp_path, gpu_id="0",
            batch=1, workers=0, imgsz=224, accepted_defect_names=("defect",),
            controller_result_json=tmp_path / "controller.json", timeout_seconds=1,
        )
