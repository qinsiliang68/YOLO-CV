from __future__ import annotations

MAIN_COMMIT = "a70ba60485dd32c2f8b4268b8f28ea2d3549f42f"
TASKBOOK_BLOB_SHA = "b201d021712e9c6614e119d35f0e14bdf405c6be"
TASKBOOK_PATH = "artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/03_preregistration_v4_sctsr/SCTSR_EXPERT_IMPLEMENTATION_TASKBOOK.md"


def source_external_references() -> list[dict[str, str]]:
    return [
        {"repository": "qinsiliang68/YOLO-CV", "commit": MAIN_COMMIT, "path": TASKBOOK_PATH, "blob_sha": TASKBOOK_BLOB_SHA, "role": "IMPLEMENTATION_TASKBOOK"},
        {"repository": "qinsiliang68/YOLO-CV", "commit": MAIN_COMMIT, "path": "YOLOv11/ultralytics/engine/trainer.py", "blob_sha": "1e5cec6b4b36f30d7d6653eb70420ae2b4cb3524", "role": "UPSTREAM_TRAINER_REFERENCE"},
        {"repository": "qinsiliang68/YOLO-CV", "commit": MAIN_COMMIT, "path": "YOLOv11/ultralytics/models/yolo/classify/train.py", "blob_sha": "779e940d8c4d9e3abd1bce42a5ac39b9127cd1d3", "role": "UPSTREAM_CLASSIFICATION_TRAINER_REFERENCE"},
        {"repository": "qinsiliang68/YOLO-CV", "commit": MAIN_COMMIT, "path": "YOLOv11/ultralytics/nn/tasks.py", "blob_sha": "bcaf6999ea8a81c82502366f2cc3fd3e921ea0ee", "role": "UPSTREAM_CLASSIFICATION_MODEL_REFERENCE"},
        {"repository": "qinsiliang68/YOLO-CV", "commit": MAIN_COMMIT, "path": "pyproject.toml", "blob_sha": "c08bf04a1242d4853272117254dd81c88d157e7b", "role": "DEPENDENCY_REFERENCE"},
    ]
