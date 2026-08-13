from __future__ import annotations

MAIN_COMMIT = "a70ba60485dd32c2f8b4268b8f28ea2d3549f42f"
TASKBOOK_BLOB_SHA = "b201d021712e9c6614e119d35f0e14bdf405c6be"
TASKBOOK_PATH = "artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/03_preregistration_v4_sctsr/SCTSR_EXPERT_IMPLEMENTATION_TASKBOOK.md"
SOURCE_TREE_INCLUDE_PATHS = (
    "stage1_sctsr_v4",
    "scripts/stage1_sctsr_v4",
    "configs/stage1_sctsr_v4",
    "tests/stage1_sctsr_v4",
    "docs/stage1_sctsr_v4",
    "pyproject.toml",
    "uv.lock",
    "requirements-sctsr-v4.txt",
    "README.md",
    TASKBOOK_PATH,
    "YOLOv11/ultralytics/engine/trainer.py",
    "YOLOv11/ultralytics/models/yolo/classify/train.py",
    "YOLOv11/ultralytics/nn/tasks.py",
    "YOLOv11/ultralytics/nn/modules/head.py",
    "YOLOv11/ultralytics/utils/loss.py",
    "YOLOv11/ultralytics/utils/torch_utils.py",
)


def source_external_references() -> list[dict[str, str]]:
    return [
        {"repository": "qinsiliang68/YOLO-CV", "commit": MAIN_COMMIT, "path": TASKBOOK_PATH, "blob_sha": TASKBOOK_BLOB_SHA, "role": "IMPLEMENTATION_TASKBOOK"},
        {"repository": "qinsiliang68/YOLO-CV", "commit": MAIN_COMMIT, "path": "YOLOv11/ultralytics/engine/trainer.py", "blob_sha": "1e5cec6b4b36f30d7d6653eb70420ae2b4cb3524", "role": "UPSTREAM_TRAINER_REFERENCE"},
        {"repository": "qinsiliang68/YOLO-CV", "commit": MAIN_COMMIT, "path": "YOLOv11/ultralytics/models/yolo/classify/train.py", "blob_sha": "779e940d8c4d9e3abd1bce42a5ac39b9127cd1d3", "role": "UPSTREAM_CLASSIFICATION_TRAINER_REFERENCE"},
        {"repository": "qinsiliang68/YOLO-CV", "commit": MAIN_COMMIT, "path": "YOLOv11/ultralytics/nn/tasks.py", "blob_sha": "bcaf6999ea8a81c82502366f2cc3fd3e921ea0ee", "role": "UPSTREAM_CLASSIFICATION_MODEL_REFERENCE"},
        {"repository": "qinsiliang68/YOLO-CV", "commit": MAIN_COMMIT, "path": "YOLOv11/ultralytics/nn/modules/head.py", "blob_sha": "c1ec140b101e0111851cda55466df6c8b737da3e", "role": "UPSTREAM_CLASSIFICATION_HEAD_REFERENCE"},
        {"repository": "qinsiliang68/YOLO-CV", "commit": MAIN_COMMIT, "path": "YOLOv11/ultralytics/utils/loss.py", "blob_sha": "d4a9dda93caff84cf0f8dae4e84e0e446e55f85e", "role": "UPSTREAM_CLASSIFICATION_LOSS_REFERENCE"},
        {"repository": "qinsiliang68/YOLO-CV", "commit": MAIN_COMMIT, "path": "YOLOv11/ultralytics/utils/torch_utils.py", "blob_sha": "537f0d5621bef6b1f2faa52148ba8dfa0280ffca", "role": "UPSTREAM_OPTIMIZER_EMA_REFERENCE"},
        {"repository": "qinsiliang68/YOLO-CV", "commit": MAIN_COMMIT, "path": "pyproject.toml", "blob_sha": "c08bf04a1242d4853272117254dd81c88d157e7b", "role": "DEPENDENCY_REFERENCE"},
    ]
