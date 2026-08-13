from __future__ import annotations

import importlib
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .baseline_reference import MAIN_COMMIT, source_external_references
from .contracts import require_synthetic_or_authorized
from .errors import ErrorCode, SctsrError
from .formal_runtime import reject_forbidden_data_role
from .replay_step_plan import ReplayStepPlan
from .serialization import load_json, sha256_file, stable_digest
from .ultralytics_overlay import run_ultralytics_classification_epoch


@dataclass(frozen=True, slots=True)
class UpstreamBinding:
    repository_root: str
    yolo_root: str
    source_git_blob_sha1: Mapping[str, str]
    source_file_sha256: Mapping[str, str]
    binding_digest: str


def bind_upstream(repository_root: str | Path, *, verify_hashes: bool = True) -> UpstreamBinding:
    root = Path(repository_root).resolve()
    expected = {row["path"]: row.get("blob_sha") for row in source_external_references() if row.get("path")}
    required = (
        "YOLOv11/ultralytics/engine/trainer.py",
        "YOLOv11/ultralytics/models/yolo/classify/train.py",
        "YOLOv11/ultralytics/nn/tasks.py",
        "YOLOv11/ultralytics/nn/modules/head.py",
        "YOLOv11/ultralytics/utils/loss.py",
        "YOLOv11/ultralytics/utils/torch_utils.py",
    )
    git_blob_sha1: dict[str, str] = {}
    file_sha256: dict[str, str] = {}
    for rel in required:
        path = root / rel
        if not path.is_file():
            raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Required upstream source is missing", artifact_path=str(path))
        content = path.read_bytes()
        observed = hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()
        if verify_hashes and expected.get(rel) and observed != expected[rel]:
            raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Upstream Git blob SHA differs from frozen main", artifact_path=str(path), observed=observed, expected=expected[rel])
        git_blob_sha1[rel] = observed
        file_sha256[rel] = sha256_file(path)
    payload = {
        "repository_root": root.as_posix(),
        "git_blob_sha1": git_blob_sha1,
        "file_sha256": file_sha256,
    }
    return UpstreamBinding(
        repository_root=root.as_posix(),
        yolo_root=(root / "YOLOv11").as_posix(),
        source_git_blob_sha1=git_blob_sha1,
        source_file_sha256=file_sha256,
        binding_digest=stable_digest(payload),
    )


def validate_upstream_manifest(binding: UpstreamBinding, manifest_path: str | Path) -> str:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "stage1.sctsr.upstream_files_manifest.v2":
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Unknown upstream source manifest schema")
    if manifest.get("repository") != "qinsiliang68/YOLO-CV" or manifest.get("commit") != MAIN_COMMIT:
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Upstream manifest baseline identity mismatch")
    if manifest.get("upstream_files_modified") is not False:
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Upstream manifest claims archived learner modifications")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Upstream manifest files must be a list")
    observed = {
        str(row.get("path")): (str(row.get("git_blob_sha1", "")), str(row.get("file_sha256", "")))
        for row in rows
        if isinstance(row, Mapping)
    }
    expected = {
        path: (binding.source_git_blob_sha1[path], binding.source_file_sha256[path])
        for path in binding.source_file_sha256
    }
    if observed != expected or len(rows) != len(expected):
        raise SctsrError(
            ErrorCode.UPSTREAM_BINDING_FAILED,
            "Upstream source manifest does not match the bound repository bytes",
            observed=observed,
            expected=expected,
        )
    return stable_digest(manifest)


def import_classification_trainer(binding: UpstreamBinding):
    if binding.yolo_root not in sys.path:
        sys.path.insert(0, binding.yolo_root)
    modules = {
        "ultralytics.engine.trainer": "YOLOv11/ultralytics/engine/trainer.py",
        "ultralytics.models.yolo.classify.train": "YOLOv11/ultralytics/models/yolo/classify/train.py",
        "ultralytics.nn.tasks": "YOLOv11/ultralytics/nn/tasks.py",
        "ultralytics.nn.modules.head": "YOLOv11/ultralytics/nn/modules/head.py",
        "ultralytics.utils.loss": "YOLOv11/ultralytics/utils/loss.py",
        "ultralytics.utils.torch_utils": "YOLOv11/ultralytics/utils/torch_utils.py",
    }
    loaded = {}
    for module_name, relative_path in modules.items():
        module = importlib.import_module(module_name)
        observed = Path(str(module.__file__)).resolve()
        expected = (Path(binding.repository_root) / relative_path).resolve()
        if observed != expected:
            raise SctsrError(
                ErrorCode.UPSTREAM_BINDING_FAILED,
                "Imported Ultralytics module is not the frozen repository source",
                artifact_path=str(observed),
                observed=observed.as_posix(),
                expected=expected.as_posix(),
            )
        if sha256_file(observed) != binding.source_file_sha256[relative_path]:
            raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Imported Ultralytics module changed after source binding", artifact_path=str(observed))
        loaded[module_name] = module
    module = loaded["ultralytics.models.yolo.classify.train"]
    return module.ClassificationTrainer


def prepare_classification_overrides(binding: UpstreamBinding, overrides: Mapping[str, Any]) -> dict[str, Any]:
    lock_path = Path(binding.repository_root) / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json"
    lock = load_json(lock_path)
    immutable = lock.get("immutable_args")
    if not isinstance(immutable, Mapping):
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Canonical training lock lacks immutable_args", artifact_path=str(lock_path))
    required_runtime = {"model", "data", "device", "project", "name", "seed", "exist_ok", "resume"}
    missing = sorted(required_runtime - set(overrides))
    if missing:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Prepared trainer lacks runtime-bound fields", observed=missing)
    clean = dict(immutable)
    for field, supplied in overrides.items():
        if field in immutable and supplied != immutable[field]:
            raise SctsrError(
                ErrorCode.CONFIGURATION_MISMATCH,
                "Trainer override conflicts with the canonical training lock",
                failing_field=field,
                observed=supplied,
                expected=immutable[field],
            )
        clean[field] = supplied
    clean["task"] = "classify"
    clean["mode"] = "train"
    return clean


def build_classification_trainer(binding: UpstreamBinding, *, overrides: Mapping[str, Any]):
    forbidden = {"test", "blind", "blind_holdout"}
    data_role = str(overrides.get("data_role", "train"))
    if data_role.lower() in forbidden:
        reject_forbidden_data_role(data_role, purpose="training")
    trainer_cls = import_classification_trainer(binding)
    clean = prepare_classification_overrides(binding, overrides)
    clean.pop("data_role", None)
    return trainer_cls(overrides=clean)


def run_prepared_branch_epoch(
    *, trainer: Any, base_loader: Sequence[Any], replay_plan: ReplayStepPlan,
    replay_provider: Callable[[Sequence[str], int, int, int], Mapping[str, Any]], epoch: int,
) -> dict[str, Any]:
    if base_loader is not trainer.train_loader:
        trainer.train_loader = base_loader
    return run_ultralytics_classification_epoch(
        trainer=trainer,
        replay_plan=replay_plan,
        replay_batch_provider=replay_provider,
        training_seed=int(replay_plan.training_seed),
        epoch=epoch,
        global_step_start=int(getattr(trainer, "global_step", 0)),
    )


def assert_formal_authorized(execution_mode: str, release_authorization: str | Path | None) -> None:
    require_synthetic_or_authorized(execution_mode, release_authorization)
