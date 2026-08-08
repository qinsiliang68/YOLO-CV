"""Immutable canonical-training argument contract for the final Stage1 campaign."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd
import yaml

from .util import atomic_write_bytes, atomic_write_json, sha256_file, stable_hash


LOCK_SCHEMA = "stage1.canonical_training_hyperparameter_lock.v1"

# These fields identify a physical execution rather than a scientific training
# choice. Their values still have to match the registered job exactly.
RUNTIME_BOUND_FIELDS = (
    "data",
    "device",
    "exist_ok",
    "model",
    "project",
    "resume",
    "save_dir",
    "seed",
)

_OMIT_FROM_MODEL_TRAIN = {"model", "save_dir"}

_REQUIRED_CANONICAL_VALUES: dict[str, Any] = {
    "task": "classify",
    "mode": "train",
    "epochs": 200,
    "batch": 128,
    "imgsz": 224,
    "workers": 4,
    "patience": 0,
    "save": True,
    "save_period": -1,
    "cache": False,
    "pretrained": True,
    "optimizer": "auto",
    "verbose": True,
    "deterministic": True,
    "cos_lr": False,
    "amp": True,
    "fraction": 1.0,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
    "auto_augment": "randaugment",
    "erasing": 0.4,
    "plots": False,
    "val": True,
    "name": "trainer",
}


class CanonicalLockError(RuntimeError):
    """Raised when a formal job differs from the canonical 240-run learner."""


@dataclass(frozen=True)
class CanonicalTrainingLock:
    path: Path
    data: dict[str, Any]
    file_sha256: str

    @property
    def immutable_args(self) -> dict[str, Any]:
        return dict(self.data["immutable_args"])

    @property
    def runtime_bound_fields(self) -> tuple[str, ...]:
        return tuple(self.data["runtime_bound_fields"])


@dataclass(frozen=True)
class CanonicalLockBuildResult:
    path: Path
    lock_file_sha256: str
    lock_payload_sha256: str
    immutable_field_count: int
    runtime_bound_field_count: int


def _mapping(path: Path, *, yaml_file: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = (
            yaml.safe_load(path.read_text(encoding="utf-8"))
            if yaml_file
            else json.loads(path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise CanonicalLockError(f"cannot parse canonical source {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CanonicalLockError(f"canonical source root is not a mapping: {path}")
    return value


def _mismatches(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {"observed": observed.get(key), "expected": value}
        for key, value in expected.items()
        if key not in observed or observed[key] != value
    }


def _validate_core(args: Mapping[str, Any]) -> None:
    missing = set(_REQUIRED_CANONICAL_VALUES) - set(args)
    if missing:
        raise CanonicalLockError(f"canonical resolved args missing required fields: {sorted(missing)}")
    mismatch = _mismatches(args, _REQUIRED_CANONICAL_VALUES)
    if mismatch:
        raise CanonicalLockError(f"canonical source differs from the verified 240-run core: {mismatch}")
    missing_runtime = set(RUNTIME_BOUND_FIELDS) - set(args)
    if missing_runtime:
        raise CanonicalLockError(f"canonical source missing runtime-bound fields: {sorted(missing_runtime)}")


def build_canonical_training_lock(
    *,
    args_yaml: str | Path,
    resolved_training_args: str | Path,
    training_execution_audit: str | Path,
    initial_checkpoint: str | Path,
    output_path: str | Path,
    source_run_id: str,
    evidence_run_count: int,
) -> CanonicalLockBuildResult:
    """Build one source-audited lock without copying checkpoints into the repo."""

    args_path = Path(args_yaml).resolve()
    resolved_path = Path(resolved_training_args).resolve()
    audit_path = Path(training_execution_audit).resolve()
    checkpoint_path = Path(initial_checkpoint).resolve()
    output = Path(output_path).resolve()
    if not source_run_id:
        raise CanonicalLockError("source_run_id must be non-empty")
    if int(evidence_run_count) != 240:
        raise CanonicalLockError("formal canonical lock must be backed by exactly 240 canonical runs")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite canonical lock: {output}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    args = _mapping(args_path, yaml_file=True)
    resolved_payload = _mapping(resolved_path)
    resolved = resolved_payload.get("resolved_args")
    if not isinstance(resolved, dict):
        raise CanonicalLockError("resolved_training_args.json has no resolved_args mapping")
    audit = _mapping(audit_path)

    declared_args_hash = str(resolved_payload.get("args_yaml_sha256", "")).upper()
    actual_args_hash = sha256_file(args_path)
    if declared_args_hash and declared_args_hash != actual_args_hash:
        raise CanonicalLockError(
            "resolved_training_args.json args_yaml_sha256 does not match canonical args.yaml"
        )
    if args != resolved:
        mismatch_keys = sorted(
            key for key in set(args) | set(resolved) if args.get(key) != resolved.get(key)
        )
        raise CanonicalLockError(
            f"canonical args.yaml and resolved args differ: {mismatch_keys}"
        )
    _validate_core(resolved)

    configured = audit.get("configured_args")
    if not isinstance(configured, dict):
        raise CanonicalLockError("training_execution_audit.json has no configured_args mapping")
    audit_expected = {
        "batch": resolved["batch"],
        "cache": resolved["cache"],
        "deterministic": resolved["deterministic"],
        "epochs": resolved["epochs"],
        "imgsz": resolved["imgsz"],
        "model": resolved["model"],
        "patience": resolved["patience"],
        "seed": resolved["seed"],
    }
    audit_mismatch = _mismatches(configured, audit_expected)
    if audit_mismatch:
        raise CanonicalLockError(f"canonical training audit disagrees with resolved args: {audit_mismatch}")
    if int(audit.get("completed_epochs", -1)) != 200 or audit.get("loss_finite") is not True:
        raise CanonicalLockError("canonical source run must be a finite, complete 200-epoch run")
    if int(audit.get("effective_batch_size", -1)) != 128:
        raise CanonicalLockError("canonical source run effective batch is not 128")

    immutable = {key: value for key, value in resolved.items() if key not in RUNTIME_BOUND_FIELDS}
    payload: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA,
        "lock_id": "STAGE1_GAPVALUE_CANONICAL_240RUN_YOLO11L_V1",
        "source_run_id": source_run_id,
        "evidence_run_count": int(evidence_run_count),
        "model_family": "yolo11l",
        "immutable_args": immutable,
        "runtime_bound_fields": list(RUNTIME_BOUND_FIELDS),
        "runtime_binding_rules": {
            "data": "registered staged dataset path; content is bound by manifest hashes",
            "device": "registered machine GPU id",
            "exist_ok": "false for a new trainer workspace; true only for registered continuation",
            "model": "initial checkpoint or exact registered resumable checkpoint",
            "project": "registered run output directory",
            "resume": "false for epoch 1 or exact registered resumable checkpoint path",
            "save_dir": "derived project/name trainer directory",
            "seed": "registered paired training seed",
        },
        "initial_checkpoint": {
            "basename": checkpoint_path.name,
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
        "source_artifacts": {
            "args_yaml": {
                "basename": args_path.name,
                "sha256": sha256_file(args_path),
            },
            "resolved_training_args": {
                "basename": resolved_path.name,
                "sha256": sha256_file(resolved_path),
            },
            "training_execution_audit": {
                "basename": audit_path.name,
                "sha256": sha256_file(audit_path),
            },
        },
        "source_resolved_field_count": len(resolved),
        "immutable_field_count": len(immutable),
        "canonical_source_runtime_example": {
            key: resolved[key] for key in RUNTIME_BOUND_FIELDS
        },
    }
    payload["lock_payload_sha256"] = stable_hash(payload)
    atomic_write_json(output, payload)
    return CanonicalLockBuildResult(
        path=output,
        lock_file_sha256=sha256_file(output),
        lock_payload_sha256=payload["lock_payload_sha256"],
        immutable_field_count=len(immutable),
        runtime_bound_field_count=len(RUNTIME_BOUND_FIELDS),
    )


def load_canonical_training_lock(path: str | Path) -> CanonicalTrainingLock:
    lock_path = Path(path).resolve()
    data = _mapping(lock_path)
    if data.get("schema_version") != LOCK_SCHEMA:
        raise CanonicalLockError(f"unsupported canonical lock schema: {data.get('schema_version')!r}")
    recorded_payload_hash = str(data.get("lock_payload_sha256", "")).upper()
    unhashed = dict(data)
    unhashed.pop("lock_payload_sha256", None)
    if recorded_payload_hash != stable_hash(unhashed):
        raise CanonicalLockError("canonical lock payload hash mismatch")
    if tuple(data.get("runtime_bound_fields", ())) != RUNTIME_BOUND_FIELDS:
        raise CanonicalLockError("canonical runtime-bound field list changed")
    immutable = data.get("immutable_args")
    if not isinstance(immutable, dict):
        raise CanonicalLockError("canonical lock immutable_args is not a mapping")
    if set(immutable) & set(RUNTIME_BOUND_FIELDS):
        raise CanonicalLockError("canonical immutable and runtime-bound fields overlap")
    _validate_core({**immutable, **data.get("canonical_source_runtime_example", {})})
    return CanonicalTrainingLock(lock_path, data, sha256_file(lock_path))


def validate_formal_training_dimensions(
    lock: CanonicalTrainingLock,
    *,
    batch: int,
    workers: int,
    imgsz: int,
    epochs: int,
) -> None:
    observed = {
        "batch": batch,
        "workers": workers,
        "imgsz": imgsz,
        "epochs": epochs,
    }
    expected = {key: lock.immutable_args[key] for key in observed}
    mismatch = _mismatches(observed, expected)
    if mismatch:
        raise CanonicalLockError(f"formal training dimensions drifted from canonical lock: {mismatch}")


def build_train_kwargs(
    lock: CanonicalTrainingLock,
    runtime_values: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_keys = set(runtime_values)
    expected_keys = set(lock.runtime_bound_fields)
    unknown = runtime_keys - expected_keys
    missing = expected_keys - runtime_keys
    if unknown:
        raise CanonicalLockError(f"unknown runtime-bound fields: {sorted(unknown)}")
    if missing:
        raise CanonicalLockError(f"missing runtime-bound fields: {sorted(missing)}")
    kwargs = dict(lock.immutable_args)
    kwargs.update(
        {
            key: runtime_values[key]
            for key in lock.runtime_bound_fields
            if key not in _OMIT_FROM_MODEL_TRAIN
        }
    )
    return kwargs


def validate_resolved_training_args(
    lock: CanonicalTrainingLock,
    resolved_args: Mapping[str, Any],
    *,
    expected_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_keys = set(expected_runtime)
    required_runtime = set(lock.runtime_bound_fields)
    if runtime_keys != required_runtime:
        raise CanonicalLockError(
            "resolved validation runtime fields mismatch: "
            f"missing={sorted(required_runtime - runtime_keys)}, "
            f"unknown={sorted(runtime_keys - required_runtime)}"
        )
    expected = {**lock.immutable_args, **dict(expected_runtime)}
    runtime_normalized_fields: list[str] = []
    if (
        expected_runtime.get("resume") not in (False, None, "")
        and expected.get("warmup_bias_lr") == 0.1
        and resolved_args.get("warmup_bias_lr") == 0.0
    ):
        expected["warmup_bias_lr"] = 0.0
        runtime_normalized_fields.append("warmup_bias_lr:0.1->0.0")
    missing = set(expected) - set(resolved_args)
    unknown = set(resolved_args) - set(expected)
    mismatch = _mismatches(resolved_args, expected)
    if missing or unknown or mismatch:
        raise CanonicalLockError(
            "resolved training args differ from canonical lock: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}, mismatches={mismatch}"
        )
    return {
        "status": "PASS",
        "canonical_lock_file_sha256": lock.file_sha256,
        "canonical_lock_payload_sha256": lock.data["lock_payload_sha256"],
        "validated_field_count": len(expected),
        "immutable_field_count": len(lock.immutable_args),
        "runtime_bound_field_count": len(lock.runtime_bound_fields),
        "runtime_normalized_fields": runtime_normalized_fields,
    }


def audit_canonical_training_corpus(
    corpus_root: str | Path,
    lock: CanonicalTrainingLock,
    *,
    output_dir: str | Path,
    expected_run_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Prove the historical run corpus is consistent with one canonical lock."""

    root = Path(corpus_root).resolve()
    output = Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"canonical corpus audit output is not empty: {output}")
    expected = expected_run_ids or tuple(f"RUN_{index:03d}" for index in range(1, 241))
    if len(set(expected)) != len(expected):
        raise CanonicalLockError("expected canonical run IDs are not unique")

    grouped: dict[str, list[Path]] = {}
    for path in root.rglob("resolved_training_args.json"):
        match = re.search(r"(?:^|[\\/])(RUN_\d{3})(?:[\\/]|$)", str(path))
        if match:
            grouped.setdefault(match.group(1), []).append(path.resolve())
    missing = sorted(set(expected) - set(grouped))
    unexpected = sorted(set(grouped) - set(expected))
    if missing or unexpected:
        raise CanonicalLockError(
            f"canonical corpus run IDs differ: missing={missing}, unexpected={unexpected}"
        )

    rows: list[dict[str, Any]] = []
    resume_normalizations = 0
    for run_id in expected:
        candidates = sorted(
            grouped[run_id],
            key=lambda path: (
                "evidence_INVALID_ARTIFACT" in str(path),
                path.parent.name != "02_logs",
                len(str(path)),
                str(path),
            ),
        )
        resolved_path = candidates[0]
        args_path = resolved_path.with_name("args.yaml")
        audit_path = resolved_path.with_name("training_execution_audit.json")
        if not args_path.is_file() or not audit_path.is_file():
            raise CanonicalLockError(f"{run_id} canonical source triplet is incomplete: {resolved_path}")
        payload = _mapping(resolved_path)
        resolved = payload.get("resolved_args")
        if not isinstance(resolved, dict):
            raise CanonicalLockError(f"{run_id} resolved_args is not a mapping")
        args = _mapping(args_path, yaml_file=True)
        if args != resolved:
            raise CanonicalLockError(f"{run_id} args.yaml differs from resolved_training_args.json")
        if str(payload.get("args_yaml_sha256", "")).upper() != sha256_file(args_path):
            raise CanonicalLockError(f"{run_id} args_yaml_sha256 mismatch")

        resumed = resolved.get("resume") not in (False, None, "")
        immutable_mismatch = _mismatches(resolved, lock.immutable_args)
        normalized_fields: list[str] = []
        if resumed and immutable_mismatch.get("warmup_bias_lr") == {
            "observed": 0.0,
            "expected": 0.1,
        }:
            immutable_mismatch.pop("warmup_bias_lr")
            normalized_fields.append("warmup_bias_lr:0.1->0.0")
            resume_normalizations += 1
        if immutable_mismatch:
            raise CanonicalLockError(
                f"{run_id} immutable canonical fields drifted: {immutable_mismatch}"
            )

        audit = _mapping(audit_path)
        configured = audit.get("configured_args")
        if not isinstance(configured, dict):
            raise CanonicalLockError(f"{run_id} training audit has no configured_args")
        audit_expected = {
            "batch": 128,
            "cache": False,
            "deterministic": True,
            "epochs": 200,
            "imgsz": 224,
            "patience": 0,
            "seed": resolved["seed"],
        }
        audit_mismatch = _mismatches(configured, audit_expected)
        if audit_mismatch:
            raise CanonicalLockError(f"{run_id} configured audit drifted: {audit_mismatch}")
        if (
            int(audit.get("completed_epochs", -1)) != 200
            or audit.get("loss_finite") is not True
            or int(audit.get("effective_batch_size", -1)) != 128
        ):
            raise CanonicalLockError(f"{run_id} is not a complete finite batch-128 canonical run")
        rows.append(
            {
                "run_id": run_id,
                "status": "PASS",
                "resumed": resumed,
                "resume_count": int(audit.get("resume_count", 0)),
                "runtime_normalized_fields": ";".join(normalized_fields),
                "resolved_field_count": len(resolved),
                "candidate_artifact_count": len(candidates),
                "selected_resolved_path": str(resolved_path),
                "args_yaml_sha256": sha256_file(args_path),
                "resolved_training_args_sha256": sha256_file(resolved_path),
                "training_execution_audit_sha256": sha256_file(audit_path),
                "canonical_lock_file_sha256": lock.file_sha256,
            }
        )

    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "CANONICAL_240RUN_HYPERPARAMETER_AUDIT.csv"
    atomic_write_bytes(detail_path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    summary = {
        "schema_version": "stage1.canonical_training_corpus_audit.v1",
        "status": "PASS",
        "corpus_root": str(root),
        "run_count": len(frame),
        "non_resumed_run_count": int((~frame.resumed).sum()),
        "resumed_run_count": int(frame.resumed.sum()),
        "resume_normalization_count": resume_normalizations,
        "resume_normalization_rule": (
            "Only warmup_bias_lr 0.1 to 0.0 is accepted after native optimizer=auto resume; "
            "the configured canonical value remains 0.1."
        ),
        "canonical_lock_path": str(lock.path),
        "canonical_lock_file_sha256": lock.file_sha256,
        "detail_csv": detail_path.name,
        "detail_csv_sha256": sha256_file(detail_path),
    }
    atomic_write_json(output / "CANONICAL_240RUN_HYPERPARAMETER_AUDIT.json", summary)
    return summary
