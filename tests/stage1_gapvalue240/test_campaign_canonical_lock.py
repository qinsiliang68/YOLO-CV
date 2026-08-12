from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from stage1_gapvalue240.campaign_canonical_lock import (
    CanonicalLockError,
    audit_canonical_training_corpus,
    build_canonical_training_lock,
    build_train_kwargs,
    load_canonical_training_lock,
    validate_formal_training_dimensions,
    validate_resolved_training_args,
)
from stage1_gapvalue240.util import sha256_file


IMMUTABLE = {
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

RUNTIME = {
    "data": r"C:\canonical\dataset",
    "device": "0",
    "model": r"C:\canonical\yolo11l-cls.pt",
    "project": r"D:\canonical\run",
    "save_dir": r"D:\canonical\run\trainer",
    "seed": 123,
    "resume": False,
    "exist_ok": False,
}


def _sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    resolved_args = {**IMMUTABLE, **RUNTIME}
    args_yaml = tmp_path / "args.yaml"
    args_yaml.write_text(yaml.safe_dump(resolved_args, sort_keys=True), encoding="utf-8")
    resolved = tmp_path / "resolved_training_args.json"
    resolved.write_text(
        json.dumps(
            {
                "schema_version": "stage1.resolved_training_args.v1",
                "args_yaml_sha256": sha256_file(args_yaml),
                "resolved_args": resolved_args,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "training_execution_audit.json"
    audit.write_text(
        json.dumps(
            {
                "completed_epochs": 200,
                "configured_args": {
                    "batch": 128,
                    "cache": False,
                    "deterministic": True,
                    "epochs": 200,
                    "imgsz": 224,
                    "model": RUNTIME["model"],
                    "patience": 0,
                    "seed": 123,
                },
                "effective_batch_size": 128,
                "expected_epochs": 200,
                "expected_steps_per_epoch": 985,
                "loss_finite": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "yolo11l-cls.pt"
    checkpoint.write_bytes(b"canonical-yolo11l")
    return args_yaml, resolved, audit, checkpoint


def _lock(tmp_path: Path):
    args_yaml, resolved, audit, checkpoint = _sources(tmp_path)
    output = tmp_path / "CANONICAL_TRAINING_LOCK_v1.json"
    result = build_canonical_training_lock(
        args_yaml=args_yaml,
        resolved_training_args=resolved,
        training_execution_audit=audit,
        initial_checkpoint=checkpoint,
        output_path=output,
        source_run_id="RUN_019",
        evidence_run_count=240,
    )
    return result, load_canonical_training_lock(output)


def test_builds_source_verified_lock_and_separates_runtime_fields(tmp_path: Path) -> None:
    result, lock = _lock(tmp_path)

    assert result.lock_file_sha256 == sha256_file(result.path)
    assert lock.data["evidence_run_count"] == 240
    assert lock.data["initial_checkpoint"]["sha256"] == sha256_file(
        tmp_path / "yolo11l-cls.pt"
    )
    assert lock.immutable_args["batch"] == 128
    assert lock.immutable_args["workers"] == 4
    assert set(lock.runtime_bound_fields) == {
        "data",
        "device",
        "exist_ok",
        "model",
        "project",
        "resume",
        "save_dir",
        "seed",
    }
    assert not set(lock.immutable_args) & set(lock.runtime_bound_fields)
    assert lock.data["source_artifacts"]["args_yaml"]["sha256"] == sha256_file(
        tmp_path / "args.yaml"
    )


def test_builder_rejects_args_yaml_and_resolved_json_drift(tmp_path: Path) -> None:
    args_yaml, resolved, audit, checkpoint = _sources(tmp_path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    payload["resolved_args"]["workers"] = 8
    resolved.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CanonicalLockError, match="args.yaml.*resolved"):
        build_canonical_training_lock(
            args_yaml=args_yaml,
            resolved_training_args=resolved,
            training_execution_audit=audit,
            initial_checkpoint=checkpoint,
            output_path=tmp_path / "lock.json",
            source_run_id="RUN_019",
            evidence_run_count=240,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("batch", 64), ("workers", 8), ("imgsz", 320), ("epochs", 199)],
)
def test_formal_dimensions_reject_any_canonical_drift(
    tmp_path: Path, field: str, value: int
) -> None:
    _, lock = _lock(tmp_path)
    observed = {"batch": 128, "workers": 4, "imgsz": 224, "epochs": 200}
    observed[field] = value

    with pytest.raises(CanonicalLockError, match=field):
        validate_formal_training_dimensions(lock, **observed)


def test_train_kwargs_are_explicit_and_runtime_values_are_whitelisted(tmp_path: Path) -> None:
    _, lock = _lock(tmp_path)
    runtime = {
        "data": str(tmp_path / "dataset"),
        "device": "1",
        "model": str(tmp_path / "base.pt"),
        "project": str(tmp_path / "out"),
        "save_dir": str(tmp_path / "out/trainer"),
        "seed": 456,
        "resume": False,
        "exist_ok": False,
    }

    kwargs = build_train_kwargs(lock, runtime)

    assert kwargs["batch"] == 128
    assert kwargs["workers"] == 4
    assert kwargs["optimizer"] == "auto"
    assert kwargs["auto_augment"] == "randaugment"
    assert kwargs["data"] == runtime["data"]
    assert kwargs["seed"] == 456
    assert "model" not in kwargs
    assert "save_dir" not in kwargs
    assert kwargs["resume"] is False

    with pytest.raises(CanonicalLockError, match="unknown runtime-bound"):
        build_train_kwargs(lock, {**runtime, "lr0": 0.1})


def test_post_training_validation_rejects_immutable_drift_but_accepts_resume_identity(
    tmp_path: Path,
) -> None:
    _, lock = _lock(tmp_path)
    resolved = {
        **lock.immutable_args,
        "data": str(tmp_path / "dataset"),
        "device": "0",
        "model": str(tmp_path / "checkpoint_epoch_0140.pt"),
        "project": str(tmp_path / "output"),
        "save_dir": str(tmp_path / "output/trainer"),
        "seed": 999,
        "resume": str(tmp_path / "checkpoint_epoch_0140.pt"),
        "exist_ok": True,
    }
    validation = validate_resolved_training_args(
        lock,
        resolved,
        expected_runtime={
            "data": resolved["data"],
            "device": "0",
            "model": resolved["model"],
            "project": resolved["project"],
            "save_dir": resolved["save_dir"],
            "seed": 999,
            "resume": resolved["resume"],
            "exist_ok": True,
        },
    )
    assert validation["status"] == "PASS"

    resumed_normalized = dict(resolved)
    resumed_normalized["warmup_bias_lr"] = 0.0
    normalized = validate_resolved_training_args(
        lock,
        resumed_normalized,
        expected_runtime={key: resumed_normalized[key] for key in lock.runtime_bound_fields},
    )
    assert normalized["runtime_normalized_fields"] == ["warmup_bias_lr:0.1->0.0"]

    resolved["weight_decay"] = 0.001
    with pytest.raises(CanonicalLockError, match="weight_decay"):
        validate_resolved_training_args(
            lock,
            resolved,
            expected_runtime={
                key: resolved[key]
                for key in lock.runtime_bound_fields
            },
        )


def test_corpus_audit_proves_cross_run_invariance_and_records_resume_normalization(
    tmp_path: Path,
) -> None:
    _, lock = _lock(tmp_path / "source")
    source = tmp_path / "source"
    corpus = tmp_path / "corpus"

    def write_run(run_id: str, *, resumed: bool) -> None:
        logs = corpus / "package" / "runs" / run_id / "attempt" / "02_logs"
        logs.mkdir(parents=True)
        resolved = json.loads((source / "resolved_training_args.json").read_text(encoding="utf-8"))
        audit = json.loads((source / "training_execution_audit.json").read_text(encoding="utf-8"))
        args = dict(resolved["resolved_args"])
        args["seed"] = 100 + int(run_id[-3:])
        args["project"] = str(logs.parent)
        args["save_dir"] = str(logs.parent / "trainer")
        audit["configured_args"]["seed"] = args["seed"]
        if resumed:
            resume = str(logs.parent / "training_state/last.pt")
            args["model"] = resume
            args["resume"] = resume
            args["warmup_bias_lr"] = 0.0
        audit["configured_args"]["model"] = args["model"]
        args_path = logs / "args.yaml"
        args_path.write_text(yaml.safe_dump(args, sort_keys=True), encoding="utf-8")
        resolved["resolved_args"] = args
        resolved["args_yaml_sha256"] = sha256_file(args_path)
        (logs / "resolved_training_args.json").write_text(
            json.dumps(resolved), encoding="utf-8"
        )
        (logs / "training_execution_audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )

    write_run("RUN_001", resumed=False)
    write_run("RUN_002", resumed=True)
    result = audit_canonical_training_corpus(
        corpus,
        lock,
        output_dir=tmp_path / "audit",
        expected_run_ids=("RUN_001", "RUN_002"),
    )

    assert result["status"] == "PASS"
    assert result["run_count"] == 2
    assert result["non_resumed_run_count"] == 1
    assert result["resumed_run_count"] == 1
    assert result["resume_normalization_count"] == 1
    assert (tmp_path / "audit/CANONICAL_240RUN_HYPERPARAMETER_AUDIT.csv").is_file()
    assert (tmp_path / "audit/CANONICAL_240RUN_HYPERPARAMETER_AUDIT.json").is_file()

    broken = corpus / "package/runs/RUN_002/attempt/02_logs/resolved_training_args.json"
    payload = json.loads(broken.read_text(encoding="utf-8"))
    payload["resolved_args"]["weight_decay"] = 0.001
    broken_args = broken.with_name("args.yaml")
    broken_args.write_text(
        yaml.safe_dump(payload["resolved_args"], sort_keys=True), encoding="utf-8"
    )
    payload["args_yaml_sha256"] = sha256_file(broken_args)
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CanonicalLockError, match="weight_decay"):
        audit_canonical_training_corpus(
            corpus,
            lock,
            output_dir=tmp_path / "bad_audit",
            expected_run_ids=("RUN_001", "RUN_002"),
        )
