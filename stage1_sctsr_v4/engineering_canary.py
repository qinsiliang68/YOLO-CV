from __future__ import annotations

"""Real-image, real-YOLO engineering canary for SCTSR v4.

This module deliberately does not expose a formal-training mode.  It exercises
one small fixed-base-step update and the evidence/recovery path using registered
training-role images.  Its output is evidence about mechanics only and can
never be interpreted as a scientific result or training authorization.
"""

import csv
import hashlib
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from .asset_registry import load_asset_registry
from .baseline_reference import SOURCE_TREE_INCLUDE_PATHS, source_external_references
from .checkpointing import (
    build_checkpoint_payload,
    checkpoint_state_digest,
    load_checkpoint,
    save_checkpoint_atomic,
)
from .columnar import validate_columnar_file, write_zstd_parquet
from .epoch_transaction import EpochTransaction
from .errors import ErrorCode, SctsrError
from .evaluation import compute_tie_safe_frontier, write_frontier_artifacts
from .prediction_artifact import PredictionRow, write_prediction_artifact
from .prediction_runtime import _probabilities_and_logits
from .recovery import find_last_complete_epoch, quarantine_inprogress, validate_recovery_pointer
from .replay_step_plan import build_replay_step_plan
from .serialization import atomic_write_json, sha256_file, stable_digest
from .source_identity import build_source_tree_manifest
from .ultralytics_overlay import UpstreamStepReceipt, run_ultralytics_classification_epoch


ENGINEERING_CANARY_SEMANTIC = "ENGINEERING_CANARY_NOT_SCIENTIFIC_RESULT"
EXPECTED_WEIGHT_SHA256 = "6B56513A5D8BDAE6B8F0A36DACAF01B26D5A522BA1B34197C3BAC9FA6463366C"
_BANNED_SPLIT_TOKENS = frozenset({"test", "blind", "holdout", "val_op"})
_REQUIRED_FALSE_FLAGS = (
    "formal_training_started",
    "assignments_generated",
    "engineering_gate_generated",
    "pilot_release_generated",
    "blind_holdout_opened",
    "test_accessed",
    "method_effectiveness_claimed",
)


@dataclass(frozen=True, slots=True)
class EngineeringCanarySample:
    sample_id: str
    label: int
    split_role: str
    manifest_path: str
    manifest_sha256: str
    manifest_row_number: int
    image_path: str
    image_bytes: int
    image_sha256: str


def _contained(path: Path, root: Path, *, role: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SctsrError(
            ErrorCode.ASSET_VALIDATION_FAILED,
            f"Engineering canary {role} escapes the registered dataset root",
            artifact_path=str(path),
        ) from exc
    return resolved


def _manifest_samples(
    dataset_root: Path,
    *,
    manifest_name: str,
    expected_split: str,
    label: int,
    limit: int,
) -> list[EngineeringCanarySample]:
    manifest = dataset_root / "manifests" / manifest_name
    if not manifest.is_file():
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Engineering canary manifest is missing", artifact_path=str(manifest))
    manifest_sha = sha256_file(manifest)
    selected: list[EngineeringCanarySample] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 2):
            observed_split = str(row.get("split", "")).strip()
            if observed_split != expected_split:
                raise SctsrError(
                    ErrorCode.TEST_ACCESS_FORBIDDEN,
                    "Engineering canary manifest row has an unexpected split role",
                    failing_field=f"{manifest_name}:{row_number}.split",
                    observed=observed_split,
                    expected=expected_split,
                )
            relative_token = str(row.get("canonical_image_relpath", "")).replace("\\", "/").strip()
            relative = Path(relative_token)
            expected_prefix = Path("Det") / "images" / expected_split
            if (
                not relative_token
                or relative.is_absolute()
                or ".." in relative.parts
                or tuple(part.lower() for part in relative.parts[:3])
                != tuple(part.lower() for part in expected_prefix.parts)
                or any(token in {part.lower() for part in relative.parts} for token in _BANNED_SPLIT_TOKENS)
            ):
                raise SctsrError(
                    ErrorCode.TEST_ACCESS_FORBIDDEN,
                    "Engineering canary image identity is outside the two allowed training roles",
                    artifact_path=relative_token,
                )
            image = _contained(dataset_root / relative, dataset_root, role="image")
            if not image.is_file():
                raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Engineering canary image is missing", artifact_path=str(image))
            sample_id = str(row.get("Filename", "")).strip()
            if not sample_id or Path(sample_id).name != sample_id or image.name != sample_id:
                raise SctsrError(
                    ErrorCode.IDENTITY_DIGEST_MISMATCH,
                    "Engineering canary manifest sample ID differs from the physical image name",
                    failing_field=f"{manifest_name}:{row_number}.Filename",
                )
            selected.append(
                EngineeringCanarySample(
                    sample_id=sample_id,
                    label=label,
                    split_role=expected_split,
                    manifest_path=manifest.resolve().as_posix(),
                    manifest_sha256=manifest_sha,
                    manifest_row_number=row_number,
                    image_path=image.as_posix(),
                    image_bytes=image.stat().st_size,
                    image_sha256=sha256_file(image),
                )
            )
            if len(selected) == limit:
                break
    if len(selected) != limit:
        raise SctsrError(
            ErrorCode.ASSET_VALIDATION_FAILED,
            "Engineering canary manifest does not contain the requested number of rows",
            observed=len(selected),
            expected=limit,
            artifact_path=str(manifest),
        )
    return selected


def select_engineering_canary_samples(
    dataset_root: str | Path,
    *,
    per_class: int = 2,
) -> tuple[EngineeringCanarySample, ...]:
    """Select a deterministic, training-only real-image microbatch."""

    if type(per_class) is not int or per_class < 1:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Engineering canary per-class count must be positive")
    root = Path(dataset_root).resolve()
    defect = _manifest_samples(
        root,
        manifest_name="train_manifest.csv",
        expected_split="train",
        label=1,
        limit=per_class,
    )
    normal = _manifest_samples(
        root,
        manifest_name="normal_train_manifest.csv",
        expected_split="normal_train",
        label=0,
        limit=per_class,
    )
    samples = tuple((*defect, *normal))
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Engineering canary selected duplicate sample IDs")
    return samples


def validate_engineering_canary_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed over the minimum mechanics and prohibited-side-effect claims."""

    required_true = (
        "real_image_bytes_verified",
        "real_yolo_weight_verified",
        "forward_passed",
        "base_backward_passed",
        "replay_backward_passed",
    )
    if report.get("schema_version") != "stage1.sctsr.engineering_canary.v1":
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Engineering canary report schema is invalid")
    if report.get("status") != "PASS" or report.get("scientific_role") != ENGINEERING_CANARY_SEMANTIC:
        raise SctsrError(ErrorCode.SYNTHETIC_RESULT_MISLABELLED, "Engineering canary report is not explicitly non-scientific")
    if int(report.get("real_image_count", 0)) < 2 or not all(report.get(field) is True for field in required_true):
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Engineering canary did not prove all real-data mechanics")
    if int(report.get("optimizer_step_count", -1)) != 1 or int(report.get("ema_update_count", -1)) != 1:
        raise SctsrError(ErrorCode.REPLAY_ADDED_OPTIMIZER_STEP, "Engineering canary must perform exactly one optimizer and EMA update")
    parquet = report.get("parquet")
    checkpoint = report.get("checkpoint")
    recovery = report.get("recovery")
    if not isinstance(parquet, Mapping) or parquet.get("canonical_parquet") is not True or parquet.get("compression") != "ZSTD":
        raise SctsrError(ErrorCode.COLUMNAR_ENGINE_UNAVAILABLE, "Engineering canary did not publish canonical Zstd Parquet")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("save_passed") is not True or checkpoint.get("reload_passed") is not True:
        raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Engineering canary checkpoint did not round-trip")
    if (
        not isinstance(recovery, Mapping)
        or recovery.get("corrupt_checkpoint_rejected") is not True
        or recovery.get("partial_generation_quarantined") is not True
    ):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Engineering canary fault recovery was not demonstrated")
    if int(report.get("frontier_point_count", -1)) != 96:
        raise SctsrError(ErrorCode.FRONTIER_INVALID, "Engineering canary frontier is not FN=0..95")
    violated = [field for field in _REQUIRED_FALSE_FLAGS if report.get(field) is not False]
    if violated:
        raise SctsrError(
            ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED,
            "Engineering canary report claims a prohibited formal side effect",
            observed=violated,
        )
    return {"status": "PASS", "scientific_role": ENGINEERING_CANARY_SEMANTIC}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy().tobytes(order="C")
    return hashlib.sha256(value).hexdigest().upper()


def _driver_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
    return ",".join(values) if values else None


def _load_augmented_tensor(sample: EngineeringCanarySample, transform: Any) -> torch.Tensor:
    with Image.open(sample.image_path) as image:
        value = transform(image.convert("RGB"))
    if not isinstance(value, torch.Tensor) or value.shape != (3, 224, 224):
        raise SctsrError(
            ErrorCode.UPSTREAM_BINDING_FAILED,
            "Frozen classification augmentation did not return a 3x224x224 tensor",
            observed=getattr(value, "shape", None),
        )
    return value


def _artifact_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def run_real_engineering_canary(
    artifact_root: str | Path,
    *,
    repository_root: str | Path,
    dataset_root: str | Path,
    training_seed: int = 20260814,
    device: str | None = None,
) -> dict[str, Any]:
    """Execute one real-image fixed-step update and evidence round trip."""

    root = Path(artifact_root).resolve()
    repository = Path(repository_root).resolve()
    dataset = Path(dataset_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Engineering canary artifact root must be new or empty", artifact_path=str(root))
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    samples = select_engineering_canary_samples(dataset, per_class=2)
    if any(token in sample.split_role.lower() for sample in samples for token in _BANNED_SPLIT_TOKENS):
        raise SctsrError(ErrorCode.TEST_ACCESS_FORBIDDEN, "Engineering canary attempted a prohibited split")

    weight = repository / "yolo11l-cls.pt"
    if not weight.is_file() or sha256_file(weight) != EXPECTED_WEIGHT_SHA256:
        raise SctsrError(
            ErrorCode.ASSET_VALIDATION_FAILED,
            "Engineering canary initial YOLO weight identity is wrong",
            artifact_path=str(weight),
            expected=EXPECTED_WEIGHT_SHA256,
        )
    yolo_root = repository / "YOLOv11"
    if str(yolo_root) not in sys.path:
        sys.path.insert(0, str(yolo_root))
    from ultralytics.data.augment import classify_augmentations
    from ultralytics.models.yolo.classify.train import ClassificationTrainer
    from ultralytics.nn.tasks import ClassificationModel, load_checkpoint as load_yolo_checkpoint
    from ultralytics.utils.torch_utils import ModelEMA

    trainer_source = Path(sys.modules[ClassificationTrainer.__module__].__file__).resolve()
    try:
        trainer_source.relative_to(yolo_root.resolve())
    except ValueError as exc:
        raise SctsrError(
            ErrorCode.UPSTREAM_BINDING_FAILED,
            "Engineering canary imported an installed Ultralytics package instead of frozen repository source",
            artifact_path=str(trainer_source),
        ) from exc

    requested_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(requested_device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "CUDA canary was requested but CUDA is unavailable")
    _seed_everything(training_seed)
    model, _ = load_yolo_checkpoint(str(weight), device=torch_device, inplace=False, fuse=False)
    ClassificationModel.reshape_outputs(model, 2)
    model.float().to(torch_device)

    transform = classify_augmentations(
        size=224,
        scale=(0.5, 1.0),
        hflip=0.5,
        vflip=0.0,
        erasing=0.4,
        auto_augment="randaugment",
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    tensors: dict[str, torch.Tensor] = {}
    base_augmentation_digests: dict[str, str] = {}
    base_augmentation_seeds: dict[str, int] = {}
    for index, sample in enumerate(samples):
        augmentation_seed = training_seed * 100 + index
        _seed_everything(augmentation_seed)
        tensor = _load_augmented_tensor(sample, transform)
        tensors[sample.sample_id] = tensor
        base_augmentation_digests[sample.sample_id] = _tensor_digest(tensor)
        base_augmentation_seeds[sample.sample_id] = augmentation_seed

    trainer = object.__new__(ClassificationTrainer)
    trainer.model = model
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.937, weight_decay=0.0005)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    trainer.scaler = torch.amp.GradScaler(torch_device.type, enabled=torch_device.type == "cuda")
    trainer.ema = ModelEMA(model)
    trainer.device = torch_device
    trainer.amp = torch_device.type == "cuda"
    trainer.batch_size = 128
    trainer.world_size = 1
    trainer.accumulate = 1
    trainer.args = SimpleNamespace(
        resume="",
        warmup_epochs=0,
        nbs=64,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        momentum=0.937,
        compile=False,
    )
    base_ids = tuple(sample.sample_id for sample in samples)
    labels_by_id = {sample.sample_id: sample.label for sample in samples}
    trainer.train_loader = [
        {
            "img": torch.stack([tensors[sample_id] for sample_id in base_ids]),
            "cls": torch.tensor([labels_by_id[sample_id] for sample_id in base_ids], dtype=torch.long),
            "sample_ids": base_ids,
            "augmentation_digests": tuple(base_augmentation_digests[sample_id] for sample_id in base_ids),
            "augmentation_seeds": tuple(base_augmentation_seeds[sample_id] for sample_id in base_ids),
        }
    ]
    upstream_optimizer_step = trainer.optimizer_step
    optimizer_calls = {"count": 0}

    def counted_optimizer_step() -> None:
        optimizer_calls["count"] += 1
        upstream_optimizer_step()

    trainer.optimizer_step = counted_optimizer_step
    replay_id = samples[0].sample_id
    plan = build_replay_step_plan(
        run_id="ENGINEERING_CANARY_REAL_IMAGE",
        arm_id="T_U_ENGINEERING_ONLY",
        training_seed=training_seed,
        epoch=121,
        schedule_family="U",
        sample_ids=(replay_id,),
        base_batch_sizes=(len(base_ids),),
    )
    replay_aug: dict[str, tuple[torch.Tensor, str, int]] = {}

    def replay_provider(ids: Sequence[str], epoch: int, step: int, seed: int) -> Mapping[str, Any]:
        if tuple(ids) != (replay_id,) or epoch != 121 or step != 0 or seed != training_seed:
            raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Engineering canary replay provider identity changed")
        replay_seed = training_seed * 1000 + epoch * 10 + step
        tensor = _load_augmented_tensor(samples[0], transform)
        digest = _tensor_digest(tensor)
        replay_aug[replay_id] = (tensor, digest, replay_seed)
        return {
            "img": tensor.unsqueeze(0),
            "cls": torch.tensor([samples[0].label], dtype=torch.long),
            "sample_ids": (replay_id,),
            "augmentation_digests": (digest,),
            "augmentation_seeds": (replay_seed,),
        }

    step_receipts: list[UpstreamStepReceipt] = []
    occurrence_events: list[Any] = []
    model_digest_before = checkpoint_state_digest(model.state_dict())
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
        torch.cuda.synchronize(torch_device)
    step_started = time.perf_counter()
    step_result = run_ultralytics_classification_epoch(
        trainer=trainer,
        replay_plan=plan,
        replay_batch_provider=replay_provider,
        training_seed=training_seed,
        epoch=121,
        global_step_start=112_560,
        step_receipt_sink=step_receipts.append,
        occurrence_event_sink=occurrence_events.append,
        replay_rate_numerator=1,
        replay_rate_denominator=120_000,
        row_generation=1,
    )
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    step_seconds = time.perf_counter() - step_started
    model_digest_after = checkpoint_state_digest(model.state_dict())
    if model_digest_before == model_digest_after:
        raise SctsrError(ErrorCode.OPTIMIZER_STEP_SKIPPED, "Engineering canary model parameters did not change")
    if optimizer_calls["count"] != 1 or step_result["optimizer_steps"] != 1 or step_result["ema_updates_delta"] != 1:
        raise SctsrError(ErrorCode.REPLAY_ADDED_OPTIMIZER_STEP, "Engineering canary optimizer/EMA count is not exactly one")
    # The production sink receives one vectorized event per BASE/REPLAY
    # microbatch.  It is expanded into one Parquet row per occurrence below;
    # event cardinality is therefore two, while row cardinality is 4 + 1.
    if len(step_receipts) != 1 or len(occurrence_events) != 2:
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Engineering canary runtime evidence row counts are wrong")

    source_manifest = build_source_tree_manifest(
        repository,
        SOURCE_TREE_INCLUDE_PATHS,
        external_references=source_external_references(),
    )
    source_digest = str(source_manifest["source_tree_digest"])
    runtime_config = repository / "configs" / "stage1_sctsr_v4" / "runtime_policy_v1.json"
    training_lock = repository / "configs" / "stage1_gapvalue240" / "CANONICAL_TRAINING_LOCK_v1.json"
    registry_path = repository / "configs" / "stage1_sctsr_v4" / "asset_registry_v1.json"
    registry = load_asset_registry(registry_path)
    base_manifest_digest = stable_digest(sorted({sample.manifest_sha256 for sample in samples}))

    transaction_root = root / "03_epoch_transactions"
    tx = EpochTransaction(
        transaction_root,
        "ENGINEERING_CANARY_REAL_IMAGE",
        121,
        1,
        required_relative_paths=(
            "checkpoint/epoch_0121.pt",
            "evidence/run_id=ENGINEERING_CANARY_REAL_IMAGE/epoch=0121/occurrences.parquet",
            "evaluation/run_id=ENGINEERING_CANARY_REAL_IMAGE/epoch=0121/predictions.parquet",
            "evaluation/run_id=ENGINEERING_CANARY_REAL_IMAGE/epoch=0121/frontier.parquet",
            "evaluation/run_id=ENGINEERING_CANARY_REAL_IMAGE/epoch=0121/frontier_summary.json",
            "step_summary.json",
        ),
    ).begin()
    occurrence_rows: list[dict[str, Any]] = []
    for event in occurrence_events:
        logits = event.logits.detach().cpu()
        ce = event.per_sample_ce.detach().cpu()
        for index, sample_id in enumerate(event.sample_ids):
            occurrence_rows.append(
                {
                    "sample_id": str(sample_id),
                    "occurrence_role": str(event.occurrence_role),
                    "label": int(event.labels[index]),
                    "logit_normal": float(logits[index, 0]),
                    "logit_defect": float(logits[index, 1]),
                    "cross_entropy": float(ce[index]),
                    "augmentation_digest": str(event.augmentation_digests[index]),
                    "augmentation_seed": int(event.augmentation_seeds[index]),
                }
            )
    occurrence_path = tx.path_for(
        "evidence/run_id=ENGINEERING_CANARY_REAL_IMAGE/epoch=0121/occurrences.parquet"
    )
    occurrence_manifest = write_zstd_parquet(
        occurrence_rows,
        occurrence_path,
        schema_version="stage1.sctsr.engineering_canary_occurrence.v1",
        require_run_epoch_partition=True,
    )
    if not occurrence_manifest.canonical_parquet or occurrence_manifest.compression != "ZSTD":
        raise SctsrError(ErrorCode.COLUMNAR_ENGINE_UNAVAILABLE, "Engineering canary did not use real Zstd Parquet")

    checkpoint_path = tx.path_for("checkpoint/epoch_0121.pt")
    checkpoint_payload = build_checkpoint_payload(
        model=model,
        ema=trainer.ema,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        epoch=121,
        global_step=int(step_result["global_step_end"]),
        base_sampler_generation=1,
        canonical_training_lock_sha256=sha256_file(training_lock),
        initial_checkpoint_sha256=EXPECTED_WEIGHT_SHA256,
        base_manifest_sha256=base_manifest_digest,
        training_seed=training_seed,
        source_tree_digest=source_digest,
        runtime_config_digest=sha256_file(runtime_config),
        asset_registry_digest=registry.digest,
    )
    checkpoint_sha = save_checkpoint_atomic(checkpoint_path, checkpoint_payload)
    reloaded = load_checkpoint(checkpoint_path, expected_sha256=checkpoint_sha, expected_epoch=121)

    model.eval()
    with torch.no_grad():
        evaluation_images = torch.stack([tensors[sample_id] for sample_id in base_ids]).to(torch_device)
        probabilities, logits = _probabilities_and_logits(model(evaluation_images), batch_size=len(base_ids))
        logits = logits.detach().cpu()
        probabilities = probabilities.detach().cpu()[:, 1]
    split_manifest_sha = stable_digest(
        [(sample.sample_id, sample.label, sample.image_sha256) for sample in samples]
    )
    predictions = tuple(
        PredictionRow(
            run_id="ENGINEERING_CANARY_REAL_IMAGE",
            arm_id="T_U_ENGINEERING_ONLY",
            training_seed=training_seed,
            split_role="synthetic",
            split_manifest_path="ENGINEERING_CANARY_TRAIN_ONLY_MICROSET",
            split_manifest_sha256=split_manifest_sha,
            sample_id=sample.sample_id,
            y_true=sample.label,
            logit_normal=float(logits[index, 0]),
            logit_defect=float(logits[index, 1]),
            p_defect_raw=float(probabilities[index]),
            checkpoint_epoch=121,
            checkpoint_sha256=checkpoint_sha,
            model_variant="MODEL_SYNTHETIC_CANARY",
            source_tree_digest=source_digest,
            prediction_generation=1,
        )
        for index, sample in enumerate(samples)
    )
    prediction_path = tx.path_for(
        "evaluation/run_id=ENGINEERING_CANARY_REAL_IMAGE/epoch=0121/predictions.parquet"
    )
    prediction_manifest, prediction_summary = write_prediction_artifact(
        predictions,
        prediction_path,
        formal_endpoint=False,
        expected_sample_labels={sample.sample_id: sample.label for sample in samples},
    )
    points, frontier_summary = compute_tie_safe_frontier(
        predictions,
        max_fn=95,
        target_tn=68_253,
        checkpoint_sha256=checkpoint_sha,
        prediction_artifact_sha256=prediction_manifest.sha256,
    )
    frontier_path = tx.path_for(
        "evaluation/run_id=ENGINEERING_CANARY_REAL_IMAGE/epoch=0121/frontier.parquet"
    )
    frontier_manifest, frontier_receipt = write_frontier_artifacts(
        points,
        frontier_summary,
        frontier_path=frontier_path,
        summary_path=tx.path_for(
            "evaluation/run_id=ENGINEERING_CANARY_REAL_IMAGE/epoch=0121/frontier_summary.json"
        ),
        evaluation_mode="synthetic",
        split_role="synthetic",
        checkpoint_epoch=121,
    )
    tx.write_json(
        "step_summary.json",
        {
            "scientific_role": ENGINEERING_CANARY_SEMANTIC,
            "step_result": step_result,
            "step_receipt": asdict(step_receipts[0]),
            "model_state_digest_before": model_digest_before,
            "model_state_digest_after": model_digest_after,
            "prediction_summary": prediction_summary,
            "frontier_summary": asdict(frontier_summary),
        },
    )
    generation = tx.commit()
    complete_checkpoint = tx.complete / "checkpoint" / "epoch_0121.pt"
    post_commit_reload = load_checkpoint(complete_checkpoint, expected_sha256=checkpoint_sha, expected_epoch=121)
    pointer = validate_recovery_pointer(root / "ROLLING_RECOVERY_POINTER.json")
    last_complete = find_last_complete_epoch(transaction_root)

    corrupt = root / "fault_injection" / "truncated_epoch_0121.pt"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    with complete_checkpoint.open("rb") as source, corrupt.open("wb") as destination:
        destination.write(source.read(4096))
        destination.flush()
        os.fsync(destination.fileno())
    corrupt_rejected = False
    try:
        load_checkpoint(corrupt, expected_epoch=121)
    except (SctsrError, EOFError, RuntimeError, OSError):
        corrupt_rejected = True
    if not corrupt_rejected:
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Truncated engineering checkpoint was not rejected")

    partial = transaction_root / "epoch_0122.generation_1.inprogress"
    partial.mkdir(parents=True)
    (partial / "half_written.json").write_bytes(b'{"partial":')
    moved = quarantine_inprogress(transaction_root, root / "09_quarantine", reason="ENGINEERING_CANARY_INJECTED_KILL")
    partial_quarantined = len(moved) == 1 and Path(moved[0]).is_dir() and not partial.exists()
    if not partial_quarantined:
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Injected partial generation was not quarantined")

    gpu_record: dict[str, Any]
    if torch_device.type == "cuda":
        properties = torch.cuda.get_device_properties(torch_device)
        gpu_record = {
            "device": str(torch_device),
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(torch_device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(torch_device)),
        }
    else:
        gpu_record = {"device": "cpu", "reason": "CUDA_UNAVAILABLE_OR_CPU_EXPLICITLY_REQUESTED"}
    report: dict[str, Any] = {
        "schema_version": "stage1.sctsr.engineering_canary.v1",
        "status": "PASS",
        "scientific_role": ENGINEERING_CANARY_SEMANTIC,
        "research_claim": "NONE_MECHANICS_ONLY",
        "repository_root": repository.as_posix(),
        "dataset_root": dataset.as_posix(),
        "artifact_root": root.as_posix(),
        "training_seed": training_seed,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_build": str(torch.version.cuda),
        "nvidia_driver": _driver_version(),
        "device": gpu_record,
        "trainer_source": trainer_source.as_posix(),
        "trainer_class": f"{type(trainer).__module__}.{type(trainer).__name__}",
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "real_image_count": len(samples),
        "real_image_bytes_verified": all(
            Path(sample.image_path).stat().st_size == sample.image_bytes
            and sha256_file(sample.image_path) == sample.image_sha256
            for sample in samples
        ),
        "real_images": [asdict(sample) for sample in samples],
        "real_yolo_weight_verified": sha256_file(weight) == EXPECTED_WEIGHT_SHA256,
        "initial_weight": {
            "path": weight.as_posix(),
            "bytes": weight.stat().st_size,
            "sha256": EXPECTED_WEIGHT_SHA256,
        },
        "augmentation": {
            "semantic": "FROZEN_CLASSIFICATION_AUGMENTATION_PARAMETERS_ENGINEERING_MICROBATCH_ONLY",
            "imgsz": 224,
            "scale": [0.5, 1.0],
            "hflip": 0.5,
            "vflip": 0.0,
            "erasing": 0.4,
            "auto_augment": "randaugment",
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "base_tensor_digests": base_augmentation_digests,
            "replay_tensor_digest": replay_aug[replay_id][1],
        },
        "source_tree_digest": source_digest,
        "source_tree_git_head": source_manifest.get("git_head"),
        "source_tree_git_dirty": source_manifest.get("git_dirty"),
        "forward_passed": len(occurrence_events) == 2,
        "base_backward_passed": any(row.occurrence_role == "BASE" for row in occurrence_events),
        "replay_backward_passed": any(row.occurrence_role == "REPLAY" for row in occurrence_events),
        "optimizer_step_count": optimizer_calls["count"],
        "ema_update_count": int(step_result["ema_updates_delta"]),
        "base_occurrences": int(step_result["base_occurrences"]),
        "replay_occurrences": int(step_result["replay_occurrences"]),
        "global_step_start": 112_560,
        "global_step_end": int(step_result["global_step_end"]),
        "step_seconds": step_seconds,
        "total_seconds": time.perf_counter() - started,
        "model_state_digest_before": model_digest_before,
        "model_state_digest_after": model_digest_after,
        "rng_restored_across_replay": step_receipts[0].rng_before_replay == step_receipts[0].rng_after_replay,
        "batchnorm_restored_across_replay": step_receipts[0].bn_before_replay == step_receipts[0].bn_after_replay,
        "parquet": {
            **occurrence_manifest.as_dict(),
            "validation": validate_columnar_file(
                tx.complete / "evidence/run_id=ENGINEERING_CANARY_REAL_IMAGE/epoch=0121/occurrences.parquet",
                expected_rows=len(occurrence_rows),
                expected_schema_version="stage1.sctsr.engineering_canary_occurrence.v1",
                expected_sha256=occurrence_manifest.sha256,
            ),
        },
        "checkpoint": {
            "save_passed": checkpoint_sha == sha256_file(complete_checkpoint),
            "reload_passed": (
                reloaded["checkpoint_payload_digest"]
                == post_commit_reload["checkpoint_payload_digest"]
                == checkpoint_payload["checkpoint_payload_digest"]
            ),
            "path": complete_checkpoint.as_posix(),
            "bytes": complete_checkpoint.stat().st_size,
            "sha256": checkpoint_sha,
            "payload_digest": checkpoint_payload["checkpoint_payload_digest"],
        },
        "transaction": {
            "generation_digest": generation["generation_digest"],
            "generation_manifest_sha256": generation["generation_manifest_sha256"],
            "receipt_chain_digest": generation["receipt_chain_digest"],
            "pointer_epoch": int(pointer["epoch"]),
            "last_complete_epoch": int(last_complete["epoch"]),
        },
        "recovery": {
            "corrupt_checkpoint_rejected": corrupt_rejected,
            "corrupt_checkpoint_path": corrupt.as_posix(),
            "partial_generation_quarantined": partial_quarantined,
            "quarantine_paths": moved,
        },
        "prediction": prediction_manifest.as_dict(),
        "frontier": frontier_manifest.as_dict(),
        "frontier_point_count": len(points),
        "frontier_summary": frontier_receipt,
        "formal_training_started": False,
        "assignments_generated": False,
        "engineering_gate_generated": False,
        "pilot_release_generated": False,
        "blind_holdout_opened": False,
        "test_accessed": False,
        "method_effectiveness_claimed": False,
    }
    validate_engineering_canary_report(report)
    atomic_write_json(root / "ENGINEERING_CANARY_REPORT.json", report)
    manifest = _artifact_manifest(root)
    atomic_write_json(
        root / "ENGINEERING_CANARY_ARTIFACT_MANIFEST.json",
        {
            "schema_version": "stage1.sctsr.engineering_canary_artifact_manifest.v1",
            "scientific_role": ENGINEERING_CANARY_SEMANTIC,
            "files": manifest,
            "file_count": len(manifest),
            "manifest_digest": stable_digest(manifest),
        },
    )
    return report
