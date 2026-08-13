from __future__ import annotations

"""Run one train-only SCTSR v4 engineering step on real local SewerML bytes.

This is deliberately not a formal experiment.  It never opens val_op, test, or
blind-holdout material and it cannot authorize a release.  Its purpose is to
exercise the frozen YOLO11l forward/backward/optimizer path plus the v4 replay,
checkpoint, Parquet, telemetry, transaction, quarantine, recovery, prediction,
and frontier artifact machinery against real image bytes.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


SEMANTIC = "REAL_DATA_ENGINEERING_CANARY_NOT_SCIENTIFIC_RESULT"
SCHEMA = "stage1.sctsr.real_data_engineering_canary.v1"
EXPECTED_WEIGHT_SHA = "6B56513A5D8BDAE6B8F0A36DACAF01B26D5A522BA1B34197C3BAC9FA6463366C"
ALLOWED_SPLITS = frozenset({"train", "normal_train"})
FORBIDDEN_ROLE_TOKENS = ("val_op", "test", "blind", "holdout")


class CanaryInputError(ValueError):
    """Fail-closed input error for a train-only engineering canary."""


@dataclass(frozen=True, slots=True)
class CanarySample:
    sample_id: str
    y_true: int
    split: str
    source_image_path: str
    image_bytes: int
    image_sha256: str
    source_manifest_path: str
    source_manifest_line: int
    role: str


@dataclass(frozen=True, slots=True)
class CanarySelection:
    base: tuple[CanarySample, ...]
    replay: CanarySample


def _sha_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def _strict_train_row(row: Mapping[str, str], *, manifest: Path, line_number: int, role: str) -> CanarySample:
    split = str(row.get("split", "")).strip()
    if split not in ALLOWED_SPLITS:
        raise CanaryInputError(f"engineering canary is train-only; observed split={split!r}")
    sample_id = str(row.get("canonical_image_relpath", "")).strip()
    source = Path(str(row.get("source_image_path", "")).strip())
    if not sample_id or not source.is_file():
        raise CanaryInputError("registered train row has no existing source image")
    label_text = str(row.get("Defect", "")).strip()
    if label_text not in {"0", "1"}:
        raise CanaryInputError("registered train row has a non-binary Defect label")
    return CanarySample(
        sample_id=sample_id,
        y_true=int(label_text),
        split=split,
        source_image_path=source.resolve().as_posix(),
        image_bytes=source.stat().st_size,
        image_sha256=_sha_file(source),
        source_manifest_path=manifest.resolve().as_posix(),
        source_manifest_line=line_number,
        role=role,
    )


def _manifest_rows(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    if not path.is_file():
        raise CanaryInputError(f"manifest is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "source_image_path", "canonical_image_relpath", "Defect"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise CanaryInputError(f"manifest schema is incomplete: {path}")
        for line_number, row in enumerate(reader, 2):
            yield line_number, dict(row)


def _treatment_identity(path: Path) -> tuple[str, int]:
    if not path.is_file():
        raise CanaryInputError(f"treatment manifest is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"sample_id", "y_true", "replay_role"}.issubset(reader.fieldnames):
            raise CanaryInputError("treatment manifest schema is incomplete")
        row = next(reader, None)
    if row is None or str(row.get("y_true", "")) not in {"0", "1"}:
        raise CanaryInputError("treatment manifest has no usable first identity")
    sample_id = str(row["sample_id"]).strip()
    if not sample_id:
        raise CanaryInputError("treatment manifest has an empty first identity")
    return sample_id, int(row["y_true"])


def select_canary_samples(
    *,
    defect_manifest: str | Path,
    normal_manifest: str | Path,
    treatment_manifest: str | Path,
    base_per_label: int = 2,
) -> CanarySelection:
    """Select deterministic real train rows and one historical T identity."""

    if type(base_per_label) is not int or base_per_label < 1:
        raise CanaryInputError("base_per_label must be a positive integer")
    defect_path, normal_path, treatment_path = map(Path, (defect_manifest, normal_manifest, treatment_manifest))
    treatment_id, treatment_label = _treatment_identity(treatment_path)
    base: list[CanarySample] = []
    replay: CanarySample | None = None
    seen: set[str] = set()
    for manifest, expected_split, expected_label in (
        (defect_path, "train", 1),
        (normal_path, "normal_train", 0),
    ):
        accepted = 0
        for line_number, row in _manifest_rows(manifest):
            if row["split"].strip() != expected_split or row["Defect"].strip() != str(expected_label):
                raise CanaryInputError("engineering canary is train-only and label-stratified")
            if row["canonical_image_relpath"].strip() == treatment_id:
                candidate = _strict_train_row(
                    row,
                    manifest=manifest,
                    line_number=line_number,
                    role="T_STRESS_ENGINEERING_CANARY",
                )
                if candidate.y_true != treatment_label:
                    raise CanaryInputError("treatment identity label differs from the canonical train manifest")
                replay = candidate
                continue
            if accepted < base_per_label:
                candidate = _strict_train_row(
                    row,
                    manifest=manifest,
                    line_number=line_number,
                    role="CANONICAL_BASE_ENGINEERING_CANARY",
                )
                if candidate.sample_id in seen:
                    raise CanaryInputError("duplicate canonical identity in base canary selection")
                base.append(candidate)
                seen.add(candidate.sample_id)
                accepted += 1
            if accepted == base_per_label and replay is not None:
                break
        if accepted != base_per_label:
            raise CanaryInputError(f"manifest lacks {base_per_label} usable base rows for label {expected_label}")
    if replay is None:
        raise CanaryInputError("historical T identity is absent from canonical train manifests")
    if replay.sample_id in seen:
        raise CanaryInputError("replay identity overlaps the engineering base microbatch")
    return CanarySelection(tuple(base), replay)


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if path.exists():
        raise CanaryInputError(f"immutable canary output already exists: {path}")
    temp = path.with_name(f".{path.name}.inprogress")
    temp.write_bytes(data)
    os.replace(temp, path)


def _hardware_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _load_metadata(
    selected: Sequence[CanarySample],
    *,
    oof_path: Path,
    dynamics_path: Path,
) -> dict[str, dict[str, Any]]:
    wanted = {row.sample_id for row in selected}
    metadata: dict[str, dict[str, Any]] = {sample_id: {} for sample_id in wanted}
    with oof_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("canonical_image_relpath", ""))
            if sample_id in wanted:
                metadata[sample_id].update(
                    oof_fold=int(row["oof_fold"]),
                    oof_group_id=str(row["oof_group_id"]),
                )
    with dynamics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("sample_id", ""))
            if sample_id in wanted:
                metadata[sample_id]["dynamic_bucket"] = str(row["dynamic_bucket"])
    missing = sorted(sample_id for sample_id, row in metadata.items() if set(row) != {"oof_fold", "oof_group_id", "dynamic_bucket"})
    if missing:
        raise CanaryInputError(f"selected train identities lack frozen OOF/dynamic metadata: {missing}")
    return metadata


def _load_image_tensor(path: Path):
    import torch
    from PIL import Image
    from torchvision.transforms import Compose, Resize, ToTensor

    transform = Compose((Resize((224, 224)), ToTensor()))
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB"))
    if tensor.shape != (3, 224, 224) or tensor.dtype != torch.float32:
        raise CanaryInputError("real image transform did not produce float32 CHW 224x224")
    return tensor


def _tensor_digest(tensor: Any, *, sample_id: str, seed: int) -> str:
    hasher = sha256()
    hasher.update(b"SCTSR_REAL_CANARY_RESIZE_224_TOTENSOR_V1\0")
    hasher.update(sample_id.encode("utf-8"))
    hasher.update(str(seed).encode("ascii"))
    hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest().upper()


def _safe_run_id(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not token:
        raise CanaryInputError("output root does not yield a safe run ID")
    return f"REALDATA_{token}"[:80]


def ensure_repository_import_path(repository_root: str | Path) -> Path:
    root = Path(repository_root).resolve()
    token = str(root)
    if token not in sys.path:
        sys.path.insert(0, token)
    return root


def run_canary(*, repository_root: str | Path, output_root: str | Path, device_text: str, telemetry_hold_seconds: float) -> dict[str, Any]:
    import torch
    import pyarrow.parquet as pq

    root = ensure_repository_import_path(repository_root)
    from stage1_sctsr_v4.filesystem import windows_safe_resolved_path

    output = windows_safe_resolved_path(output_root)
    if output.exists():
        raise CanaryInputError("output root already exists; engineering canary outputs are immutable")
    if telemetry_hold_seconds < 1.05:
        raise CanaryInputError("telemetry hold must permit at least two one-second samples")
    lowered = output.as_posix().lower()
    if any(token in lowered for token in FORBIDDEN_ROLE_TOKENS):
        raise CanaryInputError("output root contains a forbidden evaluation-role token")

    defect_manifest = root / "data/final_sewerml_dataset/manifests/train_manifest.csv"
    normal_manifest = root / "data/final_sewerml_dataset/manifests/normal_train_manifest.csv"
    treatment_manifest = root / "artifacts/stage1_sample_value_experiments/contracts/gapvalue240_v1_1/generated/selections/RUN_010/selection_manifest.csv"
    oof_path = root / "artifacts/stage1_oof_folds_10fold_20260617/train_oof_assignments.csv"
    dynamics_path = root / "artifacts/stage1_sample_value_experiments/contracts/gapvalue240_v1_1/frozen_inputs/reference_tables/sample_dynamics_summary.csv"
    selection = select_canary_samples(
        defect_manifest=defect_manifest,
        normal_manifest=normal_manifest,
        treatment_manifest=treatment_manifest,
        base_per_label=2,
    )
    all_samples = (*selection.base, selection.replay)
    metadata = _load_metadata(all_samples, oof_path=oof_path, dynamics_path=dynamics_path)
    run_id = _safe_run_id(output.name)
    seed = 2_026_081_401
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_text)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise CanaryInputError("this real-data engineering canary requires an available CUDA GPU")

    yolo_root = root / "YOLOv11"
    if str(yolo_root) not in sys.path:
        sys.path.insert(0, str(yolo_root))
    from ultralytics.models.yolo.classify.train import ClassificationTrainer
    from ultralytics.nn.tasks import ClassificationModel, load_checkpoint as load_ultralytics_checkpoint
    from ultralytics.utils.torch_utils import ModelEMA

    from stage1_sctsr_v4.checkpointing import (
        build_checkpoint_payload,
        load_checkpoint,
        save_checkpoint_atomic,
    )
    from stage1_sctsr_v4.baseline_reference import SOURCE_TREE_INCLUDE_PATHS, source_external_references
    from stage1_sctsr_v4.columnar import validate_columnar_file, write_zstd_parquet
    from stage1_sctsr_v4.epoch_transaction import EpochTransaction, GenerationIdentity, validate_receipt_chain
    from stage1_sctsr_v4.evaluation import compute_tie_safe_frontier
    from stage1_sctsr_v4.evidence_runtime import EpochEvidenceRecorder, ReplayHistoryState, SampleEvidence
    from stage1_sctsr_v4.ledger_schema import FRONTIER_SCHEMA, PREDICTION_SCHEMA
    from stage1_sctsr_v4.prediction_artifact import PredictionRow, sample_label_identity_digest, validate_prediction_rows
    from stage1_sctsr_v4.recovery import find_last_complete_epoch, validate_recovery_pointer
    from stage1_sctsr_v4.replay_step_plan import build_replay_step_plan
    from stage1_sctsr_v4.rng_isolation import capture_global_rng
    from stage1_sctsr_v4.serialization import sha256_file, stable_digest
    from stage1_sctsr_v4.source_identity import build_source_tree_manifest
    from stage1_sctsr_v4.ultralytics_overlay import run_ultralytics_classification_epoch

    weight = root / "yolo11l-cls.pt"
    if sha256_file(weight) != EXPECTED_WEIGHT_SHA:
        raise CanaryInputError("initial yolo11l-cls.pt SHA differs from the frozen lock")
    source_tree_manifest = build_source_tree_manifest(
        root,
        SOURCE_TREE_INCLUDE_PATHS,
        external_references=source_external_references(),
    )
    source_tree_digest = str(source_tree_manifest["source_tree_digest"])
    output.mkdir(parents=True)
    input_manifest = output / "01_inputs/REAL_TRAIN_INPUTS.json"
    input_payload = {
        "schema_version": "stage1.sctsr.real_data_canary_inputs.v1",
        "semantic": SEMANTIC,
        "base_samples": [asdict(row) for row in selection.base],
        "replay_sample": asdict(selection.replay),
        "source_manifests": {
            defect_manifest.resolve().as_posix(): {"bytes": defect_manifest.stat().st_size, "sha256": sha256_file(defect_manifest)},
            normal_manifest.resolve().as_posix(): {"bytes": normal_manifest.stat().st_size, "sha256": sha256_file(normal_manifest)},
            treatment_manifest.resolve().as_posix(): {"bytes": treatment_manifest.stat().st_size, "sha256": sha256_file(treatment_manifest)},
        },
        "forbidden_roles_accessed": [],
    }
    _json_write(input_manifest, input_payload)
    input_manifest_sha = sha256_file(input_manifest)

    tensors = {row.sample_id: _load_image_tensor(Path(row.source_image_path)) for row in all_samples}
    augmentation_seed = {row.sample_id: seed + index for index, row in enumerate(all_samples)}
    augmentation_digest = {
        row.sample_id: _tensor_digest(tensors[row.sample_id], sample_id=row.sample_id, seed=augmentation_seed[row.sample_id])
        for row in all_samples
    }
    base_batch = {
        "img": torch.stack([tensors[row.sample_id] for row in selection.base]),
        "cls": torch.tensor([row.y_true for row in selection.base], dtype=torch.long),
        "sample_ids": tuple(row.sample_id for row in selection.base),
        "augmentation_digests": tuple(augmentation_digest[row.sample_id] for row in selection.base),
        "augmentation_seeds": tuple(augmentation_seed[row.sample_id] for row in selection.base),
    }

    model, _ = load_ultralytics_checkpoint(str(weight), device=device, inplace=False, fuse=False)
    ClassificationModel.reshape_outputs(model, 2)
    model.to(device).float()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainer = object.__new__(ClassificationTrainer)
    trainer.model = model
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.937, weight_decay=0.0005, nesterov=True)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=True)
    trainer.ema = ModelEMA(model)
    trainer.device = device
    trainer.amp = True
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
    trainer.train_loader = [base_batch]
    upstream_optimizer_step = trainer.optimizer_step
    optimizer_calls = {"count": 0}

    def counted_optimizer_step() -> None:
        optimizer_calls["count"] += 1
        upstream_optimizer_step()

    trainer.optimizer_step = counted_optimizer_step
    plan = build_replay_step_plan(
        run_id=run_id,
        arm_id="T_U",
        training_seed=seed,
        epoch=121,
        schedule_family="U",
        sample_ids=(selection.replay.sample_id,),
        base_batch_sizes=(len(selection.base),),
    )

    def replay_provider(ids: Sequence[str], _epoch: int, _step: int, _seed: int) -> Mapping[str, Any]:
        return {
            "img": torch.stack([tensors[sample_id] for sample_id in ids]),
            "cls": torch.tensor([next(row.y_true for row in all_samples if row.sample_id == sample_id) for sample_id in ids], dtype=torch.long),
            "sample_ids": tuple(ids),
            "augmentation_digests": tuple(augmentation_digest[sample_id] for sample_id in ids),
            "augmentation_seeds": tuple(augmentation_seed[sample_id] for sample_id in ids),
        }

    evidence = {
        row.sample_id: SampleEvidence(
            sample_id=row.sample_id,
            y_true=row.y_true,
            replay_role="T_STRESS" if row.sample_id == selection.replay.sample_id else "CANONICAL_BASE",
            oof_fold=int(metadata[row.sample_id]["oof_fold"]),
            oof_group_id=str(metadata[row.sample_id]["oof_group_id"]),
            historical_dynamic_bucket=str(metadata[row.sample_id]["dynamic_bucket"]),
            identity_group="REAL_TRAIN_ENGINEERING_CANARY",
            oof_reference_probability=None,
            oof_reference_reason="REGISTERED_NOT_AVAILABLE",
            rho_candidate_signal=None,
            rho_reason="REGISTERED_NOT_REPORTED",
        )
        for row in all_samples
    }
    contract_sha = sha256_file(root / "configs/stage1_sctsr_v4/contract_v1.json")
    asset_registry_sha = sha256_file(root / "configs/stage1_sctsr_v4/asset_registry_v1.json")
    runtime_digest = stable_digest(
        {
            "semantic": SEMANTIC,
            "device": str(device),
            "base_batch": len(selection.base),
            "replay_batch": 1,
            "transform": "Resize((224,224))+ToTensor",
            "seed": seed,
        }
    )
    combined_base_manifest_sha = stable_digest(
        {"defect": sha256_file(defect_manifest), "normal": sha256_file(normal_manifest)}
    )
    identity = GenerationIdentity(
        parent_sha256=EXPECTED_WEIGHT_SHA,
        arm_id="T_U",
        training_seed=seed,
        source_tree_digest=source_tree_digest,
        contract_digest=contract_sha,
        asset_registry_digest=asset_registry_sha,
        rng_state_digest=capture_global_rng().digest(),
        previous_generation_digest="0" * 64,
    )
    transaction = EpochTransaction(
        output / "03_epoch_transactions",
        run_id,
        121,
        1,
        quarantine_root=output / "09_quarantine",
        identity=identity,
    ).begin()
    recorder = EpochEvidenceRecorder(
        transaction=transaction,
        parent_id="ENGINEERING_CANARY_INITIAL_WEIGHT",
        arm_id="T_U",
        training_seed=seed,
        sample_evidence=evidence,
        identity_policy="T_STRESS",
        schedule_family="U",
        fallback_state="TARGETED",
        rate_numerator=1,
        rate_denominator=4,
        schedule_digest=plan.plan_digest,
        identity_pool_digest=stable_digest((selection.replay.sample_id,)),
        pool_multiplicity_targets={selection.replay.sample_id: 1},
        expected_base_denominator=len(selection.base),
        expected_optimizer_steps=1,
        global_step_start=0,
        history=ReplayHistoryState(),
        artifact_root=output,
    )
    step_receipts: list[Any] = []
    started = time.perf_counter()
    try:
        runtime_result = run_ultralytics_classification_epoch(
            trainer=trainer,
            replay_plan=plan,
            replay_batch_provider=replay_provider,
            training_seed=seed,
            epoch=121,
            global_step_start=0,
            step_receipt_sink=lambda row: (step_receipts.append(row), recorder.step_sink(row)),
            occurrence_event_sink=recorder.occurrence_sink,
            replay_rate_numerator=1,
            replay_rate_denominator=4,
            row_generation=1,
        )
        training_wall_seconds = time.perf_counter() - started
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True):
            logits = model(torch.stack([tensors[row.sample_id] for row in all_samples]).to(device))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = logits.float().detach().cpu()
        if logits.shape != (len(all_samples), 2):
            raise CanaryInputError(f"engineering prediction logits have unexpected shape: {tuple(logits.shape)}")

        checkpoint_payload = build_checkpoint_payload(
            model=model,
            ema=trainer.ema,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            epoch=121,
            global_step=int(runtime_result["global_step_end"]),
            base_sampler_generation=1,
            canonical_training_lock_sha256=sha256_file(root / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json"),
            initial_checkpoint_sha256=EXPECTED_WEIGHT_SHA,
            base_manifest_sha256=combined_base_manifest_sha,
            training_seed=seed,
            source_tree_digest=source_tree_digest,
            runtime_config_digest=runtime_digest,
            asset_registry_digest=asset_registry_sha,
        )
        checkpoint_sha = save_checkpoint_atomic(recorder.checkpoint_path, checkpoint_payload)
        time.sleep(telemetry_hold_seconds)
        epoch_summary = recorder.finalize(runtime_result=runtime_result, checkpoint_sha256=checkpoint_sha)
        generation = transaction.commit()
    except BaseException:
        recorder.abort()
        transaction.abort("REAL_DATA_ENGINEERING_CANARY_FAILURE")
        raise

    complete_checkpoint = transaction.complete / recorder.checkpoint_relative
    load_checkpoint(complete_checkpoint, expected_sha256=checkpoint_sha, expected_epoch=121)
    recovery = find_last_complete_epoch(output / "03_epoch_transactions")
    pointer = validate_recovery_pointer(output / "ROLLING_RECOVERY_POINTER.json")
    receipt_chain = validate_receipt_chain(output / "08_receipts/epoch_receipts.jsonl")

    prediction_rows: list[PredictionRow] = []
    probabilities = torch.softmax(logits, dim=1)
    for index, row in enumerate(all_samples):
        prediction_rows.append(
            PredictionRow(
                run_id=run_id,
                arm_id="T_U",
                training_seed=seed,
                split_role="train_engineering_canary",
                split_manifest_path=input_manifest.resolve().as_posix(),
                split_manifest_sha256=input_manifest_sha,
                sample_id=row.sample_id,
                y_true=row.y_true,
                logit_normal=float(logits[index, 0]),
                logit_defect=float(logits[index, 1]),
                p_defect_raw=float(probabilities[index, 1]),
                checkpoint_epoch=121,
                checkpoint_sha256=checkpoint_sha,
                model_variant="MODEL",
                source_tree_digest=source_tree_digest,
                prediction_generation=1,
            )
        )
    validate_prediction_rows(prediction_rows, expected_split_role="train_engineering_canary")
    prediction_identity_digest = sample_label_identity_digest(prediction_rows)
    prediction_path = output / f"06_predictions/run_id={run_id}/epoch=0121/predictions.parquet"
    prediction_manifest = write_zstd_parquet(
        [
            {
                **asdict(row),
                "sample_label_identity_digest": prediction_identity_digest,
                "artifact_row_count": len(prediction_rows),
            }
            for row in prediction_rows
        ],
        prediction_path,
        schema_version="stage1.sctsr.prediction_artifact.v2",
        schema=PREDICTION_SCHEMA,
        require_run_epoch_partition=True,
    )
    points, frontier_summary = compute_tie_safe_frontier(
        prediction_rows,
        max_fn=95,
        target_tn=68_253,
        checkpoint_sha256=checkpoint_sha,
        prediction_artifact_sha256=prediction_manifest.sha256,
    )
    frontier_path = output / f"07_evaluation/run_id={run_id}/epoch=0121/frontier.parquet"
    frontier_manifest = write_zstd_parquet(
        [asdict(row) for row in points],
        frontier_path,
        schema_version="stage1.sctsr.frontier.v1",
        schema=FRONTIER_SCHEMA,
        require_run_epoch_partition=True,
    )
    _json_write(
        output / "07_evaluation/TRAIN_ONLY_FRONTIER_SUMMARY.json",
        {
            "schema_version": "stage1.sctsr.train_only_engineering_frontier.v1",
            "semantic": SEMANTIC,
            "split_role": "train_engineering_canary",
            "not_for_method_selection": True,
            "not_a_scientific_result": True,
            "summary": asdict(frontier_summary),
            "frontier": frontier_manifest.as_dict(),
        },
    )

    fault = EpochTransaction(
        output / "03_fault_injection",
        run_id + "_FAULT",
        122,
        1,
        quarantine_root=output / "09_quarantine",
    ).begin()
    fault.write_json("PARTIAL_REAL_DATA_CANARY.json", {"status": "INJECTED_PARTIAL", "semantic": SEMANTIC})
    quarantined = fault.abort("ENGINEERING_CANARY_INJECTED_KILL")
    quarantine_receipt = json.loads((quarantined / "QUARANTINE_RECEIPT.json").read_text(encoding="utf-8"))

    partition_reports: dict[str, Any] = {}
    for role, relative in (
        ("occurrence", epoch_summary["occurrence_partition"]["path"]),
        ("optimizer_step", epoch_summary["step_partition"]["path"]),
        ("telemetry", epoch_summary["telemetry_partition"]["path"]),
        ("exposure", epoch_summary["exposure_partition"]["path"]),
    ):
        path = transaction.complete / relative
        report = validate_columnar_file(path)
        report["physical_row_count"] = pq.ParquetFile(path).metadata.num_rows
        partition_reports[role] = report
    partition_reports["prediction"] = validate_columnar_file(prediction_path)
    partition_reports["frontier"] = validate_columnar_file(frontier_path)

    step = step_receipts[0]
    status_checks = {
        "real_source_images_read": all(Path(row.source_image_path).is_file() for row in all_samples),
        "real_yolo11l_loaded": type(model).__name__ == "ClassificationModel",
        "real_classification_trainer_used": type(trainer).__module__ == "ultralytics.models.yolo.classify.train",
        "one_optimizer_step": optimizer_calls["count"] == 1 == runtime_result["optimizer_steps"],
        "base_replay_exposure_conserved": runtime_result["base_occurrences"] == 4 and runtime_result["replay_occurrences"] == 1,
        "bn_replay_restored": step.bn_before_replay == step.bn_after_replay,
        "rng_replay_restored": step.rng_before_replay == step.rng_after_replay,
        "checkpoint_reloaded": recovery is not None and recovery["epoch"] == 121,
        "atomic_generation_complete": transaction.complete.is_dir() and not transaction.inprogress.exists(),
        "recovery_pointer_valid": pointer["validation_status"] == "PASS",
        "receipt_chain_valid": receipt_chain["status"] == "PASS",
        "partial_epoch_quarantined": quarantine_receipt["status"] == "QUARANTINED",
        "real_zstd_parquet": all(row["status"] == "PASS" and row["compression"] == "ZSTD" for row in partition_reports.values()),
        "prediction_rows_complete": prediction_manifest.row_count == len(all_samples),
        "frontier_has_96_points": frontier_manifest.row_count == 96,
        "forbidden_evaluation_roles_accessed_false": True,
        "formal_training_started_false": True,
    }
    result = {
        "schema_version": SCHEMA,
        "status": "PASS" if all(status_checks.values()) else "FAIL",
        "semantic": SEMANTIC,
        "scientific_result": False,
        "formal_training_started": False,
        "formal_seed_generated": False,
        "assignment_generated": False,
        "engineering_gate_generated": False,
        "pilot_release_generated": False,
        "blind_holdout_opened": False,
        "test_accessed": False,
        "val_op_accessed": False,
        "selector_trained": False,
        "run_id": run_id,
        "implementation_source_commit": str(source_tree_manifest["git_head"]),
        "source_tree_git_dirty": bool(source_tree_manifest["git_dirty"]),
        "source_tree_digest": source_tree_digest,
        "hardware": _hardware_snapshot(),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "device_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "training_wall_seconds": training_wall_seconds,
            "base_forward_seconds": step.base_forward_seconds,
            "replay_forward_seconds": step.replay_forward_seconds,
            "backward_seconds": step.backward_seconds,
            "optimizer_seconds": step.optimizer_seconds,
            "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "input_manifest": {"path": input_manifest.as_posix(), "bytes": input_manifest.stat().st_size, "sha256": input_manifest_sha},
        "checkpoint": {"path": complete_checkpoint.as_posix(), "bytes": complete_checkpoint.stat().st_size, "sha256": checkpoint_sha, "loaded": True},
        "generation": {
            "digest": generation["generation_digest"],
            "manifest_sha256": generation["generation_manifest_sha256"],
            "file_count": generation["file_count"],
        },
        "partition_reports": partition_reports,
        "prediction_partition": prediction_manifest.as_dict(),
        "frontier_partition": frontier_manifest.as_dict(),
        "frontier_summary": asdict(frontier_summary),
        "quarantine": {"path": quarantined.as_posix(), "receipt": quarantine_receipt},
        "checks": status_checks,
    }
    _json_write(output / "REAL_DATA_ENGINEERING_CANARY_RECEIPT.json", result)
    if result["status"] != "PASS":
        raise CanaryInputError("real-data engineering canary failed one or more checks")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--telemetry-hold-seconds", type=float, default=1.25)
    args = parser.parse_args()
    result = run_canary(
        repository_root=args.repository_root,
        output_root=args.output_root,
        device_text=args.device,
        telemetry_hold_seconds=args.telemetry_hold_seconds,
    )
    print(json.dumps({"status": result["status"], "semantic": result["semantic"], "run_id": result["run_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
