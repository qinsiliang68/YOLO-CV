from __future__ import annotations

import csv
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import cv2
import torch
from PIL import Image

from .asset_registry import (
    AssetRegistry,
    build_split_identity_bundle,
    load_asset_registry,
    load_registered_split_labels,
)
from .checkpointing import checkpoint_state_digest
from .columnar import read_columnar, validate_columnar_file, write_zstd_parquet
from .dataset_content_ledger import DATASET_CONTENT_ASSET_ID, load_registered_dataset_content_map
from .errors import ErrorCode, SctsrError
from .evaluation import compute_tie_safe_frontier, write_frontier_artifacts
from .prediction_artifact import (
    PredictionArtifactBinding,
    PredictionRow,
    validate_prediction_rows,
    write_prediction_artifact,
)
from .serialization import atomic_write_json, file_record, sha256_file, stable_digest


@dataclass(frozen=True, slots=True)
class PredictionBatch:
    images: torch.Tensor
    sample_ids: tuple[str, ...]
    labels: tuple[int, ...]

    def validate(self) -> None:
        if not isinstance(self.images, torch.Tensor) or self.images.ndim < 2:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction batch images are malformed")
        size = int(self.images.shape[0])
        if size < 1 or len(self.sample_ids) != size or len(self.labels) != size:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction batch identity cardinality mismatch")
        if any(not str(sample_id).strip() for sample_id in self.sample_ids) or len(set(self.sample_ids)) != size:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction batch has empty or duplicate sample IDs")
        if any(type(label) is not int or label not in {0, 1} for label in self.labels):
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction batch labels must be binary integers")


@dataclass(frozen=True, slots=True)
class RegisteredImageRecord:
    sample_id: str
    y_true: int
    image_path: Path
    image_bytes: int
    image_sha256: str


def load_registered_image_records(
    registry: AssetRegistry,
    *,
    repository_root: str | Path,
    dataset_root: str | Path,
    split_role: str,
    expected_content: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[RegisteredImageRecord, ...]:
    """Resolve physical images only through SHA-bound registered split CSVs."""

    root = Path(repository_root).resolve()
    data_root = Path(dataset_root).resolve()
    if not data_root.is_dir():
        raise SctsrError(
            ErrorCode.ASSET_VALIDATION_FAILED,
            "Prediction dataset root is missing",
            artifact_path=str(data_root),
        )
    expected_labels = load_registered_split_labels(registry, root, split_role)
    asset_ids = registry.split_asset_map.get(split_role)
    if not asset_ids:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Prediction split role is not registered", observed=split_role)
    asset_by_id = {record.asset_id: record for record in registry.assets}
    records: dict[str, RegisteredImageRecord] = {}
    for asset_id in asset_ids:
        asset = asset_by_id.get(asset_id)
        if asset is None or not asset.identity_column:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Prediction split component lacks an identity column")
        manifest = (root / asset.relative_path).resolve()
        try:
            manifest.relative_to(root)
        except ValueError as exc:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Prediction split manifest escapes repository") from exc
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            required = {asset.identity_column, "canonical_image_relpath"}
            if asset.label_column:
                required.add(asset.label_column)
            if not required.issubset(fields):
                raise SctsrError(
                    ErrorCode.ASSET_VALIDATION_FAILED,
                    "Prediction split manifest lacks identity/image/label fields",
                    observed=sorted(fields),
                    expected=sorted(required),
                )
            for row_index, row in enumerate(reader, start=2):
                sample_id = str(row[asset.identity_column]).replace("\\", "/")
                image_relative = str(row["canonical_image_relpath"]).replace("\\", "/")
                if not sample_id or image_relative != sample_id:
                    raise SctsrError(
                        ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                        "Prediction image path differs from its registered sample identity",
                        failing_field=f"{asset.asset_id}:row[{row_index}]",
                    )
                if sample_id not in expected_labels:
                    # The SHA-bound content-disjointness overlay removes this
                    # historical manifest row from the effective endpoint.
                    continue
                raw_label = asset.constant_label if asset.constant_label is not None else row.get(asset.label_column or "")
                try:
                    label = int(raw_label)
                except (TypeError, ValueError) as exc:
                    raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Prediction split label is not an integer") from exc
                if label != expected_labels[sample_id]:
                    raise SctsrError(
                        ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                        "Prediction manifest label differs from the effective registered split",
                        observed=label,
                        expected=expected_labels[sample_id],
                    )
                image_path = (data_root / Path(image_relative)).resolve()
                try:
                    image_path.relative_to(data_root)
                except ValueError as exc:
                    raise SctsrError(
                        ErrorCode.ASSET_VALIDATION_FAILED,
                        "Prediction image path escapes the registered dataset root",
                        artifact_path=str(image_path),
                    ) from exc
                if not image_path.is_file():
                    raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered prediction image is missing", artifact_path=str(image_path))
                image_bytes = image_path.stat().st_size
                image_sha256 = sha256_file(image_path)
                if expected_content is not None:
                    expected = expected_content.get(sample_id)
                    if (
                        expected is None
                        or image_bytes != int(expected["image_bytes"])
                        or image_sha256 != str(expected["image_sha256"]).upper()
                    ):
                        raise SctsrError(
                            ErrorCode.DATASET_CONTENT_MISMATCH,
                            "Endpoint image bytes differ from the frozen dataset content ledger",
                            observed={"bytes": image_bytes, "sha256": image_sha256},
                            expected=None if expected is None else {"bytes": int(expected["image_bytes"]), "sha256": str(expected["image_sha256"]).upper()},
                            artifact_path=str(image_path),
                        )
                if sample_id in records:
                    raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Prediction split contains a duplicate image identity", observed=sample_id)
                records[sample_id] = RegisteredImageRecord(sample_id, label, image_path, image_bytes, image_sha256)
    observed_labels = {sample_id: record.y_true for sample_id, record in records.items()}
    if observed_labels != expected_labels:
        raise SctsrError(
            ErrorCode.PREDICTION_IDENTITY_MISMATCH,
            "Resolved prediction images differ from registered split labels",
        )
    return tuple(records[sample_id] for sample_id in sorted(records))


def build_endpoint_input_binding(
    records: Sequence[RegisteredImageRecord],
    *,
    registry: AssetRegistry,
    repository_root: str | Path,
    dataset_root: str | Path,
    split_role: str,
    content_ledger_path: str | Path,
    content_ledger_sha256: str,
    content_ledger_identity_digest: str,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind the exact endpoint filesystem bytes before inference or reuse."""

    repository = Path(repository_root).resolve()
    data_root = Path(dataset_root).resolve()
    ledger = Path(content_ledger_path).resolve()
    if not ledger.is_file() or sha256_file(ledger) != str(content_ledger_sha256).upper():
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Endpoint content ledger bytes are not the registered bytes", artifact_path=str(ledger))
    if not records:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Endpoint input binding cannot be empty")
    rows = []
    for record in records:
        image = record.image_path.resolve()
        try:
            image.relative_to(data_root)
        except ValueError as exc:
            raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Endpoint image escapes its bound dataset root", artifact_path=str(image)) from exc
        rows.append(
            {
                "sample_id": record.sample_id,
                "y_true": record.y_true,
                "image_path": image.as_posix(),
                "image_bytes": record.image_bytes,
                "image_sha256": record.image_sha256,
            }
        )
    rows = sorted(rows, key=lambda row: row["sample_id"])
    payload: dict[str, Any] = {
        "schema_version": "stage1.sctsr.endpoint_input_binding.v1",
        "split_role": split_role,
        "dataset_root": data_root.as_posix(),
        "dataset_root_digest": stable_digest({"resolved_dataset_root": data_root.as_posix()}),
        "repository_root": repository.as_posix(),
        "asset_registry_digest": registry.digest,
        "content_ledger_path": ledger.as_posix(),
        "content_ledger_sha256": str(content_ledger_sha256).upper(),
        "content_ledger_identity_digest": str(content_ledger_identity_digest).upper(),
        "row_count": len(rows),
        "physical_bytes": sum(int(row["image_bytes"]) for row in rows),
        "sample_label_digest": stable_digest([(row["sample_id"], row["y_true"]) for row in rows]),
        "sample_content_digest": stable_digest(rows),
    }
    if evidence_path is None:
        payload["row_evidence"] = {"storage": "IN_MEMORY_UNIT_ONLY", "rows": rows}
    else:
        manifest = write_zstd_parquet(rows, evidence_path, schema_version="stage1.sctsr.endpoint_input_rows.v1")
        payload["row_evidence"] = {
            "storage": "PARQUET_ZSTD",
            "path": Path(manifest.path).resolve().as_posix(),
            "bytes": manifest.bytes,
            "sha256": manifest.sha256,
            "row_count": manifest.row_count,
            "schema_version": manifest.schema_version,
            "schema_digest": manifest.schema_digest,
        }
    return {**payload, "binding_digest": stable_digest(payload)}


def validate_endpoint_input_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    if binding.get("schema_version") != "stage1.sctsr.endpoint_input_binding.v1":
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Endpoint input binding schema is not registered")
    core = {key: value for key, value in binding.items() if key != "binding_digest"}
    if binding.get("binding_digest") != stable_digest(core):
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Endpoint input binding digest is invalid")
    ledger = Path(str(binding["content_ledger_path"])).resolve()
    if not ledger.is_file() or sha256_file(ledger) != binding.get("content_ledger_sha256"):
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Endpoint input content ledger changed")
    evidence = binding.get("row_evidence")
    if not isinstance(evidence, Mapping):
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Endpoint input binding has no row evidence")
    if evidence.get("storage") == "PARQUET_ZSTD":
        validate_columnar_file(
            evidence["path"],
            expected_rows=int(evidence["row_count"]),
            expected_schema_version=str(evidence["schema_version"]),
            expected_schema_digest=str(evidence["schema_digest"]),
            expected_sha256=str(evidence["sha256"]),
        )
        rows = read_columnar(evidence["path"])
    elif evidence.get("storage") == "IN_MEMORY_UNIT_ONLY":
        rows = list(evidence.get("rows", ()))
    else:
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Endpoint input row-evidence storage is invalid")
    data_root = Path(str(binding["dataset_root"])).resolve()
    observed_rows = []
    for row in rows:
        image = Path(str(row["image_path"])).resolve()
        try:
            image.relative_to(data_root)
        except ValueError as exc:
            raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Endpoint image escaped the bound dataset root after setup") from exc
        if not image.is_file():
            raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Endpoint image is missing after binding", artifact_path=str(image))
        observed = dict(row)
        observed["image_bytes"] = image.stat().st_size
        observed["image_sha256"] = sha256_file(image)
        if observed != row:
            raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Endpoint image bytes changed after binding", observed=observed, expected=row, artifact_path=str(image))
        observed_rows.append(observed)
    if len(observed_rows) != int(binding["row_count"]) or stable_digest(observed_rows) != binding.get("sample_content_digest"):
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Endpoint input aggregate digest changed")
    return {"status": "PASS", "binding_digest": binding["binding_digest"], "row_count": len(observed_rows)}


def iter_registered_image_batches(
    records: Sequence[RegisteredImageRecord],
    *,
    transform: Callable[[Image.Image], torch.Tensor],
    batch_size: int,
) -> Iterator[PredictionBatch]:
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Prediction batch size must be an integer in 1..128")
    if not records:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Registered prediction image set is empty")
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        for record in chunk:
            image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise SctsrError(
                    ErrorCode.ASSET_VALIDATION_FAILED,
                    "Registered prediction image cannot be decoded",
                    artifact_path=str(record.image_path),
                )
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = transform(Image.fromarray(rgb))
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
                raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction transform did not return one CHW tensor")
            tensors.append(tensor)
        try:
            images = torch.stack(tensors, dim=0)
        except RuntimeError as exc:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction transform returned inconsistent shapes") from exc
        yield PredictionBatch(
            images=images,
            sample_ids=tuple(record.sample_id for record in chunk),
            labels=tuple(record.y_true for record in chunk),
        )


def _endpoint_artifact_paths(run_root: Path, run_id: str) -> dict[str, Path]:
    prediction_root = run_root / "06_predictions" / f"run_id={run_id}" / "epoch=0200"
    evaluation_root = run_root / "07_evaluation" / f"run_id={run_id}" / "epoch=0200"
    return {
        "endpoint_input_rows": prediction_root / "endpoint_input_rows.parquet",
        "endpoint_input_binding": prediction_root / "endpoint_input_binding.json",
        "split_bundle": prediction_root / "split_identity_bundle.json",
        "prediction": prediction_root / "predictions.parquet",
        "prediction_summary": prediction_root / "prediction_summary.json",
        "frontier": evaluation_root / "frontier.parquet",
        "frontier_summary": evaluation_root / "frontier_summary.json",
    }


def build_formal_endpoint_receipt(
    run_root: str | Path,
    *,
    run_id: str,
    arm_id: str,
    training_seed: int,
    checkpoint_epoch: int,
    checkpoint_sha256: str,
    model_variant: str,
    endpoint_input_binding_digest: str,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    paths = _endpoint_artifact_paths(root, run_id)
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise SctsrError(
            ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE,
            "Formal endpoint receipt cannot cover an incomplete artifact set",
            observed=missing,
        )
    files = sorted((file_record(path, root=root) for path in paths.values()), key=lambda row: row["path"])
    payload: dict[str, Any] = {
        "schema_version": "stage1.sctsr.formal_endpoint_receipt.v2",
        "status": "FORMAL_ENDPOINT_COMPLETE_NOT_METHOD_SELECTION",
        "run_id": str(run_id),
        "arm_id": str(arm_id),
        "training_seed": int(training_seed),
        "split_role": "val_op",
        "checkpoint_epoch": int(checkpoint_epoch),
        "checkpoint_sha256": str(checkpoint_sha256).upper(),
        "model_variant": str(model_variant),
        "endpoint_input_binding_digest": str(endpoint_input_binding_digest).upper(),
        "selection_semantic": "ENDPOINT_ONLY_NOT_FOR_SELECTION",
        "files": files,
        "formal_training_started": True,
        "blind_holdout_opened": False,
        "test_accessed": False,
        "method_effectiveness_claimed": False,
    }
    payload["endpoint_digest"] = stable_digest(payload)
    return payload


def prepare_formal_endpoint_publication(
    run_root: str | Path,
    *,
    run_id: str,
    validate_existing: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    """Reuse a complete endpoint or preserve every partial byte before retry.

    This function is called only after the new execution attempt owns the
    latest logical-job fence.  It never deletes prior output and it refuses to
    touch a canonically completed run.
    """

    root = Path(run_root).resolve()
    if (root / "FORMAL_COMPLETION_RECEIPT.json").exists():
        raise SctsrError(
            ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE,
            "Canonical formal completion already exists; endpoint is immutable",
        )
    paths = _endpoint_artifact_paths(root, run_id)
    receipt_path = root / "08_receipts" / "FORMAL_ENDPOINT_RECEIPT.json"
    all_paths = {**paths, "receipt": receipt_path}
    existing = {name: path for name, path in all_paths.items() if path.exists()}
    if not existing:
        return {"action": "BUILD_FRESH", "quarantined_artifacts": []}
    if len(existing) == len(all_paths):
        try:
            validation = dict(validate_existing())
        except (SctsrError, OSError, ValueError):
            validation = None
        else:
            return {
                "action": "REUSE_VALID_ENDPOINT",
                "validation": validation,
                "quarantined_artifacts": [],
            }
    suffix = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.{uuid.uuid4().hex[:12]}"
    quarantine_root = root / "09_quarantine" / f"formal_endpoint.incomplete.{suffix}"
    records: list[dict[str, Any]] = []
    for name, source in sorted(existing.items()):
        relative = source.relative_to(root)
        target = quarantine_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        records.append(
            {
                "role": name,
                "source": relative.as_posix(),
                "target": target.relative_to(root).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    atomic_write_json(
        quarantine_root / "QUARANTINE_RECEIPT.json",
        {
            "schema_version": "stage1.sctsr.formal_endpoint_quarantine.v1",
            "status": "QUARANTINED_INCOMPLETE_OR_INVALID_ENDPOINT",
            "run_id": run_id,
            "artifacts": records,
            "artifact_count": len(records),
        },
    )
    return {"action": "BUILD_FRESH", "quarantined_artifacts": records}


def publish_formal_endpoint(
    *,
    model: torch.nn.Module,
    transform: Callable[[Image.Image], torch.Tensor],
    run_root: str | Path,
    repository_root: str | Path,
    dataset_root: str | Path,
    asset_registry_path: str | Path,
    checkpoint_path: str | Path,
    run_id: str,
    arm_id: str,
    model_variant: str,
    batch_size: int = 128,
    prediction_generation: int = 1,
) -> dict[str, Any]:
    """Publish one immutable E200 ``val_op`` endpoint after formal training."""

    root = Path(run_root).resolve()
    repository = Path(repository_root).resolve()
    data_root = Path(dataset_root).resolve()
    if not data_root.is_dir():
        raise SctsrError(
            ErrorCode.ASSET_VALIDATION_FAILED,
            "Formal endpoint dataset root is missing",
            artifact_path=str(data_root),
        )
    registry_source = Path(asset_registry_path).resolve()
    registry = load_asset_registry(registry_source)
    paths = _endpoint_artifact_paths(root, run_id)
    receipt_path = root / "08_receipts" / "FORMAL_ENDPOINT_RECEIPT.json"
    checkpoint = Path(checkpoint_path).resolve()
    from .checkpointing import load_checkpoint

    checkpoint_payload = load_checkpoint(checkpoint, expected_epoch=200)
    checkpoint_sha = sha256_file(checkpoint)
    from .run_validation import validate_formal_endpoint_evidence

    endpoint_manifest = {
        "run_id": run_id,
        "arm_id": arm_id,
        "training_seed": int(checkpoint_payload["training_seed"]),
        "source_tree_digest": str(checkpoint_payload["source_tree_digest"]),
        "asset_registry_digest": registry.digest,
    }
    content = load_registered_dataset_content_map(registry=registry, repository_root=repository)
    ledger_records = [record for record in registry.assets if record.asset_id == DATASET_CONTENT_ASSET_ID]
    if len(ledger_records) != 1:
        raise SctsrError(ErrorCode.DATASET_CONTENT_LEDGER_REQUIRED, "Formal endpoint requires one frozen dataset content ledger")
    ledger_record = ledger_records[0]
    ledger_path = (repository / ledger_record.relative_path).resolve()
    current_records = load_registered_image_records(
        registry,
        repository_root=repository,
        dataset_root=data_root,
        split_role="val_op",
        expected_content=content,
    )
    current_input_binding = build_endpoint_input_binding(
        current_records,
        registry=registry,
        repository_root=repository,
        dataset_root=data_root,
        split_role="val_op",
        content_ledger_path=ledger_path,
        content_ledger_sha256=ledger_record.sha256,
        content_ledger_identity_digest=str(ledger_record.identity_digest),
    )
    preparation = prepare_formal_endpoint_publication(
        root,
        run_id=run_id,
        validate_existing=lambda: validate_formal_endpoint_evidence(
            root,
            manifest=endpoint_manifest,
            checkpoint_path=checkpoint,
            repository_root=repository,
            expected_endpoint_input_binding=current_input_binding,
        ),
    )
    if preparation["action"] == "REUSE_VALID_ENDPOINT":
        return {
            **preparation["validation"],
            "dataset_root": data_root.as_posix(),
            "endpoint_publication_action": preparation["action"],
            "quarantined_artifacts": [],
        }
    build_split_identity_bundle(
        registry,
        repository_root=repository,
        registry_path=registry_source,
        split_role="val_op",
        output_path=paths["split_bundle"],
        formal_input_snapshot_path=root / "00_contract" / "FORMAL_INPUT_SNAPSHOT.json",
    )
    split_sha = sha256_file(paths["split_bundle"])
    endpoint_input_binding = build_endpoint_input_binding(
        current_records,
        registry=registry,
        repository_root=repository,
        dataset_root=data_root,
        split_role="val_op",
        content_ledger_path=ledger_path,
        content_ledger_sha256=ledger_record.sha256,
        content_ledger_identity_digest=str(ledger_record.identity_digest),
        evidence_path=paths["endpoint_input_rows"],
    )
    atomic_write_json(paths["endpoint_input_binding"], endpoint_input_binding)
    binding = PredictionArtifactBinding(
        checkpoint_path=checkpoint.as_posix(),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_epoch=200,
        model_variant=model_variant,
        source_tree_digest=str(checkpoint_payload["source_tree_digest"]),
        training_seed=int(checkpoint_payload["training_seed"]),
        split_role="val_op",
        split_manifest_path=paths["split_bundle"].as_posix(),
        split_manifest_sha256=split_sha,
        evaluation_mode="formal",
        selection_semantic="ENDPOINT_ONLY_NOT_FOR_SELECTION",
    )
    rows = infer_prediction_rows(
        model=model,
        batches=iter_registered_image_batches(current_records, transform=transform, batch_size=batch_size),
        binding=binding,
        run_id=run_id,
        arm_id=arm_id,
        prediction_generation=prediction_generation,
    )
    prediction_manifest, prediction_summary = write_prediction_artifact(
        rows,
        paths["prediction"],
        formal_endpoint=True,
        binding=binding,
        repository_root=repository,
    )
    prediction_summary["endpoint_input_binding_digest"] = endpoint_input_binding["binding_digest"]
    prediction_summary["endpoint_input_sample_content_digest"] = endpoint_input_binding["sample_content_digest"]
    atomic_write_json(paths["prediction_summary"], prediction_summary)
    points, frontier_summary = compute_tie_safe_frontier(
        rows,
        max_fn=95,
        target_tn=68_253,
        checkpoint_sha256=checkpoint_sha,
        prediction_artifact_sha256=prediction_manifest.sha256,
    )
    write_frontier_artifacts(
        points,
        frontier_summary,
        frontier_path=paths["frontier"],
        summary_path=paths["frontier_summary"],
        evaluation_mode="formal",
        split_role="val_op",
        checkpoint_epoch=200,
    )
    receipt = build_formal_endpoint_receipt(
        root,
        run_id=run_id,
        arm_id=arm_id,
        training_seed=binding.training_seed,
        checkpoint_epoch=200,
        checkpoint_sha256=checkpoint_sha,
        model_variant=model_variant,
        endpoint_input_binding_digest=endpoint_input_binding["binding_digest"],
    )
    atomic_write_json(receipt_path, receipt)
    validation = validate_formal_endpoint_evidence(
        root,
        manifest=endpoint_manifest,
        checkpoint_path=checkpoint,
        repository_root=repository,
        expected_endpoint_input_binding=endpoint_input_binding,
    )
    return {
        **validation,
        "dataset_root": data_root.as_posix(),
        "endpoint_publication_action": preparation["action"],
        "quarantined_artifacts": preparation["quarantined_artifacts"],
    }


def _ema_model_state(ema_state: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("ema_model_state", "shadow"):
        value = ema_state.get(key)
        if isinstance(value, Mapping) and value:
            return value
    tensor_values = [value for value in ema_state.values() if isinstance(value, torch.Tensor)]
    if tensor_values and len(tensor_values) == len(ema_state):
        return ema_state
    raise SctsrError(
        ErrorCode.PARENT_CHECKPOINT_INCOMPLETE,
        "EMA checkpoint state has no registered model-state representation",
    )


def prediction_checkpoint_state(payload: Mapping[str, object], model_variant: str) -> Mapping[str, object]:
    if model_variant == "MODEL":
        state = payload.get("model_state")
        if not isinstance(state, Mapping) or not state:
            raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Checkpoint MODEL state is missing")
        return state
    if model_variant == "EMA":
        state = payload.get("ema_state")
        if not isinstance(state, Mapping) or not state:
            raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Checkpoint EMA state is missing")
        return _ema_model_state(state)
    raise SctsrError(
        ErrorCode.PREDICTION_IDENTITY_MISMATCH,
        "Formal inference accepts only checkpoint MODEL or EMA state",
        observed=model_variant,
    )


def _binary_class_indices(model: torch.nn.Module) -> tuple[int, int]:
    raw = getattr(model, "names", None)
    if isinstance(raw, (list, tuple)):
        names = {index: str(value) for index, value in enumerate(raw)}
    elif isinstance(raw, Mapping):
        try:
            names = {int(index): str(value) for index, value in raw.items()}
        except (TypeError, ValueError) as exc:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Model class-name keys are not integer indices") from exc
    else:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction model has no registered class names")
    expected = {0: "no_target", 1: "target_defect"}
    if names != expected:
        raise SctsrError(
            ErrorCode.PREDICTION_IDENTITY_MISMATCH,
            "Binary class mapping differs from the frozen training dataset order",
            observed=names,
            expected=expected,
        )
    return 0, 1


def _probabilities_and_logits(output: object, *, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise SctsrError(
            ErrorCode.PREDICTION_IDENTITY_MISMATCH,
            "Prediction model must expose both probabilities and pre-softmax logits",
        )
    probabilities, logits = output
    if not isinstance(probabilities, torch.Tensor) or not isinstance(logits, torch.Tensor):
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction outputs are not tensors")
    expected_shape = (batch_size, 2)
    if tuple(probabilities.shape) != expected_shape or tuple(logits.shape) != expected_shape:
        raise SctsrError(
            ErrorCode.PREDICTION_IDENTITY_MISMATCH,
            "Prediction output shape is not two-class batch logits",
            observed={"probabilities": tuple(probabilities.shape), "logits": tuple(logits.shape)},
            expected=expected_shape,
        )
    if not torch.isfinite(probabilities).all() or not torch.isfinite(logits).all():
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction output contains non-finite values")
    expected_probabilities = logits.float().softmax(dim=1)
    if not torch.allclose(probabilities.float(), expected_probabilities, rtol=1e-5, atol=1e-7):
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction probabilities differ from raw logits")
    return expected_probabilities, logits.float()


def infer_prediction_rows(
    *,
    model: torch.nn.Module,
    batches: Iterable[PredictionBatch],
    binding: PredictionArtifactBinding,
    run_id: str,
    arm_id: str,
    prediction_generation: int,
) -> tuple[PredictionRow, ...]:
    """Run auditable inference from a model proven equal to a checkpoint state."""

    if not str(run_id).strip() or not str(arm_id).strip() or type(prediction_generation) is not int or prediction_generation < 1:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction run/arm/generation identity is invalid")
    payload = binding.validate()
    expected_state = prediction_checkpoint_state(payload, binding.model_variant)
    observed_state = model.state_dict()
    if checkpoint_state_digest(observed_state) != checkpoint_state_digest(expected_state):
        raise SctsrError(
            ErrorCode.PREDICTION_IDENTITY_MISMATCH,
            "Inference model bytes differ from the selected checkpoint MODEL/EMA state",
        )
    normal_index, defect_index = _binary_class_indices(model)
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction model has no parameters") from exc

    was_training = bool(model.training)
    rows: list[PredictionRow] = []
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                batch.validate()
                images = batch.images.to(device=device, dtype=torch.float32, non_blocking=False)
                probabilities, logits = _probabilities_and_logits(model(images), batch_size=len(batch.sample_ids))
                probabilities = probabilities.detach().cpu()
                logits = logits.detach().cpu()
                for index, (sample_id, label) in enumerate(zip(batch.sample_ids, batch.labels, strict=True)):
                    rows.append(
                        PredictionRow(
                            run_id=str(run_id),
                            arm_id=str(arm_id),
                            training_seed=binding.training_seed,
                            split_role=binding.split_role,
                            split_manifest_path=binding.split_manifest_path,
                            split_manifest_sha256=binding.split_manifest_sha256,
                            sample_id=str(sample_id),
                            y_true=int(label),
                            logit_normal=float(logits[index, normal_index]),
                            logit_defect=float(logits[index, defect_index]),
                            p_defect_raw=float(probabilities[index, defect_index]),
                            checkpoint_epoch=binding.checkpoint_epoch,
                            checkpoint_sha256=binding.checkpoint_sha256,
                            model_variant=binding.model_variant,
                            source_tree_digest=binding.source_tree_digest,
                            prediction_generation=prediction_generation,
                        )
                    )
    finally:
        model.train(was_training)
    result = tuple(rows)
    validate_prediction_rows(result, expected_split_role=binding.split_role)
    return result
