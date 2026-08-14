from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import ErrorCode, SctsrError
from .serialization import _fsync_directory, atomic_write_bytes, atomic_write_json, load_json, sha256_file, stable_digest


RUNBOOK_SCHEMA = "stage1.sctsr.runbook_manifest.v1"
RUN_INTENT_SCHEMA = "stage1.sctsr.run_intent_acknowledgement.v1"
RUN_INTENT_BINDING_SCHEMA = "stage1.sctsr.run_intent_binding.v1"
RUN_INTENT_SNAPSHOT_SCHEMA = "stage1.sctsr.run_intent_snapshot.v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_SHA_FIELDS = (
    "formal_identity_sha256",
    "trainer_overrides_sha256",
    "identity_manifest_sha256",
    "source_tree_digest",
    "contract_digest",
    "asset_registry_digest",
    "runtime_config_digest",
    "dataset_content_identity_digest",
    "parent_checkpoint_sha256",
    "schedule_digest",
    "identity_pool_binding_digest",
    "release_manifest_sha256",
    "execution_token_sha256",
    "claim_registry_root_digest",
    "resume_checkpoint_sha256",
    "resume_receipt_digest",
    "runbook_manifest_sha256",
)
_ZERO_ALLOWED = {
    "schedule_digest",
    "identity_pool_binding_digest",
    "resume_checkpoint_sha256",
    "resume_receipt_digest",
}
_BRANCH_ARMS = {
    "NR",
    "R1_U",
    "R2_U",
    "T_U",
    "R2_F",
    "T_F",
    "T_TO_R2_AT_160",
    "T_TO_NR_AT_160",
}


REQUIRED_ACKNOWLEDGEMENT_STATEMENTS = (
    "understood_research_question",
    "understood_no_method_effectiveness_claim",
    "understood_t_is_stress_set_not_validated_selector",
    "understood_r2_is_random_control_and_current_spec_is_blocked",
    "understood_nr_is_no_replay_base_process",
    "understood_only_preregistered_treatment_may_differ",
    "understood_base_order_augmentation_steps_scheduler_ema_must_match",
    "understood_e200_ema_val_op_is_fixed_endpoint",
    "understood_val_op_cannot_select_method_stop_checkpoint_or_threshold",
    "understood_test_and_blind_access_are_forbidden",
    "understood_exit_zero_is_not_canonical_completion",
    "verified_exact_source_asset_dataset_parent_schedule_and_pool_identities",
    "verified_gpu_runtime_and_disk_capacity",
    "understood_oom_kill_disk_and_partial_write_stop_rules",
    "understood_resume_requires_contiguous_receipts_and_new_token",
    "will_not_substitute_latest_similar_or_automatically_discovered_files",
)


@dataclass(frozen=True, slots=True)
class RunIntentContext:
    action: str
    run_role: str
    logical_run_id: str
    arm_id: str
    training_seed: int
    dataset_root: str
    output_root: str
    formal_identity_sha256: str
    trainer_overrides_sha256: str
    identity_manifest_sha256: str
    source_tree_digest: str
    contract_digest: str
    asset_registry_digest: str
    runtime_config_digest: str
    dataset_content_identity_digest: str
    parent_checkpoint_sha256: str
    schedule_digest: str
    identity_pool_binding_digest: str
    release_manifest_sha256: str
    execution_token_sha256: str
    claim_registry_root_digest: str
    resume_checkpoint_sha256: str
    resume_receipt_digest: str
    runbook_manifest_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunIntentContext":
        expected = {field.name for field in fields(cls)}
        if set(value) != expected:
            raise SctsrError(
                ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED,
                "Run-intent context fields do not exactly match the registered schema",
                observed={"missing": sorted(expected - set(value)), "extra": sorted(set(value) - expected)},
            )
        if type(value.get("training_seed")) is not int:
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent training seed must be a JSON integer")
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as exc:
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent context values are invalid") from exc

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.action not in {"START", "RESUME"}:
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run intent action must be START or RESUME")
        if self.run_role not in {"COMMON_PARENT", "BRANCH"}:
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run intent role is invalid")
        if self.run_role == "COMMON_PARENT" and self.arm_id != "COMMON_PARENT_NR":
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Common parent acknowledgement must use COMMON_PARENT_NR")
        if self.run_role == "BRANCH" and self.arm_id not in _BRANCH_ARMS:
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Branch acknowledgement arm is invalid")
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run intent training seed is invalid")
        if not _IDENTIFIER.fullmatch(self.logical_run_id):
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run intent logical run ID is noncanonical")
        for field in ("dataset_root", "output_root"):
            token = str(getattr(self, field))
            if not Path(token).is_absolute() or _has_placeholder(token):
                raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run intent path is not an explicit absolute path", failing_field=field, observed=token)
        for field in _SHA_FIELDS:
            token = str(getattr(self, field)).upper()
            if len(token) != 64 or any(character not in "0123456789ABCDEF" for character in token):
                raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run intent contains an invalid SHA-256", failing_field=field)
            if token == "0" * 64 and field not in _ZERO_ALLOWED:
                raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run intent uses a zero placeholder for a required identity", failing_field=field)
        if self.run_role == "COMMON_PARENT" and (self.schedule_digest != "0" * 64 or self.identity_pool_binding_digest != "0" * 64):
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Common parent must have zero schedule and pool bindings")
        if self.run_role == "BRANCH" and self.schedule_digest == "0" * 64:
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Branch acknowledgement must bind a schedule")
        if self.action == "START" and (self.resume_checkpoint_sha256 != "0" * 64 or self.resume_receipt_digest != "0" * 64):
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "START acknowledgement may not bind resume state")
        if self.action == "RESUME" and (self.resume_checkpoint_sha256 == "0" * 64 or self.resume_receipt_digest == "0" * 64):
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "RESUME acknowledgement must bind checkpoint and receipt state")


def _has_placeholder(value: str) -> bool:
    upper = value.upper()
    return any(token in upper for token in ("<", ">", "REPLACE", "PLACEHOLDER", "TBD", "TODO"))


def build_runbook_manifest(
    *,
    repository_root: str | Path,
    document_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_path)
    if output.exists():
        raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Runbook manifest is immutable; choose a new output path")
    if not document_paths:
        raise SctsrError(ErrorCode.RUNBOOK_IDENTITY_MISMATCH, "Runbook manifest requires at least one document")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in document_paths:
        path = Path(value).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise SctsrError(ErrorCode.RUNBOOK_IDENTITY_MISMATCH, "Runbook document escapes repository root", artifact_path=str(path)) from exc
        if relative in seen or not path.is_file() or path.suffix.lower() != ".md":
            raise SctsrError(ErrorCode.RUNBOOK_IDENTITY_MISMATCH, "Runbook document is missing, duplicated, or not Markdown", artifact_path=str(path))
        seen.add(relative)
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    records.sort(key=lambda row: row["path"])
    core = {
        "schema_version": RUNBOOK_SCHEMA,
        "documents": records,
        "document_count": len(records),
        "total_bytes": sum(int(row["bytes"]) for row in records),
    }
    payload = {**core, "runbook_digest": stable_digest(core)}
    atomic_write_json(output, payload)
    return payload


def validate_runbook_manifest(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = load_json(source)
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "documents", "document_count", "total_bytes", "runbook_digest"}:
        raise SctsrError(ErrorCode.RUNBOOK_IDENTITY_MISMATCH, "Runbook manifest schema fields are invalid")
    documents = raw.get("documents")
    if raw.get("schema_version") != RUNBOOK_SCHEMA or not isinstance(documents, list) or not documents:
        raise SctsrError(ErrorCode.RUNBOOK_IDENTITY_MISMATCH, "Runbook manifest schema or document list is invalid")
    root = Path(repository_root).resolve()
    recomputed: list[dict[str, Any]] = []
    paths: set[str] = set()
    for row in documents:
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise SctsrError(ErrorCode.RUNBOOK_IDENTITY_MISMATCH, "Runbook document record is invalid")
        relative = str(row["path"])
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise SctsrError(ErrorCode.RUNBOOK_IDENTITY_MISMATCH, "Runbook document path escapes repository root", observed=relative) from exc
        if relative in paths or not target.is_file():
            raise SctsrError(ErrorCode.RUNBOOK_IDENTITY_MISMATCH, "Runbook document is missing or duplicated", observed=relative)
        paths.add(relative)
        record = {"path": relative, "bytes": target.stat().st_size, "sha256": sha256_file(target)}
        if dict(row) != record:
            raise SctsrError(ErrorCode.RUNBOOK_IDENTITY_MISMATCH, "Runbook document bytes changed", observed=relative)
        recomputed.append(record)
    recomputed.sort(key=lambda row: row["path"])
    core = {
        "schema_version": RUNBOOK_SCHEMA,
        "documents": recomputed,
        "document_count": len(recomputed),
        "total_bytes": sum(int(row["bytes"]) for row in recomputed),
    }
    if int(raw["document_count"]) != len(recomputed) or int(raw["total_bytes"]) != core["total_bytes"] or raw["runbook_digest"] != stable_digest(core):
        raise SctsrError(ErrorCode.RUNBOOK_IDENTITY_MISMATCH, "Runbook manifest digest/count/bytes are invalid")
    return {
        "status": "PASS",
        "manifest_path": source.as_posix(),
        "manifest_bytes": source.stat().st_size,
        "manifest_sha256": sha256_file(source),
        "runbook_digest": raw["runbook_digest"],
        "document_count": len(recomputed),
        "total_bytes": core["total_bytes"],
        "documents": recomputed,
    }


def _utc_token(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Acknowledgement timestamp must be timezone-aware UTC")
    utc = value.astimezone(timezone.utc).replace(microsecond=0)
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_run_intent_acknowledgement(
    *,
    context: RunIntentContext,
    acknowledgement_id: str,
    operator_agent_id: str,
    machine_id: str,
    created_at_utc: datetime,
) -> dict[str, Any]:
    context.validate()
    for field, value in (
        ("acknowledgement_id", acknowledgement_id),
        ("operator_agent_id", operator_agent_id),
        ("machine_id", machine_id),
    ):
        if not _IDENTIFIER.fullmatch(value) or _has_placeholder(value):
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Acknowledgement identifier is invalid", failing_field=field)
    core = {
        "schema_version": RUN_INTENT_SCHEMA,
        "acknowledgement_id": acknowledgement_id,
        "created_at_utc": _utc_token(created_at_utc),
        "operator_agent_id": operator_agent_id,
        "machine_id": machine_id,
        **context.as_dict(),
        "statements": {statement: True for statement in REQUIRED_ACKNOWLEDGEMENT_STATEMENTS},
    }
    return {**core, "acknowledgement_digest": stable_digest(core)}


def validate_run_intent_acknowledgement(
    path: str | Path,
    *,
    expected_context: RunIntentContext,
    runbook_manifest_path: str | Path,
    repository_root: str | Path,
    now_utc: datetime | None = None,
    maximum_age: timedelta = timedelta(days=7),
    enforce_freshness: bool = True,
) -> dict[str, Any]:
    expected_context.validate()
    source = Path(path).resolve()
    raw = load_json(source)
    context_fields = {field.name for field in fields(RunIntentContext)}
    expected_fields = {
        "schema_version",
        "acknowledgement_id",
        "created_at_utc",
        "operator_agent_id",
        "machine_id",
        "statements",
        "acknowledgement_digest",
    } | context_fields
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        raise SctsrError(
            ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED,
            "Run-intent acknowledgement fields do not exactly match the registered schema",
            observed={"missing": sorted(expected_fields - set(raw) if isinstance(raw, Mapping) else expected_fields), "extra": sorted(set(raw) - expected_fields if isinstance(raw, Mapping) else [])},
        )
    if raw.get("schema_version") != RUN_INTENT_SCHEMA:
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent acknowledgement schema version is invalid")
    for field in ("acknowledgement_id", "operator_agent_id", "machine_id"):
        value = str(raw[field])
        if not _IDENTIFIER.fullmatch(value) or _has_placeholder(value):
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent acknowledgement identifier is invalid", failing_field=field)
    statements = raw.get("statements")
    if not isinstance(statements, Mapping) or set(statements) != set(REQUIRED_ACKNOWLEDGEMENT_STATEMENTS) or any(value is not True for value in statements.values()):
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Every required run-intent statement must be explicitly true")
    observed_context = {field: raw[field] for field in context_fields}
    if observed_context != expected_context.as_dict():
        mismatch = {
            field: {"observed": observed_context.get(field), "expected": expected_context.as_dict()[field]}
            for field in sorted(context_fields)
            if observed_context.get(field) != expected_context.as_dict()[field]
        }
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent acknowledgement targets a different job", observed=mismatch)
    core = {key: value for key, value in raw.items() if key != "acknowledgement_digest"}
    if raw["acknowledgement_digest"] != stable_digest(core):
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent acknowledgement digest is invalid")
    try:
        created = datetime.strptime(str(raw["created_at_utc"]), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent timestamp is not canonical UTC") from exc
    now = datetime.now(timezone.utc) if now_utc is None else now_utc.astimezone(timezone.utc)
    if enforce_freshness and (created > now + timedelta(minutes=5) or now - created > maximum_age):
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent acknowledgement is future-dated or stale", observed=raw["created_at_utc"])
    runbook = validate_runbook_manifest(runbook_manifest_path, repository_root=repository_root)
    if runbook["manifest_sha256"] != expected_context.runbook_manifest_sha256:
        raise SctsrError(ErrorCode.RUNBOOK_IDENTITY_MISMATCH, "Acknowledgement runbook manifest SHA differs from verified bytes")
    return {
        "schema_version": "stage1.sctsr.run_intent_validation.v1",
        "status": "PASS",
        "acknowledgement_path": source.as_posix(),
        "acknowledgement_bytes": source.stat().st_size,
        "acknowledgement_sha256": sha256_file(source),
        "acknowledgement_digest": raw["acknowledgement_digest"],
        "acknowledgement_id": raw["acknowledgement_id"],
        "operator_agent_id": raw["operator_agent_id"],
        "machine_id": raw["machine_id"],
        "all_required_statements_true": True,
        "runbook_manifest_sha256": runbook["manifest_sha256"],
        "runbook_digest": runbook["runbook_digest"],
        "context_digest": stable_digest(expected_context),
        "validated_at_utc": _utc_token(now),
    }


def build_run_intent_binding(
    *,
    acknowledgement_path: str | Path,
    runbook_manifest_path: str | Path,
    repository_root: str | Path,
    context: RunIntentContext,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the pre-claim acknowledgement result to immutable external bytes."""

    root = Path(repository_root).resolve()
    acknowledgement = Path(acknowledgement_path).resolve()
    runbook = Path(runbook_manifest_path).resolve()
    context.validate()
    for role, path in (("acknowledgement", acknowledgement), ("runbook_manifest", runbook)):
        if not path.is_file() or path.is_symlink():
            raise SctsrError(
                ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED,
                "Run-intent external input is missing or indirect",
                failing_field=role,
                artifact_path=path.as_posix(),
            )
    expected = {
        "status": "PASS",
        "acknowledgement_sha256": sha256_file(acknowledgement),
        "runbook_manifest_sha256": sha256_file(runbook),
        "context_digest": stable_digest(context),
    }
    mismatch = {
        field: {"observed": validation.get(field), "expected": value}
        for field, value in expected.items()
        if validation.get(field) != value
    }
    if mismatch:
        raise SctsrError(
            ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED,
            "Run-intent validation result does not bind the supplied context and files",
            observed=mismatch,
        )
    core = {
        "schema_version": RUN_INTENT_BINDING_SCHEMA,
        "repository_root": root.as_posix(),
        "context": context.as_dict(),
        "context_digest": stable_digest(context),
        "acknowledgement": {
            "path": acknowledgement.as_posix(),
            "bytes": acknowledgement.stat().st_size,
            "sha256": sha256_file(acknowledgement),
        },
        "runbook_manifest": {
            "path": runbook.as_posix(),
            "bytes": runbook.stat().st_size,
            "sha256": sha256_file(runbook),
        },
        "validation": dict(validation),
        "validation_digest": stable_digest(dict(validation)),
    }
    return {**core, "binding_digest": stable_digest(core)}


def validate_run_intent_binding(
    binding: Mapping[str, Any],
    *,
    enforce_freshness: bool,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "repository_root",
        "context",
        "context_digest",
        "acknowledgement",
        "runbook_manifest",
        "validation",
        "validation_digest",
        "binding_digest",
    }
    if set(binding) != expected_fields:
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent binding schema fields are invalid")
    core = {key: value for key, value in binding.items() if key != "binding_digest"}
    if binding.get("schema_version") != RUN_INTENT_BINDING_SCHEMA or binding.get("binding_digest") != stable_digest(core):
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent binding digest is invalid")
    root = Path(str(binding["repository_root"])).resolve()
    if not root.is_dir():
        raise SctsrError(ErrorCode.RUNBOOK_IDENTITY_MISMATCH, "Run-intent repository root is missing")
    context_raw = binding.get("context")
    if not isinstance(context_raw, Mapping):
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent binding context is invalid")
    context = RunIntentContext.from_mapping(context_raw)
    context.validate()
    if binding.get("context_digest") != stable_digest(context):
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent context digest is invalid")
    rows: dict[str, Path] = {}
    for role in ("acknowledgement", "runbook_manifest"):
        row = binding.get(role)
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent external byte record is invalid", failing_field=role)
        path = Path(str(row["path"])).resolve()
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent external bytes changed", failing_field=role)
        rows[role] = path
    validation = binding.get("validation")
    if not isinstance(validation, Mapping) or binding.get("validation_digest") != stable_digest(dict(validation)):
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent validation receipt digest is invalid")
    checked = validate_run_intent_acknowledgement(
        rows["acknowledgement"],
        expected_context=context,
        runbook_manifest_path=rows["runbook_manifest"],
        repository_root=root,
        enforce_freshness=enforce_freshness,
    )
    for field in ("acknowledgement_sha256", "runbook_manifest_sha256", "context_digest"):
        if checked.get(field) != validation.get(field):
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent revalidation differs from its original receipt", failing_field=field)
    return {
        "status": "PASS",
        "binding_digest": binding["binding_digest"],
        "context": context.as_dict(),
        "acknowledgement_id": checked["acknowledgement_id"],
        "acknowledgement_sha256": checked["acknowledgement_sha256"],
        "runbook_manifest_sha256": checked["runbook_manifest_sha256"],
        "runbook_digest": checked["runbook_digest"],
    }


def publish_run_intent_snapshot(run_root: str | Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically append one START/RESUME acknowledgement to the run evidence."""

    checked = validate_run_intent_binding(binding, enforce_freshness=True)
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Run root must exist before publishing run intent")
    attempts_root = root / "00_contract" / "run_intent_attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in attempts_root.iterdir() if path.is_dir())
    if any(not path.name.startswith("attempt_") for path in existing):
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent attempt directory contains an incomplete or foreign entry")
    previous_digest = "0" * 64
    if existing:
        prior = load_json(existing[-1] / "RUN_INTENT_SNAPSHOT.json")
        previous_digest = str(prior.get("snapshot_digest", ""))
    attempt_index = len(existing) + 1
    acknowledgement_id = str(checked["acknowledgement_id"])
    final_name = f"attempt_{attempt_index:04d}_{acknowledgement_id}"
    final_root = attempts_root / final_name
    if final_root.exists():
        raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Run-intent attempt already exists")
    staging = attempts_root / f".{final_name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    acknowledgement_source = Path(str(binding["acknowledgement"]["path"]))
    runbook_source = Path(str(binding["runbook_manifest"]["path"]))
    acknowledgement_copy = staging / "RUN_INTENT_ACKNOWLEDGEMENT.json"
    runbook_copy = staging / "RUNBOOK_MANIFEST.json"
    binding_copy = staging / "RUN_INTENT_BINDING.json"
    atomic_write_bytes(acknowledgement_copy, acknowledgement_source.read_bytes())
    atomic_write_bytes(runbook_copy, runbook_source.read_bytes())
    atomic_write_json(binding_copy, dict(binding))
    copied = {
        "acknowledgement": {"path": acknowledgement_copy.name, "bytes": acknowledgement_copy.stat().st_size, "sha256": sha256_file(acknowledgement_copy)},
        "runbook_manifest": {"path": runbook_copy.name, "bytes": runbook_copy.stat().st_size, "sha256": sha256_file(runbook_copy)},
        "binding": {"path": binding_copy.name, "bytes": binding_copy.stat().st_size, "sha256": sha256_file(binding_copy)},
    }
    core = {
        "schema_version": RUN_INTENT_SNAPSHOT_SCHEMA,
        "attempt_index": attempt_index,
        "attempt_directory": final_name,
        "acknowledgement_id": acknowledgement_id,
        "action": checked["context"]["action"],
        "previous_snapshot_digest": previous_digest,
        "binding_digest": checked["binding_digest"],
        "copied_files": copied,
    }
    snapshot = {**core, "snapshot_digest": stable_digest(core)}
    atomic_write_json(staging / "RUN_INTENT_SNAPSHOT.json", snapshot)
    staging.rename(final_root)
    _fsync_directory(attempts_root)
    return {**snapshot, "snapshot_path": (final_root / "RUN_INTENT_SNAPSHOT.json").as_posix()}


def validate_run_intent_snapshot_chain(
    run_root: str | Path,
    *,
    repository_root: str | Path,
    expected_latest_snapshot_digest: str | None = None,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    attempts_root = root / "00_contract" / "run_intent_attempts"
    if not attempts_root.is_dir():
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Formal run lacks run-intent attempt evidence")
    attempt_roots = sorted(path for path in attempts_root.iterdir() if path.is_dir())
    if not attempt_roots or any(not path.name.startswith("attempt_") for path in attempt_roots):
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent snapshot chain is empty or contains an incomplete entry")
    previous = "0" * 64
    records: list[dict[str, Any]] = []
    expected_repository = Path(repository_root).resolve()
    for index, attempt_root in enumerate(attempt_roots, start=1):
        path = attempt_root / "RUN_INTENT_SNAPSHOT.json"
        raw = load_json(path)
        expected_fields = {
            "schema_version", "attempt_index", "attempt_directory", "acknowledgement_id", "action",
            "previous_snapshot_digest", "binding_digest", "copied_files", "snapshot_digest",
        }
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent snapshot schema is invalid")
        core = {key: value for key, value in raw.items() if key != "snapshot_digest"}
        if (
            raw.get("schema_version") != RUN_INTENT_SNAPSHOT_SCHEMA
            or raw.get("attempt_index") != index
            or raw.get("attempt_directory") != attempt_root.name
            or raw.get("previous_snapshot_digest") != previous
            or raw.get("snapshot_digest") != stable_digest(core)
        ):
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent snapshot chain or digest is invalid")
        copied = raw.get("copied_files")
        if not isinstance(copied, Mapping) or set(copied) != {"acknowledgement", "runbook_manifest", "binding"}:
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent copied-file set is incomplete")
        for role, row in copied.items():
            if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
                raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent copied-file record is invalid", failing_field=role)
            file_path = (attempt_root / str(row["path"])).resolve()
            try:
                file_path.relative_to(attempt_root)
            except ValueError as exc:
                raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent copied file escapes its attempt root") from exc
            if not file_path.is_file() or file_path.stat().st_size != row["bytes"] or sha256_file(file_path) != row["sha256"]:
                raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent copied bytes changed", failing_field=role)
        binding = load_json(attempt_root / str(copied["binding"]["path"]))
        if Path(str(binding.get("repository_root", ""))).resolve() != expected_repository:
            raise SctsrError(ErrorCode.RUNBOOK_IDENTITY_MISMATCH, "Run-intent snapshot references another repository root")
        checked = validate_run_intent_binding(binding, enforce_freshness=False)
        if checked["binding_digest"] != raw["binding_digest"] or checked["acknowledgement_id"] != raw["acknowledgement_id"]:
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent snapshot differs from its binding")
        previous = str(raw["snapshot_digest"])
        records.append({"attempt_index": index, "acknowledgement_id": raw["acknowledgement_id"], "action": raw["action"], "snapshot_digest": previous})
    if expected_latest_snapshot_digest is not None and previous != expected_latest_snapshot_digest:
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Run-intent latest snapshot digest differs from terminal evidence")
    return {
        "status": "PASS",
        "attempt_count": len(records),
        "latest_snapshot_digest": previous,
        "attempts": records,
    }


def derive_formal_run_intent_context(
    *,
    runbook_manifest_path: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
    action: str,
    run_role: str,
    logical_run_id: str,
    arm_id: str,
    training_seed: int,
    formal_identity_path: str | Path,
    trainer_overrides_path: str | Path,
    identity_manifest_path: str | Path,
    source_tree_digest: str,
    contract_digest: str,
    asset_registry_digest: str,
    runtime_config_digest: str,
    asset_registry_path: str | Path,
    parent_checkpoint_sha256: str,
    schedule_digest: str,
    identity_pool_binding_digest: str,
    release_manifest_path: str | Path,
    execution_token_path: str | Path,
    claim_registry_root: str | Path,
    resume_checkpoint_sha256: str,
    resume_receipt_digest: str,
) -> RunIntentContext:
    """Derive the exact job context without claiming a token or constructing a trainer.

    The function deliberately reads only already-frozen control bytes.  It
    does not construct an Ultralytics trainer and does not create a claim.
    """

    from .asset_registry import load_asset_registry
    from .formal_execution import output_root_digest

    root = Path(repository_root).resolve()
    overrides_path = Path(trainer_overrides_path).resolve()
    overrides = load_json(overrides_path)
    if not isinstance(overrides, Mapping) or not isinstance(overrides.get("data"), str):
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Trainer overrides do not expose one explicit dataset root")
    dataset_root = Path(str(overrides["data"]))
    dataset_root = (root / dataset_root).resolve() if not dataset_root.is_absolute() else dataset_root.resolve()
    if not dataset_root.is_dir():
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Acknowledged dataset root does not exist", artifact_path=dataset_root.as_posix())
    registry = load_asset_registry(asset_registry_path)
    content_assets = [record for record in registry.assets if record.asset_id == "dataset_content_ledger"]
    if len(content_assets) != 1 or not isinstance(content_assets[0].identity_digest, str):
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Asset registry does not uniquely declare dataset-content identity")
    descriptor_path = Path(claim_registry_root).resolve() / "CLAIM_REGISTRY.json"
    descriptor = load_json(descriptor_path)
    expected_registry_root_digest = output_root_digest(claim_registry_root)
    if not isinstance(descriptor, Mapping) or descriptor.get("registry_root_digest") != expected_registry_root_digest:
        raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Acknowledged execution-claim registry root is invalid")
    runbook_path = Path(runbook_manifest_path).resolve()
    validate_runbook_manifest(runbook_path, repository_root=root)
    context = RunIntentContext(
        action=action,
        run_role=run_role,
        logical_run_id=logical_run_id,
        arm_id=arm_id,
        training_seed=training_seed,
        dataset_root=dataset_root.as_posix(),
        output_root=Path(output_root).resolve().as_posix(),
        formal_identity_sha256=sha256_file(formal_identity_path),
        trainer_overrides_sha256=sha256_file(overrides_path),
        identity_manifest_sha256=sha256_file(identity_manifest_path),
        source_tree_digest=source_tree_digest,
        contract_digest=contract_digest,
        asset_registry_digest=asset_registry_digest,
        runtime_config_digest=runtime_config_digest,
        dataset_content_identity_digest=str(content_assets[0].identity_digest).upper(),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        schedule_digest=schedule_digest,
        identity_pool_binding_digest=identity_pool_binding_digest,
        release_manifest_sha256=sha256_file(release_manifest_path),
        execution_token_sha256=sha256_file(execution_token_path),
        claim_registry_root_digest=expected_registry_root_digest,
        resume_checkpoint_sha256=resume_checkpoint_sha256,
        resume_receipt_digest=resume_receipt_digest,
        runbook_manifest_sha256=sha256_file(runbook_path),
    )
    context.validate()
    return context


def prepare_formal_run_intent_binding(
    *,
    acknowledgement_path: str | Path,
    runbook_manifest_path: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
    action: str,
    run_role: str,
    logical_run_id: str,
    arm_id: str,
    training_seed: int,
    formal_identity_path: str | Path,
    trainer_overrides_path: str | Path,
    identity_manifest_path: str | Path,
    source_tree_digest: str,
    contract_digest: str,
    asset_registry_digest: str,
    runtime_config_digest: str,
    asset_registry_path: str | Path,
    parent_checkpoint_sha256: str,
    schedule_digest: str,
    identity_pool_binding_digest: str,
    release_manifest_path: str | Path,
    execution_token_path: str | Path,
    claim_registry_root: str | Path,
    resume_checkpoint_sha256: str,
    resume_receipt_digest: str,
) -> dict[str, Any]:
    """Derive, validate and bind the exact operator intent before a token claim."""

    root = Path(repository_root).resolve()
    runbook_path = Path(runbook_manifest_path).resolve()
    context = derive_formal_run_intent_context(
        runbook_manifest_path=runbook_path,
        repository_root=root,
        output_root=output_root,
        action=action,
        run_role=run_role,
        logical_run_id=logical_run_id,
        arm_id=arm_id,
        training_seed=training_seed,
        formal_identity_path=formal_identity_path,
        trainer_overrides_path=trainer_overrides_path,
        identity_manifest_path=identity_manifest_path,
        source_tree_digest=source_tree_digest,
        contract_digest=contract_digest,
        asset_registry_digest=asset_registry_digest,
        runtime_config_digest=runtime_config_digest,
        asset_registry_path=asset_registry_path,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        schedule_digest=schedule_digest,
        identity_pool_binding_digest=identity_pool_binding_digest,
        release_manifest_path=release_manifest_path,
        execution_token_path=execution_token_path,
        claim_registry_root=claim_registry_root,
        resume_checkpoint_sha256=resume_checkpoint_sha256,
        resume_receipt_digest=resume_receipt_digest,
    )
    validation = validate_run_intent_acknowledgement(
        acknowledgement_path,
        expected_context=context,
        runbook_manifest_path=runbook_path,
        repository_root=root,
    )
    return build_run_intent_binding(
        acknowledgement_path=acknowledgement_path,
        runbook_manifest_path=runbook_path,
        repository_root=root,
        context=context,
        validation=validation,
    )
