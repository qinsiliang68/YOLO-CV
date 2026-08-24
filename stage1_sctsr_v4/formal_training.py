from __future__ import annotations

import math
import os
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .branch_lineage import BranchLineage
from .base_rng import BaseEpochRngReceipt, prepare_counter_domain_base_loader
from .checkpointing import build_checkpoint_payload, load_checkpoint, save_checkpoint_atomic
from .contracts import require_synthetic_or_authorized
from .epoch_transaction import EpochTransaction, GenerationIdentity, validate_receipt_chain
from .errors import ErrorCode, SctsrError
from .evidence_runtime import EpochEvidenceRecorder, ReplayHistoryState, SampleEvidence, sample_evidence_from_trainer
from .filesystem import windows_safe_resolved_path
from .formal_execution import (
    build_execution_job_bindings,
    execute_fenced_finalization,
    execution_fence_guard,
    publish_execution_claim_snapshot,
    validate_execution_claim_binding,
)
from .formal_completion import publish_formal_completion
from .logical_artifact_index import LogicalArtifactEntry, LogicalArtifactIndex
from .recovery import FormalResumeContext
from .replay_step_plan import build_replay_step_plan
from .rng_isolation import capture_global_rng
from .run_intent import publish_run_intent_snapshot, validate_run_intent_binding
from .schedule import SchedulePlan, schedule_to_dict
from .serialization import _fsync_directory, atomic_write_bytes, atomic_write_json, canonical_json_bytes, load_json, sha256_file, stable_digest
from .ultralytics_overlay import run_ultralytics_classification_epoch


CANONICAL_BATCH_SIZE = 128
CANONICAL_BASE_DENOMINATOR = 120_000
CANONICAL_BASE_STEPS = 938
KEEP_CHECKPOINTS = {120, 140, 150, 160, 180, 200}
FORMAL_AMP_INITIAL_SCALE = 65_536.0
FORMAL_AMP_GROWTH_INTERVAL = 1_000_000_000


def _lock_formal_amp_scaler_growth(trainer: Any) -> None:
    """Keep the formal AMP scale at its frozen, known-good initial value.

    Dynamic scale growth can make an otherwise stable long run skip a later
    optimizer update.  A skipped update invalidates the fixed 938-update epoch,
    and restoring the previous epoch also restores the scale that deterministically
    causes the same skip.  Formal runs therefore retain the prepared trainer's
    initial scale and disable growth for the bounded E1-E200 trajectory.  Backoff
    remains enabled so any genuine overflow still fails closed in the overlay.
    """

    if not bool(getattr(trainer, "amp", False)):
        return
    scaler = getattr(trainer, "scaler", None)
    if (
        scaler is None
        or not callable(getattr(scaler, "state_dict", None))
        or not callable(getattr(scaler, "load_state_dict", None))
    ):
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Formal AMP trainer has no stateful GradScaler")
    if callable(getattr(scaler, "is_enabled", None)) and not bool(scaler.is_enabled()):
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Formal AMP trainer has a disabled GradScaler")
    state = dict(scaler.state_dict())
    required = {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
    if not required.issubset(state):
        raise SctsrError(
            ErrorCode.UPSTREAM_BINDING_FAILED,
            "Formal AMP GradScaler state is incomplete",
            observed=sorted(state),
            expected=sorted(required),
        )
    if float(state["scale"]) != FORMAL_AMP_INITIAL_SCALE:
        raise SctsrError(
            ErrorCode.CONFIGURATION_MISMATCH,
            "Formal AMP GradScaler must start and resume at the frozen stable scale",
            failing_field="scaler_state.scale",
            observed=float(state["scale"]),
            expected=FORMAL_AMP_INITIAL_SCALE,
        )
    state["growth_interval"] = FORMAL_AMP_GROWTH_INTERVAL
    scaler.load_state_dict(state)
    verified = dict(scaler.state_dict())
    if int(verified.get("growth_interval", -1)) != FORMAL_AMP_GROWTH_INTERVAL:
        raise SctsrError(
            ErrorCode.UPSTREAM_BINDING_FAILED,
            "Formal AMP GradScaler rejected the frozen growth interval",
            observed=verified.get("growth_interval"),
            expected=FORMAL_AMP_GROWTH_INTERVAL,
        )


def publish_formal_input_snapshot(
    run_root: str | Path,
    external_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy every authorization input into the fresh run before epoch one.

    The snapshot is intentionally small: datasets and checkpoints remain in
    the asset registry and are re-hashed separately.  This directory contains
    the exact control-plane bytes that authorized the run, while retaining the
    external paths needed to detect post-start replacement at closeout.
    """

    from .formal_cli import FORMAL_AUTHORIZATION_INPUT_ROLES, validate_external_file_binding

    root = Path(run_root).resolve()
    validated = validate_external_file_binding(
        external_binding,
        required_roles=FORMAL_AUTHORIZATION_INPUT_ROLES,
    )
    files: dict[str, dict[str, Any]] = {}
    for role in FORMAL_AUTHORIZATION_INPUT_ROLES:
        external = Path(validated["files"][role]["path"])
        suffix = external.suffix.lower() if external.suffix.lower() in {".json", ".csv", ".md", ".yaml", ".yml", ".toml"} else ".bin"
        folder = "01_assets" if role == "asset_registry" else "00_contract"
        relative = Path(folder) / f"{role}{suffix}"
        destination = root / relative
        if destination.exists():
            raise SctsrError(
                ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE,
                "Formal input snapshot destination already exists",
                artifact_path=str(destination),
            )
        atomic_write_bytes(destination, external.read_bytes())
        files[role] = {
            "external_path": external.resolve().as_posix(),
            "snapshot_relative_path": relative.as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    core = {
        "schema_version": "stage1.sctsr.formal_input_snapshot.v1",
        "external_binding": dict(external_binding),
        "files": files,
    }
    snapshot = {**core, "snapshot_digest": stable_digest(core)}
    atomic_write_json(root / "00_contract" / "FORMAL_INPUT_SNAPSHOT.json", snapshot)
    return snapshot


def validate_formal_input_snapshot(
    run_root: str | Path,
    *,
    expected_external_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-hash both frozen copies and still-referenced external inputs."""

    from .formal_cli import FORMAL_AUTHORIZATION_INPUT_ROLES, validate_external_file_binding

    root = Path(run_root).resolve()
    path = root / "00_contract" / "FORMAL_INPUT_SNAPSHOT.json"
    if not path.is_file():
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Formal input snapshot manifest is missing")
    raw = load_json(path)
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "external_binding", "files", "snapshot_digest"}:
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Formal input snapshot schema is invalid")
    core = {key: value for key, value in raw.items() if key != "snapshot_digest"}
    if raw.get("schema_version") != "stage1.sctsr.formal_input_snapshot.v1" or raw.get("snapshot_digest") != stable_digest(core):
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Formal input snapshot digest is invalid")
    external = raw.get("external_binding")
    if not isinstance(external, Mapping) or (expected_external_binding is not None and dict(external) != dict(expected_external_binding)):
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Formal input snapshot references a different authorization binding")
    validated = validate_external_file_binding(external, required_roles=FORMAL_AUTHORIZATION_INPUT_ROLES)
    rows = raw.get("files")
    if not isinstance(rows, Mapping) or set(rows) != set(FORMAL_AUTHORIZATION_INPUT_ROLES):
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Formal input snapshot file set is incomplete")
    for role in FORMAL_AUTHORIZATION_INPUT_ROLES:
        row = rows[role]
        if not isinstance(row, Mapping) or set(row) != {"external_path", "snapshot_relative_path", "bytes", "sha256"}:
            raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Formal input snapshot row is invalid", failing_field=role)
        relative = Path(str(row["snapshot_relative_path"]))
        snapshot_file = (root / relative).resolve()
        try:
            snapshot_file.relative_to(root)
        except ValueError as exc:
            raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Formal input snapshot path escapes its run root") from exc
        expected = validated["files"][role]
        if (
            str(row["external_path"]) != expected["path"]
            or not snapshot_file.is_file()
            or snapshot_file.stat().st_size != row["bytes"]
            or row["bytes"] != expected["bytes"]
            or sha256_file(snapshot_file) != row["sha256"]
            or row["sha256"] != expected["sha256"]
        ):
            raise SctsrError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Formal input snapshot bytes differ from the authorization input",
                failing_field=role,
            )
    return {
        **raw,
        "status": "PASS",
        "external_binding_digest": external["binding_digest"],
        "manifest_sha256": sha256_file(path),
    }


@dataclass(frozen=True, slots=True)
class FormalIdentity:
    training_seed: int
    canonical_training_lock_sha256: str
    initial_checkpoint_sha256: str
    base_manifest_sha256: str
    source_tree_digest: str
    runtime_config_digest: str
    asset_registry_digest: str
    contract_digest: str | None = None
    seed_registry_digest: str | None = None

    def validate(self, *, formal: bool) -> None:
        if type(self.training_seed) is not int or not 0 <= self.training_seed < 2**63:
            raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Formal identity training seed is invalid")
        required = {
            "canonical_training_lock_sha256": self.canonical_training_lock_sha256,
            "initial_checkpoint_sha256": self.initial_checkpoint_sha256,
            "base_manifest_sha256": self.base_manifest_sha256,
            "source_tree_digest": self.source_tree_digest,
            "runtime_config_digest": self.runtime_config_digest,
            "asset_registry_digest": self.asset_registry_digest,
        }
        if formal:
            required.update(
                {
                    "contract_digest": self.contract_digest,
                    "seed_registry_digest": self.seed_registry_digest,
                }
            )
        invalid = [
            name
            for name, value in required.items()
            if not isinstance(value, str)
            or len(value) != 64
            or value != value.upper()
            or any(character not in "0123456789ABCDEF" for character in value)
        ]
        if invalid:
            raise SctsrError(
                ErrorCode.CONFIGURATION_MISMATCH,
                "Prepared run identity contains missing, placeholder, or noncanonical digests",
                observed=invalid,
            )

    @property
    def effective_contract_digest(self) -> str:
        # Backward-compatible only for synthetic unit fixtures. Formal mode
        # requires ``contract_digest`` explicitly through ``validate``.
        return self.contract_digest or self.runtime_config_digest


def _base_batch_sizes(trainer: Any) -> tuple[int, ...]:
    dataset_size = len(trainer.train_loader.dataset)
    if dataset_size != CANONICAL_BASE_DENOMINATOR:
        raise SctsrError(
            ErrorCode.DENOMINATOR_IDENTITY_MISMATCH,
            "Formal base dataset must contain exactly the frozen 120,000 optimizer-visible occurrences",
            observed=dataset_size, expected=CANONICAL_BASE_DENOMINATOR,
        )
    full, tail = divmod(dataset_size, CANONICAL_BATCH_SIZE)
    sizes = [CANONICAL_BATCH_SIZE] * full
    if tail:
        sizes.append(tail)
    if len(sizes) != CANONICAL_BASE_STEPS:
        raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Formal base process must have 938 steps", observed=len(sizes), expected=CANONICAL_BASE_STEPS)
    if len(trainer.train_loader) != CANONICAL_BASE_STEPS:
        raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Upstream DataLoader does not have 938 base batches")
    return tuple(sizes)


def _empty_replay_provider(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
    raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "No-replay parent requested a replay batch")


def _restore_trainer(payload: Mapping[str, Any], trainer: Any) -> None:
    trainer.model.load_state_dict(payload["model_state"])
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    trainer.scheduler.load_state_dict(payload["scheduler_state"])
    trainer.scaler.load_state_dict(payload["scaler_state"])
    ema_state = payload.get("ema_state", {})
    if getattr(trainer, "ema", None) is not None:
        if hasattr(trainer.ema, "load_state_dict") and ema_state:
            trainer.ema.load_state_dict(ema_state)
        elif hasattr(trainer.ema, "ema") and isinstance(ema_state, Mapping):
            model_state = ema_state.get("ema_model_state", ema_state.get("shadow", {}))
            if model_state:
                trainer.ema.ema.load_state_dict(model_state)
        trainer.ema.updates = int(payload.get("ema_updates", 0))
    from .rng_isolation import restore_global_rng
    restore_global_rng(payload["rng_state"])


def _checkpoint_payload(trainer: Any, identity: FormalIdentity, *, epoch: int, global_step: int) -> dict[str, Any]:
    return build_checkpoint_payload(
        model=trainer.model, ema=getattr(trainer, "ema", None), optimizer=trainer.optimizer,
        scheduler=trainer.scheduler, scaler=trainer.scaler, epoch=epoch, global_step=global_step,
        base_sampler_generation=epoch, canonical_training_lock_sha256=identity.canonical_training_lock_sha256,
        initial_checkpoint_sha256=identity.initial_checkpoint_sha256, base_manifest_sha256=identity.base_manifest_sha256,
        training_seed=identity.training_seed, source_tree_digest=identity.source_tree_digest,
        runtime_config_digest=identity.runtime_config_digest, asset_registry_digest=identity.asset_registry_digest,
    )


def _evidence_enabled(execution_mode: str, requested: bool | None) -> bool:
    enabled = execution_mode == "formal" if requested is None else bool(requested)
    if execution_mode == "formal" and not enabled:
        raise SctsrError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Formal SCTSR execution may not disable the taskbook evidence chain",
        )
    return enabled


def _prepare_run_root_after_upstream_setup(root_value: str | Path, *, execution_mode: str) -> Path:
    """Open one new run root without accepting pre-existing experiment data.

    A formal Ultralytics trainer creates only ``<run>/trainer`` during its
    explicitly validated setup stage.  The SCTSR runner may adopt that fresh
    subtree, but no other pre-existing sibling is allowed.  Synthetic/unit
    runners do not perform upstream setup and therefore create an absent root.
    """

    root = windows_safe_resolved_path(root_value)
    if execution_mode == "formal":
        if not root.is_dir():
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Formal run root was not created by prepared upstream setup", artifact_path=str(root))
        entries = list(root.iterdir())
        if len(entries) != 1 or entries[0].name != "trainer" or not entries[0].is_dir():
            raise SctsrError(
                ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE,
                "Formal run root contains data outside the fresh upstream trainer subtree",
                observed=sorted(path.name for path in entries),
                expected=["trainer"],
                artifact_path=str(root),
            )
        return root
    root.mkdir(parents=True, exist_ok=False)
    return root


def _revalidate_prepared_dataset_bindings(
    prepared_trainer_binding: Mapping[str, Any],
    *,
    trainer: Any | None = None,
) -> None:
    from .dataset_adapter import revalidate_materialized_dataset_binding

    dataset = prepared_trainer_binding.get("dataset_binding")
    if not isinstance(dataset, Mapping):
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Prepared trainer lacks materialized dataset evidence")
    dataset_digest = stable_digest(dict(dataset))
    marker_name = "_sctsr_fresh_materialized_dataset_binding_digest"
    if trainer is not None and getattr(trainer, marker_name, None) == dataset_digest:
        # build_prepared_trainer created and byte-bound these loaders in this
        # invocation. Consume the one-shot marker instead of reading every
        # hardlink again immediately before the adjacent run call.
        delattr(trainer, marker_name)
        return
    for field in ("train_materialized_content_binding", "val_model_materialized_content_binding"):
        binding = dataset.get(field)
        if not isinstance(binding, Mapping):
            raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Prepared trainer lacks a row-level materialized dataset binding", failing_field=field)
        revalidate_materialized_dataset_binding(binding)


_RESUME_STABLE_TRAINER_BINDING_FIELDS = (
    "upstream_binding_digest",
    "canonical_training_lock_sha256",
    "initial_checkpoint_sha256",
    "scientific_overrides_digest",
    "identity_manifest_binding",
    "dataset_binding",
    "dataset_content_binding",
    "training_seed",
)


def _resume_stable_trainer_binding_value(field: str, value: Any) -> Any:
    """Remove only location-derived fields from a resume identity comparison.

    A freshly prepared resume trainer writes its row-level dataset evidence
    below the immutable resume setup root.  That location must differ from the
    original trainer root, while the evidence bytes and all scientific dataset
    fields must remain identical.  The materialized binding digest also
    changes because it signs the evidence path, so compare the underlying
    identity after independently validating each binding's signed ledger.
    """

    if field != "dataset_binding" or not isinstance(value, Mapping):
        return value
    normalized = dict(value)
    for binding_field in (
        "train_materialized_content_binding",
        "val_model_materialized_content_binding",
    ):
        binding = normalized.get(binding_field)
        if not isinstance(binding, Mapping):
            continue
        stable_binding = dict(binding)
        stable_binding.pop("binding_digest", None)
        evidence = stable_binding.get("evidence")
        if isinstance(evidence, Mapping):
            stable_evidence = dict(evidence)
            stable_evidence.pop("path", None)
            stable_binding["evidence"] = stable_evidence
        normalized[binding_field] = stable_binding
    return normalized


def _validate_resume_root_and_bindings(
    *,
    output_root: str | Path,
    context: FormalResumeContext,
    identity: FormalIdentity,
    release_expected_bindings: Mapping[str, str] | None,
    prepared_trainer_binding: Mapping[str, Any] | None,
    expected_run_id: str,
    expected_arm_id: str,
) -> tuple[Path, dict[str, Any]]:
    canonical_output = Path(output_root).resolve()
    canonical_context = Path(context.run_root).resolve()
    root = windows_safe_resolved_path(canonical_output)
    if canonical_output != canonical_context or context.run_id != expected_run_id or context.arm_id != expected_arm_id:
        raise SctsrError(
            ErrorCode.RESUME_GENERATION_MISMATCH,
            "Resume context targets a different run root, run ID, or arm",
        )
    if context.training_seed != identity.training_seed:
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Resume context training seed differs from the prepared identity")
    if (root / "FORMAL_COMPLETION_RECEIPT.json").exists():
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "A canonically completed run may not enter resume")
    published_control = [
        path
        for path in (root / "RUN_MANIFEST.json", root / "PARENT_RECEIPT.json", root / "BRANCH_RECEIPT.json")
        if path.exists()
    ]
    if published_control and not context.terminal_epoch_complete:
        raise SctsrError(
            ErrorCode.RESUME_GENERATION_MISMATCH,
            "A nonterminal resume found finalization control artifacts",
            observed=[path.name for path in published_control],
        )
    for state_path in (root / "PARENT_RECEIPT.json", root / "BRANCH_RECEIPT.json"):
        if not state_path.exists():
            continue
        state = load_json(state_path)
        allowed = {
            "FORMAL_PARENT_EPOCHS_COMPLETE_PENDING_FINALIZATION",
            "FORMAL_BRANCH_EPOCHS_COMPLETE_PENDING_ENDPOINT",
            "FORMAL_BRANCH_ENDPOINT_COMPLETE_PENDING_COMMIT",
        }
        if state.get("status") not in allowed:
            raise SctsrError(
                ErrorCode.RESUME_GENERATION_MISMATCH,
                "Finalization recovery found an unregistered run-state status",
                observed=state.get("status"),
            )
    formal_identity = load_json(root / "FORMAL_IDENTITY.json")
    if formal_identity != asdict(identity):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Run-root formal identity differs from the resume identity")
    authorization = load_json(root / "FORMAL_AUTHORIZATION_BINDING.json")
    if authorization != dict(release_expected_bindings or {}):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Run-root authorization binding changed before resume")
    original_binding = load_json(root / "PREPARED_TRAINER_BINDING.json")
    if prepared_trainer_binding is None:
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Resume requires a newly revalidated prepared trainer")
    mismatch = {
        field: {"original": original_binding.get(field), "resume": prepared_trainer_binding.get(field)}
        for field in _RESUME_STABLE_TRAINER_BINDING_FIELDS
        if _resume_stable_trainer_binding_value(field, original_binding.get(field))
        != _resume_stable_trainer_binding_value(field, prepared_trainer_binding.get(field))
    }
    if mismatch:
        raise SctsrError(
            ErrorCode.UPSTREAM_BINDING_FAILED,
            "Resume prepared trainer differs in scientific or dataset identity",
            observed=mismatch,
        )
    return root, original_binding


def _append_resume_binding_receipt(
    *,
    root: Path,
    context: FormalResumeContext,
    original_binding: Mapping[str, Any],
    resume_binding: Mapping[str, Any],
) -> str:
    path = root / "08_receipts" / "resume_bindings.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = "0" * 64
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.endswith("\n"):
                        raise ValueError("missing newline")
                    existing = json.loads(line)
                    claimed = str(existing.pop("receipt_digest"))
                    if existing.get("previous_receipt_digest") != previous or stable_digest(existing) != claimed:
                        raise ValueError(f"chain mismatch at {line_number}")
                    previous = claimed
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
            raise SctsrError(
                ErrorCode.RESUME_GENERATION_MISMATCH,
                "Resume prepared-trainer receipt chain is corrupt",
                artifact_path=str(path),
            ) from exc
    row = {
        "schema_version": "stage1.sctsr.resume_prepared_trainer_receipt.v1",
        "status": "REVALIDATED_BEFORE_RESUME",
        "run_id": context.run_id,
        "arm_id": context.arm_id,
        "training_seed": context.training_seed,
        "resume_epoch": context.resume_epoch,
        "resume_context_digest": context.as_dict()["resume_context_digest"],
        "original_prepared_trainer_binding_digest": original_binding.get("binding_digest"),
        "resume_prepared_trainer_binding": dict(resume_binding),
        "resume_prepared_trainer_binding_digest": resume_binding.get("binding_digest"),
        "previous_receipt_digest": previous,
    }
    row["receipt_digest"] = stable_digest(row)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)
    return str(row["receipt_digest"])


def _restore_resume_checkpoint(
    *,
    trainer: Any,
    identity: FormalIdentity,
    context: FormalResumeContext,
) -> Mapping[str, Any]:
    checkpoint_path = windows_safe_resolved_path(context.checkpoint_path)
    payload = load_checkpoint(
        checkpoint_path,
        expected_sha256=context.checkpoint_sha256,
        expected_epoch=context.last_complete_epoch,
    )
    _assert_expected_checkpoint_payload(payload, identity)
    if payload["rng_state"].digest() != context.checkpoint_rng_digest or int(payload["global_step"]) != context.global_step:
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Resume checkpoint changed after preflight")
    _restore_trainer(payload, trainer)
    return payload


def _abort_failed_epoch(*, recorder: EpochEvidenceRecorder | None, transaction: EpochTransaction, primary_error: BaseException) -> None:
    """Attempt every cleanup step without replacing the epoch's root cause."""

    cleanup_errors: list[tuple[str, BaseException]] = []
    if recorder is not None:
        try:
            recorder.abort()
        except BaseException as exc:  # cleanup must continue through telemetry/writer failure.
            cleanup_errors.append(("evidence recorder abort", exc))
    try:
        transaction.abort("EPOCH_RUNTIME_OR_EVIDENCE_FAILURE")
    except BaseException as exc:  # preserve root cause and expose quarantine failure as a note.
        cleanup_errors.append(("epoch transaction abort", exc))
    for role, error in cleanup_errors:
        primary_error.add_note(f"SCTSR cleanup failure during {role}: {type(error).__name__}: {error}")


def _run_transactional_epoch(
    *,
    trainer: Any,
    identity: FormalIdentity,
    root: Path,
    run_id: str,
    parent_id: str,
    arm_id: str,
    epoch: int,
    replay_plan: Any,
    replay_batch_provider: Callable[[Sequence[str], int, int, int], Mapping[str, Any]],
    global_step_start: int,
    identity_policy: str,
    schedule_family: str,
    fallback_state: str,
    rate_numerator: int,
    rate_denominator: int,
    schedule_digest: str,
    identity_pool_digest: str,
    pool_multiplicity_targets: Mapping[str, int],
    sample_evidence: Mapping[str, SampleEvidence],
    history: ReplayHistoryState,
    previous_checkpoint_sha256: str,
    previous_generation_digest: str,
    base_rng_receipt: BaseEpochRngReceipt,
    publication_guard: Callable[[], Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run, checkpoint, validate and publish exactly one epoch generation."""

    generation = 1
    transaction = EpochTransaction(
        root / "03_epoch_transactions",
        run_id,
        epoch,
        generation,
        quarantine_root=root / "09_quarantine",
        identity=GenerationIdentity(
            parent_sha256=previous_checkpoint_sha256,
            arm_id=arm_id,
            training_seed=identity.training_seed,
            source_tree_digest=identity.source_tree_digest,
            contract_digest=identity.effective_contract_digest,
            asset_registry_digest=identity.asset_registry_digest,
            rng_state_digest=capture_global_rng().digest(),
            previous_generation_digest=previous_generation_digest,
        ),
    ).begin()
    recorder: EpochEvidenceRecorder | None = None
    try:
        recorder = EpochEvidenceRecorder(
            transaction=transaction,
            parent_id=parent_id,
            arm_id=arm_id,
            training_seed=identity.training_seed,
            sample_evidence=sample_evidence,
            identity_policy=identity_policy,
            schedule_family=schedule_family,
            fallback_state=fallback_state,
            rate_numerator=rate_numerator,
            rate_denominator=rate_denominator,
            schedule_digest=schedule_digest,
            identity_pool_digest=identity_pool_digest,
            pool_multiplicity_targets=pool_multiplicity_targets,
            expected_base_denominator=CANONICAL_BASE_DENOMINATOR,
            expected_optimizer_steps=CANONICAL_BASE_STEPS,
            global_step_start=global_step_start,
            history=history,
            artifact_root=root,
        )
        result = run_ultralytics_classification_epoch(
            trainer=trainer,
            replay_plan=replay_plan,
            replay_batch_provider=replay_batch_provider,
            training_seed=identity.training_seed,
            epoch=epoch,
            global_step_start=global_step_start,
            step_receipt_sink=recorder.step_sink,
            occurrence_event_sink=recorder.occurrence_sink,
            replay_rate_numerator=rate_numerator,
            replay_rate_denominator=rate_denominator,
            row_generation=generation,
        )
        result["base_rng_domain_receipt"] = base_rng_receipt.as_dict()
        checkpoint_sha = save_checkpoint_atomic(
            recorder.checkpoint_path,
            _checkpoint_payload(
                trainer,
                identity,
                epoch=epoch,
                global_step=int(result["global_step_end"]),
            ),
        )
        evidence_summary = recorder.finalize(runtime_result=result, checkpoint_sha256=checkpoint_sha)
        if publication_guard is None:
            generation_manifest = transaction.commit()
        else:
            with publication_guard():
                generation_manifest = transaction.commit()
    except BaseException as exc:
        _abort_failed_epoch(recorder=recorder, transaction=transaction, primary_error=exc)
        raise
    published_checkpoint = transaction.complete / recorder.checkpoint_relative
    if sha256_file(published_checkpoint) != checkpoint_sha:
        raise SctsrError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Published epoch checkpoint differs from the transaction-bound checkpoint",
            artifact_path=str(published_checkpoint),
        )
    evidence_receipt = {
        "status": "EPOCH_GENERATION_COMPLETE",
        "generation_digest": generation_manifest["generation_digest"],
        "generation_manifest_sha256": generation_manifest["generation_manifest_sha256"],
        "receipt_chain_digest": generation_manifest["receipt_chain_digest"],
        "checkpoint_path": published_checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "evidence_summary_digest": stable_digest(evidence_summary),
        "transaction_complete_path": transaction.complete.as_posix(),
    }
    return result, evidence_receipt


def _assert_expected_checkpoint_payload(payload: Mapping[str, Any], identity: FormalIdentity) -> None:
    expected = {
        "training_seed": identity.training_seed,
        "canonical_training_lock_sha256": identity.canonical_training_lock_sha256,
        "initial_checkpoint_sha256": identity.initial_checkpoint_sha256,
        "base_manifest_sha256": identity.base_manifest_sha256,
        "source_tree_digest": identity.source_tree_digest,
        "runtime_config_digest": identity.runtime_config_digest,
        "asset_registry_digest": identity.asset_registry_digest,
    }
    for field, value in expected.items():
        observed = payload.get(field)
        if observed != value:
            raise SctsrError(
                ErrorCode.BRANCH_LINEAGE_MISMATCH,
                "Parent checkpoint content differs from the prepared branch identity",
                failing_field=field,
                observed=observed,
                expected=value,
            )


def _validate_parent_completion_receipt(
    parent_checkpoint: str | Path,
    parent_sha: str,
    training_seed: int,
    *,
    require_formal: bool = False,
) -> dict[str, Any]:
    checkpoint = Path(parent_checkpoint)
    candidates = tuple(parent / "PARENT_RECEIPT.json" for parent in checkpoint.parents[:7])
    receipt_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if receipt_path is None:
        raise SctsrError(
            ErrorCode.BRANCH_LINEAGE_MISMATCH,
            "Child startup requires the canonical completed parent receipt, not a bare checkpoint path",
            artifact_path=str(checkpoint),
        )
    from .serialization import load_json

    receipt = load_json(receipt_path)
    expected = {
        "parent_id": f"PARENT_{training_seed}",
        "training_seed": training_seed,
        "checkpoint_sha256": parent_sha,
        "epoch_end": 120,
        "best_pt_used": False,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise SctsrError(
                ErrorCode.BRANCH_LINEAGE_MISMATCH,
                "Parent completion receipt differs from the selected checkpoint",
                failing_field=field,
                observed=receipt.get(field),
                expected=value,
                artifact_path=str(receipt_path),
            )
    effective_status = receipt.get("status")
    completion = None
    if effective_status == "FORMAL_PARENT_EPOCHS_COMPLETE_PENDING_FINALIZATION":
        try:
            from .formal_completion import validate_formal_completion

            completion = validate_formal_completion(receipt_path.parent, expected_run_role="COMMON_PARENT")
            effective_status = completion["receipt"]["status"]
        except SctsrError as exc:
            raise SctsrError(
                ErrorCode.BRANCH_LINEAGE_MISMATCH,
                "Parent has E120 evidence but lacks a valid atomic completion receipt",
                artifact_path=str(receipt_path),
            ) from exc
    if effective_status not in {"FORMAL_PARENT_COMPLETE", "IMPLEMENTED_NOT_FORMALLY_RUN"}:
        raise SctsrError(ErrorCode.BRANCH_LINEAGE_MISMATCH, "Parent receipt is not complete", artifact_path=str(receipt_path))
    if require_formal and effective_status != "FORMAL_PARENT_COMPLETE":
        raise SctsrError(
            ErrorCode.BRANCH_LINEAGE_MISMATCH,
            "A formal branch may only descend from a formally completed common parent",
            observed=effective_status,
            expected="FORMAL_PARENT_COMPLETE",
            artifact_path=str(receipt_path),
        )
    return {
        **receipt,
        "status": effective_status,
        "formal_completion": None if completion is None else completion["receipt"],
        "_receipt_path": receipt_path.as_posix(),
    }


def _manifest_entry(
    *,
    physical_root: Path,
    logical_run_id: str,
    logical_epoch: int,
    owner: str,
    physical_run_id: str,
    source_tree_digest: str,
    lineage_digest: str,
) -> LogicalArtifactEntry:
    from .serialization import load_json

    manifest_path = physical_root / "03_epoch_transactions" / f"epoch_{logical_epoch:04d}.generation_1.complete" / "GENERATION_MANIFEST.json"
    if not manifest_path.is_file():
        raise SctsrError(
            ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH,
            "Logical timeline references a missing completed epoch generation",
            epoch=logical_epoch,
            artifact_path=str(manifest_path),
        )
    manifest = load_json(manifest_path)
    checkpoint_rows = [row for row in manifest.get("files", []) if str(row.get("path", "")).endswith(".pt")]
    if len(checkpoint_rows) != 1:
        raise SctsrError(
            ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH,
            "Epoch generation must contain exactly one transaction-bound checkpoint",
            epoch=logical_epoch,
            observed=len(checkpoint_rows),
        )
    return LogicalArtifactEntry(
        logical_run_id=logical_run_id,
        logical_epoch=logical_epoch,
        physical_owner_type=owner,
        physical_run_id=physical_run_id,
        artifact_relative_path=manifest_path.relative_to(physical_root).as_posix(),
        artifact_sha256=sha256_file(manifest_path),
        checkpoint_sha256=str(checkpoint_rows[0]["sha256"]),
        source_tree_digest=source_tree_digest,
        lineage_digest=lineage_digest,
    )


def _publish_complete_logical_timeline(
    *,
    child_root: Path,
    parent_root: Path,
    lineage: BranchLineage,
    identity: FormalIdentity,
) -> dict[str, Any]:
    from .serialization import load_json

    entries = [
        _manifest_entry(
            physical_root=parent_root,
            logical_run_id=lineage.logical_run_id,
            logical_epoch=epoch,
            owner="PARENT",
            physical_run_id=lineage.parent_id,
            source_tree_digest=identity.source_tree_digest,
            lineage_digest="NOT_APPLICABLE_PARENT",
        )
        for epoch in range(1, 121)
    ]
    entries.extend(
        _manifest_entry(
            physical_root=child_root,
            logical_run_id=lineage.logical_run_id,
            logical_epoch=epoch,
            owner="CHILD",
            physical_run_id=lineage.logical_run_id,
            source_tree_digest=identity.source_tree_digest,
            lineage_digest=lineage.lineage_digest,
        )
        for epoch in range(121, 201)
    )
    logical = LogicalArtifactIndex(entries)
    logical.validate(require_complete_timeline=True, logical_run_id=lineage.logical_run_id)
    generation_index_path = (
        child_root / "ARTIFACT_INDEX_GENERATIONS.json"
        if (child_root / "ARTIFACT_INDEX_GENERATIONS.json").is_file()
        else child_root / "ARTIFACT_INDEX.json"
    )
    generation_index = load_json(generation_index_path)
    combined = {
        "schema_version": "stage1.sctsr.combined_artifact_index.v1",
        "epoch_generations": generation_index["epoch_generations"],
        "epoch_generation_index_digest": generation_index["epoch_generation_index_digest"],
        "logical_run_id": lineage.logical_run_id,
        "logical_timeline": [asdict(entry) for entry in sorted(entries, key=lambda item: item.logical_epoch)],
        "logical_timeline_digest": logical.digest,
        "physical_parent_root": parent_root.as_posix(),
        "physical_child_root": child_root.as_posix(),
    }
    index_path = child_root / "ARTIFACT_INDEX_LOGICAL.json"
    atomic_write_json(index_path, combined)
    return {
        "path": index_path.as_posix(),
        "sha256": sha256_file(index_path),
        "logical_timeline_digest": logical.digest,
        "logical_epoch_count": len(entries),
    }


def publish_formal_run_manifest_and_indexes(
    *,
    root: Path,
    identity: FormalIdentity,
    run_role: str,
    run_id: str,
    arm_id: str,
    release_authorization: str | Path | Mapping[str, Any],
    release_expected_bindings: Mapping[str, str],
    execution_claim_binding: Mapping[str, Any],
    execution_claim_snapshot: Mapping[str, Any],
    run_intent_snapshot: Mapping[str, Any],
    prepared_trainer_binding: Mapping[str, Any],
) -> dict[str, Any]:
    from .run_validation import build_artifact_index
    from .serialization import load_json

    generation_index_path = (
        root / "ARTIFACT_INDEX_GENERATIONS.json"
        if (root / "ARTIFACT_INDEX_GENERATIONS.json").is_file()
        else root / "ARTIFACT_INDEX.json"
    )
    if not generation_index_path.is_file():
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Formal run lacks its epoch-generation index")
    generation_index = load_json(generation_index_path)
    if generation_index.get("schema_version") != "stage1.sctsr.epoch_artifact_index.v1":
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Formal epoch-generation index schema is invalid")
    registered_generation_index = root / "ARTIFACT_INDEX_GENERATIONS.json"
    if registered_generation_index.is_file():
        if load_json(registered_generation_index) != generation_index:
            raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Published generation index changed during finalization recovery")
    else:
        atomic_write_json(registered_generation_index, generation_index)
    release = release_authorization if isinstance(release_authorization, Mapping) else load_json(release_authorization)
    release_sha = stable_digest(release) if isinstance(release_authorization, Mapping) else sha256_file(release_authorization)
    formal_input_snapshot = validate_formal_input_snapshot(root)
    adapter_binding = prepared_trainer_binding.get("adapter_import_binding")
    binary_binding = prepared_trainer_binding.get("binary_classification_binding")
    if not isinstance(adapter_binding, Mapping) or not isinstance(binary_binding, Mapping):
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Formal finalization lacks adapter or binary preflight identity")
    adapter_core = {key: value for key, value in adapter_binding.items() if key != "adapter_binding_digest"}
    binary_core = {key: value for key, value in binary_binding.items() if key != "binary_contract_digest"}
    if (
        adapter_binding.get("adapter_binding_digest") != stable_digest(adapter_core)
        or binary_binding.get("binary_contract_digest") != stable_digest(binary_core)
    ):
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Adapter or binary preflight binding digest changed before finalization")
    adapter_path = Path(str(adapter_binding["adapter_origin"])).resolve()
    repository_root = adapter_path.parents[2]
    if (
        not adapter_path.is_file()
        or adapter_path.is_symlink()
        or adapter_path.stat().st_size != adapter_binding["adapter_bytes"]
        or sha256_file(adapter_path) != adapter_binding["adapter_sha256"]
    ):
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "SCTSR adapter bytes changed before formal finalization")
    source_snapshot_row = formal_input_snapshot["files"]["source_tree_manifest"]
    source_snapshot_path = root / source_snapshot_row["snapshot_relative_path"]
    from .source_identity import validate_source_tree_manifest

    live_source = validate_source_tree_manifest(source_snapshot_path, repository_root, require_clean=True)
    manifest = {
        "schema_version": "stage1.sctsr.formal_run_manifest.v2",
        "execution_mode": "formal",
        "run_role": run_role,
        "run_id": run_id,
        "arm_id": arm_id,
        "training_seed": identity.training_seed,
        "source_tree_digest": identity.source_tree_digest,
        "repository_root": repository_root.as_posix(),
        "runtime_environment_digest": live_source["runtime_environment_digest"],
        "runtime_environment": live_source["runtime_environment"],
        "adapter_import_binding": dict(adapter_binding),
        "binary_classification_binding": dict(binary_binding),
        "contract_digest": identity.effective_contract_digest,
        "asset_registry_digest": identity.asset_registry_digest,
        "runtime_config_digest": identity.runtime_config_digest,
        "seed_registry_digest": identity.seed_registry_digest,
        "release_id": release.get("release_id"),
        "release_key_id": release.get("key_id"),
        "release_manifest_sha256": release_sha,
        "release_expected_bindings": dict(release_expected_bindings),
        "formal_input_snapshot_digest": formal_input_snapshot["snapshot_digest"],
        "formal_input_external_binding_digest": formal_input_snapshot["external_binding_digest"],
        "execution_id": execution_claim_binding["execution_id"],
        "execution_claim_sha256": execution_claim_binding["claim_sha256"],
        "execution_job_binding_digest": execution_claim_binding["job_binding_digest"],
        "execution_attempt_snapshot_digest": execution_claim_snapshot["snapshot_digest"],
        "run_intent_snapshot_digest": run_intent_snapshot["snapshot_digest"],
        "run_intent_acknowledgement_id": run_intent_snapshot["acknowledgement_id"],
        "formal_training_authorized": True,
        "formal_training_started": True,
        "engineering_gate_generated": False,
        "assignments_generated": False,
        "pilot_release_generated": False,
        "blind_holdout_opened": False,
        "selector_trained": False,
        "method_effectiveness_claimed": False,
        "test_accessed": False,
        "best_pt_used": False,
    }
    atomic_write_json(root / "RUN_MANIFEST.json", manifest)
    return {
        "run_manifest_path": (root / "RUN_MANIFEST.json").as_posix(),
        "run_manifest_sha256": sha256_file(root / "RUN_MANIFEST.json"),
        "artifact_index_path": generation_index_path.as_posix(),
        "generation_index_path": registered_generation_index.as_posix(),
        "generation_index_sha256": sha256_file(registered_generation_index),
    }


def validate_formal_run_runtime_identity(run_root: str | Path) -> dict[str, Any]:
    """Re-probe the live runtime and adapter bytes immediately before completion."""

    root = Path(run_root).resolve()
    manifest_path = root / "RUN_MANIFEST.json"
    manifest = load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Formal run manifest is missing during runtime closeout")
    repository_root = Path(str(manifest.get("repository_root", ""))).resolve()
    formal_input_snapshot = validate_formal_input_snapshot(root)
    source_row = formal_input_snapshot["files"]["source_tree_manifest"]
    source_path = root / source_row["snapshot_relative_path"]
    from .source_identity import validate_source_tree_manifest

    live = validate_source_tree_manifest(source_path, repository_root, require_clean=True)
    if (
        manifest.get("runtime_environment_digest") != live["runtime_environment_digest"]
        or manifest.get("runtime_environment") != live["runtime_environment"]
    ):
        raise SctsrError(
            ErrorCode.SOURCE_TREE_MISMATCH,
            "Live runtime changed between authorization and formal completion",
            observed=live["runtime_environment_digest"],
            expected=manifest.get("runtime_environment_digest"),
        )
    adapter = manifest.get("adapter_import_binding")
    binary = manifest.get("binary_classification_binding")
    if not isinstance(adapter, Mapping) or not isinstance(binary, Mapping):
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Formal run manifest lacks adapter or binary preflight binding")
    adapter_core = {key: value for key, value in adapter.items() if key != "adapter_binding_digest"}
    binary_core = {key: value for key, value in binary.items() if key != "binary_contract_digest"}
    adapter_path = Path(str(adapter.get("adapter_origin", ""))).resolve()
    if (
        adapter.get("adapter_binding_digest") != stable_digest(adapter_core)
        or binary.get("binary_contract_digest") != stable_digest(binary_core)
        or not adapter_path.is_file()
        or adapter_path.is_symlink()
        or adapter_path.stat().st_size != adapter.get("adapter_bytes")
        or sha256_file(adapter_path) != adapter.get("adapter_sha256")
    ):
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Adapter or binary preflight identity changed before completion")
    return {
        "status": "PASS",
        "runtime_environment_digest": live["runtime_environment_digest"],
        "adapter_binding_digest": adapter["adapter_binding_digest"],
        "binary_contract_digest": binary["binary_contract_digest"],
    }


def run_prepared_common_parent(
    *, trainer: Any, identity: FormalIdentity, output_root: str | Path,
    release_authorization: str | Path | None, execution_mode: str = "formal",
    collect_epoch_evidence: bool | None = None,
    release_trust_policy: str | Path | Mapping[str, Any] | None = None,
    release_expected_bindings: Mapping[str, str] | None = None,
    release_verification_secret: bytes | str | None = None,
    prepared_trainer_binding: Mapping[str, Any] | None = None,
    formal_input_binding: Mapping[str, Any] | None = None,
    execution_claim_binding: Mapping[str, Any] | None = None,
    run_intent_binding: Mapping[str, Any] | None = None,
    resume_context: FormalResumeContext | None = None,
) -> dict[str, Any]:
    """Run the formal E1-E120 no-replay common parent on a prepared trainer.

    A prepared trainer has its model, optimizer, scheduler, scaler, EMA and
    canonical base DataLoader initialized, but has not entered the upstream
    training loop. The function deliberately bypasses upstream final_eval so
    ``best.pt`` can never influence SCTSR.
    """

    identity.validate(formal=execution_mode == "formal")
    require_synthetic_or_authorized(
        execution_mode,
        release_authorization,
        trust_policy=release_trust_policy,
        expected_bindings=release_expected_bindings,
        verification_secret=release_verification_secret,
    )
    execution_job = build_execution_job_bindings(
        action="RESUME" if resume_context is not None else "START",
        run_role="COMMON_PARENT",
        logical_run_id=f"PARENT_{identity.training_seed}",
        arm_id="COMMON_PARENT_NR",
        training_seed=identity.training_seed,
        output_root=output_root,
        parent_checkpoint_sha256=identity.initial_checkpoint_sha256,
        resume_checkpoint_sha256="0" * 64 if resume_context is None else resume_context.checkpoint_sha256,
        lineage_digest=stable_digest({"role": "NOT_APPLICABLE_COMMON_PARENT"}),
        schedule_digest=stable_digest({"role": "COMMON_PARENT_NR", "epochs": [1, 120]}),
        resume_from_receipt_digest="0" * 64 if resume_context is None else resume_context.receipt_chain_digest,
    )
    if execution_mode == "formal":
        if execution_claim_binding is None:
            raise SctsrError(ErrorCode.FORMAL_EXECUTION_TOKEN_INVALID, "Formal parent lacks an atomic execution claim")
        validate_execution_claim_binding(
            execution_claim_binding,
            expected_job_bindings=execution_job,
            require_token_file=True,
        )
        if run_intent_binding is None:
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Formal parent lacks its exact run-intent acknowledgement")
        validate_run_intent_binding(run_intent_binding, enforce_freshness=True)
    sizes = _base_batch_sizes(trainer)
    evidence_enabled = _evidence_enabled(execution_mode, collect_epoch_evidence)
    sample_evidence = sample_evidence_from_trainer(trainer) if evidence_enabled else {}
    if execution_mode == "formal" and prepared_trainer_binding is None:
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Formal parent lacks its prepared trainer binding")
    if execution_mode == "formal":
        _revalidate_prepared_dataset_bindings(dict(prepared_trainer_binding or {}), trainer=trainer)
    if execution_mode == "formal" and formal_input_binding is None:
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Formal parent lacks its immutable authorization-input binding")
    if resume_context is not None and execution_mode != "formal":
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Only a formally authorized run may use formal resume")
    resume_binding_receipt_digest = None
    if resume_context is None:
        root = _prepare_run_root_after_upstream_setup(output_root, execution_mode=execution_mode)
        original_trainer_binding = dict(prepared_trainer_binding or {})
    else:
        if (resume_context.epoch_start, resume_context.epoch_end) != (1, 120):
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Common-parent resume context has the wrong epoch range")
        root, original_trainer_binding = _validate_resume_root_and_bindings(
            output_root=output_root,
            context=resume_context,
            identity=identity,
            release_expected_bindings=release_expected_bindings,
            prepared_trainer_binding=prepared_trainer_binding,
            expected_run_id=f"PARENT_{identity.training_seed}",
            expected_arm_id="COMMON_PARENT_NR",
        )
        _restore_resume_checkpoint(trainer=trainer, identity=identity, context=resume_context)
        if resume_context.history.cumulative_occurrences != 0 or resume_context.history.counts:
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Common-parent resume history contains replay")
        resume_binding_receipt_digest = _append_resume_binding_receipt(
            root=root,
            context=resume_context,
            original_binding=original_trainer_binding,
            resume_binding=dict(prepared_trainer_binding or {}),
        )
    if execution_mode == "formal":
        _lock_formal_amp_scaler_growth(trainer)
    execution_claim_snapshot = (
        publish_execution_claim_snapshot(root, dict(execution_claim_binding), expected_job_bindings=execution_job)
        if execution_mode == "formal"
        else None
    )
    run_intent_snapshot = (
        publish_run_intent_snapshot(root, dict(run_intent_binding))
        if execution_mode == "formal"
        else None
    )
    if execution_mode == "formal" and resume_context is None:
        formal_input_snapshot = publish_formal_input_snapshot(root, dict(formal_input_binding))
        atomic_write_json(root / "FORMAL_IDENTITY.json", asdict(identity))
        atomic_write_json(root / "FORMAL_AUTHORIZATION_BINDING.json", dict(release_expected_bindings or {}))
        atomic_write_json(root / "PREPARED_TRAINER_BINDING.json", dict(prepared_trainer_binding or {}))
    elif execution_mode == "formal":
        formal_input_snapshot = validate_formal_input_snapshot(
            root,
            expected_external_binding=formal_input_binding,
        )
    else:
        formal_input_snapshot = None
    global_step = 0 if resume_context is None else resume_context.global_step
    start_epoch = 1 if resume_context is None else resume_context.resume_epoch
    epoch_receipts = []
    history = ReplayHistoryState() if resume_context is None else resume_context.history
    previous_checkpoint_sha = identity.initial_checkpoint_sha256 if resume_context is None else resume_context.checkpoint_sha256
    previous_generation_digest = (
        stable_digest({"role": "COMMON_PARENT_START", "initial_checkpoint_sha256": identity.initial_checkpoint_sha256})
        if resume_context is None
        else resume_context.previous_generation_digest
    )
    final_evidence: dict[str, Any] | None = None
    no_replay_schedule_digest = stable_digest({"role": "COMMON_PARENT_NR", "epochs": [1, 120]})
    no_replay_pool_digest = stable_digest({"role": "NO_REPLAY_IDENTITY_POOL"})
    for epoch in range(start_epoch, 121):
        sampler = getattr(trainer.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch - 1)
        base_rng_receipt = prepare_counter_domain_base_loader(
            trainer,
            training_seed=identity.training_seed,
            epoch=epoch,
        ) if evidence_enabled else None
        plan = build_replay_step_plan(
            run_id=f"PARENT_{identity.training_seed}", arm_id="COMMON_PARENT_NR",
            training_seed=identity.training_seed, epoch=epoch, schedule_family="NR",
            sample_ids=(), base_batch_sizes=sizes,
        )
        if evidence_enabled:
            result, final_evidence = _run_transactional_epoch(
                trainer=trainer,
                identity=identity,
                root=root,
                run_id=f"PARENT_{identity.training_seed}",
                parent_id=f"PARENT_{identity.training_seed}",
                arm_id="COMMON_PARENT_NR",
                epoch=epoch,
                replay_plan=plan,
                replay_batch_provider=_empty_replay_provider,
                global_step_start=global_step,
                identity_policy="NONE",
                schedule_family="NR",
                fallback_state="NOT_APPLICABLE",
                rate_numerator=0,
                rate_denominator=1000,
                schedule_digest=no_replay_schedule_digest,
                identity_pool_digest=no_replay_pool_digest,
                pool_multiplicity_targets={},
                sample_evidence=sample_evidence,
                history=history,
                previous_checkpoint_sha256=previous_checkpoint_sha,
                previous_generation_digest=previous_generation_digest,
                base_rng_receipt=base_rng_receipt,
                publication_guard=(
                    lambda: execution_fence_guard(
                        dict(execution_claim_binding or {}),
                        expected_job_bindings=execution_job,
                    )
                    if execution_mode == "formal"
                    else None
                ),
            )
            previous_checkpoint_sha = final_evidence["checkpoint_sha256"]
            previous_generation_digest = final_evidence["generation_digest"]
        else:
            result = run_ultralytics_classification_epoch(
                trainer=trainer, replay_plan=plan, replay_batch_provider=_empty_replay_provider,
                training_seed=identity.training_seed, epoch=epoch, global_step_start=global_step,
            )
        global_step = int(result["global_step_end"])
        if result["optimizer_steps"] != CANONICAL_BASE_STEPS or result["replay_occurrences"] != 0:
            raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Common parent violated fixed base process", observed=result)
        epoch_receipts.append(
            {**result, "epoch_evidence": final_evidence if evidence_enabled else "NOT_COLLECTED_SYNTHETIC_ONLY"}
        )
    if evidence_enabled:
        if final_evidence is None:
            if resume_context is None or not resume_context.terminal_epoch_complete:
                raise SctsrError(
                    ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE,
                    "Common parent reached finalization without E120 epoch evidence",
                )
            checkpoint = Path(resume_context.checkpoint_path)
            checkpoint_sha = resume_context.checkpoint_sha256
        else:
            checkpoint = Path(final_evidence["checkpoint_path"])
            checkpoint_sha = str(final_evidence["checkpoint_sha256"])
    else:
        checkpoint = root / "epoch_0120.pt"
        checkpoint_sha = save_checkpoint_atomic(checkpoint, _checkpoint_payload(trainer, identity, epoch=120, global_step=global_step))
    try:
        os.chmod(checkpoint, 0o444)
    except OSError:
        pass
    receipt_chain = validate_receipt_chain(root / "08_receipts" / "epoch_receipts.jsonl") if evidence_enabled else None
    receipt = {
        "schema_version": "stage1.sctsr.formal_parent_receipt.v3",
        "status": (
            "IMPLEMENTED_NOT_FORMALLY_RUN"
            if execution_mode != "formal"
            else "FORMAL_PARENT_EPOCHS_COMPLETE_PENDING_FINALIZATION"
        ),
        "parent_id": f"PARENT_{identity.training_seed}", "training_seed": identity.training_seed,
        "epoch_start": 1, "epoch_end": 120, "global_step": global_step,
        "checkpoint_path": checkpoint.as_posix(), "checkpoint_sha256": checkpoint_sha,
        "epoch_receipt_digest": receipt_chain["receipt_chain_digest"] if receipt_chain is not None else stable_digest(epoch_receipts), "best_pt_used": False,
        "epoch_evidence_enabled": evidence_enabled,
        "recovery_pointer_path": (root / "ROLLING_RECOVERY_POINTER.json").as_posix() if evidence_enabled else None,
        "prepared_trainer_binding_digest": original_trainer_binding.get("binding_digest"),
        "resume_binding_receipt_digest": resume_binding_receipt_digest,
        "resumed_from_epoch": None if resume_context is None else resume_context.last_complete_epoch,
        "formal_input_snapshot_digest": None if formal_input_snapshot is None else formal_input_snapshot["snapshot_digest"],
        "execution_id": None if execution_claim_binding is None else execution_claim_binding.get("execution_id"),
        "execution_attempt_snapshot_digest": None if execution_claim_snapshot is None else execution_claim_snapshot["snapshot_digest"],
        "run_intent_snapshot_digest": None if run_intent_snapshot is None else run_intent_snapshot["snapshot_digest"],
        "run_intent_acknowledgement_id": None if run_intent_snapshot is None else run_intent_snapshot["acknowledgement_id"],
    }
    atomic_write_json(root / "PARENT_RECEIPT.json", receipt)
    if execution_mode == "formal":
        if release_expected_bindings is None or release_authorization is None:
            raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "Formal parent finalization lacks release bindings")
        def finalize_parent() -> dict[str, Any]:
            final_indexes = publish_formal_run_manifest_and_indexes(
                root=root,
                identity=identity,
                run_role="COMMON_PARENT",
                run_id=f"PARENT_{identity.training_seed}",
                arm_id="COMMON_PARENT_NR",
                release_authorization=release_authorization,
                release_expected_bindings=release_expected_bindings,
                execution_claim_binding=dict(execution_claim_binding or {}),
                execution_claim_snapshot=dict(execution_claim_snapshot or {}),
                run_intent_snapshot=dict(run_intent_snapshot or {}),
                prepared_trainer_binding=dict(prepared_trainer_binding or {}),
            )
            finalized_receipt = {**receipt, "final_indexes": final_indexes}
            atomic_write_json(root / "PARENT_RECEIPT.json", finalized_receipt)
            validate_formal_run_runtime_identity(root)
            from .run_validation import build_artifact_index
            atomic_write_json(root / "ARTIFACT_INDEX.json", build_artifact_index(root))
            completion = publish_formal_completion(
                root,
                run_role="COMMON_PARENT",
                run_id=f"PARENT_{identity.training_seed}",
                arm_id="COMMON_PARENT_NR",
                training_seed=identity.training_seed,
                terminal_epoch=120,
                fixed_checkpoint_sha256=checkpoint_sha,
            )
            return {**finalized_receipt, "status": completion["status"], "formal_completion": completion}

        return execute_fenced_finalization(
            dict(execution_claim_binding or {}),
            expected_job_bindings=execution_job,
            operation=finalize_parent,
        )
    return receipt


def run_prepared_branch(
    *, trainer: Any, identity: FormalIdentity, parent_checkpoint: str | Path,
    lineage: BranchLineage, schedule: SchedulePlan,
    replay_batch_provider: Callable[[Sequence[str], int, int, int], Mapping[str, Any]],
    output_root: str | Path, release_authorization: str | Path,
    execution_mode: str = "formal", collect_epoch_evidence: bool | None = None,
    release_trust_policy: str | Path | Mapping[str, Any] | None = None,
    release_expected_bindings: Mapping[str, str] | None = None,
    release_verification_secret: bytes | str | None = None,
    identity_pool_binding: Mapping[str, Any] | None = None,
    parent_artifact_index_binding: Mapping[str, Any] | None = None,
    prepared_trainer_binding: Mapping[str, Any] | None = None,
    formal_input_binding: Mapping[str, Any] | None = None,
    execution_claim_binding: Mapping[str, Any] | None = None,
    run_intent_binding: Mapping[str, Any] | None = None,
    resume_context: FormalResumeContext | None = None,
) -> dict[str, Any]:
    identity.validate(formal=execution_mode == "formal")
    require_synthetic_or_authorized(
        execution_mode,
        release_authorization,
        trust_policy=release_trust_policy,
        expected_bindings=release_expected_bindings,
        verification_secret=release_verification_secret,
    )
    parent_sha = sha256_file(parent_checkpoint)
    execution_job = build_execution_job_bindings(
        action="RESUME" if resume_context is not None else "START",
        run_role="BRANCH",
        logical_run_id=lineage.logical_run_id,
        arm_id=schedule.arm_id.value,
        training_seed=identity.training_seed,
        output_root=output_root,
        parent_checkpoint_sha256=parent_sha,
        resume_checkpoint_sha256="0" * 64 if resume_context is None else resume_context.checkpoint_sha256,
        lineage_digest=lineage.lineage_digest,
        schedule_digest=schedule.plan_digest,
        resume_from_receipt_digest="0" * 64 if resume_context is None else resume_context.receipt_chain_digest,
    )
    if execution_mode == "formal":
        if execution_claim_binding is None:
            raise SctsrError(ErrorCode.FORMAL_EXECUTION_TOKEN_INVALID, "Formal branch lacks an atomic execution claim")
        validate_execution_claim_binding(
            execution_claim_binding,
            expected_job_bindings=execution_job,
            require_token_file=True,
        )
        if run_intent_binding is None:
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Formal branch lacks its exact run-intent acknowledgement")
        validate_run_intent_binding(run_intent_binding, enforce_freshness=True)
    parent_receipt = _validate_parent_completion_receipt(
        parent_checkpoint,
        parent_sha,
        identity.training_seed,
        require_formal=execution_mode == "formal",
    )
    if execution_mode == "formal":
        if prepared_trainer_binding is None:
            raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Formal branch lacks its prepared trainer binding")
        _revalidate_prepared_dataset_bindings(dict(prepared_trainer_binding), trainer=trainer)
    lineage.validate(
        parent_sha=parent_sha, training_seed=identity.training_seed, arm_id=schedule.arm_id.value,
        source_digest=identity.source_tree_digest,
        contract_digest=identity.effective_contract_digest if execution_mode == "formal" else lineage.child_contract_digest,
    )
    payload = load_checkpoint(parent_checkpoint, expected_sha256=parent_sha, expected_epoch=120)
    _assert_expected_checkpoint_payload(payload, identity)
    _restore_trainer(payload, trainer)
    sizes = _base_batch_sizes(trainer)
    evidence_enabled = _evidence_enabled(execution_mode, collect_epoch_evidence)
    sample_evidence = sample_evidence_from_trainer(trainer) if evidence_enabled else {}
    if execution_mode == "formal" and (
        identity_pool_binding is None
        or parent_artifact_index_binding is None
        or prepared_trainer_binding is None
        or formal_input_binding is None
    ):
        raise SctsrError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Formal branch lacks immutable trainer, identity-pool, or parent-index bindings",
        )
    if resume_context is not None and execution_mode != "formal":
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Only a formally authorized run may use formal resume")
    resume_binding_receipt_digest = None
    if resume_context is None:
        root = _prepare_run_root_after_upstream_setup(output_root, execution_mode=execution_mode)
        original_trainer_binding = dict(prepared_trainer_binding or {})
    else:
        if (resume_context.epoch_start, resume_context.epoch_end) != (121, 200):
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Branch resume context has the wrong epoch range")
        root, original_trainer_binding = _validate_resume_root_and_bindings(
            output_root=output_root,
            context=resume_context,
            identity=identity,
            release_expected_bindings=release_expected_bindings,
            prepared_trainer_binding=prepared_trainer_binding,
            expected_run_id=lineage.logical_run_id,
            expected_arm_id=schedule.arm_id.value,
        )
        snapshots = {
            "BRANCH_LINEAGE.json": asdict(lineage),
            "SCHEDULE.json": schedule_to_dict(schedule),
            "IDENTITY_POOL_BINDING.json": dict(identity_pool_binding or {}),
            "PARENT_ARTIFACT_INDEX_BINDING.json": dict(parent_artifact_index_binding or {}),
        }
        for filename, expected_snapshot in snapshots.items():
            if load_json(root / filename) != expected_snapshot:
                raise SctsrError(
                    ErrorCode.RESUME_GENERATION_MISMATCH,
                    "Formal branch snapshot changed before resume",
                    failing_field=filename,
                )
        _restore_resume_checkpoint(trainer=trainer, identity=identity, context=resume_context)
        resume_binding_receipt_digest = _append_resume_binding_receipt(
            root=root,
            context=resume_context,
            original_binding=original_trainer_binding,
            resume_binding=dict(prepared_trainer_binding or {}),
        )
    if execution_mode == "formal":
        _lock_formal_amp_scaler_growth(trainer)
    execution_claim_snapshot = (
        publish_execution_claim_snapshot(root, dict(execution_claim_binding), expected_job_bindings=execution_job)
        if execution_mode == "formal"
        else None
    )
    run_intent_snapshot = (
        publish_run_intent_snapshot(root, dict(run_intent_binding))
        if execution_mode == "formal"
        else None
    )
    if execution_mode == "formal" and resume_context is None:
        formal_input_snapshot = publish_formal_input_snapshot(root, dict(formal_input_binding))
        atomic_write_json(root / "FORMAL_IDENTITY.json", asdict(identity))
        atomic_write_json(root / "FORMAL_AUTHORIZATION_BINDING.json", dict(release_expected_bindings or {}))
        atomic_write_json(root / "BRANCH_LINEAGE.json", asdict(lineage))
        atomic_write_json(root / "SCHEDULE.json", schedule_to_dict(schedule))
        atomic_write_json(root / "IDENTITY_POOL_BINDING.json", dict(identity_pool_binding))
        atomic_write_json(root / "PARENT_ARTIFACT_INDEX_BINDING.json", dict(parent_artifact_index_binding))
        atomic_write_json(root / "PREPARED_TRAINER_BINDING.json", dict(prepared_trainer_binding))
    elif execution_mode == "formal":
        formal_input_snapshot = validate_formal_input_snapshot(
            root,
            expected_external_binding=formal_input_binding,
        )
    else:
        formal_input_snapshot = None
    global_step = int(payload["global_step"]) if resume_context is None else resume_context.global_step
    start_epoch = 121 if resume_context is None else resume_context.resume_epoch
    parent_sha_before = sha256_file(parent_checkpoint)
    epoch_receipts = []
    checkpoints: dict[int, dict[str, str]] = {}
    history = ReplayHistoryState() if resume_context is None else resume_context.history
    previous_checkpoint_sha = parent_sha if resume_context is None else resume_context.checkpoint_sha256
    previous_generation_digest = (
        stable_digest({"role": "BRANCH_START", "parent_checkpoint_sha256": parent_sha, "lineage_digest": lineage.lineage_digest})
        if resume_context is None
        else resume_context.previous_generation_digest
    )
    if resume_context is not None:
        for epoch in sorted(KEEP_CHECKPOINTS & set(range(121, resume_context.last_complete_epoch + 1))):
            path = (
                root
                / "03_epoch_transactions"
                / f"epoch_{epoch:04d}.generation_1.complete"
                / "05_checkpoints"
                / f"rolling_epoch_{epoch:04d}.generation_1.pt"
            )
            if not path.is_file():
                raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Retained key checkpoint is missing before resume", epoch=epoch)
            checkpoints[epoch] = {"path": path.as_posix(), "sha256": sha256_file(path)}
    multiplicity = schedule.multiplicity()
    for epoch in range(start_epoch, 201):
        sampler = getattr(trainer.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch - 1)
        base_rng_receipt = prepare_counter_domain_base_loader(
            trainer,
            training_seed=identity.training_seed,
            epoch=epoch,
        ) if evidence_enabled else None
        epoch_plan = schedule.epoch(epoch)
        plan = build_replay_step_plan(
            run_id=lineage.logical_run_id, arm_id=schedule.arm_id.value,
            training_seed=identity.training_seed, epoch=epoch,
            schedule_family=epoch_plan.schedule_family, sample_ids=epoch_plan.sample_ids,
            base_batch_sizes=sizes,
        )
        if evidence_enabled:
            result, evidence = _run_transactional_epoch(
                trainer=trainer,
                identity=identity,
                root=root,
                run_id=lineage.logical_run_id,
                parent_id=lineage.parent_id,
                arm_id=schedule.arm_id.value,
                epoch=epoch,
                replay_plan=plan,
                replay_batch_provider=replay_batch_provider,
                global_step_start=global_step,
                identity_policy=epoch_plan.identity_policy,
                schedule_family=epoch_plan.schedule_family,
                fallback_state=epoch_plan.fallback_state,
                rate_numerator=epoch_plan.rate.numerator,
                rate_denominator=epoch_plan.rate.denominator,
                schedule_digest=schedule.plan_digest,
                identity_pool_digest=schedule.identity_pool_digest,
                pool_multiplicity_targets=multiplicity,
                sample_evidence=sample_evidence,
                history=history,
                previous_checkpoint_sha256=previous_checkpoint_sha,
                previous_generation_digest=previous_generation_digest,
                base_rng_receipt=base_rng_receipt,
                publication_guard=(
                    lambda: execution_fence_guard(
                        dict(execution_claim_binding or {}),
                        expected_job_bindings=execution_job,
                    )
                    if execution_mode == "formal"
                    else None
                ),
            )
            previous_checkpoint_sha = evidence["checkpoint_sha256"]
            previous_generation_digest = evidence["generation_digest"]
        else:
            result = run_ultralytics_classification_epoch(
                trainer=trainer, replay_plan=plan, replay_batch_provider=replay_batch_provider,
                training_seed=identity.training_seed, epoch=epoch, global_step_start=global_step,
            )
        global_step = int(result["global_step_end"])
        epoch_receipts.append({**result, "epoch_evidence": evidence if evidence_enabled else "NOT_COLLECTED_SYNTHETIC_ONLY"})
        if epoch in KEEP_CHECKPOINTS:
            if evidence_enabled:
                path = Path(evidence["checkpoint_path"])
                sha = str(evidence["checkpoint_sha256"])
            else:
                path = root / f"epoch_{epoch:04d}.pt"
                sha = save_checkpoint_atomic(path, _checkpoint_payload(trainer, identity, epoch=epoch, global_step=global_step))
            checkpoints[epoch] = {"path": path.as_posix(), "sha256": sha}
    if sha256_file(parent_checkpoint) != parent_sha_before:
        raise SctsrError(ErrorCode.CHILD_MUTATED_PARENT, "Branch mutated the read-only parent checkpoint")
    if execution_mode == "formal" and parent_receipt.get("epoch_evidence_enabled") is not True:
        raise SctsrError(
            ErrorCode.BRANCH_LINEAGE_MISMATCH,
            "Formal branch requires a parent with complete E1-E120 transaction evidence",
        )
    logical_index = None
    if evidence_enabled:
        logical_index = _publish_complete_logical_timeline(
            child_root=root,
            parent_root=Path(parent_receipt["_receipt_path"]).parent,
            lineage=lineage,
            identity=identity,
        )
    receipt_chain = validate_receipt_chain(root / "08_receipts" / "epoch_receipts.jsonl") if evidence_enabled else None
    receipt = {
        "schema_version": "stage1.sctsr.formal_branch_receipt.v3",
        "status": (
            "FORMAL_BRANCH_EPOCHS_COMPLETE_PENDING_ENDPOINT"
            if execution_mode == "formal"
            else "IMPLEMENTED_NOT_FORMALLY_RUN"
        ), "logical_run_id": lineage.logical_run_id,
        "arm_id": schedule.arm_id.value, "training_seed": identity.training_seed,
        "parent_checkpoint_sha256": parent_sha, "lineage_digest": lineage.lineage_digest,
        "epoch_start": 121, "epoch_end": 200, "global_step": global_step,
        "checkpoints": checkpoints, "fixed_formal_endpoint": checkpoints[200],
        "epoch_receipt_digest": receipt_chain["receipt_chain_digest"] if receipt_chain is not None else stable_digest(epoch_receipts), "best_pt_used": False,
        "epoch_evidence_enabled": evidence_enabled,
        "recovery_pointer_path": (root / "ROLLING_RECOVERY_POINTER.json").as_posix() if evidence_enabled else None,
        "logical_artifact_index": logical_index,
        "identity_pool_binding_digest": None if identity_pool_binding is None else identity_pool_binding.get("binding_digest"),
        "parent_artifact_index_binding_digest": None if parent_artifact_index_binding is None else parent_artifact_index_binding.get("binding_digest"),
        "prepared_trainer_binding_digest": original_trainer_binding.get("binding_digest"),
        "resume_binding_receipt_digest": resume_binding_receipt_digest,
        "resumed_from_epoch": None if resume_context is None else resume_context.last_complete_epoch,
        "formal_input_snapshot_digest": None if formal_input_snapshot is None else formal_input_snapshot["snapshot_digest"],
        "execution_id": None if execution_claim_binding is None else execution_claim_binding.get("execution_id"),
        "execution_attempt_snapshot_digest": None if execution_claim_snapshot is None else execution_claim_snapshot["snapshot_digest"],
        "run_intent_snapshot_digest": None if run_intent_snapshot is None else run_intent_snapshot["snapshot_digest"],
        "run_intent_acknowledgement_id": None if run_intent_snapshot is None else run_intent_snapshot["acknowledgement_id"],
    }
    atomic_write_json(root / "BRANCH_RECEIPT.json", receipt)
    if execution_mode == "formal":
        if release_expected_bindings is None:
            raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "Formal branch finalization lacks release bindings")
        return {
            **receipt,
            "_finalization_context": {
                "execution_claim_snapshot": dict(execution_claim_snapshot or {}),
                "run_intent_snapshot": dict(run_intent_snapshot or {}),
            },
        }
    return receipt
