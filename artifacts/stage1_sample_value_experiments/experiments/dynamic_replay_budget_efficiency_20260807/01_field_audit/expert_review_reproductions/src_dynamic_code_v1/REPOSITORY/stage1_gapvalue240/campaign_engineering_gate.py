"""Fail-closed engineering-gate v2 evidence binding and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

from .errors import ValidationError
from .util import atomic_write_json, sha256_file, stable_hash


ENGINEERING_GATE_SCHEMA_V2 = "stage1.dynamic_campaign_engineering_gate.v2"
EVIDENCE_ENVELOPE_SCHEMA = "stage1.validation_evidence_envelope.v1"

REQUIRED_EVIDENCE_SCHEMAS: dict[str, str] = {
    "canonical_lock_validation": "stage1.canonical_lock_validation.v1",
    "queue_v2_validation": "stage1.dynamic_campaign_run_queue.v2",
    "standalone_entry_validation": "stage1.standalone_entry_validation.v1",
    "assignment_reassignment_validation": "stage1.assignment_reassignment_validation.v1",
    "source_tree_immutability_validation": "stage1.source_tree_immutability_validation.v1",
    "local_real_data_smoke": "stage1.local_real_data_smoke_validation.v1",
    "crossed_numerical_parity": "stage1.crossed_numerical_parity_validation.v1",
    "failure_injection": "stage1.failure_injection_validation.v1",
    "all_epoch_telemetry": "stage1.all_epoch_telemetry_validation.v1",
    "lease_concurrency_validation": "stage1.lease_concurrency_validation.v1",
    "lease_fencing_validation": "stage1.lease_fencing_validation.v1",
    "coordination_root_canary": "stage1.coordination_root_canary_aggregate.v1",
    "ten_machine_real_data_canary": "stage1.ten_machine_real_data_canary_aggregate.v1",
    "disk_gpu_preflight": "stage1.disk_gpu_preflight_validation.v1",
    "documentation_handoff_validation": "stage1.documentation_handoff_validation.v1",
}


def _sha(value: str, label: str) -> str:
    text = str(value).upper()
    if len(text) != 64 or any(character not in "0123456789ABCDEF" for character in text):
        raise ValidationError(f"invalid {label} SHA-256")
    return text


@dataclass(frozen=True)
class ValidationIdentity:
    source_tree_sha256: str
    queue_registry_sha256: str
    canonical_lock_file_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_tree_sha256", _sha(self.source_tree_sha256, "source tree"))
        object.__setattr__(self, "queue_registry_sha256", _sha(self.queue_registry_sha256, "queue"))
        object.__setattr__(
            self,
            "canonical_lock_file_sha256",
            _sha(self.canonical_lock_file_sha256, "canonical lock"),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "source_tree_sha256": self.source_tree_sha256,
            "queue_registry_sha256": self.queue_registry_sha256,
            "canonical_lock_file_sha256": self.canonical_lock_file_sha256,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ValidationIdentity":
        return cls(
            source_tree_sha256=str(payload.get("source_tree_sha256", "")),
            queue_registry_sha256=str(payload.get("queue_registry_sha256", "")),
            canonical_lock_file_sha256=str(payload.get("canonical_lock_file_sha256", "")),
        )


def _within(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValidationError(f"{label} path escapes allowed evidence root: {resolved}") from exc
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def bind_validation_evidence(
    report_path: str | Path,
    *,
    evidence_type: str,
    expected_schema: str,
    identity: ValidationIdentity,
    output_path: str | Path,
    allowed_root: str | Path,
) -> Path:
    """Bind one immutable lower-level PASS report to campaign identities."""

    root = Path(allowed_root).resolve()
    report = _within(root, Path(report_path), "validation payload")
    output = _within(root, Path(output_path), "validation envelope")
    if evidence_type not in REQUIRED_EVIDENCE_SCHEMAS:
        raise ValidationError(f"unknown engineering-gate evidence type: {evidence_type}")
    if REQUIRED_EVIDENCE_SCHEMAS[evidence_type] != expected_schema:
        raise ValidationError(f"evidence schema contract differs for {evidence_type}")
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"unreadable validation payload: {report}") from exc
    if payload.get("schema_version") != expected_schema:
        raise ValidationError(
            f"validation payload schema mismatch for {evidence_type}: {payload.get('schema_version')}"
        )
    if payload.get("status") != "PASS":
        raise ValidationError(f"validation payload is not PASS: {evidence_type}")
    envelope = {
        "schema_version": EVIDENCE_ENVELOPE_SCHEMA,
        "status": "PASS",
        "evidence_type": evidence_type,
        "payload_relpath": _relative(root, report),
        "payload_sha256": sha256_file(report),
        "payload_schema": expected_schema,
        "payload_status": "PASS",
        "identity": identity.as_dict(),
        "created_at_unix": time.time(),
    }
    atomic_write_json(output, envelope, overwrite=True)
    return output


def _validate_envelope(
    envelope_path: Path,
    *,
    evidence_type: str,
    expected_identity: ValidationIdentity,
    allowed_root: Path,
) -> dict[str, Any]:
    envelope_path = _within(allowed_root, envelope_path, "validation envelope")
    try:
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"unreadable validation envelope: {envelope_path}") from exc
    expected_schema = REQUIRED_EVIDENCE_SCHEMAS[evidence_type]
    expected_envelope = {
        "schema_version": EVIDENCE_ENVELOPE_SCHEMA,
        "status": "PASS",
        "evidence_type": evidence_type,
        "payload_schema": expected_schema,
        "payload_status": "PASS",
    }
    mismatches = {
        key: {"expected": value, "observed": envelope.get(key)}
        for key, value in expected_envelope.items()
        if envelope.get(key) != value
    }
    try:
        observed_identity = ValidationIdentity.from_mapping(envelope.get("identity", {}))
    except ValidationError as exc:
        mismatches["identity"] = {"expected": expected_identity.as_dict(), "observed": str(exc)}
    else:
        if observed_identity != expected_identity:
            mismatches["identity"] = {
                "expected": expected_identity.as_dict(),
                "observed": observed_identity.as_dict(),
            }
    relative = Path(str(envelope.get("payload_relpath", "")))
    if relative.is_absolute() or ".." in relative.parts:
        mismatches["payload_relpath"] = {"expected": "safe relative path", "observed": str(relative)}
        payload_path = allowed_root / "__invalid__"
    else:
        payload_path = _within(allowed_root, allowed_root / relative, "validation payload")
    if not payload_path.is_file():
        mismatches["payload_exists"] = {"expected": True, "observed": False}
    else:
        actual_sha = sha256_file(payload_path)
        if actual_sha != str(envelope.get("payload_sha256", "")).upper():
            mismatches["payload_sha256"] = {
                "expected": envelope.get("payload_sha256"),
                "observed": actual_sha,
            }
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception as exc:
            mismatches["payload_json"] = {"expected": "readable JSON", "observed": str(exc)}
        else:
            if payload.get("schema_version") != expected_schema:
                mismatches["payload_schema_actual"] = {
                    "expected": expected_schema,
                    "observed": payload.get("schema_version"),
                }
            if payload.get("status") != "PASS":
                mismatches["payload_status_actual"] = {
                    "expected": "PASS",
                    "observed": payload.get("status"),
                }
    if mismatches:
        raise ValidationError(f"engineering gate evidence {evidence_type} invalid: {mismatches}")
    return {
        "envelope_relpath": _relative(allowed_root, envelope_path),
        "envelope_sha256": sha256_file(envelope_path),
        "payload_relpath": _relative(allowed_root, payload_path),
        "payload_sha256": sha256_file(payload_path),
        "payload_schema": expected_schema,
    }


def build_engineering_gate_v2(
    evidence_paths: Mapping[str, str | Path],
    *,
    expected_identity: ValidationIdentity,
    allowed_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(allowed_root).resolve()
    output = _within(root, Path(output_path), "engineering gate")
    issues = []
    if set(evidence_paths) != set(REQUIRED_EVIDENCE_SCHEMAS):
        missing = sorted(set(REQUIRED_EVIDENCE_SCHEMAS) - set(evidence_paths))
        extra = sorted(set(evidence_paths) - set(REQUIRED_EVIDENCE_SCHEMAS))
        issues.append(f"evidence registry mismatch: missing={missing}, extra={extra}")
    validated: dict[str, Any] = {}
    if not issues:
        for evidence_type in REQUIRED_EVIDENCE_SCHEMAS:
            try:
                validated[evidence_type] = _validate_envelope(
                    Path(evidence_paths[evidence_type]),
                    evidence_type=evidence_type,
                    expected_identity=expected_identity,
                    allowed_root=root,
                )
            except ValidationError as exc:
                issues.append(str(exc))
    report = {
        "schema_version": ENGINEERING_GATE_SCHEMA_V2,
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "created_at_unix": time.time(),
        "identity": expected_identity.as_dict(),
        "required_evidence_types": list(REQUIRED_EVIDENCE_SCHEMAS),
        "evidence": validated,
        "evidence_digest": stable_hash(validated),
        "validation_complete": not issues and len(validated) == len(REQUIRED_EVIDENCE_SCHEMAS),
    }
    atomic_write_json(output, report, overwrite=True)
    if issues:
        raise ValidationError(f"engineering gate build failed; see {output}")
    return report


def validate_engineering_gate_v2(
    gate_path: str | Path,
    *,
    expected_identity: ValidationIdentity,
    allowed_root: str | Path,
) -> dict[str, Any]:
    root = Path(allowed_root).resolve()
    gate_file = _within(root, Path(gate_path), "engineering gate")
    try:
        gate = json.loads(gate_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"engineering gate is unreadable: {gate_file}") from exc
    issues = []
    if gate.get("schema_version") != ENGINEERING_GATE_SCHEMA_V2:
        issues.append("engineering gate schema mismatch")
    if gate.get("status") != "PASS" or gate.get("validation_complete") is not True:
        issues.append("engineering gate top-level state is not validated PASS")
    try:
        gate_identity = ValidationIdentity.from_mapping(gate.get("identity", {}))
    except ValidationError as exc:
        issues.append(str(exc))
    else:
        if gate_identity != expected_identity:
            issues.append("engineering gate identity mismatch")
    evidence = gate.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(REQUIRED_EVIDENCE_SCHEMAS):
        issues.append("engineering gate evidence set is incomplete")
    validated: dict[str, Any] = {}
    if not issues:
        for evidence_type, row in evidence.items():
            relative = Path(str(row.get("envelope_relpath", "")))
            if relative.is_absolute() or ".." in relative.parts:
                issues.append(f"unsafe envelope path for {evidence_type}")
                continue
            try:
                validated[evidence_type] = _validate_envelope(
                    root / relative,
                    evidence_type=evidence_type,
                    expected_identity=expected_identity,
                    allowed_root=root,
                )
            except ValidationError as exc:
                issues.append(str(exc))
                continue
            if validated[evidence_type] != row:
                issues.append(f"engineering gate cached evidence differs for {evidence_type}")
    if not issues and stable_hash(validated) != str(gate.get("evidence_digest", "")).upper():
        issues.append("engineering gate evidence digest mismatch")
    if issues:
        raise ValidationError(f"engineering gate validation failed: {issues}")
    return gate


__all__ = [
    "ENGINEERING_GATE_SCHEMA_V2",
    "EVIDENCE_ENVELOPE_SCHEMA",
    "REQUIRED_EVIDENCE_SCHEMAS",
    "ValidationIdentity",
    "bind_validation_evidence",
    "build_engineering_gate_v2",
    "validate_engineering_gate_v2",
]
