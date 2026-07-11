from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .errors import ValidationError
from .machine import MachineConfig
from .runtime_contract import RuntimeContract, validate_runtime_links
from .util import atomic_write_json, sha256_file, stable_hash


IMAGE_VERIFICATION_MODES = {"none", "existence", "sha256"}


def _update_length_prefixed(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _identity_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        _update_length_prefixed(digest, value)
    return digest.hexdigest().upper()


def _machine_value(machine: MachineConfig | dict[str, Any], key: str) -> Any:
    return machine.data.get(key) if isinstance(machine, MachineConfig) else machine.get(key)


def _machine_path(machine: MachineConfig | dict[str, Any], key: str) -> Path:
    value = _machine_value(machine, key)
    if value in (None, ""):
        raise ValidationError(f"Machine config missing required asset path: {key}")
    return Path(str(value)).expanduser().resolve()


def _snapshot_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "runtime_contract_id": report["runtime_contract_id"],
        "runtime_contract_sha256": report["runtime_contract_sha256"],
        "science_contract_file_sha256": report["science_contract_file_sha256"],
        "science_contract_semantic_sha256": report["science_contract_semantic_sha256"],
        "frozen_matrix_sha256": report["frozen_matrix_sha256"],
        "selection_index_sha256": report["selection_index_sha256"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "machine_id": report["machine_id"],
        "dataset_root": report["dataset_root"],
        "manifests": report["manifests"],
        "total_manifest_rows": report["total_manifest_rows"],
        "global_unique_id_count": report["global_unique_id_count"],
        "images": report["images"],
    }


def _image_audit(
    dataset_root: Path,
    identities: list[str],
    verification_mode: str,
) -> tuple[dict[str, Any], list[str]]:
    if verification_mode not in IMAGE_VERIFICATION_MODES:
        raise ValidationError(
            f"Unknown image verification mode {verification_mode!r}; expected one of {sorted(IMAGE_VERIFICATION_MODES)}"
        )
    if verification_mode == "none":
        return {
            "verification_mode": "none",
            "expected_count": len(identities),
            "checked_count": 0,
            "missing_count": 0,
            "non_file_count": 0,
            "total_size_bytes": None,
            "asset_digest_sha256": None,
            "content_hashes_computed": False,
        }, []

    digest = hashlib.sha256()
    missing: list[str] = []
    non_files: list[str] = []
    checked = 0
    total_size = 0
    for identity in sorted(identities):
        relative = Path(identity.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            missing.append(f"unsafe:{identity}")
            continue
        image = (dataset_root / relative).resolve()
        try:
            image.relative_to(dataset_root.resolve())
        except ValueError:
            missing.append(f"unsafe:{identity}")
            continue
        if not image.exists():
            missing.append(identity)
            continue
        if not image.is_file():
            non_files.append(identity)
            continue
        size = image.stat().st_size
        checked += 1
        total_size += size
        _update_length_prefixed(digest, identity)
        digest.update(size.to_bytes(8, "big", signed=False))
        if verification_mode == "sha256":
            _update_length_prefixed(digest, sha256_file(image))
    issues: list[str] = []
    if missing:
        issues.append(
            f"Missing or unsafe image paths: count={len(missing)}, examples={missing[:5]}"
        )
    if non_files:
        issues.append(f"Image paths are not files: count={len(non_files)}, examples={non_files[:5]}")
    return {
        "verification_mode": verification_mode,
        "expected_count": len(identities),
        "checked_count": checked,
        "missing_count": len(missing),
        "non_file_count": len(non_files),
        "total_size_bytes": total_size,
        "asset_digest_sha256": digest.hexdigest().upper(),
        "content_hashes_computed": verification_mode == "sha256",
    }, issues


def build_machine_asset_report(
    contract: RuntimeContract,
    machine: MachineConfig | dict[str, Any],
    output: str | Path,
    *,
    image_verification: str = "existence",
    overwrite: bool = False,
) -> dict[str, Any]:
    repo_root = _machine_path(machine, "repo_root")
    linked = validate_runtime_links(contract, repo_root)
    dataset_root = _machine_path(machine, "dataset_root")
    machine_id = str(_machine_value(machine, "machine_id"))
    spec = contract.data["machine_assets"]
    identity_column = str(spec["identity_column"])
    split_column = str(spec["split_column"])
    label_column = str(spec["label_column"])
    required_columns = {str(value) for value in spec["required_columns"]}
    issues: list[str] = []
    records: dict[str, Any] = {}
    all_identities: list[str] = []
    seen: dict[str, str] = {}
    overlap_examples: list[dict[str, str]] = []

    for role, role_spec in spec["manifest_roles"].items():
        key = str(role_spec["machine_config_key"])
        path = _machine_path(machine, key)
        if not path.is_file():
            issues.append(f"Missing manifest for {role}: {path}")
            continue
        try:
            frame = pd.read_csv(path, dtype={identity_column: "string"})
        except Exception as exc:
            issues.append(f"Unable to read manifest {role}: {exc}")
            continue
        columns = [str(value) for value in frame.columns]
        missing_columns = required_columns - set(columns)
        if missing_columns:
            issues.append(f"Manifest {role} missing columns: {sorted(missing_columns)}")
            continue
        identities = frame[identity_column]
        if identities.isna().any() or (identities.str.len() == 0).any():
            issues.append(f"Manifest {role} has null or empty identities")
            continue
        ids = identities.astype(str).tolist()
        duplicate_count = int(identities.duplicated().sum())
        if duplicate_count:
            issues.append(f"Manifest {role} has {duplicate_count} duplicate identities")
        actual_rows = len(frame)
        expected_rows = int(role_spec["expected_rows"])
        if actual_rows != expected_rows:
            issues.append(
                f"Manifest {role} row count mismatch: expected={expected_rows}, actual={actual_rows}"
            )
        splits = sorted(str(value) for value in frame[split_column].dropna().unique())
        expected_split = str(role_spec["expected_split"])
        if splits != [expected_split]:
            issues.append(
                f"Manifest {role} split mismatch: expected={[expected_split]}, actual={splits}"
            )
        expected_label = int(role_spec["expected_label"])
        labels = pd.to_numeric(frame[label_column], errors="coerce")
        invalid_label_count = int((labels.isna() | (labels != expected_label)).sum())
        if invalid_label_count:
            issues.append(
                f"Manifest {role} has {invalid_label_count} rows not labeled {expected_label}"
            )
        for identity in ids:
            previous = seen.get(identity)
            if previous is not None and previous != role:
                if len(overlap_examples) < 20:
                    overlap_examples.append(
                        {"sample_id": identity, "first_role": previous, "second_role": role}
                    )
            else:
                seen[identity] = role
        all_identities.extend(ids)
        records[str(role)] = {
            "machine_config_key": key,
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": actual_rows,
            "expected_rows": expected_rows,
            "columns": columns,
            "column_count": len(columns),
            "id_digest_sha256": _identity_digest(ids),
            "split_values": splits,
            "expected_split": expected_split,
            "expected_label": expected_label,
            "duplicate_id_count": duplicate_count,
            "invalid_label_count": invalid_label_count,
        }

    if overlap_examples:
        issues.append(f"Detected cross-manifest identity overlap: examples={overlap_examples[:5]}")
    expected_roles = {str(value) for value in spec["manifest_roles"]}
    missing_roles = expected_roles - set(records)
    if missing_roles:
        issues.append(f"Machine asset report missing manifest roles: {sorted(missing_roles)}")
    expected_total = int(spec["expected_total_rows"])
    if len(all_identities) != expected_total:
        issues.append(
            f"Total manifest row count mismatch: expected={expected_total}, actual={len(all_identities)}"
        )
    image_report, image_issues = _image_audit(dataset_root, all_identities, image_verification)
    issues.extend(image_issues)
    checkpoint = _machine_path(machine, "base_checkpoint")
    if not checkpoint.is_file():
        issues.append(f"Missing base checkpoint: {checkpoint}")
        checkpoint_sha = None
    else:
        checkpoint_sha = sha256_file(checkpoint)
        expected_checkpoint_sha = str(contract.data["checkpoint"]["sha256"]).upper()
        if checkpoint_sha != expected_checkpoint_sha:
            issues.append(
                "Base checkpoint SHA-256 mismatch: "
                f"expected={expected_checkpoint_sha}, actual={checkpoint_sha}"
            )

    report: dict[str, Any] = {
        "report_version": "1.0.0",
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "runtime_contract_id": contract.runtime_contract_id,
        "runtime_contract_sha256": contract.sha256,
        "science_contract_file_sha256": linked["science_contract"]["file_sha256"],
        "science_contract_semantic_sha256": linked["science_contract"]["semantic_sha256"],
        "frozen_matrix_sha256": linked["queue"]["frozen_matrix"]["sha256"],
        "selection_index_sha256": linked["queue"]["selection_index"]["sha256"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "machine_id": machine_id,
        "dataset_root": str(dataset_root),
        "manifests": records,
        "total_manifest_rows": len(all_identities),
        "global_unique_id_count": len(seen),
        "cross_manifest_overlap_examples": overlap_examples,
        "images": image_report,
    }
    report["snapshot_id"] = stable_hash(_snapshot_payload(report))
    atomic_write_json(output, report, overwrite=overwrite)
    if issues:
        raise ValidationError(f"Machine asset validation failed: {issues[:3]}")
    return report


def validate_machine_asset_report(
    contract: RuntimeContract,
    report_path: str | Path,
    *,
    expected_machine_id: str | None = None,
    minimum_image_verification: str = "existence",
) -> dict[str, Any]:
    path = Path(report_path).resolve()
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"Unable to read machine asset report {path}: {exc}") from exc
    if report.get("status") != "PASS" or report.get("issues"):
        raise ValidationError(f"Machine asset report is not PASS: {report.get('issues')}")
    if report.get("runtime_contract_id") != contract.runtime_contract_id:
        raise ValidationError("Machine asset report runtime contract ID mismatch")
    if str(report.get("runtime_contract_sha256", "")).upper() != contract.sha256:
        raise ValidationError("Machine asset report runtime contract SHA-256 mismatch")
    science = contract.data["science_contract"]
    queue = contract.data["queue"]
    expected_links = {
        "science_contract_file_sha256": str(science["file_sha256"]).upper(),
        "science_contract_semantic_sha256": str(science["semantic_sha256"]).upper(),
        "frozen_matrix_sha256": str(queue["frozen_matrix"]["sha256"]).upper(),
        "selection_index_sha256": str(queue["selection_index"]["sha256"]).upper(),
        "checkpoint_sha256": str(contract.data["checkpoint"]["sha256"]).upper(),
    }
    for key, expected in expected_links.items():
        if str(report.get(key, "")).upper() != expected:
            raise ValidationError(f"Machine asset report {key} mismatch")
    if expected_machine_id is not None and report.get("machine_id") != expected_machine_id:
        raise ValidationError(
            f"Machine asset report machine ID mismatch: expected={expected_machine_id}, actual={report.get('machine_id')}"
        )
    expected_roles = {str(value) for value in contract.data["machine_assets"]["manifest_roles"]}
    if set(report.get("manifests", {})) != expected_roles:
        raise ValidationError("Machine asset report manifest role set mismatch")
    asset_spec = contract.data["machine_assets"]
    expected_total = int(asset_spec["expected_total_rows"])
    if int(report.get("total_manifest_rows", -1)) != expected_total:
        raise ValidationError("Machine asset report total row count mismatch")
    if int(report.get("global_unique_id_count", -1)) != expected_total:
        raise ValidationError("Machine asset report does not prove global identity disjointness")
    required_columns = {str(value) for value in asset_spec["required_columns"]}
    for role, role_spec in asset_spec["manifest_roles"].items():
        record = report["manifests"][role]
        if int(record.get("rows", -1)) != int(role_spec["expected_rows"]):
            raise ValidationError(f"Machine asset report row count mismatch for {role}")
        if int(record.get("duplicate_id_count", -1)) != 0:
            raise ValidationError(f"Machine asset report duplicate IDs for {role}")
        if int(record.get("invalid_label_count", -1)) != 0:
            raise ValidationError(f"Machine asset report invalid labels for {role}")
        if set(record.get("columns", [])) < required_columns:
            raise ValidationError(f"Machine asset report missing required columns for {role}")
        if record.get("split_values") != [str(role_spec["expected_split"])]:
            raise ValidationError(f"Machine asset report split mismatch for {role}")
    verification_order = {"none": 0, "existence": 1, "sha256": 2}
    if minimum_image_verification not in verification_order:
        raise ValidationError(f"Unknown minimum image verification: {minimum_image_verification}")
    image_record = report.get("images", {})
    actual_mode = str(image_record.get("verification_mode", ""))
    if actual_mode not in verification_order:
        raise ValidationError(f"Unknown machine asset report image verification: {actual_mode}")
    if verification_order[actual_mode] < verification_order[minimum_image_verification]:
        raise ValidationError(
            f"Machine asset report image verification {actual_mode} is below required {minimum_image_verification}"
        )
    if int(image_record.get("expected_count", -1)) != expected_total:
        raise ValidationError("Machine asset report image expected count mismatch")
    if int(image_record.get("missing_count", -1)) != 0 or int(image_record.get("non_file_count", -1)) != 0:
        raise ValidationError("Machine asset report contains missing or non-file images")
    if actual_mode != "none" and int(image_record.get("checked_count", -1)) != expected_total:
        raise ValidationError("Machine asset report did not check every image")
    expected_snapshot = stable_hash(_snapshot_payload(report))
    if str(report.get("snapshot_id", "")).upper() != expected_snapshot:
        raise ValidationError(
            f"Machine asset report snapshot ID mismatch: expected={expected_snapshot}, actual={report.get('snapshot_id')}"
        )
    return {
        "status": "PASS",
        "machine_id": report["machine_id"],
        "snapshot_id": expected_snapshot,
        "runtime_contract_sha256": contract.sha256,
        "image_verification": report["images"]["verification_mode"],
        "image_count": report["images"]["checked_count"],
        "report_path": str(path),
    }
