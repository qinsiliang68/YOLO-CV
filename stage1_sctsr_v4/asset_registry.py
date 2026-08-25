from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import ErrorCode, SctsrError
from .serialization import atomic_write_json, load_json, sha256_file, stable_digest


EXTERNAL_FORMAL_REGISTRY_SCOPE = "EXTERNAL_FORMAL_INPUT_SNAPSHOT"


@dataclass(frozen=True, slots=True)
class AssetRecord:
    asset_id: str
    relative_path: str
    sha256: str
    bytes: int
    role: str
    required: bool = True
    row_count: int | None = None
    identity_digest: str | None = None
    group_semantic: str | None = None
    identity_digest_algorithm: str | None = None
    identity_column: str | None = None
    label_column: str | None = None
    constant_label: int | None = None
    replay_role_column: str | None = None
    split_column: str | None = None
    allowed_split_values: tuple[str, ...] = ()
    identity_universe: str | None = None
    mutual_exclusion_group: str | None = None
    declared_row_count_field: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AssetRecord":
        return cls(
            asset_id=str(value["asset_id"]),
            relative_path=str(value["relative_path"]),
            sha256=str(value["sha256"]).upper(),
            bytes=int(value["bytes"]),
            role=str(value["role"]),
            required=bool(value.get("required", True)),
            row_count=None if value.get("row_count") is None else int(value["row_count"]),
            identity_digest=value.get("identity_digest"),
            group_semantic=value.get("group_semantic"),
            identity_digest_algorithm=value.get("identity_digest_algorithm"),
            identity_column=value.get("identity_column"),
            label_column=value.get("label_column"),
            constant_label=None if value.get("constant_label") is None else int(value["constant_label"]),
            replay_role_column=value.get("replay_role_column"),
            split_column=value.get("split_column"),
            allowed_split_values=tuple(str(item) for item in value.get("allowed_split_values", ())),
            identity_universe=value.get("identity_universe"),
            mutual_exclusion_group=value.get("mutual_exclusion_group"),
            declared_row_count_field=value.get("declared_row_count_field"),
        )


@dataclass(frozen=True, slots=True)
class AssetRegistry:
    schema_version: str
    base_denominator: int
    assets: tuple[AssetRecord, ...]
    val_target_available: bool = False
    base_identity_digest: str | None = None
    base_identity_digest_algorithm: str | None = None
    base_asset_ids: tuple[str, ...] = ()
    split_asset_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()
    content_exclusion_asset_id: str | None = None

    @property
    def digest(self) -> str:
        return stable_digest(self)

    @property
    def split_asset_map(self) -> dict[str, tuple[str, ...]]:
        output: dict[str, tuple[str, ...]] = {}
        for role, asset_ids in self.split_asset_ids:
            if role in output:
                raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered split role is duplicated", observed=role)
            output[role] = asset_ids
        return output

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AssetRegistry":
        return cls(
            schema_version=str(value["schema_version"]),
            base_denominator=int(value["base_denominator"]),
            assets=tuple(AssetRecord.from_mapping(row) for row in value.get("assets", [])),
            val_target_available=bool(value.get("val_target_available", False)),
            base_identity_digest=None if value.get("base_identity_digest") is None else str(value["base_identity_digest"]).upper(),
            base_identity_digest_algorithm=value.get("base_identity_digest_algorithm"),
            base_asset_ids=tuple(str(item) for item in value.get("base_asset_ids", ())),
            split_asset_ids=tuple(
                (str(role), tuple(str(item) for item in asset_ids))
                for role, asset_ids in sorted(dict(value.get("split_asset_ids", {})).items())
            ),
            content_exclusion_asset_id=(
                None
                if value.get("content_exclusion_asset_id") is None
                else str(value["content_exclusion_asset_id"])
            ),
        )


SAMPLE_LABEL_DIGEST = "SHA256_SORTED_SAMPLE_ID_PIPE_LABEL_LF"
REPLAY_ROLE_DIGEST = "SHA256_SORTED_REPLAY_ROLE_TAB_SAMPLE_ID_LF"


@dataclass(frozen=True, slots=True)
class _ContentEvidence:
    row_count: int
    identities: Mapping[str, int]
    identity_digest: str | None
    split_values: tuple[str, ...]


def _sample_label_digest(identities: Mapping[str, int]) -> str:
    digest = hashlib.sha256()
    for sample_id, label in sorted(identities.items()):
        digest.update(f"{sample_id.replace(chr(92), '/')}|{int(label)}\n".encode("utf-8"))
    return digest.hexdigest().upper()


def _csv_content(record: AssetRecord, path: Path) -> _ContentEvidence:
    identities: dict[str, int] = {}
    role_identity: list[tuple[str, str]] = []
    split_values: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {field for field in (record.identity_column, record.label_column, record.replay_role_column, record.split_column) if field}
        missing = sorted(required - fields)
        if missing:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered CSV columns are missing", artifact_path=str(path), observed=missing)
        for row in reader:
            count += 1
            if record.split_column:
                split_values.add(str(row[record.split_column]))
            if record.identity_column:
                sample_id = str(row[record.identity_column]).replace("\\", "/")
                if not sample_id:
                    raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Asset contains an empty sample identity", artifact_path=str(path))
                if sample_id in identities:
                    raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Asset contains duplicate sample identities", artifact_path=str(path), observed=sample_id)
                raw_label = record.constant_label if record.constant_label is not None else row.get(record.label_column or "")
                if raw_label in (None, ""):
                    raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Asset identity has no registered label", artifact_path=str(path), observed=sample_id)
                try:
                    label = int(raw_label)
                except (TypeError, ValueError) as exc:
                    raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Asset label is not an integer", artifact_path=str(path), observed=raw_label) from exc
                if label not in (0, 1):
                    raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "SCTSR base labels must be binary", artifact_path=str(path), observed=label)
                identities[sample_id] = label
                if record.replay_role_column:
                    role_identity.append((str(row[record.replay_role_column]), sample_id))
    observed_digest: str | None = None
    if record.identity_digest_algorithm == SAMPLE_LABEL_DIGEST:
        observed_digest = _sample_label_digest(identities)
    elif record.identity_digest_algorithm == REPLAY_ROLE_DIGEST:
        digest = hashlib.sha256()
        for role, sample_id in sorted(role_identity):
            digest.update(role.encode("utf-8"))
            digest.update(b"\t")
            digest.update(sample_id.encode("utf-8"))
            digest.update(b"\n")
        observed_digest = digest.hexdigest().upper()
    elif record.identity_digest_algorithm is not None:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Unknown identity digest algorithm", artifact_path=str(path), observed=record.identity_digest_algorithm)
    return _ContentEvidence(count, identities, observed_digest, tuple(sorted(split_values)))


def _content_evidence(record: AssetRecord, path: Path) -> _ContentEvidence | None:
    if path.suffix.lower() == ".csv" and (record.row_count is not None or record.identity_column or record.split_column):
        return _csv_content(record, path)
    if path.suffix.lower() == ".json" and record.declared_row_count_field:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping) or record.declared_row_count_field not in payload:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered JSON row-count field is missing", artifact_path=str(path), observed=record.declared_row_count_field)
        return _ContentEvidence(int(payload[record.declared_row_count_field]), {}, None, ())
    return None


def validate_asset_registry(registry: AssetRegistry, repository_root: str | Path, *, verify_large_files: bool = True) -> dict[str, Any]:
    if registry.schema_version != "stage1.sctsr.asset_registry.v1":
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Unknown asset registry schema")
    if registry.base_denominator <= 0:
        raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Asset registry base denominator must be positive")
    if registry.val_target_available:
        raise SctsrError(ErrorCode.BLOCKED_BY_VAL_TARGET, "Current repository does not have an independent val_target")
    root = Path(repository_root).resolve()
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    evidence_by_id: dict[str, _ContentEvidence] = {}
    for record in registry.assets:
        if record.asset_id in seen_ids:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Duplicate asset_id", observed=record.asset_id)
        seen_ids.add(record.asset_id)
        path = (root / record.relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Asset path escapes repository root", artifact_path=str(path)) from exc
        if not path.exists():
            if record.required:
                raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Required frozen asset is missing", artifact_path=str(path))
            results.append({"asset_id": record.asset_id, "status": "OPTIONAL_MISSING"})
            continue
        observed_bytes = path.stat().st_size
        if observed_bytes != record.bytes:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Asset byte count mismatch", artifact_path=str(path), observed=observed_bytes, expected=record.bytes)
        observed_sha = sha256_file(path) if verify_large_files else "NOT_COMPUTED"
        if verify_large_files and observed_sha != record.sha256:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Asset SHA-256 mismatch", artifact_path=str(path), observed=observed_sha, expected=record.sha256)
        evidence = _content_evidence(record, path)
        if evidence is not None:
            if record.row_count is not None and evidence.row_count != record.row_count:
                raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Asset row count mismatch", artifact_path=str(path), observed=evidence.row_count, expected=record.row_count)
            if record.identity_digest is not None and evidence.identity_digest != record.identity_digest.upper():
                raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Asset identity digest mismatch", artifact_path=str(path), observed=evidence.identity_digest, expected=record.identity_digest.upper())
            if record.allowed_split_values and set(evidence.split_values) != set(record.allowed_split_values):
                raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Asset split role differs from the registry", artifact_path=str(path), observed=evidence.split_values, expected=record.allowed_split_values)
            evidence_by_id[record.asset_id] = evidence
        results.append({
            "asset_id": record.asset_id,
            "status": "PASS",
            "bytes": observed_bytes,
            "sha256": observed_sha,
            "row_count": None if evidence is None else evidence.row_count,
            "identity_digest": None if evidence is None else evidence.identity_digest,
            "split_values": [] if evidence is None else list(evidence.split_values),
        })

    if registry.base_asset_ids:
        missing_base = sorted(set(registry.base_asset_ids) - set(evidence_by_id))
        if missing_base:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Base identity assets lack verified content evidence", observed=missing_base)
        base_identity: dict[str, int] = {}
        for asset_id in registry.base_asset_ids:
            for sample_id, label in evidence_by_id[asset_id].identities.items():
                if sample_id in base_identity:
                    raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Canonical base components are not mutually exclusive", observed=sample_id)
                base_identity[sample_id] = label
        if len(base_identity) != registry.base_denominator:
            raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Base denominator is not derived from the registered canonical identities", observed=len(base_identity), expected=registry.base_denominator)
        if registry.base_identity_digest_algorithm != SAMPLE_LABEL_DIGEST:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Canonical base identity digest algorithm is not registered", observed=registry.base_identity_digest_algorithm, expected=SAMPLE_LABEL_DIGEST)
        observed_base_digest = _sample_label_digest(base_identity)
        if observed_base_digest != registry.base_identity_digest:
            raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Canonical base identity digest mismatch", observed=observed_base_digest, expected=registry.base_identity_digest)
        for record in registry.assets:
            evidence = evidence_by_id.get(record.asset_id)
            if evidence is None or not record.identity_universe:
                continue
            ids = set(evidence.identities)
            if record.identity_universe in {"CANONICAL_BASE_FULL", "REFERENCE_FULL"} and evidence.identities != base_identity:
                raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Full-universe asset differs from canonical base identity/labels", artifact_path=record.relative_path)
            if record.identity_universe == "T_SUBSET" and not ids.issubset(base_identity):
                raise SctsrError(ErrorCode.IDENTITY_NOT_IN_BASE, "T stress identity is outside canonical base", artifact_path=record.relative_path, observed=sorted(ids - set(base_identity))[:20])

    exclusion_groups: dict[str, list[tuple[str, set[str]]]] = {}
    for record in registry.assets:
        evidence = evidence_by_id.get(record.asset_id)
        if evidence is not None and record.mutual_exclusion_group:
            exclusion_groups.setdefault(record.mutual_exclusion_group, []).append((record.asset_id, set(evidence.identities)))
    for group, members in exclusion_groups.items():
        for index, (left_id, left) in enumerate(members):
            for right_id, right in members[index + 1 :]:
                overlap = left & right
                if overlap:
                    raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Mutually exclusive assets overlap", failing_field=group, observed={"left": left_id, "right": right_id, "example": sorted(overlap)[:20]})

    from .dataset_disjointness import load_registered_content_exclusions, scientific_split_role

    exclusions = load_registered_content_exclusions(registry, root)
    exclusions_by_role: dict[str, set[str]] = {}
    for sample_id, row in exclusions.items():
        exclusions_by_role.setdefault(scientific_split_role(row["split_role"]), set()).add(sample_id)

    allowed_split_roles = {"val_model", "val_cal", "val_op"}
    registered_splits: dict[str, dict[str, Any]] = {}
    split_identities_seen: dict[str, str] = {}
    asset_by_id = {record.asset_id: record for record in registry.assets}
    split_map = registry.split_asset_map
    for split_role, asset_ids in split_map.items():
        if split_role.lower() in {"test", "blind_test", "blind_holdout", "holdout_test"}:
            raise SctsrError(ErrorCode.TEST_ACCESS_FORBIDDEN, "Test or blind holdout may not enter the SCTSR v4 asset registry")
        if split_role not in allowed_split_roles:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered validation split role is unknown", observed=split_role)
        if not asset_ids or len(asset_ids) != len(set(asset_ids)):
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered validation split components are empty or duplicated", observed={split_role: asset_ids})
        combined: dict[str, int] = {}
        components: list[dict[str, Any]] = []
        for asset_id in asset_ids:
            record = asset_by_id.get(asset_id)
            evidence = evidence_by_id.get(asset_id)
            if record is None or evidence is None:
                raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered split component lacks verified content evidence", observed={split_role: asset_id})
            if split_role.upper() not in record.role.upper():
                raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Split component role does not match its registry split", observed=record.role, expected=split_role)
            for sample_id, label in evidence.identities.items():
                if sample_id in combined:
                    raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered split components overlap", observed={"split_role": split_role, "sample_id": sample_id})
                prior_role = split_identities_seen.get(sample_id)
                if prior_role is not None:
                    raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered validation splits are not identity-disjoint", observed={"sample_id": sample_id, "roles": [prior_role, split_role]})
                combined[sample_id] = label
                split_identities_seen[sample_id] = split_role
            components.append(
                {
                    "asset_id": asset_id,
                    "relative_path": record.relative_path,
                    "sha256": record.sha256,
                    "row_count": evidence.row_count,
                    "identity_digest": evidence.identity_digest,
                }
            )
        excluded_ids = exclusions_by_role.get(split_role, set())
        unknown_exclusions = sorted(excluded_ids - set(combined))
        if unknown_exclusions:
            raise SctsrError(
                ErrorCode.DATASET_SPLIT_CONTENT_LEAKAGE,
                "Content exclusion references an identity outside its registered split",
                observed={"split_role": split_role, "sample_ids": unknown_exclusions[:20]},
            )
        effective = {sample_id: label for sample_id, label in combined.items() if sample_id not in excluded_ids}
        if not effective:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Effective registered validation split is empty", observed=split_role)
        registered_splits[split_role] = {
            "asset_ids": list(asset_ids),
            "raw_row_count": len(combined),
            "excluded_content_rows": len(excluded_ids),
            "row_count": len(effective),
            "sample_label_identity_digest": _sample_label_digest(effective),
            "components": components,
        }

    return {
        "status": "PASS",
        "registry_digest": registry.digest,
        "base_denominator": registry.base_denominator,
        "base_identity_digest": registry.base_identity_digest,
        "assets": results,
        "registered_splits": registered_splits,
        "val_target_available": False,
    }


def load_asset_registry(path: str | Path) -> AssetRegistry:
    return AssetRegistry.from_mapping(load_json(path))


def load_registered_split_labels(
    registry: AssetRegistry,
    repository_root: str | Path,
    split_role: str,
) -> dict[str, int]:
    role = str(split_role)
    if role.lower() in {"test", "blind_test", "blind_holdout", "holdout_test"}:
        raise SctsrError(ErrorCode.TEST_ACCESS_FORBIDDEN, "Test or blind holdout split loading is forbidden")
    asset_ids = registry.split_asset_map.get(role)
    if not asset_ids:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Requested validation split is not registered", observed=role)
    root = Path(repository_root).resolve()
    asset_by_id = {record.asset_id: record for record in registry.assets}
    labels: dict[str, int] = {}
    for asset_id in asset_ids:
        record = asset_by_id.get(asset_id)
        if record is None:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered split component is missing", observed=asset_id)
        path = (root / record.relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered split component escapes repository", artifact_path=str(path)) from exc
        if not path.is_file() or path.stat().st_size != record.bytes or sha256_file(path) != record.sha256:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered split component identity changed", artifact_path=str(path))
        evidence = _content_evidence(record, path)
        if evidence is None or evidence.row_count != record.row_count or evidence.identity_digest != record.identity_digest:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered split component content evidence changed", artifact_path=str(path))
        for sample_id, label in evidence.identities.items():
            if sample_id in labels:
                raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered split components overlap", observed=sample_id)
            labels[sample_id] = label
    from .dataset_disjointness import load_registered_content_exclusions, scientific_split_role

    exclusions = load_registered_content_exclusions(registry, root)
    excluded_ids = {
        sample_id
        for sample_id, row in exclusions.items()
        if scientific_split_role(row["split_role"]) == role
    }
    unknown = sorted(excluded_ids - set(labels))
    if unknown:
        raise SctsrError(
            ErrorCode.DATASET_SPLIT_CONTENT_LEAKAGE,
            "Content exclusion references an identity outside the requested split",
            observed=unknown[:20],
        )
    labels = {sample_id: label for sample_id, label in labels.items() if sample_id not in excluded_ids}
    if not labels:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Registered validation split is empty", observed=role)
    return labels


def _validate_external_formal_registry_snapshot(
    snapshot_path: str | Path,
    registry_path: str | Path,
) -> dict[str, Any]:
    """Bind an external registry to the immutable formal-input snapshot."""

    source = Path(snapshot_path).resolve()
    registry_source = Path(registry_path).resolve()
    if not source.is_file():
        raise SctsrError(
            ErrorCode.ASSET_VALIDATION_FAILED,
            "Formal input snapshot is missing for the external asset registry",
            artifact_path=str(source),
        )
    run_root = source.parent.parent.resolve()
    if source != (run_root / "00_contract" / "FORMAL_INPUT_SNAPSHOT.json").resolve():
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Formal input snapshot path is not canonical")
    raw = load_json(source)
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "external_binding", "files", "snapshot_digest"}:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Formal input snapshot schema is invalid")
    core = {key: value for key, value in raw.items() if key != "snapshot_digest"}
    if raw.get("schema_version") != "stage1.sctsr.formal_input_snapshot.v1" or raw.get("snapshot_digest") != stable_digest(core):
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Formal input snapshot digest is invalid")

    external = raw.get("external_binding")
    if not isinstance(external, Mapping) or set(external) != {"schema_version", "required_roles", "files", "binding_digest"}:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Formal external-input binding schema is invalid")
    external_core = {key: value for key, value in external.items() if key != "binding_digest"}
    if external.get("schema_version") != "stage1.sctsr.external_file_binding.v1" or external.get("binding_digest") != stable_digest(external_core):
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Formal external-input binding digest is invalid")
    external_files = external.get("files")
    snapshot_files = raw.get("files")
    if not isinstance(external_files, Mapping) or not isinstance(snapshot_files, Mapping):
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Formal asset-registry snapshot row is missing")
    external_row = external_files.get("asset_registry")
    snapshot_row = snapshot_files.get("asset_registry")
    if (
        not isinstance(external_row, Mapping)
        or set(external_row) != {"path", "bytes", "sha256"}
        or not isinstance(snapshot_row, Mapping)
        or set(snapshot_row) != {"external_path", "snapshot_relative_path", "bytes", "sha256"}
    ):
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Formal asset-registry snapshot row is invalid")

    relative = Path(str(snapshot_row["snapshot_relative_path"]))
    snapshot_copy = (run_root / relative).resolve()
    try:
        snapshot_copy.relative_to(run_root)
    except ValueError as exc:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Formal asset-registry snapshot copy escapes the run root") from exc
    expected_external = Path(str(snapshot_row["external_path"])).resolve()
    expected_bytes = int(snapshot_row["bytes"])
    expected_sha = str(snapshot_row["sha256"])
    if (
        registry_source != expected_external
        or Path(str(external_row["path"])).resolve() != expected_external
        or int(external_row["bytes"]) != expected_bytes
        or str(external_row["sha256"]) != expected_sha
        or not registry_source.is_file()
        or not snapshot_copy.is_file()
        or registry_source.stat().st_size != expected_bytes
        or snapshot_copy.stat().st_size != expected_bytes
        or sha256_file(registry_source) != expected_sha
        or sha256_file(snapshot_copy) != expected_sha
    ):
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "External asset registry differs from its formal input snapshot")
    return {
        "asset_registry_path_scope": EXTERNAL_FORMAL_REGISTRY_SCOPE,
        "asset_registry_snapshot_relative_path": relative.as_posix(),
        "formal_input_snapshot_path": source.as_posix(),
        "formal_input_snapshot_sha256": sha256_file(source),
        "formal_input_snapshot_digest": raw["snapshot_digest"],
    }


def build_split_identity_bundle(
    registry: AssetRegistry,
    *,
    repository_root: str | Path,
    registry_path: str | Path,
    split_role: str,
    output_path: str | Path,
    formal_input_snapshot_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    registry_source = Path(registry_path).resolve()
    output = Path(output_path)
    if output.exists():
        raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Split identity bundle is immutable; choose a new output path", artifact_path=str(output))
    external_snapshot_binding: dict[str, Any] | None = None
    try:
        registry_relative = registry_source.relative_to(root).as_posix()
    except ValueError as exc:
        if formal_input_snapshot_path is None:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Asset registry path escapes repository root", artifact_path=str(registry_source)) from exc
        external_snapshot_binding = _validate_external_formal_registry_snapshot(formal_input_snapshot_path, registry_source)
        registry_relative = registry_source.as_posix()
    if not registry_source.is_file():
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Asset registry file is missing", artifact_path=str(registry_source))
    registered = load_asset_registry(registry_source)
    if registered.digest != registry.digest:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "In-memory and on-disk asset registries differ")
    report = validate_asset_registry(registry, root, verify_large_files=True)
    labels = load_registered_split_labels(registry, root, split_role)
    split_report = report["registered_splits"].get(split_role)
    if split_report is None:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Requested split is absent from validated registry", observed=split_role)
    payload: dict[str, Any] = {
        "schema_version": "stage1.sctsr.split_identity_bundle.v1",
        "split_role": split_role,
        "asset_registry_path": registry_relative,
        "asset_registry_sha256": sha256_file(registry_source),
        "asset_registry_digest": registry.digest,
        "components": split_report["components"],
        "row_count": len(labels),
        "sample_label_identity_digest": _sample_label_digest(labels),
        "formal_training_started": False,
        "blind_holdout_opened": False,
    }
    if external_snapshot_binding is not None:
        payload.update(external_snapshot_binding)
    payload["bundle_digest"] = stable_digest(payload)
    atomic_write_json(output, payload)
    return payload


def load_split_identity_bundle(
    path: str | Path,
    *,
    repository_root: str | Path,
    expected_split_role: str,
    allow_external_formal_registry: bool = False,
    expected_asset_registry_digest: str | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    source = Path(path)
    raw = load_json(source)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != "stage1.sctsr.split_identity_bundle.v1":
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction split identity bundle schema is invalid")
    if raw.get("split_role") != expected_split_role:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction split identity bundle role mismatch")
    if expected_split_role.lower() in {"test", "blind_test", "blind_holdout", "holdout_test"}:
        raise SctsrError(ErrorCode.TEST_ACCESS_FORBIDDEN, "Test or blind holdout split bundle is forbidden")
    expected_digest = stable_digest({key: value for key, value in raw.items() if key != "bundle_digest"})
    if raw.get("bundle_digest") != expected_digest:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction split identity bundle digest mismatch")
    root = Path(repository_root).resolve()
    scope = raw.get("asset_registry_path_scope")
    if scope is None:
        registry_path = (root / str(raw.get("asset_registry_path", ""))).resolve()
        try:
            registry_path.relative_to(root)
        except ValueError as exc:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Split bundle registry path escapes repository root") from exc
    elif scope == EXTERNAL_FORMAL_REGISTRY_SCOPE:
        if not allow_external_formal_registry or expected_asset_registry_digest is None:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "External formal asset registry was not explicitly authorized for this prediction")
        registry_path = Path(str(raw.get("asset_registry_path", ""))).resolve()
        snapshot_binding = _validate_external_formal_registry_snapshot(
            str(raw.get("formal_input_snapshot_path", "")),
            registry_path,
        )
        for field, value in snapshot_binding.items():
            if raw.get(field) != value:
                raise SctsrError(
                    ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                    "Split bundle formal-input snapshot binding changed",
                    failing_field=field,
                )
        if str(raw.get("asset_registry_digest", "")).upper() != str(expected_asset_registry_digest).upper():
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "External split bundle asset registry differs from its checkpoint")
    else:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Split bundle asset registry path scope is invalid")
    if not registry_path.is_file() or sha256_file(registry_path) != raw.get("asset_registry_sha256"):
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Split bundle asset registry identity changed")
    registry = load_asset_registry(registry_path)
    if registry.digest != raw.get("asset_registry_digest"):
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Split bundle asset registry digest changed")
    report = validate_asset_registry(registry, root, verify_large_files=True)
    labels = load_registered_split_labels(registry, root, expected_split_role)
    split_report = report["registered_splits"][expected_split_role]
    expected = {
        "components": split_report["components"],
        "row_count": len(labels),
        "sample_label_identity_digest": _sample_label_digest(labels),
    }
    for field, value in expected.items():
        if raw.get(field) != value:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Split identity bundle content differs from the validated registry", failing_field=field)
    return labels, dict(raw)
