"""Read-only inventory and integrity audit for external expert deliveries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterable
import zipfile


class ExpertAuditError(RuntimeError):
    """Raised when delivery evidence is malformed, unsafe, or inconsistent."""


@dataclass(frozen=True)
class ArchiveValidation:
    summary: dict[str, Any]
    members: tuple[dict[str, Any], ...]


_SHA_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$")
_HASH_CHUNK = 8 * 1024 * 1024

_DYNAMIC_TAR = "Stage1_DynamicReplay_ExpertReturn_20260808_144555.tar.gz"
_DYNAMIC_SIDECAR = f"{_DYNAMIC_TAR}.sha256"
_DYNAMIC_VALIDATION = (
    "Stage1_DynamicReplay_ExpertReturn_20260808_144555_PACKAGE_VALIDATION.json"
)
_BUDGET_LEDGER = "Stage1_BudgetedReplay_Learnability_20260809_v1.0.0_SHA256SUMS.txt"
_REVIEW_LEDGER = "Stage1_BudgetedReplay_v1.0.0_Review_SHA256SUMS.txt"

_REQUIRED_BUDGET_SOURCE = frozenset(
    {
        "Stage1_BudgetedReplay_Learnability_20260809_v1.0.0.tar.gz",
        "Stage1_BudgetedReplay_Learnability_20260809_v1.0.0.zip",
        "stage1_budgeted_replay-1.0.0-py3-none-any.whl",
    }
)

_DEFAULT_EXPECTED = (
    (_DYNAMIC_TAR, "expert_dynamic_return_source", True),
    (_DYNAMIC_SIDECAR, "expert_dynamic_return_sha_ledger", False),
    (_DYNAMIC_VALIDATION, "expert_dynamic_return_package_validation", False),
    (_BUDGET_LEDGER, "budgeted_replay_release_sha_ledger", False),
    ("Stage1_BudgetedReplay_Learnability_20260809_v1.0.0.tar.gz", "budgeted_replay_source_tar", True),
    ("Stage1_BudgetedReplay_Learnability_20260809_v1.0.0.zip", "budgeted_replay_source_zip", True),
    ("stage1_budgeted_replay-1.0.0-py3-none-any.whl", "budgeted_replay_wheel", True),
    ("Stage1_BudgetedReplay_Learnability_20260809_v1.0.0_RELEASE_VALIDATION.json", "budgeted_replay_release_validation", False),
    ("Stage1_BudgetedReplay_Learnability_20260809_v1.0.0_POSTPACKAGE_VALIDATION.json", "budgeted_replay_postpackage_validation", False),
    ("Stage1_BudgetedReplay_Learnability_20260809_v1.0.0_DELIVERY_REPORT.md", "budgeted_replay_delivery_report", False),
    (_REVIEW_LEDGER, "budgeted_replay_review_sha_ledger", False),
    ("Stage1_BudgetedReplay_v1.0.0_Independent_Review.md", "independent_no_go_review", False),
    ("Stage1_BudgetedReplay_v1.0.0_Findings.json", "independent_review_findings", False),
    ("Stage1_BudgetedReplay_v1.0.0_GO_NO_GO_Checklist.md", "independent_review_checklist", False),
    ("Stage1_BudgetedReplay_v1.0.0_Code_Evidence.txt", "independent_review_code_evidence", False),
    ("Stage1_BudgetedReplay_v1.0.0_Review_Environment.txt", "independent_review_environment", False),
    ("Stage1_BudgetedReplay_v1.0.0_Review_Evidence.zip", "independent_review_evidence_archive", False),
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _filesystem_io_path(path: Path) -> Path:
    """Use Win32 extended paths so verified extraction does not truncate evidence."""

    if os.name != "nt":
        return path
    absolute = str(path.resolve())
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def _remove_extraction_tree(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(_filesystem_io_path(path))
    if path.exists():
        raise ExpertAuditError(f"temporary extraction cleanup failed: {path}")


def _copy_and_hash(source: BinaryIO, destination: Path | None = None) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    output = None
    if destination is not None:
        io_destination = _filesystem_io_path(destination)
        io_destination.parent.mkdir(parents=True, exist_ok=True)
        output = io_destination.open("xb")
    try:
        for chunk in iter(lambda: source.read(_HASH_CHUNK), b""):
            size += len(chunk)
            digest.update(chunk)
            if output is not None:
                output.write(chunk)
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
    finally:
        if output is not None:
            output.close()
    return size, digest.hexdigest().upper()


def parse_sha256_ledger(path: str | Path) -> dict[str, str]:
    """Parse a sha256sum-style ledger and reject ambiguous identities."""

    source = Path(path)
    rows: dict[str, str] = {}
    for line_number, raw in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        match = _SHA_LINE.match(raw)
        if match is None:
            raise ExpertAuditError(f"invalid sha256 ledger line {line_number}: {source}")
        digest, filename = match.groups()
        digest = digest.upper()
        existing = rows.get(filename)
        if existing is not None and existing != digest:
            raise ExpertAuditError(f"conflicting sha256 entries for {filename}")
        rows[filename] = digest
    return rows


def _assert_safe_member(name: str) -> PurePosixPath:
    normalized = str(name).replace("\\", "/")
    member = PurePosixPath(normalized)
    if (
        member.is_absolute()
        or not member.parts
        or ".." in member.parts
        or ":" in member.parts[0]
    ):
        raise ExpertAuditError(f"unsafe archive member: {name}")
    return member


def _tar_regular_members(tf: tarfile.TarFile) -> list[tarfile.TarInfo]:
    regular: list[tarfile.TarInfo] = []
    for member in tf.getmembers():
        _assert_safe_member(member.name)
        if member.isdir():
            continue
        if not member.isfile():
            raise ExpertAuditError(f"unsupported archive member type: {member.name}")
        regular.append(member)
    return regular


def _scan_tar(
    archive: Path,
    *,
    extract_root: Path | None,
) -> tuple[list[dict[str, Any]], list[tarfile.TarInfo]]:
    rows: list[dict[str, Any]] = []
    with tarfile.open(archive, "r:*") as tf:
        members = _tar_regular_members(tf)
        for member in members:
            stream = tf.extractfile(member)
            if stream is None:
                raise ExpertAuditError(f"cannot read archive member: {member.name}")
            destination = extract_root.joinpath(*_assert_safe_member(member.name).parts) if extract_root else None
            size, digest = _copy_and_hash(stream, destination)
            rows.append(
                {
                    "container_filename": archive.name,
                    "member_path": member.name,
                    "size_bytes": size,
                    "sha256": digest,
                    "manifest_expected_sha256": "",
                    "manifest_hash_match": "",
                }
            )
    return rows, members


def validate_tar_archive(
    archive: str | Path, *, extract_root: str | Path | None = None
) -> ArchiveValidation:
    source = Path(archive)
    destination = Path(extract_root) if extract_root is not None else None
    rows, members = _scan_tar(source, extract_root=destination)
    return ArchiveValidation(
        summary={
            "status": "PASS",
            "archive_integrity": "PASS",
            "archive_member_count": len(members),
            "regular_file_count": len(rows),
            "manifest_member_count": None,
            "extraction_result": "FULL_EXTRACT_PASS" if destination else "STREAM_VALIDATION_PASS",
        },
        members=tuple(rows),
    )


def validate_manifested_tar(
    archive: str | Path, *, extract_root: str | Path | None = None
) -> ArchiveValidation:
    """Safely extract a tar and verify every payload row in FILE_MANIFEST.csv."""

    source = Path(archive)
    destination = Path(extract_root) if extract_root is not None else None
    with tarfile.open(source, "r:*") as tf:
        regular = _tar_regular_members(tf)
        csv_members = [item for item in regular if item.name.endswith("/FILE_MANIFEST.csv")]
        json_members = [item for item in regular if item.name.endswith("/FILE_MANIFEST.json")]
        if len(csv_members) != 1 or len(json_members) != 1:
            raise ExpertAuditError("manifested tar must contain one FILE_MANIFEST.csv and JSON")
        csv_stream = tf.extractfile(csv_members[0])
        json_stream = tf.extractfile(json_members[0])
        if csv_stream is None or json_stream is None:
            raise ExpertAuditError("cannot read package manifest")
        manifest_rows = list(
            csv.DictReader(io.TextIOWrapper(csv_stream, encoding="utf-8-sig", newline=""))
        )
        manifest_json = json.load(json_stream)

    required = {"relative_path", "size_bytes", "sha256"}
    if not manifest_rows or not required.issubset(manifest_rows[0]):
        raise ExpertAuditError("package manifest has missing required columns")
    declared: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        relative = _assert_safe_member(row["relative_path"]).as_posix()
        if relative in declared:
            raise ExpertAuditError(f"duplicate package manifest path: {relative}")
        declared[relative] = row
    if int(manifest_json.get("manifest_entry_count", -1)) != len(declared):
        raise ExpertAuditError("manifest entry count mismatch")

    scanned_rows, members = _scan_tar(source, extract_root=destination)
    root = PurePosixPath(csv_members[0].name).parent
    actual: dict[str, dict[str, Any]] = {}
    manifest_files = {"FILE_MANIFEST.csv", "FILE_MANIFEST.json"}
    for row in scanned_rows:
        member_path = PurePosixPath(str(row["member_path"]))
        try:
            relative = member_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ExpertAuditError(f"archive member outside package root: {member_path}") from exc
        actual[relative] = row

    missing = sorted(set(declared) - set(actual))
    extra = sorted(set(actual) - set(declared) - manifest_files)
    if missing:
        raise ExpertAuditError(f"manifest missing archive members: {missing[:3]}")
    if extra:
        raise ExpertAuditError(f"unmanifested archive members: {extra[:3]}")

    manifested_bytes = 0
    for relative, expected in declared.items():
        observed = actual[relative]
        expected_size = int(expected["size_bytes"])
        expected_hash = str(expected["sha256"]).strip().upper()
        manifested_bytes += expected_size
        if int(observed["size_bytes"]) != expected_size:
            raise ExpertAuditError(f"manifest size mismatch: {relative}")
        if str(observed["sha256"]).upper() != expected_hash:
            raise ExpertAuditError(f"manifest hash mismatch: {relative}")
        observed["manifest_expected_sha256"] = expected_hash
        observed["manifest_hash_match"] = True
    if "manifested_bytes" in manifest_json and int(manifest_json["manifested_bytes"]) != manifested_bytes:
        raise ExpertAuditError("manifested byte count mismatch")

    return ArchiveValidation(
        summary={
            "status": "PASS",
            "archive_integrity": "PASS",
            "archive_member_count": len(members),
            "regular_file_count": len(scanned_rows),
            "manifest_member_count": len(declared),
            "manifested_bytes": manifested_bytes,
            "extraction_result": "FULL_EXTRACT_PASS" if destination else "STREAM_VALIDATION_PASS",
        },
        members=tuple(scanned_rows),
    )


def extract_manifested_tar_subset(
    archive: str | Path,
    output_dir: str | Path,
    *,
    relative_paths: Iterable[str],
) -> ArchiveValidation:
    """Extract selected payloads only after the entire package manifest validates."""

    source = Path(archive).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence directory: {output}")
    requested = tuple(_assert_safe_member(path).as_posix() for path in relative_paths)
    if not requested or len(requested) != len(set(requested)):
        raise ExpertAuditError("subset paths must be non-empty and unique")

    full = validate_manifested_tar(source)
    manifest_members = [
        row for row in full.members if str(row["member_path"]).endswith("/FILE_MANIFEST.csv")
    ]
    if len(manifest_members) != 1:
        raise ExpertAuditError("validated package has no unique FILE_MANIFEST.csv")
    package_root = PurePosixPath(str(manifest_members[0]["member_path"])).parent
    by_relative: dict[str, dict[str, Any]] = {}
    for row in full.members:
        member_path = PurePosixPath(str(row["member_path"]))
        try:
            relative = member_path.relative_to(package_root).as_posix()
        except ValueError:
            continue
        if row.get("manifest_expected_sha256"):
            by_relative[relative] = row
    missing = sorted(set(requested) - set(by_relative))
    if missing:
        raise ExpertAuditError(f"requested subset is not in package manifest: {missing}")

    selected_rows: list[dict[str, Any]] = []
    output.mkdir(parents=True)
    try:
        with tarfile.open(source, "r:*") as tf:
            members = {member.name: member for member in _tar_regular_members(tf)}
            for relative in sorted(requested):
                verified = by_relative[relative]
                member_name = str(verified["member_path"])
                member = members.get(member_name)
                if member is None:
                    raise ExpertAuditError(f"verified archive member disappeared: {member_name}")
                stream = tf.extractfile(member)
                if stream is None:
                    raise ExpertAuditError(f"cannot read selected archive member: {member_name}")
                destination = output.joinpath(*PurePosixPath(relative).parts)
                size, digest = _copy_and_hash(stream, destination)
                if size != int(verified["size_bytes"]) or digest != str(verified["sha256"]):
                    raise ExpertAuditError(f"selected member identity mismatch: {relative}")
                selected_rows.append(
                    {
                        "container_filename": source.name,
                        "relative_path": relative,
                        "size_bytes": size,
                        "sha256": digest,
                        "manifest_expected_sha256": str(
                            verified["manifest_expected_sha256"]
                        ),
                        "manifest_hash_match": True,
                    }
                )
    except Exception:
        _remove_extraction_tree(output)
        raise

    return ArchiveValidation(
        summary={
            "status": "PASS",
            "archive_sha256": _sha256_path(source),
            "full_manifest_member_count": full.summary["manifest_member_count"],
            "selected_file_count": len(selected_rows),
            "extraction_result": "VERIFIED_SUBSET_EXTRACT_PASS",
        },
        members=tuple(selected_rows),
    )


def validate_zip_archive(
    archive: str | Path,
    *,
    expected_member_hashes: dict[str, str] | None = None,
    extract_root: str | Path | None = None,
) -> ArchiveValidation:
    source = Path(archive)
    destination = Path(extract_root) if extract_root is not None else None
    expected = {name: digest.upper() for name, digest in (expected_member_hashes or {}).items()}
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(source) as zf:
        bad_crc = zf.testzip()
        if bad_crc is not None:
            raise ExpertAuditError(f"zip CRC failure: {bad_crc}")
        basename_counts: dict[str, int] = {}
        for info in zf.infolist():
            _assert_safe_member(info.filename)
            basename_counts[PurePosixPath(info.filename).name] = (
                basename_counts.get(PurePosixPath(info.filename).name, 0) + 1
            )
        for info in zf.infolist():
            member = _assert_safe_member(info.filename)
            if info.is_dir():
                continue
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and stat.S_ISLNK(mode):
                raise ExpertAuditError(f"unsupported archive member type: {info.filename}")
            destination_path = destination.joinpath(*member.parts) if destination else None
            with zf.open(info) as stream:
                size, digest = _copy_and_hash(stream, destination_path)
            basename = member.name
            manifest_hash = expected.get(basename, "") if basename_counts[basename] == 1 else ""
            rows.append(
                {
                    "container_filename": source.name,
                    "member_path": info.filename,
                    "size_bytes": size,
                    "sha256": digest,
                    "manifest_expected_sha256": manifest_hash,
                    "manifest_hash_match": digest == manifest_hash if manifest_hash else "",
                }
            )
            if manifest_hash and digest != manifest_hash:
                raise ExpertAuditError(f"manifest hash mismatch: {info.filename}")
    return ArchiveValidation(
        summary={
            "status": "PASS",
            "archive_integrity": "PASS",
            "archive_member_count": len(zf.infolist()),
            "regular_file_count": len(rows),
            "manifest_member_count": sum(bool(row["manifest_expected_sha256"]) for row in rows),
            "extraction_result": "FULL_EXTRACT_PASS" if destination else "STREAM_VALIDATION_PASS",
        },
        members=tuple(rows),
    )


def _load_expected_hashes(downloads: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for ledger_name in (_BUDGET_LEDGER, _REVIEW_LEDGER):
        ledger = downloads / ledger_name
        if ledger.is_file():
            for filename, digest in parse_sha256_ledger(ledger).items():
                existing = expected.get(filename)
                if existing is not None and existing != digest:
                    raise ExpertAuditError(f"conflicting expected hash across ledgers: {filename}")
                expected[filename] = digest
    sidecar = downloads / _DYNAMIC_SIDECAR
    if sidecar.is_file():
        expected.update(parse_sha256_ledger(sidecar))
    return expected


def _atomic_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".md", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _archive_kind(filename: str) -> str | None:
    lower = filename.lower()
    if lower.endswith((".zip", ".whl")):
        return "zip"
    if lower.endswith((".tar.gz", ".tgz", ".tar")):
        return "tar"
    return None


def audit_expert_deliveries(
    downloads_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Publish immutable inventory evidence without executing delivered code."""

    downloads = Path(downloads_dir).resolve()
    output = Path(output_dir).resolve()
    inventory_path = output / "expert_v1_inventory.csv"
    validation_path = output / "expert_v1_hash_validation.json"
    member_path = output / "expert_archive_member_manifest.csv"
    readme_path = output / "README.md"
    output_manifest_path = output / "expert_audit_output_manifest.csv"
    for target in (
        inventory_path,
        validation_path,
        member_path,
        readme_path,
        output_manifest_path,
    ):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")

    expected_hashes = _load_expected_hashes(downloads)
    expected_specs: dict[str, tuple[str, bool]] = {
        filename: (role, required_source) for filename, role, required_source in _DEFAULT_EXPECTED
    }
    for filename in expected_hashes:
        expected_specs.setdefault(filename, ("ledger_declared_artifact", filename in _REQUIRED_BUDGET_SOURCE))

    archive_results: dict[str, ArchiveValidation] = {}
    archive_errors: dict[str, str] = {}
    archive_members: list[dict[str, Any]] = []
    temp_root = Path(tempfile.mkdtemp(prefix="stage1-expert-delivery-audit-"))
    try:
        for filename in sorted(expected_specs):
            source = downloads / filename
            kind = _archive_kind(filename)
            if kind is None or not source.is_file():
                continue
            try:
                extract_root = temp_root / re.sub(r"[^A-Za-z0-9_.-]+", "_", filename)
                if filename == _DYNAMIC_TAR:
                    result = validate_manifested_tar(source, extract_root=extract_root)
                elif kind == "tar":
                    result = validate_tar_archive(source, extract_root=extract_root)
                else:
                    result = validate_zip_archive(
                        source,
                        expected_member_hashes=expected_hashes if "Review_Evidence" in filename else None,
                        extract_root=extract_root,
                    )
                archive_results[filename] = result
                archive_members.extend(result.members)
            except Exception as exc:
                archive_errors[filename] = f"{type(exc).__name__}: {exc}"
    finally:
        _remove_extraction_tree(temp_root)

    by_basename: dict[str, list[dict[str, Any]]] = {}
    for row in archive_members:
        by_basename.setdefault(PurePosixPath(str(row["member_path"])).name, []).append(row)

    inventory: list[dict[str, Any]] = []
    for index, filename in enumerate(sorted(expected_specs), 1):
        role, required_source = expected_specs[filename]
        direct = downloads / filename
        expected_hash = expected_hashes.get(filename, "")
        observed_path = ""
        observed_hash = ""
        size_bytes: int | str = ""
        present = False
        archive_name = ""
        if direct.is_file():
            present = True
            observed_path = str(direct)
            observed_hash = _sha256_path(direct)
            size_bytes = direct.stat().st_size
            if expected_hash:
                status = "PRESENT_AND_VERIFIED" if observed_hash == expected_hash else "PRESENT_BUT_HASH_MISMATCH"
            else:
                status = "PRESENT_UNVERIFIED_NO_EXPECTED_HASH"
        else:
            candidates = by_basename.get(filename, [])
            verified = [row for row in candidates if not expected_hash or row["sha256"] == expected_hash]
            if len(verified) == 1 and not required_source:
                member = verified[0]
                present = True
                archive_name = str(member["container_filename"])
                observed_path = f"{archive_name}::{member['member_path']}"
                observed_hash = str(member["sha256"])
                size_bytes = int(member["size_bytes"])
                status = (
                    "PRESENT_AS_ARCHIVE_MEMBER_VERIFIED"
                    if expected_hash
                    else "PRESENT_AS_ARCHIVE_MEMBER_UNVERIFIED"
                )
            elif required_source:
                status = "REPORT_ONLY_SOURCE_MISSING"
            else:
                status = "EXPECTED_AUXILIARY_MISSING"

        archive_summary = archive_results.get(filename)
        if archive_summary is None and archive_name:
            archive_summary = archive_results.get(archive_name)
        archive_error = archive_errors.get(filename, "")
        summary = archive_summary.summary if archive_summary else {}
        inventory.append(
            {
                "artifact_id": f"EXPERT-{index:03d}",
                "expected_filename": filename,
                "observed_path": observed_path,
                "present_or_missing": "PRESENT" if present else "MISSING",
                "size_bytes": size_bytes,
                "sha256": observed_hash,
                "expected_sha256": expected_hash,
                "hash_match": (
                    str(observed_hash == expected_hash).lower()
                    if expected_hash and observed_hash
                    else "not_available"
                ),
                "archive_integrity": summary.get("archive_integrity", "FAIL" if archive_error else "NOT_ARCHIVE"),
                "manifest_member_count": summary.get("manifest_member_count", ""),
                "extraction_result": summary.get("extraction_result", "FAIL" if archive_error else "NOT_ARCHIVE"),
                "evidence_role": role,
                "required_source": str(required_source).lower(),
                "status": status if not archive_error else "ARCHIVE_VALIDATION_FAILED",
                "validation_error": archive_error,
            }
        )

    expected_names = set(expected_specs)
    observed_relevant = sorted(
        path
        for path in downloads.iterdir()
        if path.is_file()
        and (
            path.name.startswith("Stage1_BudgetedReplay")
            or path.name.startswith("Stage1_DynamicReplay")
            or path.name.startswith("YOLO-CV_push-info")
        )
        and path.name not in expected_names
    )
    for path in observed_relevant:
        inventory.append(
            {
                "artifact_id": f"OBSERVED-{len(inventory) + 1:03d}",
                "expected_filename": path.name,
                "observed_path": str(path),
                "present_or_missing": "PRESENT",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_path(path),
                "expected_sha256": "",
                "hash_match": "not_available",
                "archive_integrity": "NOT_AUDITED_UNLISTED_COPY",
                "manifest_member_count": "",
                "extraction_result": "NOT_AUDITED_UNLISTED_COPY",
                "evidence_role": "unlisted_related_artifact",
                "required_source": "false",
                "status": "PRESENT_UNLISTED",
                "validation_error": "",
            }
        )

    missing_sources = [row for row in inventory if row["status"] == "REPORT_ONLY_SOURCE_MISSING"]
    mismatches = [
        row
        for row in inventory
        if row["status"] in {"PRESENT_BUT_HASH_MISMATCH", "ARCHIVE_VALIDATION_FAILED"}
    ]
    source_level_ready = not missing_sources and not mismatches
    if mismatches:
        overall = "FAIL"
    elif missing_sources:
        overall = "INCOMPLETE_SOURCE_MISSING"
    elif any(row["status"] == "EXPECTED_AUXILIARY_MISSING" for row in inventory):
        overall = "PASS_WITH_AUXILIARY_GAPS"
    else:
        overall = "PASS"

    member_fields = [
        "container_filename",
        "member_path",
        "size_bytes",
        "sha256",
        "manifest_expected_sha256",
        "manifest_hash_match",
    ]
    inventory_fields = [
        "artifact_id",
        "expected_filename",
        "observed_path",
        "present_or_missing",
        "size_bytes",
        "sha256",
        "expected_sha256",
        "hash_match",
        "archive_integrity",
        "manifest_member_count",
        "extraction_result",
        "evidence_role",
        "required_source",
        "status",
        "validation_error",
    ]
    receipt = {
        "schema_version": "stage1.expert_delivery_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloads_dir": str(downloads),
        "output_dir": str(output),
        "status": overall,
        "source_level_audit_ready": source_level_ready,
        "expected_artifact_count": len(expected_specs),
        "observed_inventory_rows": len(inventory),
        "archive_member_rows": len(archive_members),
        "missing_required_source_count": len(missing_sources),
        "hash_or_archive_failure_count": len(mismatches),
        "missing_required_sources": [row["expected_filename"] for row in missing_sources],
        "archive_summaries": {name: result.summary for name, result in sorted(archive_results.items())},
        "archive_errors": archive_errors,
        "executed_delivered_code": False,
        "temporary_extractions_retained": False,
        "outputs": [
            "expert_v1_inventory.csv",
            "expert_archive_member_manifest.csv",
            "expert_v1_hash_validation.json",
            "README.md",
            "expert_audit_output_manifest.csv",
        ],
    }
    _atomic_csv(inventory_path, inventory, inventory_fields)
    _atomic_csv(member_path, archive_members, member_fields)
    _atomic_json(validation_path, receipt)
    missing_lines = "\n".join(
        f"- `{filename}`" for filename in receipt["missing_required_sources"]
    ) or "- None"
    readme = (
        "# Expert Delivery Audit\n\n"
        f"- Status: `{overall}`\n"
        f"- Expected artifact identities: {len(expected_specs)}\n"
        f"- Inventory rows: {len(inventory)}\n"
        f"- Archive member rows: {len(archive_members)}\n"
        f"- Required source artifacts missing: {len(missing_sources)}\n"
        f"- Hash or archive failures: {len(mismatches)}\n"
        "- Delivered code executed: no\n"
        "- Temporary extraction retained: no\n\n"
        "## Missing Required Source Artifacts\n\n"
        f"{missing_lines}\n\n"
        "A report or an archived code excerpt does not satisfy a missing source archive. "
        "The three missing BudgetedReplay source carriers keep source-level comparison blocked.\n\n"
        "## Reproduce\n\n"
        "```powershell\n"
        "uv run python scripts/stage1_dynamic_replay_v3/audit_expert_deliveries.py `\n"
        "  --downloads-dir C:\\Users\\28898\\Downloads `\n"
        "  --output-dir <NEW_VERSIONED_AUDIT_DIRECTORY>\n"
        "```\n\n"
        "Outputs are immutable by default. Use a new versioned directory for a later rerun.\n"
    )
    _atomic_text(readme_path, readme)
    manifest_rows = []
    for artifact in (readme_path, member_path, validation_path, inventory_path):
        manifest_rows.append(
            {
                "relative_path": artifact.name,
                "size_bytes": artifact.stat().st_size,
                "sha256": _sha256_path(artifact),
            }
        )
    _atomic_csv(
        output_manifest_path,
        manifest_rows,
        ["relative_path", "size_bytes", "sha256"],
    )
    return receipt
